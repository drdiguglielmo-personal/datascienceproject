"""
Binary (no-draw) model experiment.

Addresses Diego's presentation comment: "draws are the hardest class; try a
win/loss subset (e.g. knockout stage) to see how good the model is".

Two variants are reported:

  1. Binary all-stages: drop draws from train and test, predict
     1 = home team win, 0 = away team win on the remaining 2022 matches (n=54).

  2. Knockout-stage only: filter both train and test to knockout matches
     (round of 16 onward). World Cup knockouts are naturally binary because
     draws are resolved by extra time / PKs (the four historical "draws"
     pre-1942 from replayed matches are dropped). 2022 knockouts: n=16.

The temporal walk-forward split is intentionally identical to the 3-class
notebook (`Part2_Models_and_Results.ipynb`):
  * val_years = (2010, 2014, 2018), train = year < val_year
  * StandardScaler fit only on the training fold
  * 2022 = held-out test
Draw filtering is applied AFTER splitting so no future-tournament info
leaks into training.
"""

from __future__ import annotations

import os
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import RocCurveDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_CLEAN_DIR = PROJECT_ROOT / "data_clean"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_STATE = 42

# Mirror the 3-class notebook split exactly.
VAL_YEARS: tuple[int, ...] = (2010, 2014, 2018)
TEST_YEAR = 2022

# Reference numbers from the 3-class tuned RF on the same 2022 test set.
THREECLASS_RF_TEST_ACC = 0.625
THREECLASS_RF_TEST_F1 = 0.578

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def _pick_label(options: list[str], present: set[str]) -> str:
    for o in options:
        if o in present:
            return o
    raise ValueError(f"None of {options} found in result labels: {sorted(present)}")


def load_features() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df_train = pd.read_csv(DATA_CLEAN_DIR / "features_train.csv")
    df_test = pd.read_csv(DATA_CLEAN_DIR / "features_test.csv")

    # Same 39-feature selection as the 3-class notebook.
    drop_cols = ["match_date", "year", "home_team_name", "away_team_name", "result"]
    exclude_features = {
        "home_fifa_points", "away_fifa_points", "fifa_points_diff",
        "home_qual_win_rate", "away_qual_win_rate", "qual_win_rate_diff",
        "home_squad_value", "away_squad_value", "squad_value_diff",
        "home_rolling_xg", "away_rolling_xg", "rolling_xg_diff",
        "home_intl_elo", "away_intl_elo", "intl_elo_diff",
        "home_intl_rolling5_win_rate", "away_intl_rolling5_win_rate",
        "home_intl_rolling5_goals_pg", "away_intl_rolling5_goals_pg",
        "intl_h2h_home_wins", "intl_h2h_away_wins", "intl_h2h_draws",
        "intl_h2h_total", "intl_h2h_home_win_rate",
        "home_intl_hist_win_rate", "away_intl_hist_win_rate",
        "home_intl_hist_draw_rate", "away_intl_hist_draw_rate",
        "home_intl_hist_goals_per_game", "away_intl_hist_goals_per_game",
        "home_intl_hist_goals_conceded_per_game", "away_intl_hist_goals_conceded_per_game",
        "home_intl_hist_matches_played", "away_intl_hist_matches_played",
        "intl_hist_win_rate_diff", "intl_hist_goals_per_game_diff",
        "intl_elo_x_form_diff",
        "home_intl_attack_x_away_defense", "away_intl_attack_x_home_defense",
    }

    feature_cols = [c for c in df_train.columns if c not in drop_cols and c not in exclude_features]
    return df_train, df_test, feature_cols


def _binary_targets(df: pd.DataFrame, draw_label: str, home_win_label: str) -> tuple[pd.DataFrame, np.ndarray]:
    sub = df[df["result"] != draw_label].copy()
    y = (sub["result"] == home_win_label).astype(int).values
    return sub, y


def temporal_cv_binary(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    draw_label: str,
    home_win_label: str,
    model_fn,
    *,
    knockout_only: bool = False,
    val_years: tuple[int, ...] = VAL_YEARS,
) -> pd.DataFrame:
    rows: list[dict] = []
    for val_year in val_years:
        tr = df_train[df_train["year"] < val_year].copy()
        va = df_train[df_train["year"] == val_year].copy()

        if knockout_only:
            tr = tr[tr["is_knockout"] == 1]
            va = va[va["is_knockout"] == 1]

        # Filter draws AFTER the temporal split so 2022 info never leaks
        # into the training fold.
        tr = tr[tr["result"] != draw_label]
        va = va[va["result"] != draw_label]

        X_tr = tr[feature_cols].values
        y_tr = (tr["result"] == home_win_label).astype(int).values
        X_va = va[feature_cols].values
        y_va = (va["result"] == home_win_label).astype(int).values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = model_fn()
        model.fit(X_tr_s, y_tr)

        y_hat = model.predict(X_va_s)
        y_proba = model.predict_proba(X_va_s)[:, 1] if hasattr(model, "predict_proba") else None

        # Some folds may have only one class (e.g., very few knockout away wins
        # in a single tournament) which makes ROC-AUC undefined.
        if y_proba is not None and len(np.unique(y_va)) == 2:
            auc = float(roc_auc_score(y_va, y_proba))
        else:
            auc = float("nan")

        rows.append(
            {
                "val_year": val_year,
                "n_train": int(len(y_tr)),
                "n_val": int(len(y_va)),
                "accuracy": float(accuracy_score(y_va, y_hat)),
                "f1": float(f1_score(y_va, y_hat, zero_division=0)),
                "roc_auc": auc,
            }
        )

    return pd.DataFrame(rows)


def make_logreg() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_variant(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    draw_label: str,
    home_win_label: str,
    *,
    variant_name: str,
    knockout_only: bool,
) -> tuple[pd.DataFrame, dict]:
    """Run the binary experiment for one variant and return per-fold CV
    plus a summary dict with test metrics and predictions for plotting."""
    if knockout_only:
        train_pool = df_train[df_train["is_knockout"] == 1].copy()
        test_pool = df_test[df_test["is_knockout"] == 1].copy()
    else:
        train_pool = df_train.copy()
        test_pool = df_test.copy()

    train_bin, y_train = _binary_targets(train_pool, draw_label, home_win_label)
    test_bin, y_test = _binary_targets(test_pool, draw_label, home_win_label)

    X_train = train_bin[feature_cols].values
    X_test = test_bin[feature_cols].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    print(f"\n=== {variant_name} ===")
    print(
        f"  Train: n={len(train_bin)}  (home_win={int(y_train.sum())}, "
        f"away_win={int((1 - y_train).sum())})"
    )
    print(
        f"  Test:  n={len(test_bin)}   (home_win={int(y_test.sum())}, "
        f"away_win={int((1 - y_test).sum())})"
    )

    summary_rows: list[dict] = []
    fitted_models: dict[str, object] = {}
    fold_tables: dict[str, pd.DataFrame] = {}

    for name, fn in [
        ("Logistic Regression (balanced)", make_logreg),
        ("Random Forest (tuned, balanced)", make_rf),
    ]:
        tcv = temporal_cv_binary(
            df_train,
            feature_cols,
            draw_label,
            home_win_label,
            fn,
            knockout_only=knockout_only,
        )
        fold_tables[name] = tcv

        model = fn()
        model.fit(X_tr_s, y_train)
        y_hat = model.predict(X_te_s)
        y_proba = model.predict_proba(X_te_s)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat, zero_division=0)
        if y_proba is not None and len(np.unique(y_test)) == 2:
            auc = float(roc_auc_score(y_test, y_proba))
        else:
            auc = float("nan")

        print(f"\n  -- {name} --")
        print("  Temporal CV folds (no-draw):")
        print(
            tcv.to_string(
                index=False,
                formatters={k: "{:.3f}".format for k in ["accuracy", "f1", "roc_auc"]},
            )
        )
        print(
            "  Temporal means: "
            f"acc={tcv['accuracy'].mean():.3f}, "
            f"f1={tcv['f1'].mean():.3f}, "
            f"auc={tcv['roc_auc'].mean():.3f}"
        )
        print(f"  2022 test:      acc={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}")

        summary_rows.append(
            {
                "variant": variant_name,
                "model": name,
                "n_train": int(len(train_bin)),
                "n_test": int(len(test_bin)),
                "temporal_acc_mean": float(tcv["accuracy"].mean()),
                "temporal_f1_mean": float(tcv["f1"].mean()),
                "temporal_auc_mean": float(tcv["roc_auc"].mean()),
                "test_acc": float(acc),
                "test_f1": float(f1),
                "test_auc": auc,
            }
        )
        fitted_models[name] = model

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, {
        "models": fitted_models,
        "scaler": scaler,
        "X_test_scaled": X_te_s,
        "y_test": y_test,
        "fold_tables": fold_tables,
    }


def slice_evaluation(
    df_test: pd.DataFrame,
    feature_cols: list[str],
    draw_label: str,
    home_win_label: str,
    fitted_model,
    fitted_scaler: StandardScaler,
) -> dict:
    """Take a model trained on the all-stages no-draw set and evaluate it
    only on 2022 knockout matches. This is variant (b) without retraining."""
    ko_test = df_test[df_test["is_knockout"] == 1].copy()
    ko_test = ko_test[ko_test["result"] != draw_label]
    y_te = (ko_test["result"] == home_win_label).astype(int).values
    X_te = fitted_scaler.transform(ko_test[feature_cols].values)

    y_hat = fitted_model.predict(X_te)
    y_proba = fitted_model.predict_proba(X_te)[:, 1] if hasattr(fitted_model, "predict_proba") else None

    return {
        "n_test": int(len(ko_test)),
        "y_true": y_te,
        "y_pred": y_hat,
        "y_proba": y_proba,
        "test_acc": float(accuracy_score(y_te, y_hat)),
        "test_f1": float(f1_score(y_te, y_hat, zero_division=0)),
        "test_auc": float(roc_auc_score(y_te, y_proba))
        if y_proba is not None and len(np.unique(y_te)) == 2
        else float("nan"),
    }


def assert_split_matches_3class(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Sanity-check that the temporal split mirrors the 3-class notebook."""
    test_years = set(df_test["year"].unique())
    assert test_years == {TEST_YEAR}, f"Expected test set to be {TEST_YEAR}, got {sorted(test_years)}"
    for vy in VAL_YEARS:
        assert vy in df_train["year"].values, f"Validation year {vy} missing from training data"
        assert (df_train["year"] < vy).any(), f"No training data before {vy}"
    print("[split-check] Temporal split matches the 3-class notebook exactly:")
    print(f"  val_years = {VAL_YEARS}, test_year = {TEST_YEAR}")
    print(f"  train years range: {df_train['year'].min()}-{df_train['year'].max()}")
    print(f"  test years        : {sorted(test_years)}")


def plot_metric_comparison(all_summaries: pd.DataFrame, out_path: pathlib.Path) -> None:
    """Grouped bar chart comparing variants on test acc / f1 / auc."""
    melted = all_summaries.melt(
        id_vars=["variant", "model"],
        value_vars=["test_acc", "test_f1", "test_auc"],
        var_name="metric",
        value_name="value",
    )
    metric_titles = {
        "test_acc": "2022 Test Accuracy",
        "test_f1": "2022 Test F1",
        "test_auc": "2022 Test ROC-AUC",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric in zip(axes, ["test_acc", "test_f1", "test_auc"]):
        sub = melted[melted["metric"] == metric]
        sns.barplot(data=sub, x="variant", y="value", hue="model", ax=ax)
        ax.set_title(metric_titles[metric])
        ax.set_xlabel("")
        ax.set_ylabel(metric_titles[metric])
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        if metric == "test_acc":
            # Reference line: 3-class tuned RF achieved 62.5% on the SAME 2022 test set.
            ax.axhline(
                THREECLASS_RF_TEST_ACC,
                color="black",
                linestyle=":",
                linewidth=1.2,
                label=f"3-class RF baseline ({THREECLASS_RF_TEST_ACC:.1%})",
            )
            ax.legend(loc="lower right", fontsize=8)
        elif metric == "test_f1":
            ax.axhline(
                THREECLASS_RF_TEST_F1,
                color="black",
                linestyle=":",
                linewidth=1.2,
                label=f"3-class RF baseline ({THREECLASS_RF_TEST_F1:.2f})",
            )
            ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(
    summaries: list[tuple[str, dict]],
    out_path: pathlib.Path,
) -> None:
    """ROC curves for each variant on its own 2022 test slice."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for variant_name, info in summaries:
        for model_name, model in info["models"].items():
            y_test = info["y_test"]
            X_test = info["X_test_scaled"]
            if len(np.unique(y_test)) < 2:
                continue
            label = f"{variant_name} :: {model_name.split(' ')[0]}"
            RocCurveDisplay.from_estimator(model, X_test, y_test, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title("ROC Curves — 2022 test (binary variants)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_knockout_confusion(slice_info: dict, out_path: pathlib.Path) -> None:
    cm = confusion_matrix(slice_info["y_true"], slice_info["y_pred"], labels=[1, 0])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["pred home", "pred away"],
        yticklabels=["actual home", "actual away"],
        ax=ax,
    )
    ax.set_title(
        f"2022 knockout-only confusion (n={slice_info['n_test']})\n"
        f"acc={slice_info['test_acc']:.3f}, f1={slice_info['test_f1']:.3f}, "
        f"auc={slice_info['test_auc']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df_train, df_test, feature_cols = load_features()
    assert_split_matches_3class(df_train, df_test)
    print(f"[features] using {len(feature_cols)} feature columns")

    labels = set(df_train["result"].unique())
    draw_label = _pick_label(["draw", "Draw"], labels)
    home_win_label = _pick_label(["home team win", "Home team win"], labels)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    # Variant 1: binary, all stages (existing experiment).
    summary_all, info_all = evaluate_variant(
        df_train,
        df_test,
        feature_cols,
        draw_label,
        home_win_label,
        variant_name="Binary all-stages",
        knockout_only=False,
    )

    # Variant 2: binary, knockout stage only (literal reviewer suggestion).
    summary_ko, info_ko = evaluate_variant(
        df_train,
        df_test,
        feature_cols,
        draw_label,
        home_win_label,
        variant_name="Binary knockout-only",
        knockout_only=True,
    )

    # Slice analysis: model trained on full no-draw set, evaluated on 2022 knockouts.
    rf_full = info_all["models"]["Random Forest (tuned, balanced)"]
    slice_info = slice_evaluation(
        df_test,
        feature_cols,
        draw_label,
        home_win_label,
        rf_full,
        info_all["scaler"],
    )
    print("\n=== Slice: trained on all-stages no-draw, tested on 2022 knockouts only ===")
    print(
        f"  n_test={slice_info['n_test']}  "
        f"acc={slice_info['test_acc']:.3f}  "
        f"f1={slice_info['test_f1']:.3f}  "
        f"auc={slice_info['test_auc']:.3f}"
    )

    # Combined summary table.
    all_summary = pd.concat([summary_all, summary_ko], ignore_index=True)
    slice_row = pd.DataFrame(
        [
            {
                "variant": "Slice: full-train → 2022-KO",
                "model": "Random Forest (tuned, balanced)",
                "n_train": int(summary_all.iloc[1]["n_train"]),
                "n_test": slice_info["n_test"],
                "temporal_acc_mean": float("nan"),
                "temporal_f1_mean": float("nan"),
                "temporal_auc_mean": float("nan"),
                "test_acc": slice_info["test_acc"],
                "test_f1": slice_info["test_f1"],
                "test_auc": slice_info["test_auc"],
            }
        ]
    )
    full_summary = pd.concat([all_summary, slice_row], ignore_index=True)

    print("\n=== Final comparison vs 3-class tuned RF baseline ===")
    print(
        f"  3-class RF on 2022 (n=64): acc={THREECLASS_RF_TEST_ACC:.3f}, "
        f"macro_f1={THREECLASS_RF_TEST_F1:.3f}"
    )
    print(
        full_summary[
            [
                "variant",
                "model",
                "n_train",
                "n_test",
                "test_acc",
                "test_f1",
                "test_auc",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    full_summary.to_csv(DATA_CLEAN_DIR / "binary_no_draw_summary.csv", index=False)

    # ----- Figures -----
    plot_metric_comparison(all_summary, FIGURES_DIR / "binary_no_draw_test_metrics.png")
    plot_roc_curves(
        [("All-stages", info_all), ("Knockout-only", info_ko)],
        FIGURES_DIR / "binary_no_draw_roc.png",
    )
    plot_knockout_confusion(slice_info, FIGURES_DIR / "binary_no_draw_knockout_confusion.png")

    print("\nSaved figures:")
    print(f"  - {FIGURES_DIR / 'binary_no_draw_test_metrics.png'}")
    print(f"  - {FIGURES_DIR / 'binary_no_draw_roc.png'}")
    print(f"  - {FIGURES_DIR / 'binary_no_draw_knockout_confusion.png'}")
    print(f"  - {DATA_CLEAN_DIR / 'binary_no_draw_summary.csv'}")


if __name__ == "__main__":
    main()
