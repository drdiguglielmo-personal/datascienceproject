"""
Binary (no-draw) model experiment.

This script trains and evaluates a second model where draws are removed and the
target is binary:
  - 1 = home team win
  - 0 = away team win

It reuses the engineered feature CSVs produced by scripts/feature_engineering.py.
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
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_CLEAN_DIR = PROJECT_ROOT / "data_clean"
FIGURES_DIR = PROJECT_ROOT / "figures"

RANDOM_STATE = 42

# Ensure matplotlib/fontconfig caches are writable inside the project.
# This prevents failures in sandboxed or restricted environments.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

# Import plotting libraries after environment is set
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

    # Keep feature selection consistent with the Part 2 notebook
    drop_cols = ["match_date", "year", "home_team_name", "away_team_name", "result"]
    exclude_features = {
        # FIFA rankings
        "home_fifa_points", "away_fifa_points", "fifa_points_diff",
        # Qualifying record
        "home_qual_win_rate", "away_qual_win_rate", "qual_win_rate_diff",
        # Squad market value
        "home_squad_value", "away_squad_value", "squad_value_diff",
        # StatsBomb xG
        "home_rolling_xg", "away_rolling_xg", "rolling_xg_diff",
        # International features
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


def temporal_cv_binary(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    draw_label: str,
    home_win_label: str,
    model_fn,
    val_years: tuple[int, ...] = (2010, 2014, 2018),
) -> pd.DataFrame:
    rows: list[dict] = []
    for val_year in val_years:
        tr = df_train[df_train["year"] < val_year].copy()
        va = df_train[df_train["year"] == val_year].copy()

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

        rows.append(
            {
                "val_year": val_year,
                "n_val": int(len(y_va)),
                "accuracy": float(accuracy_score(y_va, y_hat)),
                "f1": float(f1_score(y_va, y_hat)),
                "roc_auc": float(roc_auc_score(y_va, y_proba)) if y_proba is not None else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    # Some environments emit numeric overflow warnings during linear algebra even when
    # the final metrics are computed successfully. Keep output focused on results.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df_train, df_test, feature_cols = load_features()

    labels = set(df_train["result"].unique())
    draw_label = _pick_label(["draw", "Draw"], labels)
    home_win_label = _pick_label(["home team win", "Home team win"], labels)

    # Filter draws
    train_bin = df_train[df_train["result"] != draw_label].copy()
    test_bin = df_test[df_test["result"] != draw_label].copy()

    X_train = train_bin[feature_cols].values
    y_train = (train_bin["result"] == home_win_label).astype(int).values
    X_test = test_bin[feature_cols].values
    y_test = (test_bin["result"] == home_win_label).astype(int).values

    print("Binary dataset (draws removed)")
    print(f"  Train rows: {len(train_bin)} | home wins: {int(y_train.sum())} | away wins: {int((1-y_train).sum())}")
    print(f"  Test rows:  {len(test_bin)} | home wins: {int(y_test.sum())} | away wins: {int((1-y_test).sum())}")

    def make_logreg():
        # Use liblinear to avoid occasional overflow warnings on small/collinear datasets
        return LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    def make_rf():
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    models = [
        ("Logistic Regression (balanced)", make_logreg),
        ("Random Forest (tuned, balanced)", make_rf),
    ]

    # Ensure figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set(style="whitegrid")

    summary_rows: list[dict] = []

    for name, fn in models:
        print(f"\n== {name} ==")
        tcv = temporal_cv_binary(df_train, feature_cols, draw_label, home_win_label, fn)
        print("Temporal CV (no-draw) by year:")
        print(tcv.to_string(index=False, formatters={k: "{:.3f}".format for k in ["accuracy", "f1", "roc_auc"]}))
        print(
            "Temporal means:",
            f"acc={tcv['accuracy'].mean():.3f}, f1={tcv['f1'].mean():.3f}, auc={tcv['roc_auc'].mean():.3f}",
        )

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        model = fn()
        model.fit(X_tr_s, y_train)
        y_hat = model.predict(X_te_s)
        y_proba = model.predict_proba(X_te_s)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        print(f"Test (2022, no-draw): acc={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}")

        summary_rows.append(
            {
                "model": name,
                "temporal_acc_mean": float(tcv["accuracy"].mean()),
                "temporal_f1_mean": float(tcv["f1"].mean()),
                "temporal_auc_mean": float(tcv["roc_auc"].mean()),
                "test_acc": float(acc),
                "test_f1": float(f1),
                "test_auc": float(auc),
            }
        )

    # --- Save visuals (test metrics bar chart + ROC curves) ---
    summary = pd.DataFrame(summary_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(
        axes,
        ["test_acc", "test_f1", "test_auc"],
        ["2022 Test Accuracy (no-draw)", "2022 Test F1 (no-draw)", "2022 Test ROC-AUC (no-draw)"],
    ):
        sns.barplot(data=summary, x="model", y=metric, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    out_bar = FIGURES_DIR / "binary_no_draw_test_metrics.png"
    fig.savefig(out_bar, dpi=200)
    plt.close(fig)

    # ROC curves on 2022 test (no-draw)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, fn in models:
        m = fn()
        m.fit(X_tr_s, y_train)
        RocCurveDisplay.from_estimator(m, X_te_s, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title("ROC Curves (2022 test, draws removed)")
    fig.tight_layout()
    out_roc = FIGURES_DIR / "binary_no_draw_roc.png"
    fig.savefig(out_roc, dpi=200)
    plt.close(fig)

    print("\nSaved figures:")
    print(f"  - {out_bar}")
    print(f"  - {out_roc}")


if __name__ == "__main__":
    main()

