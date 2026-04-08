"""
Phase 2: Expanded training on competitive international matches.

Instead of training on 900 World Cup matches with 37 WC-only features,
this script trains on ~27K competitive international matches with 31
features computed from the full 49K-match international history.

The key insight: adding more features to 900 rows caused overfitting.
Adding more ROWS makes those same features viable.

in:
  - data_clean/international_results.csv  (49K international matches)

out:
  - data_clean/features_expanded_train.csv  (~27K rows x 36 cols)
  - data_clean/features_expanded_test.csv   (64 rows x 36 cols, WC 2022)
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

# Import helpers from the existing feature engineering module
_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
import feature_engineering as fe  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_CLEAN_DIR = PROJECT_ROOT / "data_clean"

# ── Tournament classification ──────────────────────────────────────

CONTINENTAL_TOURNAMENTS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Gold Cup",
    "OFC Nations Cup",
    "Confederations Cup",
}

# Sample weights: WC matters most, then continental, then qualifiers.
# These are used during model training so that a WC match contributes
# more to the loss than, say, a CECAFA Cup match.
SAMPLE_WEIGHT_MAP = {
    "FIFA World Cup": 3.0,
    "Confederations Cup": 2.0,
    # Continental championships
    "UEFA Euro": 2.0,
    "Copa América": 2.0,
    "African Cup of Nations": 2.0,
    "AFC Asian Cup": 2.0,
    "CONCACAF Gold Cup": 2.0,
    "OFC Nations Cup": 1.5,
    # Qualifications
    "FIFA World Cup qualification": 1.5,
    "UEFA Euro qualification": 1.2,
    "Copa América qualification": 1.0,
    "African Cup of Nations qualification": 1.0,
    "AFC Asian Cup qualification": 1.0,
    "CONCACAF Gold Cup qualification": 1.0,
    "OFC Nations Cup qualification": 1.0,
    # Modern competitive
    "UEFA Nations League": 1.2,
}
DEFAULT_SAMPLE_WEIGHT = 0.8  # smaller regional tournaments

# ── Feature columns (31 intl features + 4 context) ────────────────

FEATURE_COLS = [
    # International ELO (3)
    "home_intl_elo",
    "away_intl_elo",
    "intl_elo_diff",
    # International rolling form — last 5 matches (4)
    "home_intl_rolling5_win_rate",
    "away_intl_rolling5_win_rate",
    "home_intl_rolling5_goals_pg",
    "away_intl_rolling5_goals_pg",
    # International head-to-head (5)
    "intl_h2h_home_wins",
    "intl_h2h_away_wins",
    "intl_h2h_draws",
    "intl_h2h_total",
    "intl_h2h_home_win_rate",
    # International historical performance (12)
    "home_intl_hist_win_rate",
    "away_intl_hist_win_rate",
    "home_intl_hist_draw_rate",
    "away_intl_hist_draw_rate",
    "home_intl_hist_goals_per_game",
    "away_intl_hist_goals_per_game",
    "home_intl_hist_goals_conceded_per_game",
    "away_intl_hist_goals_conceded_per_game",
    "home_intl_hist_matches_played",
    "away_intl_hist_matches_played",
    "intl_hist_win_rate_diff",
    "intl_hist_goals_per_game_diff",
    # Interaction features (3)
    "intl_elo_x_form_diff",
    "home_intl_attack_x_away_defense",
    "away_intl_attack_x_home_defense",
    # Match context (4)
    "is_neutral",
    "is_world_cup",
    "is_continental",
    "is_qualifying",
]

META_COLS = ["match_date", "home_team_name", "away_team_name", "tournament"]
TARGET_COL = "result"
WEIGHT_COL = "sample_weight"


# ── Fill missing values ────────────────────────────────────────────

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values for cold-start teams (same strategy as main pipeline)."""
    df = df.copy()

    fill_map = {
        # Rate features -> uniform 3-class prior
        "home_intl_hist_win_rate": 0.33,
        "away_intl_hist_win_rate": 0.33,
        "home_intl_hist_draw_rate": 0.33,
        "away_intl_hist_draw_rate": 0.33,
        "intl_h2h_home_win_rate": 0.33,
        "home_intl_rolling5_win_rate": 0.33,
        "away_intl_rolling5_win_rate": 0.33,
        # Count features -> zero
        "home_intl_hist_matches_played": 0,
        "away_intl_hist_matches_played": 0,
        "intl_h2h_home_wins": 0,
        "intl_h2h_away_wins": 0,
        "intl_h2h_draws": 0,
        "intl_h2h_total": 0,
        # Goals/rate features -> zero
        "home_intl_hist_goals_per_game": 0.0,
        "away_intl_hist_goals_per_game": 0.0,
        "home_intl_hist_goals_conceded_per_game": 0.0,
        "away_intl_hist_goals_conceded_per_game": 0.0,
        "home_intl_rolling5_goals_pg": 0.0,
        "away_intl_rolling5_goals_pg": 0.0,
        # Diff features -> zero
        "intl_hist_win_rate_diff": 0.0,
        "intl_hist_goals_per_game_diff": 0.0,
        # ELO -> starting value
        "home_intl_elo": 1500.0,
        "away_intl_elo": 1500.0,
        "intl_elo_diff": 0.0,
        # Interactions -> zero
        "intl_elo_x_form_diff": 0.0,
        "home_intl_attack_x_away_defense": 0.0,
        "away_intl_attack_x_home_defense": 0.0,
    }

    for col, val in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df


# ── Main pipeline ──────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── 1. Load all international results ──────────────────────────
    print("Loading international results...")
    intl = fe.load_international_data()
    if intl is None:
        print(f"ERROR: Place international_results.csv at {fe.INTL_DATA_PATH}")
        return
    print(f"  {len(intl)} matches ({intl['match_date'].min().date()} "
          f"to {intl['match_date'].max().date()})")

    # ── 2. Split train / test ──────────────────────────────────────
    # Train: competitive (non-friendly) matches before WC 2022
    # Test:  FIFA World Cup 2022 matches (same 64 as original pipeline)
    wc_2022_start = pd.Timestamp("2022-11-20")

    is_friendly = intl["tournament"].str.contains(
        "Friendly", case=False, na=False
    )

    train_mask = (~is_friendly) & (intl["match_date"] < wc_2022_start)
    train_raw = intl[train_mask].copy()

    # Use the ORIGINAL WC test set for fair comparison.
    # The intl dataset has different match ordering on same-day matches
    # and records penalty-shootout matches as draws, so we must use
    # the authoritative WC dataset for test labels.
    wc_test_path = DATA_CLEAN_DIR / "matches_test.csv"
    test_raw = pd.read_csv(wc_test_path)
    test_raw["match_date"] = pd.to_datetime(test_raw["match_date"])
    # Add columns needed for context features
    test_raw["tournament"] = "FIFA World Cup"
    test_raw["neutral"] = True  # all WC matches are at neutral venues except host

    print(f"\n  Train: {len(train_raw):,} competitive matches (before WC 2022)")
    print(f"  Test:  {len(test_raw)} FIFA World Cup 2022 matches (from WC dataset)")

    if len(test_raw) != 64:
        print(f"  WARNING: Expected 64 WC 2022 test matches, got {len(test_raw)}")

    # Show train composition
    print("\n  Train composition (top 10 tournament types):")
    for t, n in train_raw["tournament"].value_counts().head(10).items():
        pct = n / len(train_raw) * 100
        print(f"    {t:45s} {n:6,} ({pct:4.1f}%)")

    # ── 3. Combine for feature computation ─────────────────────────
    train_raw["_split"] = "train"
    test_raw["_split"] = "test"
    df = pd.concat([train_raw, test_raw], ignore_index=True)
    df = df.sort_values("match_date").reset_index(drop=True)

    # ── 4. Compute intl features from full 49K history ─────────────
    # The lookup structures are built from ALL matches (including
    # friendlies — they contribute to ELO and form even though we
    # don't train on them).  Queries use bisect to ensure only
    # pre-match data is used (no leakage).

    print("\nComputing international ELO ratings...")
    df = fe.compute_intl_elo(df, intl)

    print("Computing international rolling form (last 5 matches)...")
    df = fe.compute_intl_rolling_form(df, intl)

    print("Computing international head-to-head records...")
    df = fe.compute_intl_h2h(df, intl)

    print("Computing international historical performance...")
    df = fe.compute_intl_history(df, intl)

    # ── 5. Interaction features ────────────────────────────────────
    print("Computing interaction features...")
    df["intl_elo_x_form_diff"] = (
        df["intl_elo_diff"]
        * (df["home_intl_rolling5_win_rate"] - df["away_intl_rolling5_win_rate"])
    )
    df["home_intl_attack_x_away_defense"] = (
        df["home_intl_hist_goals_per_game"]
        * df["away_intl_hist_goals_conceded_per_game"]
    )
    df["away_intl_attack_x_home_defense"] = (
        df["away_intl_hist_goals_per_game"]
        * df["home_intl_hist_goals_conceded_per_game"]
    )

    # ── 6. Context features ────────────────────────────────────────
    print("Adding context features...")
    df["is_neutral"] = df["neutral"].astype(int) if "neutral" in df.columns else 0
    df["is_world_cup"] = (df["tournament"] == "FIFA World Cup").astype(int)
    df["is_continental"] = df["tournament"].isin(CONTINENTAL_TOURNAMENTS).astype(int)
    df["is_qualifying"] = df["tournament"].str.contains(
        "qualification", case=False, na=False
    ).astype(int)

    # ── 7. Sample weights ──────────────────────────────────────────
    df[WEIGHT_COL] = (
        df["tournament"].map(SAMPLE_WEIGHT_MAP).fillna(DEFAULT_SAMPLE_WEIGHT)
    )

    # ── 8. Fill missing values ─────────────────────────────────────
    print("Filling missing values...")
    df = fill_missing(df)

    # ── 9. Build output and save ───────────────────────────────────
    output_cols = META_COLS + FEATURE_COLS + [TARGET_COL, WEIGHT_COL]
    output = df[output_cols].copy()

    train_out = output[df["_split"] == "train"].reset_index(drop=True)
    test_out = output[df["_split"] == "test"].reset_index(drop=True)

    train_path = DATA_CLEAN_DIR / "features_expanded_train.csv"
    test_path = DATA_CLEAN_DIR / "features_expanded_test.csv"

    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    elapsed = time.time() - t0
    print(f"\nSaved: {train_path}  {train_out.shape}")
    print(f"Saved: {test_path}  {test_out.shape}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Time: {elapsed:.1f}s")

    # ── 10. Validation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    train_nan = train_out[FEATURE_COLS].isna().sum().sum()
    test_nan = test_out[FEATURE_COLS].isna().sum().sum()
    print(f"\nNaN — train: {train_nan}, test: {test_nan}")

    print("\nTrain class distribution:")
    for cls in ["home team win", "away team win", "draw"]:
        n = (train_out[TARGET_COL] == cls).sum()
        print(f"  {cls:15s}  {n:6,}  ({n / len(train_out):.1%})")

    print(f"\nTest class distribution:")
    for cls in ["home team win", "away team win", "draw"]:
        n = (test_out[TARGET_COL] == cls).sum()
        print(f"  {cls:15s}  {n:3}  ({n / len(test_out):.1%})")

    # ── 11. Model evaluation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    X_train = train_out[FEATURE_COLS].values
    X_test = test_out[FEATURE_COLS].values
    y_train = train_out[TARGET_COL].values
    y_test = test_out[TARGET_COL].values
    sw = train_out[WEIGHT_COL].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── RF without sample weights ──
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    rf.fit(X_train_s, y_train)
    y_pred = rf.predict(X_test_s)
    print(f"\n--- Random Forest (balanced, no sample weights) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
    print(classification_report(y_test, y_pred, digits=3))

    # ── RF with tournament sample weights ──
    rf_w = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    rf_w.fit(X_train_s, y_train, sample_weight=sw)
    y_pred_w = rf_w.predict(X_test_s)
    print(f"--- Random Forest (balanced + tournament weights) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_w):.3f}")
    print(f"Macro F1: {f1_score(y_test, y_pred_w, average='macro'):.3f}")
    print(classification_report(y_test, y_pred_w, digits=3))

    # ── XGBoost ──
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)

        # With tournament weights + draw upweighting
        sw_draw = sw.copy()
        sw_draw[y_train == "draw"] *= 1.5

        xgb = XGBClassifier(
            max_depth=4,
            n_estimators=300,
            learning_rate=0.05,
            reg_alpha=0.5,
            reg_lambda=2.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
        )
        xgb.fit(X_train_s, y_tr_enc, sample_weight=sw_draw)
        y_pred_xgb = le.inverse_transform(xgb.predict(X_test_s))
        print(f"--- XGBoost (tournament weights + draw 1.5x) ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.3f}")
        print(f"Macro F1: {f1_score(y_test, y_pred_xgb, average='macro'):.3f}")
        print(classification_report(y_test, y_pred_xgb, digits=3))
    except ImportError:
        print("\nXGBoost not installed — skipping.")

    # ── Feature importance (from weighted RF) ──
    importances = rf_w.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    print("--- Top 10 features (RF with tournament weights) ---")
    for name, imp in feat_imp[:10]:
        print(f"  {imp:.4f}  {name}")

    # ── Comparison with original pipeline ──
    print("\n" + "=" * 60)
    print("COMPARISON WITH ORIGINAL PIPELINE")
    print("=" * 60)
    print(f"\n{'':30s}  {'Original':>10s}  {'Expanded':>10s}")
    print(f"{'Training rows':30s}  {'900':>10s}  {f'{len(train_out):,}':>10s}")
    print(f"{'Features':30s}  {'37':>10s}  {f'{len(FEATURE_COLS)}':>10s}")
    print(f"{'RF Test Accuracy':30s}  {'0.609':>10s}  {accuracy_score(y_test, y_pred_w):.3f}{'':>5s}")
    print(f"{'RF Test Macro F1':30s}  {'0.556':>10s}  {f1_score(y_test, y_pred_w, average='macro'):.3f}{'':>5s}")


if __name__ == "__main__":
    main()
