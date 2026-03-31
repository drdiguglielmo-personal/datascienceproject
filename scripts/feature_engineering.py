"""
Feature engineering for World Cup match prediction.

Computes leakage-safe historical features using expanding windows
with shift(1) so that only past matches inform the current row.

in:
  - data_clean/matches_train.csv   (900 rows, 1930-2018)
  - data_clean/matches_test.csv    (64 rows, 2022)

out:
  - data_clean/features_train.csv  (engineered features + result + metadata)
  - data_clean/features_test.csv   (same structure)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_CLEAN_DIR = PROJECT_ROOT / "data_clean"

# ---------------------------------------------------------------------------
# Post-match columns that must NOT appear in the final output.
# They are used only to derive historical features.
# ---------------------------------------------------------------------------
POST_MATCH_COLS = [
    "home_team_score",
    "away_team_score",
    "home_team_score_margin",
    "away_team_score_margin",
    "extra_time",
    "penalty_shootout",
    "score_penalties",
    "home_team_score_penalties",
    "away_team_score_penalties",
    "home_team_win",
    "away_team_win",
    "draw",
    "score",
]

# ---------------------------------------------------------------------------
# Stage ordinal mapping
# ---------------------------------------------------------------------------
STAGE_ORDINAL = {
    "group stage": 0,
    "second group stage": 1,
    "final round": 2,
    "round of 16": 2,
    "quarter-finals": 3,
    "semi-finals": 4,
    "third-place match": 5,
    "final": 6,
}

# ---------------------------------------------------------------------------
# Host country name -> team name overrides for known mismatches.
# In our data the names already match, but we keep this map for safety.
# ---------------------------------------------------------------------------
HOST_TEAM_ALIASES: dict[str, list[str]] = {
    "South Korea": ["South Korea", "Korea Republic"],
}


# ===================================================================
# 1. Load and concatenate
# ===================================================================
def load_data() -> pd.DataFrame:
    """Load train + test, concatenate, and sort chronologically."""
    train = pd.read_csv(DATA_CLEAN_DIR / "matches_train.csv")
    test = pd.read_csv(DATA_CLEAN_DIR / "matches_test.csv")

    train["_split"] = "train"
    test["_split"] = "test"

    df = pd.concat([train, test], ignore_index=True)
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)
    return df


# ===================================================================
# 2a. Team historical performance (expanding window)
# ===================================================================
def compute_team_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build long-form (one row per team-match), compute expanding-window
    aggregates shifted by 1, then pivot back to wide form and join.

    Returns the original df with new historical columns attached.
    """
    # Create a unique row id to join back later
    df = df.copy()
    df["_row_id"] = range(len(df))

    # ----- build long form: home perspective -----
    home = df[["_row_id", "match_date", "home_team_name",
               "home_team_score", "away_team_score",
               "home_team_win", "away_team_win", "draw"]].copy()
    home = home.rename(columns={
        "home_team_name": "team",
        "home_team_score": "goals_for",
        "away_team_score": "goals_against",
        "home_team_win": "win",
        "away_team_win": "loss",
        "draw": "is_draw",
    })
    home["perspective"] = "home"

    # ----- away perspective -----
    away = df[["_row_id", "match_date", "away_team_name",
               "away_team_score", "home_team_score",
               "away_team_win", "home_team_win", "draw"]].copy()
    away = away.rename(columns={
        "away_team_name": "team",
        "away_team_score": "goals_for",
        "home_team_score": "goals_against",
        "away_team_win": "win",
        "home_team_win": "loss",
        "draw": "is_draw",
    })
    away["perspective"] = "away"

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values("match_date").reset_index(drop=True)

    # Aggregate each team's stats per date first, then compute a
    # date-level cumulative sum and shift by one *date* so that all
    # matches on the same day see only stats from strictly earlier dates.
    cumulative_cols = ["win", "loss", "is_draw", "goals_for", "goals_against"]
    for col in cumulative_cols:
        long[col] = long[col].astype(float)

    # Sum stats per (team, match_date) -- handles multiple matches on same day
    daily = (
        long.groupby(["team", "match_date"])[cumulative_cols]
        .sum()
        .sort_index(level="match_date")
    )

    # Cumulative sum within each team, shifted by 1 date-slot
    daily_cum = daily.groupby(level="team").cumsum().groupby(level="team").shift(1)

    # Map the date-level cumulative sums back to every row in long
    long = long.merge(
        daily_cum.rename(columns={c: f"cum_{c}" for c in cumulative_cols}),
        left_on=["team", "match_date"],
        right_index=True,
        how="left",
    )

    long["cum_wins"] = long["cum_win"]
    long["cum_draws"] = long["cum_is_draw"]
    long["cum_goals_for"] = long["cum_goals_for"]
    long["cum_goals_against"] = long["cum_goals_against"]
    long["cum_matches"] = long["cum_win"] + long["cum_loss"] + long["cum_is_draw"]

    # Derive rates
    long["hist_win_rate"] = long["cum_wins"] / long["cum_matches"]
    long["hist_draw_rate"] = long["cum_draws"] / long["cum_matches"]
    long["hist_goals_per_game"] = long["cum_goals_for"] / long["cum_matches"]
    long["hist_goals_conceded_per_game"] = long["cum_goals_against"] / long["cum_matches"]
    long["hist_matches_played"] = long["cum_matches"]

    feature_cols = [
        "hist_win_rate",
        "hist_draw_rate",
        "hist_goals_per_game",
        "hist_goals_conceded_per_game",
        "hist_matches_played",
    ]

    # Split back to home / away and merge
    home_feats = long.loc[long["perspective"] == "home", ["_row_id"] + feature_cols].copy()
    home_feats = home_feats.rename(columns={c: f"home_{c}" for c in feature_cols})

    away_feats = long.loc[long["perspective"] == "away", ["_row_id"] + feature_cols].copy()
    away_feats = away_feats.rename(columns={c: f"away_{c}" for c in feature_cols})

    df = df.merge(home_feats, on="_row_id", how="left")
    df = df.merge(away_feats, on="_row_id", how="left")

    # Difference features
    df["hist_win_rate_diff"] = df["home_hist_win_rate"] - df["away_hist_win_rate"]
    df["hist_goals_per_game_diff"] = (
        df["home_hist_goals_per_game"] - df["away_hist_goals_per_game"]
    )

    df = df.drop(columns=["_row_id"])
    return df


# ===================================================================
# 2b. Match context
# ===================================================================
def compute_match_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add stage_ordinal (is_group_stage and is_knockout already exist)."""
    df = df.copy()
    df["is_group_stage"] = df["group_stage"]
    df["is_knockout"] = df["knockout_stage"]
    df["stage_ordinal"] = df["stage_name"].map(STAGE_ORDINAL).fillna(0).astype(int)
    return df


# ===================================================================
# 2c. Host advantage
# ===================================================================
def compute_host_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """Flag whether home or away team is the host nation."""
    df = df.copy()

    # Build a set of acceptable team names for each host country
    def _team_matches_host(team_name: str, country_name: str) -> int:
        if team_name == country_name:
            return 1
        aliases = HOST_TEAM_ALIASES.get(country_name, [])
        return 1 if team_name in aliases else 0

    df["home_is_host"] = df.apply(
        lambda r: _team_matches_host(r["home_team_name"], r["country_name"]), axis=1
    )
    df["away_is_host"] = df.apply(
        lambda r: _team_matches_host(r["away_team_name"], r["country_name"]), axis=1
    )
    return df


# ===================================================================
# 2d. Team World Cup experience (distinct prior tournament years)
# ===================================================================
def compute_wc_experience(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team in each row, count how many distinct World Cup years
    that team appeared in *before* the current tournament year.
    """
    df = df.copy()

    # Build a mapping: team -> sorted list of years they appeared in
    home_appearances = df[["home_team_name", "year"]].rename(
        columns={"home_team_name": "team"}
    )
    away_appearances = df[["away_team_name", "year"]].rename(
        columns={"away_team_name": "team"}
    )
    all_appearances = pd.concat([home_appearances, away_appearances], ignore_index=True)
    team_years = all_appearances.groupby("team")["year"].apply(
        lambda s: sorted(s.unique())
    ).to_dict()

    def _prior_appearances(team: str, current_year: int) -> int:
        years = team_years.get(team, [])
        return sum(1 for y in years if y < current_year)

    df["home_wc_appearances"] = df.apply(
        lambda r: _prior_appearances(r["home_team_name"], r["year"]), axis=1
    )
    df["away_wc_appearances"] = df.apply(
        lambda r: _prior_appearances(r["away_team_name"], r["year"]), axis=1
    )
    df["wc_appearances_diff"] = df["home_wc_appearances"] - df["away_wc_appearances"]
    return df


# ===================================================================
# 2e. Head-to-head record
# ===================================================================
def compute_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, count prior meetings between the two teams
    (regardless of who was home/away in those prior matches).
    Compute h2h_home_wins, h2h_away_wins, h2h_draws, h2h_total,
    and h2h_home_win_rate.
    """
    df = df.copy()
    df = df.sort_values("match_date").reset_index(drop=True)

    # Precompute pair key for every row
    # Normalise pair as a frozenset so home/away order doesn't matter
    pair_keys = []
    for _, row in df.iterrows():
        pair_keys.append(frozenset([row["home_team_name"], row["away_team_name"]]))
    df["_pair"] = pair_keys

    # For each row, we need to know who won from the *current* match's
    # home team perspective across all prior meetings.
    h2h_home_wins = []
    h2h_away_wins = []
    h2h_draws = []

    # Accumulator: (team_a, team_b) frozen -> list of (home_team, result)
    history: dict[frozenset, list[tuple[str, str, str]]] = {}

    for idx, row in df.iterrows():
        pair = row["_pair"]
        home_team = row["home_team_name"]
        away_team = row["away_team_name"]

        prior = history.get(pair, [])

        hw = 0
        aw = 0
        dr = 0
        for prev_home, prev_away, prev_result in prior:
            if prev_result == "draw":
                dr += 1
            elif prev_result == "home team win":
                # prev_home won; is that our current home or away?
                if prev_home == home_team:
                    hw += 1
                else:
                    aw += 1
            elif prev_result == "away team win":
                # prev_away won
                if prev_away == home_team:
                    hw += 1
                else:
                    aw += 1

        h2h_home_wins.append(hw)
        h2h_away_wins.append(aw)
        h2h_draws.append(dr)

        # Record current match for future rows
        if pair not in history:
            history[pair] = []
        history[pair].append((home_team, away_team, row["result"]))

    df["h2h_home_wins"] = h2h_home_wins
    df["h2h_away_wins"] = h2h_away_wins
    df["h2h_draws"] = h2h_draws
    df["h2h_total"] = df["h2h_home_wins"] + df["h2h_away_wins"] + df["h2h_draws"]
    df["h2h_home_win_rate"] = (df["h2h_home_wins"] / df["h2h_total"]).fillna(0.33)

    df = df.drop(columns=["_pair"])
    return df


# ===================================================================
# 2f. ELO ratings
# ===================================================================
def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ELO ratings for each team using match results.

    All teams start at 1500. For each match (sorted chronologically),
    the pre-match ELO is recorded as the feature, then ratings are
    updated based on the result. K-factor is 32.

    Produces columns: home_elo, away_elo, elo_diff.
    """
    df = df.copy()
    df = df.sort_values("match_date").reset_index(drop=True)

    K = 32
    ratings: dict[str, float] = {}

    home_elo_list: list[float] = []
    away_elo_list: list[float] = []

    for _, row in df.iterrows():
        home_team = row["home_team_name"]
        away_team = row["away_team_name"]

        # Pre-match ratings (feature values)
        r_home = ratings.get(home_team, 1500.0)
        r_away = ratings.get(away_team, 1500.0)
        home_elo_list.append(r_home)
        away_elo_list.append(r_away)

        # Expected scores
        e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))
        e_away = 1.0 / (1.0 + 10.0 ** ((r_home - r_away) / 400.0))

        # Actual scores based on result
        result = row["result"]
        if result == "home team win":
            s_home = 1.0
            s_away = 0.0
        elif result == "away team win":
            s_home = 0.0
            s_away = 1.0
        else:  # draw
            s_home = 0.5
            s_away = 0.5

        # Update ratings
        ratings[home_team] = r_home + K * (s_home - e_home)
        ratings[away_team] = r_away + K * (s_away - e_away)

    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    return df


# ===================================================================
# 2g. Rolling recent form (last 5 WC matches)
# ===================================================================
def compute_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute win rate and goals per game over their
    last 5 World Cup matches prior to the current match.

    Uses a long-form approach: builds per-team match history sorted
    chronologically, computes rolling(5) with shift(1) to avoid
    leakage, then merges back.

    Produces: home_rolling5_win_rate, away_rolling5_win_rate,
              home_rolling5_goals_pg, away_rolling5_goals_pg.
    """
    df = df.copy()
    df["_row_id"] = range(len(df))

    # ----- build long form: home perspective -----
    home = df[["_row_id", "match_date", "home_team_name",
               "home_team_score", "home_team_win"]].copy()
    home = home.rename(columns={
        "home_team_name": "team",
        "home_team_score": "goals_for",
        "home_team_win": "win",
    })
    home["perspective"] = "home"

    # ----- away perspective -----
    away = df[["_row_id", "match_date", "away_team_name",
               "away_team_score", "away_team_win"]].copy()
    away = away.rename(columns={
        "away_team_name": "team",
        "away_team_score": "goals_for",
        "away_team_win": "win",
    })
    away["perspective"] = "away"

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values("match_date").reset_index(drop=True)

    for col in ["win", "goals_for"]:
        long[col] = long[col].astype(float)

    # For each team, compute rolling(5) win rate and goals per game,
    # shifted by 1 so the current match is excluded.
    long = long.sort_values(["team", "match_date"]).reset_index(drop=True)

    long["rolling5_win_rate"] = (
        long.groupby("team")["win"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )
    long["rolling5_goals_pg"] = (
        long.groupby("team")["goals_for"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # Compute all-time rate per team (for cold-start fallback)
    team_alltime_win = long.groupby("team")["win"].transform("mean")
    team_alltime_goals = long.groupby("team")["goals_for"].transform("mean")

    # Fill NaN (teams with no prior matches) with all-time rate, then
    # fall back to 0.33 / 0.0 for complete cold-start
    long["rolling5_win_rate"] = (
        long["rolling5_win_rate"]
        .fillna(team_alltime_win)
        .fillna(0.33)
    )
    long["rolling5_goals_pg"] = (
        long["rolling5_goals_pg"]
        .fillna(team_alltime_goals)
        .fillna(0.0)
    )

    rolling_cols = ["rolling5_win_rate", "rolling5_goals_pg"]

    # Split back to home / away and merge
    home_feats = long.loc[
        long["perspective"] == "home", ["_row_id"] + rolling_cols
    ].copy()
    home_feats = home_feats.rename(columns={
        "rolling5_win_rate": "home_rolling5_win_rate",
        "rolling5_goals_pg": "home_rolling5_goals_pg",
    })

    away_feats = long.loc[
        long["perspective"] == "away", ["_row_id"] + rolling_cols
    ].copy()
    away_feats = away_feats.rename(columns={
        "rolling5_win_rate": "away_rolling5_win_rate",
        "rolling5_goals_pg": "away_rolling5_goals_pg",
    })

    df = df.merge(home_feats, on="_row_id", how="left")
    df = df.merge(away_feats, on="_row_id", how="left")

    df = df.drop(columns=["_row_id"])
    return df


# ===================================================================
# 2h. Rest days
# ===================================================================
def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute the number of days since each team's
    previous World Cup match. Only matches with a strictly earlier
    match_date are considered.

    Produces: home_rest_days, away_rest_days.
    NaN (first match ever) is filled with the median rest days.
    Values are capped at 365 to avoid extreme outliers.
    """
    df = df.copy()
    df = df.sort_values("match_date").reset_index(drop=True)

    # Build long form to track each team's match dates
    home = df[["match_date", "home_team_name"]].rename(
        columns={"home_team_name": "team"}
    )
    away = df[["match_date", "away_team_name"]].rename(
        columns={"away_team_name": "team"}
    )
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values("match_date").reset_index(drop=True)

    # For each team, compute days since their previous match
    long["prev_date"] = long.groupby("team")["match_date"].shift(1)
    long["rest_days"] = (long["match_date"] - long["prev_date"]).dt.days

    # Build lookup: for each (team, match_date), the rest days
    # Use the long-form data to create a mapping
    # Since a team could play multiple matches on the same date (rare),
    # we take the minimum rest days in that case.
    rest_lookup = (
        long.dropna(subset=["rest_days"])
        .groupby(["team", "match_date"])["rest_days"]
        .min()
        .to_dict()
    )

    home_rest = []
    away_rest = []
    for _, row in df.iterrows():
        h_rest = rest_lookup.get((row["home_team_name"], row["match_date"]), np.nan)
        a_rest = rest_lookup.get((row["away_team_name"], row["match_date"]), np.nan)
        home_rest.append(h_rest)
        away_rest.append(a_rest)

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest

    # Fill NaN with median, then cap at 365
    all_rest = pd.Series(home_rest + away_rest).dropna()
    median_rest = all_rest.median() if len(all_rest) > 0 else 30.0

    df["home_rest_days"] = df["home_rest_days"].fillna(median_rest)
    df["away_rest_days"] = df["away_rest_days"].fillna(median_rest)

    df["home_rest_days"] = df["home_rest_days"].clip(upper=365)
    df["away_rest_days"] = df["away_rest_days"].clip(upper=365)

    return df


# ===================================================================
# 2i. Interaction features
# ===================================================================
def compute_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute interaction features that combine ELO, rolling form,
    and historical performance metrics.

    Must be called AFTER compute_team_history, compute_elo_ratings,
    and compute_rolling_form.

    Produces:
      - home_attack_x_away_defense
      - away_attack_x_home_defense
      - elo_x_form_diff
    """
    df = df.copy()

    df["home_attack_x_away_defense"] = (
        df["home_hist_goals_per_game"] * df["away_hist_goals_conceded_per_game"]
    )
    df["away_attack_x_home_defense"] = (
        df["away_hist_goals_per_game"] * df["home_hist_goals_conceded_per_game"]
    )
    df["elo_x_form_diff"] = (
        df["elo_diff"]
        * (df["home_rolling5_win_rate"] - df["away_rolling5_win_rate"])
    )

    return df


# ===================================================================
# 3. Handle NaN / cold-start
# ===================================================================
RATE_FEATURES = [
    "home_hist_win_rate",
    "away_hist_win_rate",
    "home_hist_draw_rate",
    "away_hist_draw_rate",
    "h2h_home_win_rate",
]

COUNT_FEATURES = [
    "home_hist_matches_played",
    "away_hist_matches_played",
    "home_wc_appearances",
    "away_wc_appearances",
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "h2h_total",
]

DIFF_FEATURES_RATE = [
    "hist_win_rate_diff",
    "hist_goals_per_game_diff",
]

GOALS_RATE_FEATURES = [
    "home_hist_goals_per_game",
    "away_hist_goals_per_game",
    "home_hist_goals_conceded_per_game",
    "away_hist_goals_conceded_per_game",
]

# New feature fill groups
ELO_FEATURES_BASE = ["home_elo", "away_elo"]
ELO_FEATURES_DIFF = ["elo_diff"]

ROLLING_FORM_WIN_RATE = [
    "home_rolling5_win_rate",
    "away_rolling5_win_rate",
]

ROLLING_FORM_GOALS = [
    "home_rolling5_goals_pg",
    "away_rolling5_goals_pg",
]

REST_DAY_FEATURES = [
    "home_rest_days",
    "away_rest_days",
]

INTERACTION_FEATURES = [
    "home_attack_x_away_defense",
    "away_attack_x_home_defense",
    "elo_x_form_diff",
]


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values for cold-start teams."""
    df = df.copy()
    for col in RATE_FEATURES:
        df[col] = df[col].fillna(0.33)
    for col in COUNT_FEATURES:
        df[col] = df[col].fillna(0)
    for col in DIFF_FEATURES_RATE:
        df[col] = df[col].fillna(0.0)
    for col in GOALS_RATE_FEATURES:
        df[col] = df[col].fillna(0.0)
    # wc_appearances_diff is derived from counts, should have no NaN but be safe
    df["wc_appearances_diff"] = df["wc_appearances_diff"].fillna(0)

    # ELO features
    for col in ELO_FEATURES_BASE:
        df[col] = df[col].fillna(1500.0)
    for col in ELO_FEATURES_DIFF:
        df[col] = df[col].fillna(0.0)

    # Rolling form features
    for col in ROLLING_FORM_WIN_RATE:
        df[col] = df[col].fillna(0.33)
    for col in ROLLING_FORM_GOALS:
        df[col] = df[col].fillna(0.0)

    # Rest days: fill with median of existing values
    for col in REST_DAY_FEATURES:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if pd.notna(median_val) else 30.0)

    # Interaction features
    for col in INTERACTION_FEATURES:
        df[col] = df[col].fillna(0.0)

    return df


# ===================================================================
# 4-5. Select output columns, split, and save
# ===================================================================
FEATURE_COLS = [
    # Team historical performance
    "home_hist_win_rate",
    "away_hist_win_rate",
    "home_hist_draw_rate",
    "away_hist_draw_rate",
    "home_hist_goals_per_game",
    "away_hist_goals_per_game",
    "home_hist_goals_conceded_per_game",
    "away_hist_goals_conceded_per_game",
    "home_hist_matches_played",
    "away_hist_matches_played",
    "hist_win_rate_diff",
    "hist_goals_per_game_diff",
    # Match context
    "is_group_stage",
    "is_knockout",
    "stage_ordinal",
    # Host advantage
    "home_is_host",
    "away_is_host",
    # World Cup experience
    "home_wc_appearances",
    "away_wc_appearances",
    "wc_appearances_diff",
    # Head-to-head
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "h2h_total",
    "h2h_home_win_rate",
    # ELO ratings
    "home_elo",
    "away_elo",
    "elo_diff",
    # Rolling form (last 5)
    "home_rolling5_win_rate",
    "away_rolling5_win_rate",
    "home_rolling5_goals_pg",
    "away_rolling5_goals_pg",
    # Rest days
    "home_rest_days",
    "away_rest_days",
    # Interaction features
    "home_attack_x_away_defense",
    "away_attack_x_home_defense",
    "elo_x_form_diff",
]

META_COLS = [
    "match_date",
    "year",
    "home_team_name",
    "away_team_name",
]

TARGET_COL = "result"


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the columns that belong in the output."""
    output_cols = META_COLS + FEATURE_COLS + [TARGET_COL]
    return df[output_cols].copy()


# ===================================================================
# 6. Summary
# ===================================================================
def print_summary(train_out: pd.DataFrame, test_out: pd.DataFrame) -> None:
    """Print a summary of engineered features."""
    print("=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    print(f"\nTrain shape: {train_out.shape}")
    print(f"Test shape:  {test_out.shape}")

    print(f"\nFeature columns ({len(FEATURE_COLS)}):")
    for col in FEATURE_COLS:
        print(f"  - {col}")

    print("\n--- Train NaN counts ---")
    nan_counts = train_out[FEATURE_COLS].isna().sum()
    print(nan_counts[nan_counts > 0].to_string() if nan_counts.sum() > 0 else "  None")

    print("\n--- Test NaN counts ---")
    nan_counts = test_out[FEATURE_COLS].isna().sum()
    print(nan_counts[nan_counts > 0].to_string() if nan_counts.sum() > 0 else "  None")

    print("\n--- Train sample (first 3 rows) ---")
    print(train_out.head(3).to_string())

    print("\n--- Test sample (first 3 rows) ---")
    print(test_out.head(3).to_string())
    print()


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    """Run the full feature engineering pipeline."""
    print("Loading data...")
    df = load_data()
    print(f"  Combined: {df.shape[0]} rows")

    print("Computing team historical performance...")
    df = compute_team_history(df)

    print("Computing match context...")
    df = compute_match_context(df)

    print("Computing host advantage...")
    df = compute_host_advantage(df)

    print("Computing World Cup experience...")
    df = compute_wc_experience(df)

    print("Computing head-to-head records...")
    df = compute_head_to_head(df)

    print("Computing ELO ratings...")
    df = compute_elo_ratings(df)

    print("Computing rolling form (last 5 matches)...")
    df = compute_rolling_form(df)

    print("Computing rest days...")
    df = compute_rest_days(df)

    print("Computing interaction features...")
    df = compute_interactions(df)

    print("Filling missing values...")
    df = fill_missing(df)

    # Build output (only engineered features + metadata + target)
    output = build_output(df)

    # Split back into train / test
    train_out = output[df["_split"] == "train"].reset_index(drop=True)
    test_out = output[df["_split"] == "test"].reset_index(drop=True)

    # Save
    DATA_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(DATA_CLEAN_DIR / "features_train.csv", index=False)
    test_out.to_csv(DATA_CLEAN_DIR / "features_test.csv", index=False)
    print(f"\nSaved: {DATA_CLEAN_DIR / 'features_train.csv'}")
    print(f"Saved: {DATA_CLEAN_DIR / 'features_test.csv'}")

    print_summary(train_out, test_out)


if __name__ == "__main__":
    main()
