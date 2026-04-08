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

import bisect
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

# ---------------------------------------------------------------------------
# International dataset: team name normalisation
# Maps names used in the Kaggle international-results dataset to the
# canonical names used in our World Cup data.
# ---------------------------------------------------------------------------
INTL_TEAM_NAME_MAP: dict[str, str] = {
    # Kaggle intl dataset name  ->  WC dataset canonical name
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Eswatini": "Swaziland",
    "Timor-Leste": "East Timor",
    "Congo DR": "DR Congo",
    "China PR": "China",
    "Cabo Verde": "Cape Verde",
    "Brunei Darussalam": "Brunei",
    "Curaçao": "Curacao",
    "São Tomé and Príncipe": "São Tomé e Príncipe",
    "Burma": "Myanmar",
    "Rhodesia": "Zimbabwe",
    "Dahomey": "Benin",
    "Upper Volta": "Burkina Faso",
    "Western Samoa": "Samoa",
    # Historical team mappings (Kaggle uses different names)
    "German DR": "East Germany",
}

# Reverse lookup: WC team name -> international data team name.
# Used when querying intl lookup structures with a WC team name.
# "West Germany" is the big one — 62 WC matches.  The Kaggle dataset
# uses "Germany" for the combined history (FIFA treats them as the same
# footballing association).
WC_TO_INTL_LOOKUP: dict[str, str] = {
    "West Germany": "Germany",
}
# Remaining WC teams with no intl equivalent get default fill values:
# - "Dutch East Indies" (1 WC match, 1938)
# - "Soviet Union" (31 WC matches — Kaggle has no Soviet entry)
# - "Serbia and Montenegro" (3 WC matches, 2006)
# - "Zaire" (3 WC matches, 1974)


def _intl_name(wc_team: str) -> str:
    """Map a WC team name to the name used in international data lookups."""
    return WC_TO_INTL_LOOKUP.get(wc_team, wc_team)

# ---------------------------------------------------------------------------
# Team -> confederation mapping for all 85 WC teams
# ---------------------------------------------------------------------------
TEAM_CONFEDERATION: dict[str, str] = {
    # UEFA (Europe)
    "Austria": "UEFA", "Belgium": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Bulgaria": "UEFA", "Croatia": "UEFA", "Czech Republic": "UEFA",
    "Czechoslovakia": "UEFA", "Denmark": "UEFA", "East Germany": "UEFA",
    "England": "UEFA", "France": "UEFA", "Germany": "UEFA",
    "Greece": "UEFA", "Hungary": "UEFA", "Iceland": "UEFA",
    "Israel": "UEFA", "Italy": "UEFA", "Netherlands": "UEFA",
    "Northern Ireland": "UEFA", "Norway": "UEFA", "Poland": "UEFA",
    "Portugal": "UEFA", "Republic of Ireland": "UEFA", "Romania": "UEFA",
    "Russia": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA",
    "Serbia and Montenegro": "UEFA", "Slovakia": "UEFA", "Slovenia": "UEFA",
    "Soviet Union": "UEFA", "Spain": "UEFA", "Sweden": "UEFA",
    "Switzerland": "UEFA", "Turkey": "UEFA", "Ukraine": "UEFA",
    "Wales": "UEFA", "West Germany": "UEFA", "Yugoslavia": "UEFA",
    # CONMEBOL (South America)
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    # CAF (Africa)
    "Algeria": "CAF", "Angola": "CAF", "Cameroon": "CAF",
    "Egypt": "CAF", "Ghana": "CAF", "Ivory Coast": "CAF",
    "Morocco": "CAF", "Nigeria": "CAF", "Senegal": "CAF",
    "South Africa": "CAF", "Togo": "CAF", "Tunisia": "CAF",
    "Zaire": "CAF",
    # AFC (Asia)
    "Australia": "AFC", "China": "AFC", "Iran": "AFC",
    "Iraq": "AFC", "Japan": "AFC", "Kuwait": "AFC",
    "North Korea": "AFC", "Qatar": "AFC", "Saudi Arabia": "AFC",
    "South Korea": "AFC", "United Arab Emirates": "AFC",
    # CONCACAF (North/Central America & Caribbean)
    "Canada": "CONCACAF", "Costa Rica": "CONCACAF", "Cuba": "CONCACAF",
    "El Salvador": "CONCACAF", "Haiti": "CONCACAF", "Honduras": "CONCACAF",
    "Jamaica": "CONCACAF", "Mexico": "CONCACAF", "Panama": "CONCACAF",
    "Trinidad and Tobago": "CONCACAF", "United States": "CONCACAF",
    # OFC (Oceania) — note: Australia moved from OFC to AFC in 2006
    "New Zealand": "OFC",
    # Historical
    "Dutch East Indies": "AFC",
}

# Host country -> confederation for each WC year
HOST_CONFEDERATION: dict[int, str] = {
    1930: "CONMEBOL",   # Uruguay
    1934: "UEFA",       # Italy
    1938: "UEFA",       # France
    1950: "CONMEBOL",   # Brazil
    1954: "UEFA",       # Switzerland
    1958: "UEFA",       # Sweden
    1962: "CONMEBOL",   # Chile
    1966: "UEFA",       # England
    1970: "CONCACAF",   # Mexico
    1974: "UEFA",       # Germany
    1978: "CONMEBOL",   # Argentina
    1982: "UEFA",       # Spain
    1986: "CONCACAF",   # Mexico
    1990: "UEFA",       # Italy
    1994: "CONCACAF",   # United States
    1998: "UEFA",       # France
    2002: "AFC",        # South Korea / Japan
    2006: "UEFA",       # Germany
    2010: "CAF",        # South Africa
    2014: "CONMEBOL",   # Brazil
    2018: "UEFA",       # Russia
    2022: "AFC",        # Qatar
}

# FIFA rankings: name normalisation
# Maps FIFA ranking team names -> WC canonical names
FIFA_RANK_NAME_MAP: dict[str, str] = {
    "China PR": "China",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Korea DPR": "North Korea",
    "Korea Republic": "South Korea",
    "USA": "United States",
}

FIFA_RANKINGS_PATH = DATA_CLEAN_DIR / "fifa_rankings.csv"

# ---------------------------------------------------------------------------
# Transfermarkt: team name normalisation + paths
# Maps Transfermarkt country_of_citizenship -> WC canonical names
# ---------------------------------------------------------------------------
TM_COUNTRY_MAP: dict[str, str] = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cote d'Ivoire": "Ivory Coast",
    "Korea, North": "North Korea",
    "Korea, South": "South Korea",
    "Republic of Ireland": "Republic of Ireland",  # same
    "IR Iran": "Iran",
    # Serbia and Montenegro dissolved in 2006; Transfermarkt uses "Serbia"
    # for post-2006 players.  Pre-2006 WC matches with S&M get NaN.
}

TM_PLAYERS_PATH = DATA_CLEAN_DIR / "players.csv"
TM_VALUATIONS_PATH = DATA_CLEAN_DIR / "player_valuations.csv"

# StatsBomb event data (per-team per-match stats for WC 2018 + 2022)
STATSBOMB_STATS_PATH = DATA_CLEAN_DIR / "statsbomb_wc_stats.csv"

# WC start dates for squad valuation windows
WC_START_DATES: dict[int, str] = {
    2006: "2006-06-09",
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}

# K-factors for international ELO by tournament type.
# Higher K gives more weight to the result.
INTL_ELO_K_FACTORS: dict[str, int] = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "UEFA Euro": 50,
    "UEFA Euro qualification": 40,
    "Copa América": 50,
    "Copa América qualification": 40,
    "African Cup of Nations": 50,
    "African Cup of Nations qualification": 40,
    "AFC Asian Cup": 50,
    "AFC Asian Cup qualification": 40,
    "CONCACAF Gold Cup": 50,
    "UEFA Nations League": 45,
    "Confederations Cup": 45,
    "Friendly": 20,
}
INTL_ELO_K_DEFAULT = 30  # fallback for unlisted tournaments

INTL_DATA_PATH = DATA_CLEAN_DIR / "international_results.csv"


def normalize_team_name(name: str) -> str:
    """Map known variant team names to the canonical form used in WC data."""
    return INTL_TEAM_NAME_MAP.get(name, name)


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


def load_international_data() -> pd.DataFrame | None:
    """
    Load the ~47K international football results dataset.

    Returns None if the file is not present (features will be skipped).
    Normalises team names to match the WC dataset and derives result
    columns compatible with the WC schema.
    """
    if not INTL_DATA_PATH.exists():
        return None

    intl = pd.read_csv(INTL_DATA_PATH)

    # Rename columns to match WC schema
    intl = intl.rename(columns={
        "date": "match_date",
        "home_team": "home_team_name",
        "away_team": "away_team_name",
        "home_score": "home_team_score",
        "away_score": "away_team_score",
    })

    intl["match_date"] = pd.to_datetime(intl["match_date"])

    # Normalise team names
    intl["home_team_name"] = intl["home_team_name"].map(normalize_team_name)
    intl["away_team_name"] = intl["away_team_name"].map(normalize_team_name)

    # Derive result columns (matching WC data format)
    intl["result"] = np.where(
        intl["home_team_score"] > intl["away_team_score"],
        "home team win",
        np.where(
            intl["home_team_score"] < intl["away_team_score"],
            "away team win",
            "draw",
        ),
    )
    intl["home_team_win"] = (intl["result"] == "home team win").astype(int)
    intl["away_team_win"] = (intl["result"] == "away team win").astype(int)
    intl["draw"] = (intl["result"] == "draw").astype(int)

    intl = intl.sort_values("match_date").reset_index(drop=True)
    return intl


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
# 2i-α. Home continent advantage  (Phase 3a)
# ===================================================================
def compute_continent_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag whether each team is playing on their home continent.

    Uses the team's FIFA confederation and the host country's
    confederation for the tournament year.

    Produces: home_on_home_continent, away_on_home_continent.
    """
    df = df.copy()

    def _on_home_continent(team: str, year: int) -> int:
        team_conf = TEAM_CONFEDERATION.get(team)
        host_conf = HOST_CONFEDERATION.get(year)
        if team_conf is None or host_conf is None:
            return 0
        return 1 if team_conf == host_conf else 0

    df["home_on_home_continent"] = df.apply(
        lambda r: _on_home_continent(r["home_team_name"], r["year"]), axis=1
    )
    df["away_on_home_continent"] = df.apply(
        lambda r: _on_home_continent(r["away_team_name"], r["year"]), axis=1
    )

    return df


# ===================================================================
# 2i-β. FIFA World Rankings  (Phase 3b)
# ===================================================================
def _load_fifa_rankings() -> pd.DataFrame | None:
    """Load and normalise FIFA world rankings. Returns None if missing."""
    if not FIFA_RANKINGS_PATH.exists():
        return None

    rk = pd.read_csv(FIFA_RANKINGS_PATH)
    rk["date"] = pd.to_datetime(rk["date"])
    rk["team"] = rk["team"].map(
        lambda t: FIFA_RANK_NAME_MAP.get(t, t)
    )
    rk = rk.sort_values("date").reset_index(drop=True)
    return rk


def _build_ranking_lookup(
    rk: pd.DataFrame,
) -> dict[str, list[tuple[float, float]]]:
    """
    Build per-team chronological list of (date_ordinal, total_points).
    """
    lookup: dict[str, list[tuple[float, float]]] = {}
    for _, row in rk.iterrows():
        team = row["team"]
        lookup.setdefault(team, []).append(
            (row["date"].toordinal(), float(row["total_points"]))
        )
    return lookup


def _lookup_ranking_before(
    lookup: dict[str, list[tuple[float, float]]],
    team: str,
    date_ord: float,
) -> float:
    """Binary-search for the most recent ranking points before a date."""
    entries = lookup.get(team)
    if not entries:
        return np.nan
    dates = [e[0] for e in entries]
    idx = bisect.bisect_right(dates, date_ord) - 1
    if idx < 0:
        return np.nan
    return entries[idx][1]


def compute_fifa_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach FIFA world ranking points to each WC match.

    Rankings are available from 1992-12 onward, so matches before
    ~1994 will get NaN (filled later by fill_missing).

    Produces: home_fifa_points, away_fifa_points, fifa_points_diff.
    """
    df = df.copy()

    rk = _load_fifa_rankings()
    if rk is None:
        df["home_fifa_points"] = np.nan
        df["away_fifa_points"] = np.nan
        df["fifa_points_diff"] = np.nan
        return df

    print("  Building FIFA ranking lookup...")
    lookup = _build_ranking_lookup(rk)

    home_pts = []
    away_pts = []
    for _, row in df.iterrows():
        date_ord = row["match_date"].toordinal()
        home_pts.append(_lookup_ranking_before(lookup, row["home_team_name"], date_ord))
        away_pts.append(_lookup_ranking_before(lookup, row["away_team_name"], date_ord))

    df["home_fifa_points"] = home_pts
    df["away_fifa_points"] = away_pts
    df["fifa_points_diff"] = df["home_fifa_points"] - df["away_fifa_points"]

    return df


# ===================================================================
# 2i-γ. Qualifying path strength  (Phase 3c)
# ===================================================================
def compute_qualifying_record(
    df: pd.DataFrame, intl: pd.DataFrame | None
) -> pd.DataFrame:
    """
    For each team in each WC match, compute their win rate in the
    qualifying campaign for that World Cup.

    Qualifying cycles are determined by the match dates relative to
    each WC year.  Host nations (who don't qualify) and teams in
    pre-qualification-era WCs get NaN, filled by fill_missing.

    Produces: home_qual_win_rate, away_qual_win_rate, qual_win_rate_diff.
    """
    df = df.copy()

    if intl is None:
        df["home_qual_win_rate"] = np.nan
        df["away_qual_win_rate"] = np.nan
        df["qual_win_rate_diff"] = np.nan
        return df

    # Extract WC qualification matches
    qual = intl[intl["tournament"] == "FIFA World Cup qualification"].copy()

    # Map each qualifying match to its target WC year.
    # Qualifiers typically run in the 3-4 years before each WC.
    wc_years = sorted(df["year"].unique())

    def _match_to_wc_year(match_date: pd.Timestamp) -> int | None:
        """Find which WC this qualifying match is for."""
        for wc_yr in wc_years:
            # Qualifiers for a WC typically start ~3.5 years before
            # and end ~0.5 year before the tournament
            window_start = pd.Timestamp(f"{wc_yr - 4}-01-01")
            window_end = pd.Timestamp(f"{wc_yr}-01-01")
            if window_start <= match_date < window_end:
                return wc_yr
        # For matches outside known WC windows, use the next WC
        for wc_yr in wc_years:
            if match_date < pd.Timestamp(f"{wc_yr}-01-01"):
                return wc_yr
        return None

    qual["wc_year"] = qual["match_date"].apply(_match_to_wc_year)
    qual = qual.dropna(subset=["wc_year"])
    qual["wc_year"] = qual["wc_year"].astype(int)

    # Build per-team, per-WC qualifying record
    # Expand to long form (one row per team per match)
    home_q = qual[["wc_year", "home_team_name", "home_team_win"]].rename(
        columns={"home_team_name": "team", "home_team_win": "win"}
    )
    away_q = qual[["wc_year", "away_team_name", "away_team_win"]].rename(
        columns={"away_team_name": "team", "away_team_win": "win"}
    )
    long_q = pd.concat([home_q, away_q], ignore_index=True)

    # Compute win rate per (team, wc_year)
    qual_record = (
        long_q.groupby(["team", "wc_year"])["win"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "wins", "count": "matches"})
    )
    qual_record["qual_win_rate"] = qual_record["wins"] / qual_record["matches"]
    qual_lookup = qual_record["qual_win_rate"].to_dict()

    # Attach to WC matches
    df["home_qual_win_rate"] = df.apply(
        lambda r: qual_lookup.get(
            (_intl_name(r["home_team_name"]), r["year"]), np.nan
        ),
        axis=1,
    )
    df["away_qual_win_rate"] = df.apply(
        lambda r: qual_lookup.get(
            (_intl_name(r["away_team_name"]), r["year"]), np.nan
        ),
        axis=1,
    )
    df["qual_win_rate_diff"] = df["home_qual_win_rate"] - df["away_qual_win_rate"]

    return df


# ===================================================================
# 2i-δ. Squad market value from Transfermarkt  (Phase 3d)
# ===================================================================
def _build_squad_value_lookup() -> dict[tuple[str, int], float] | None:
    """
    Build a lookup: (wc_team_name, wc_year) -> squad market value in EUR.

    For each WC year with valuation data (2006-2022), finds each
    country's top 23 most valuable players using their most recent
    Transfermarkt valuation within a 2-year window before the WC.

    Returns None if the Transfermarkt data files are not present.
    """
    if not TM_PLAYERS_PATH.exists() or not TM_VALUATIONS_PATH.exists():
        return None

    players = pd.read_csv(
        TM_PLAYERS_PATH,
        usecols=["player_id", "country_of_citizenship"],
    )
    valuations = pd.read_csv(TM_VALUATIONS_PATH)
    valuations["date"] = pd.to_datetime(valuations["date"])

    # Join nationality onto valuations
    val = valuations.merge(
        players[["player_id", "country_of_citizenship"]],
        on="player_id",
        how="left",
    )
    val = val.dropna(subset=["country_of_citizenship", "market_value_in_eur"])

    # Normalise Transfermarkt country names to WC canonical names
    val["country"] = val["country_of_citizenship"].map(
        lambda c: TM_COUNTRY_MAP.get(c, c)
    )

    lookup: dict[tuple[str, int], float] = {}

    for wc_year, start_str in WC_START_DATES.items():
        cutoff = pd.Timestamp(start_str)
        window_start = cutoff - pd.DateOffset(years=2)

        # Valuations in the 2-year window before the WC
        mask = (val["date"] >= window_start) & (val["date"] < cutoff)
        window = val.loc[mask]

        if window.empty:
            continue

        # Most recent valuation per player within the window
        latest = (
            window.sort_values("date")
            .groupby("player_id")
            .last()
            .reset_index()
        )

        # For each country, sum the top 23 players' values
        for country, group in latest.groupby("country"):
            if len(group) < 11:
                # Fewer than 11 valued players = unreliable estimate
                continue
            top23 = group.nlargest(23, "market_value_in_eur")
            total = top23["market_value_in_eur"].sum()
            lookup[(country, wc_year)] = total

    return lookup


def compute_squad_market_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach Transfermarkt squad market value to each WC match.

    Uses the top 23 most valuable players per nationality, valued
    at the most recent Transfermarkt valuation within 2 years before
    each World Cup.

    Coverage: WC 2006-2022 (earlier WCs get NaN, filled by fill_missing).

    Produces: home_squad_value, away_squad_value, squad_value_diff.
    Values are in millions of EUR for readability.
    """
    df = df.copy()

    lookup = _build_squad_value_lookup()

    if lookup is None:
        print("  Transfermarkt data not found — skipping squad values.")
        df["home_squad_value"] = np.nan
        df["away_squad_value"] = np.nan
        df["squad_value_diff"] = np.nan
        return df

    # Convert to millions for readability
    lookup_m = {k: v / 1e6 for k, v in lookup.items()}

    home_vals = []
    away_vals = []
    for _, row in df.iterrows():
        h = lookup_m.get((row["home_team_name"], row["year"]), np.nan)
        a = lookup_m.get((row["away_team_name"], row["year"]), np.nan)
        home_vals.append(h)
        away_vals.append(a)

    df["home_squad_value"] = home_vals
    df["away_squad_value"] = away_vals
    df["squad_value_diff"] = df["home_squad_value"] - df["away_squad_value"]

    # Report coverage
    n_covered = df["home_squad_value"].notna().sum()
    print(f"  Squad value coverage: {n_covered}/{len(df)} matches "
          f"({n_covered / len(df):.0%})")

    return df


# ===================================================================
# 2i-ε. StatsBomb within-tournament rolling xG  (Phase 3e)
# ===================================================================
def compute_statsbomb_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute within-tournament rolling xG and pass completion
    from StatsBomb event data.

    For each WC match, uses only the team's prior matches in the
    SAME tournament (leakage-safe). Coverage: WC 2018 + 2022 only.

    Produces: home_rolling_xg, away_rolling_xg, rolling_xg_diff.
    """
    df = df.copy()

    if not STATSBOMB_STATS_PATH.exists():
        print("  StatsBomb data not found — skipping.")
        df["home_rolling_xg"] = np.nan
        df["away_rolling_xg"] = np.nan
        df["rolling_xg_diff"] = np.nan
        return df

    sb = pd.read_csv(STATSBOMB_STATS_PATH)
    sb["match_date"] = pd.to_datetime(sb["match_date"])

    # Build per-(team, wc_year) chronological match lists
    # Each entry: (date_ordinal, xg, pass_pct)
    team_tournament_log: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for _, row in sb.iterrows():
        key = (row["team"], int(row["wc_year"]))
        team_tournament_log.setdefault(key, []).append(
            (row["match_date"].toordinal(), float(row["xg"]))
        )
    # Sort each team's matches chronologically
    for key in team_tournament_log:
        team_tournament_log[key].sort(key=lambda x: x[0])

    def _rolling_xg(team: str, year: int, date_ord: float) -> float:
        """Average xG from prior matches in the same WC tournament."""
        entries = team_tournament_log.get((team, year))
        if not entries:
            return np.nan
        # Find all entries strictly before this date
        prior = [e for e in entries if e[0] < date_ord]
        if not prior:
            return np.nan
        return np.mean([e[1] for e in prior])

    home_xg, away_xg = [], []
    for _, row in df.iterrows():
        date_ord = row["match_date"].toordinal()
        year = row["year"]
        h = _rolling_xg(row["home_team_name"], year, date_ord)
        a = _rolling_xg(row["away_team_name"], year, date_ord)
        home_xg.append(h)
        away_xg.append(a)

    df["home_rolling_xg"] = home_xg
    df["away_rolling_xg"] = away_xg
    df["rolling_xg_diff"] = df["home_rolling_xg"] - df["away_rolling_xg"]

    n_covered = df["home_rolling_xg"].notna().sum()
    n_both = (df["home_rolling_xg"].notna() & df["away_rolling_xg"].notna()).sum()
    print(f"  StatsBomb xG coverage: {n_both}/{len(df)} matches with both teams "
          f"({n_both / len(df):.0%})")

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

    # International interaction features (only if intl features exist)
    if "intl_elo_diff" in df.columns:
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

    return df


# ===================================================================
# 2j. International ELO ratings (from all international matches)
# ===================================================================
def _build_intl_elo_history(intl: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
    """
    Process all international matches chronologically and return a
    per-team ELO history: team -> [(ordinal_date, elo_after), ...].

    Uses variable K-factors by tournament importance.
    """
    ratings: dict[str, float] = {}
    # team -> list of (ordinal_date, elo_after_this_date)
    history: dict[str, list[tuple[float, float]]] = {}

    for _, row in intl.iterrows():
        home = row["home_team_name"]
        away = row["away_team_name"]
        date_ord = row["match_date"].toordinal()
        tournament = row.get("tournament", "")

        # Look up K-factor for this tournament type
        k = INTL_ELO_K_FACTORS.get(tournament, INTL_ELO_K_DEFAULT)

        r_home = ratings.get(home, 1500.0)
        r_away = ratings.get(away, 1500.0)

        # Expected scores
        e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home) / 400.0))
        e_away = 1.0 - e_home

        # Actual scores
        result = row["result"]
        if result == "home team win":
            s_home, s_away = 1.0, 0.0
        elif result == "away team win":
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5

        # Update
        new_home = r_home + k * (s_home - e_home)
        new_away = r_away + k * (s_away - e_away)
        ratings[home] = new_home
        ratings[away] = new_away

        # Record post-match ELO
        history.setdefault(home, []).append((date_ord, new_home))
        history.setdefault(away, []).append((date_ord, new_away))

    return history


def _lookup_elo_before(
    history: dict[str, list[tuple[float, float]]],
    team: str,
    date_ord: float,
) -> float:
    """
    Binary-search the ELO history to find the team's rating
    just before the given date.  Returns 1500.0 if no prior data.
    """
    entries = history.get(team)
    if not entries:
        return 1500.0
    # entries are sorted by date_ord (built in chronological order)
    dates = [e[0] for e in entries]
    idx = bisect.bisect_left(dates, date_ord) - 1
    if idx < 0:
        return 1500.0
    return entries[idx][1]


def compute_intl_elo(df: pd.DataFrame, intl: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ELO ratings from ALL international matches and attach
    them to each WC match row.

    Uses variable K-factors: WC (60), continental cups (50),
    qualifiers (40), friendlies (20), other (30).

    Produces: home_intl_elo, away_intl_elo, intl_elo_diff.
    """
    df = df.copy()

    print("  Building ELO history from international data...")
    elo_history = _build_intl_elo_history(intl)

    home_elos = []
    away_elos = []
    for _, row in df.iterrows():
        date_ord = row["match_date"].toordinal()
        home_elos.append(
            _lookup_elo_before(elo_history, _intl_name(row["home_team_name"]), date_ord)
        )
        away_elos.append(
            _lookup_elo_before(elo_history, _intl_name(row["away_team_name"]), date_ord)
        )

    df["home_intl_elo"] = home_elos
    df["away_intl_elo"] = away_elos
    df["intl_elo_diff"] = df["home_intl_elo"] - df["away_intl_elo"]

    return df


# ===================================================================
# 2k. International rolling form (last 5 international matches)
# ===================================================================
def _build_team_match_log(intl: pd.DataFrame) -> dict[str, list[tuple[float, float, float]]]:
    """
    Build per-team chronological match log from international data.

    Returns: team -> [(ordinal_date, win_flag, goals_for), ...]
    """
    log: dict[str, list[tuple[float, float, float]]] = {}

    for _, row in intl.iterrows():
        date_ord = row["match_date"].toordinal()
        home = row["home_team_name"]
        away = row["away_team_name"]

        log.setdefault(home, []).append(
            (date_ord, float(row["home_team_win"]), float(row["home_team_score"]))
        )
        log.setdefault(away, []).append(
            (date_ord, float(row["away_team_win"]), float(row["away_team_score"]))
        )

    return log


def compute_intl_rolling_form(df: pd.DataFrame, intl: pd.DataFrame) -> pd.DataFrame:
    """
    For each WC match, compute each team's win rate and goals per game
    over their last 5 international matches (any tournament) before
    the match date.

    Produces: home_intl_rolling5_win_rate, away_intl_rolling5_win_rate,
              home_intl_rolling5_goals_pg, away_intl_rolling5_goals_pg.
    """
    df = df.copy()

    print("  Building team match logs from international data...")
    match_log = _build_team_match_log(intl)

    def _rolling5(team: str, date_ord: float) -> tuple[float, float]:
        """Return (win_rate, goals_pg) from last 5 matches before date."""
        entries = match_log.get(team)
        if not entries:
            return 0.33, 0.0
        dates = [e[0] for e in entries]
        idx = bisect.bisect_left(dates, date_ord)
        # Take up to 5 matches before this date
        start = max(0, idx - 5)
        recent = entries[start:idx]
        if not recent:
            return 0.33, 0.0
        wins = sum(e[1] for e in recent)
        goals = sum(e[2] for e in recent)
        n = len(recent)
        return wins / n, goals / n

    h_wr, h_gpg, a_wr, a_gpg = [], [], [], []
    for _, row in df.iterrows():
        date_ord = row["match_date"].toordinal()
        hw, hg = _rolling5(_intl_name(row["home_team_name"]), date_ord)
        aw, ag = _rolling5(_intl_name(row["away_team_name"]), date_ord)
        h_wr.append(hw)
        h_gpg.append(hg)
        a_wr.append(aw)
        a_gpg.append(ag)

    df["home_intl_rolling5_win_rate"] = h_wr
    df["away_intl_rolling5_win_rate"] = a_wr
    df["home_intl_rolling5_goals_pg"] = h_gpg
    df["away_intl_rolling5_goals_pg"] = a_gpg

    return df


# ===================================================================
# 2l. International head-to-head record
# ===================================================================
def compute_intl_h2h(df: pd.DataFrame, intl: pd.DataFrame) -> pd.DataFrame:
    """
    For each WC match between Team A and Team B, count all prior
    international meetings (any tournament) and compute H2H stats
    from the current home team's perspective.

    Produces: intl_h2h_home_wins, intl_h2h_away_wins, intl_h2h_draws,
              intl_h2h_total, intl_h2h_home_win_rate.
    """
    df = df.copy()

    # Build lookup: frozenset(team_a, team_b) -> list of (date_ord, home, away, result)
    print("  Building H2H lookup from international data...")
    h2h_log: dict[frozenset, list[tuple[float, str, str, str]]] = {}
    for _, row in intl.iterrows():
        pair = frozenset([row["home_team_name"], row["away_team_name"]])
        h2h_log.setdefault(pair, []).append((
            row["match_date"].toordinal(),
            row["home_team_name"],
            row["away_team_name"],
            row["result"],
        ))

    # Pre-sort each pair's log by date
    for pair in h2h_log:
        h2h_log[pair].sort(key=lambda x: x[0])

    hw_list, aw_list, dr_list = [], [], []
    for _, row in df.iterrows():
        home_team_intl = _intl_name(row["home_team_name"])
        away_team_intl = _intl_name(row["away_team_name"])
        pair = frozenset([home_team_intl, away_team_intl])
        date_ord = row["match_date"].toordinal()

        entries = h2h_log.get(pair, [])
        # Find entries before this date using bisect
        dates = [e[0] for e in entries]
        idx = bisect.bisect_left(dates, date_ord)
        prior = entries[:idx]

        hw = 0
        aw = 0
        dr = 0
        for _, prev_home, prev_away, prev_result in prior:
            if prev_result == "draw":
                dr += 1
            elif prev_result == "home team win":
                if prev_home == home_team_intl:
                    hw += 1
                else:
                    aw += 1
            elif prev_result == "away team win":
                if prev_away == home_team_intl:
                    hw += 1
                else:
                    aw += 1

        hw_list.append(hw)
        aw_list.append(aw)
        dr_list.append(dr)

    df["intl_h2h_home_wins"] = hw_list
    df["intl_h2h_away_wins"] = aw_list
    df["intl_h2h_draws"] = dr_list
    df["intl_h2h_total"] = (
        df["intl_h2h_home_wins"] + df["intl_h2h_away_wins"] + df["intl_h2h_draws"]
    )
    df["intl_h2h_home_win_rate"] = (
        df["intl_h2h_home_wins"] / df["intl_h2h_total"]
    ).fillna(0.33)

    return df


# ===================================================================
# 2m. International historical performance (expanding window)
# ===================================================================
def compute_intl_history(df: pd.DataFrame, intl: pd.DataFrame) -> pd.DataFrame:
    """
    For each WC match, compute expanding-window historical stats from
    ALL international matches before the match date.

    Produces: home/away_intl_hist_win_rate, home/away_intl_hist_draw_rate,
              home/away_intl_hist_goals_per_game,
              home/away_intl_hist_goals_conceded_per_game,
              home/away_intl_hist_matches_played,
              intl_hist_win_rate_diff, intl_hist_goals_per_game_diff.
    """
    df = df.copy()

    # Build per-team cumulative stats: team -> list of
    # (date_ord, cum_wins, cum_draws, cum_gf, cum_ga, cum_matches)
    print("  Building cumulative history from international data...")
    team_cumulative: dict[str, list[tuple[float, float, float, float, float, float]]] = {}

    # Accumulators
    cum: dict[str, list[float]] = {}  # team -> [wins, draws, gf, ga, matches]

    for _, row in intl.iterrows():
        date_ord = row["match_date"].toordinal()
        home = row["home_team_name"]
        away = row["away_team_name"]

        # Update home team
        if home not in cum:
            cum[home] = [0.0, 0.0, 0.0, 0.0, 0.0]
        c = cum[home]
        c[0] += float(row["home_team_win"])
        c[1] += float(row["draw"])
        c[2] += float(row["home_team_score"])
        c[3] += float(row["away_team_score"])
        c[4] += 1.0
        team_cumulative.setdefault(home, []).append(
            (date_ord, c[0], c[1], c[2], c[3], c[4])
        )

        # Update away team
        if away not in cum:
            cum[away] = [0.0, 0.0, 0.0, 0.0, 0.0]
        c = cum[away]
        c[0] += float(row["away_team_win"])
        c[1] += float(row["draw"])
        c[2] += float(row["away_team_score"])
        c[3] += float(row["home_team_score"])
        c[4] += 1.0
        team_cumulative.setdefault(away, []).append(
            (date_ord, c[0], c[1], c[2], c[3], c[4])
        )

    def _lookup_cumulative(team: str, date_ord: float) -> tuple[float, ...]:
        """Return (wins, draws, gf, ga, matches) up to but not including date."""
        entries = team_cumulative.get(team)
        if not entries:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        dates = [e[0] for e in entries]
        idx = bisect.bisect_left(dates, date_ord) - 1
        if idx < 0:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        e = entries[idx]
        return e[1], e[2], e[3], e[4], e[5]

    cols = {
        "home_intl_hist_win_rate": [],
        "home_intl_hist_draw_rate": [],
        "home_intl_hist_goals_per_game": [],
        "home_intl_hist_goals_conceded_per_game": [],
        "home_intl_hist_matches_played": [],
        "away_intl_hist_win_rate": [],
        "away_intl_hist_draw_rate": [],
        "away_intl_hist_goals_per_game": [],
        "away_intl_hist_goals_conceded_per_game": [],
        "away_intl_hist_matches_played": [],
    }

    for _, row in df.iterrows():
        date_ord = row["match_date"].toordinal()

        for side, team_col in [("home", "home_team_name"), ("away", "away_team_name")]:
            wins, draws, gf, ga, matches = _lookup_cumulative(
                _intl_name(row[team_col]), date_ord
            )
            if matches > 0:
                cols[f"{side}_intl_hist_win_rate"].append(wins / matches)
                cols[f"{side}_intl_hist_draw_rate"].append(draws / matches)
                cols[f"{side}_intl_hist_goals_per_game"].append(gf / matches)
                cols[f"{side}_intl_hist_goals_conceded_per_game"].append(ga / matches)
            else:
                cols[f"{side}_intl_hist_win_rate"].append(np.nan)
                cols[f"{side}_intl_hist_draw_rate"].append(np.nan)
                cols[f"{side}_intl_hist_goals_per_game"].append(np.nan)
                cols[f"{side}_intl_hist_goals_conceded_per_game"].append(np.nan)
            cols[f"{side}_intl_hist_matches_played"].append(matches)

    for col_name, values in cols.items():
        df[col_name] = values

    # Difference features
    df["intl_hist_win_rate_diff"] = (
        df["home_intl_hist_win_rate"] - df["away_intl_hist_win_rate"]
    )
    df["intl_hist_goals_per_game_diff"] = (
        df["home_intl_hist_goals_per_game"] - df["away_intl_hist_goals_per_game"]
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

# --- International feature fill groups (only present when intl data loaded) ---
INTL_RATE_FEATURES = [
    "home_intl_hist_win_rate",
    "away_intl_hist_win_rate",
    "home_intl_hist_draw_rate",
    "away_intl_hist_draw_rate",
    "intl_h2h_home_win_rate",
]

INTL_COUNT_FEATURES = [
    "home_intl_hist_matches_played",
    "away_intl_hist_matches_played",
    "intl_h2h_home_wins",
    "intl_h2h_away_wins",
    "intl_h2h_draws",
    "intl_h2h_total",
]

INTL_GOALS_RATE_FEATURES = [
    "home_intl_hist_goals_per_game",
    "away_intl_hist_goals_per_game",
    "home_intl_hist_goals_conceded_per_game",
    "away_intl_hist_goals_conceded_per_game",
]

INTL_DIFF_FEATURES = [
    "intl_hist_win_rate_diff",
    "intl_hist_goals_per_game_diff",
]

INTL_ELO_BASE = ["home_intl_elo", "away_intl_elo"]
INTL_ELO_DIFF = ["intl_elo_diff"]

INTL_ROLLING_WIN_RATE = [
    "home_intl_rolling5_win_rate",
    "away_intl_rolling5_win_rate",
]

INTL_ROLLING_GOALS = [
    "home_intl_rolling5_goals_pg",
    "away_intl_rolling5_goals_pg",
]

INTL_INTERACTION_FEATURES = [
    "intl_elo_x_form_diff",
    "home_intl_attack_x_away_defense",
    "away_intl_attack_x_home_defense",
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

    # Helper for safe column fills (columns may not exist if data is missing)
    def _safe_fill(columns: list[str], value: float) -> None:
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna(value)

    # --- Phase 3 features ---
    # Continent advantage: 0 = not on home continent (safe default)
    _safe_fill(["home_on_home_continent", "away_on_home_continent"], 0)
    # FIFA rankings: fill with median of existing values (pre-1994 matches get NaN)
    for col in ["home_fifa_points", "away_fifa_points"]:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0.0)
    _safe_fill(["fifa_points_diff"], 0.0)
    # Qualifying record: 0.33 uniform prior (host nations don't qualify)
    _safe_fill(["home_qual_win_rate", "away_qual_win_rate"], 0.33)
    _safe_fill(["qual_win_rate_diff"], 0.0)
    # Squad market value: fill with median (pre-2006 matches get NaN)
    for col in ["home_squad_value", "away_squad_value"]:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0.0)
    _safe_fill(["squad_value_diff"], 0.0)
    # StatsBomb xG: fill with median (only WC 2018+2022 have data)
    for col in ["home_rolling_xg", "away_rolling_xg"]:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if pd.notna(median_val) else 1.0)
    _safe_fill(["rolling_xg_diff"], 0.0)

    # --- International features (only fill if columns exist) ---

    _safe_fill(INTL_RATE_FEATURES, 0.33)
    _safe_fill(INTL_COUNT_FEATURES, 0)
    _safe_fill(INTL_GOALS_RATE_FEATURES, 0.0)
    _safe_fill(INTL_DIFF_FEATURES, 0.0)
    _safe_fill(INTL_ELO_BASE, 1500.0)
    _safe_fill(INTL_ELO_DIFF, 0.0)
    _safe_fill(INTL_ROLLING_WIN_RATE, 0.33)
    _safe_fill(INTL_ROLLING_GOALS, 0.0)
    _safe_fill(INTL_INTERACTION_FEATURES, 0.0)

    return df


# ===================================================================
# 4-5. Select output columns, split, and save
# ===================================================================
FEATURE_COLS_BASE = [
    # Team historical performance (WC-only)
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
    # Head-to-head (WC-only)
    "h2h_home_wins",
    "h2h_away_wins",
    "h2h_draws",
    "h2h_total",
    "h2h_home_win_rate",
    # ELO ratings (WC-only)
    "home_elo",
    "away_elo",
    "elo_diff",
    # Rolling form — last 5 WC matches
    "home_rolling5_win_rate",
    "away_rolling5_win_rate",
    "home_rolling5_goals_pg",
    "away_rolling5_goals_pg",
    # Rest days
    "home_rest_days",
    "away_rest_days",
    # Continent advantage (Phase 3a)
    "home_on_home_continent",
    "away_on_home_continent",
    # FIFA world ranking points (Phase 3b)
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
    # Qualifying path strength (Phase 3c)
    "home_qual_win_rate",
    "away_qual_win_rate",
    "qual_win_rate_diff",
    # Squad market value (Phase 3d)
    "home_squad_value",
    "away_squad_value",
    "squad_value_diff",
    # StatsBomb within-tournament rolling xG (Phase 3e)
    "home_rolling_xg",
    "away_rolling_xg",
    "rolling_xg_diff",
    # Interaction features (WC-only)
    "home_attack_x_away_defense",
    "away_attack_x_home_defense",
    "elo_x_form_diff",
]

FEATURE_COLS_INTL = [
    # International ELO
    "home_intl_elo",
    "away_intl_elo",
    "intl_elo_diff",
    # International rolling form — last 5 international matches
    "home_intl_rolling5_win_rate",
    "away_intl_rolling5_win_rate",
    "home_intl_rolling5_goals_pg",
    "away_intl_rolling5_goals_pg",
    # International head-to-head
    "intl_h2h_home_wins",
    "intl_h2h_away_wins",
    "intl_h2h_draws",
    "intl_h2h_total",
    "intl_h2h_home_win_rate",
    # International historical performance
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
    # International interaction features
    "intl_elo_x_form_diff",
    "home_intl_attack_x_away_defense",
    "away_intl_attack_x_home_defense",
]

# FEATURE_COLS is set dynamically in main() based on whether intl data is loaded.
# Default to base-only for backward compatibility.
FEATURE_COLS = list(FEATURE_COLS_BASE)

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
    global FEATURE_COLS  # noqa: PLW0603

    print("Loading data...")
    df = load_data()
    print(f"  Combined: {df.shape[0]} rows")

    # --- Try to load international data ---
    print("\nLoading international results data...")
    intl = load_international_data()
    has_intl = intl is not None
    if has_intl:
        print(f"  Loaded {len(intl)} international matches "
              f"({intl['match_date'].min().date()} to {intl['match_date'].max().date()})")
        # Validate team name coverage
        wc_teams = set(df["home_team_name"].unique()) | set(df["away_team_name"].unique())
        intl_teams = set(intl["home_team_name"].unique()) | set(intl["away_team_name"].unique())
        missing = wc_teams - intl_teams
        if missing:
            print(f"  WARNING: {len(missing)} WC teams not found in intl data: {sorted(missing)}")
        else:
            print(f"  All {len(wc_teams)} WC teams found in international data.")
    else:
        print("  international_results.csv not found — skipping intl features.")
        print(f"  Place the file at: {INTL_DATA_PATH}")

    # --- WC-only features (existing pipeline, unchanged) ---
    print("\nComputing team historical performance (WC-only)...")
    df = compute_team_history(df)

    print("Computing match context...")
    df = compute_match_context(df)

    print("Computing host advantage...")
    df = compute_host_advantage(df)

    print("Computing World Cup experience...")
    df = compute_wc_experience(df)

    print("Computing head-to-head records (WC-only)...")
    df = compute_head_to_head(df)

    print("Computing ELO ratings (WC-only)...")
    df = compute_elo_ratings(df)

    print("Computing rolling form — last 5 WC matches...")
    df = compute_rolling_form(df)

    print("Computing rest days...")
    df = compute_rest_days(df)

    # --- Phase 3 features ---
    print("\nComputing continent advantage...")
    df = compute_continent_advantage(df)

    print("Computing FIFA world ranking points...")
    df = compute_fifa_rankings(df)

    print("Computing qualifying path strength...")
    df = compute_qualifying_record(df, intl)

    print("Computing squad market value...")
    df = compute_squad_market_value(df)

    print("Computing StatsBomb within-tournament rolling xG...")
    df = compute_statsbomb_rolling(df)

    # --- International features (new, only if data available) ---
    if has_intl:
        print("\nComputing international ELO ratings...")
        df = compute_intl_elo(df, intl)

        print("Computing international rolling form — last 5 intl matches...")
        df = compute_intl_rolling_form(df, intl)

        print("Computing international head-to-head records...")
        df = compute_intl_h2h(df, intl)

        print("Computing international historical performance...")
        df = compute_intl_history(df, intl)

    print("\nComputing interaction features...")
    df = compute_interactions(df)

    print("Filling missing values...")
    df = fill_missing(df)

    # Set FEATURE_COLS based on what was computed
    if has_intl:
        FEATURE_COLS = FEATURE_COLS_BASE + FEATURE_COLS_INTL
    else:
        FEATURE_COLS = list(FEATURE_COLS_BASE)

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
