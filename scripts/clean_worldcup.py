"""
clean and split world cup match data

Inputs:
  - files_needed/matches.csv
  - files_needed/tournaments.csv

new files from this script:
  - data_clean/matches_train.csv    # mens world cup matches, 1930–2018
  - data_clean/matches_test.csv     # mens world cup matches, 2022

Usage:
  python3 scripts/clean_worldcup.py
"""

from __future__ import annotations

import pathlib  # for working with filesystem paths in an OS-independent way

# External library for data manipulation (install via `pip install pandas`)
import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES_NEEDED_DIR = PROJECT_ROOT / "files_needed"
DATA_CLEAN_DIR = PROJECT_ROOT / "data_clean"


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw CSV files from the `files_needed/` folder.

    Returns
    -------
    matches : DataFrame
        Full matches table from Fjelstul's database.
    tournaments : DataFrame
        Full tournaments table (used to filter to mens tournaments and get year).
    """
    matches_path = FILES_NEEDED_DIR / "matches.csv"
    tournaments_path = FILES_NEEDED_DIR / "tournaments.csv"

    matches = pd.read_csv(matches_path)
    tournaments = pd.read_csv(tournaments_path)

    return matches, tournaments


def filter_mens_world_cup(
    matches: pd.DataFrame, tournaments: pd.DataFrame
) -> pd.DataFrame:
    """
    Keep only mens matches.

    tournaments where `tournament_name` contains
    FIFA Men's World Cup
    """
    # filter tournaments down to just men's World Cups
    mens_tournaments = tournaments[
        tournaments["tournament_name"].str.contains("FIFA Men's World Cup", na=False)
    ].copy()

    # use the tournament IDs from that subset to filter matches
    mens_ids = mens_tournaments["tournament_id"].unique()
    matches_men = matches[matches["tournament_id"].isin(mens_ids)].copy()

    # bring in the tournament year for convenience (used for train/test split)
    matches_men = matches_men.merge(
        mens_tournaments[["tournament_id", "year"]],
        on="tournament_id",
        how="left",
    )

    return matches_men


def basic_type_cleaning(matches: pd.DataFrame) -> pd.DataFrame:
    """
    minimal, safe type cleaning

    - Parse match_date as datetime (date)
    - Leave match_time as string
    - Ensure key boolean-ish columns are 0/1 integers
    """
    df = matches.copy()

    # Dates: parse to datetime; errors='coerce' turns invalid values into NaT
    # (missing or malformed dates become NaT instead of crashing)
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    # Boolean-like columns that are coded as 0/1 in the CSV
    bool_cols = [
        "group_stage",
        "knockout_stage",
        "replayed",
        "replay",
        "extra_time",
        "penalty_shootout",
        "home_team_win",
        "away_team_win",
        "draw",
    ]
    for col in bool_cols:
        if col in df.columns:
            # convert to numeric first, then fill missing with 0 and cast to small int
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    # scores: ensure they are numeric
    score_cols = [
        "home_team_score",
        "away_team_score",
        "home_team_score_margin",
        "away_team_score_margin",
        "home_team_score_penalties",
        "away_team_score_penalties",
    ]
    for col in score_cols:
        if col in df.columns:
            # keep as numeric (floats) so we can do arithmetic or checks later
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def split_train_test(matches_men: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split mens matches into train (1930–2018) and test (2022).

    Expects a 'year' column already present (via merge from tournaments).
    """
    if "year" not in matches_men.columns:
        raise ValueError(
            "Expected a 'year' column in matches_men. "
            "Make sure you merged in tournaments before splitting."
        )

    train_mask = (matches_men["year"] >= 1930) & (matches_men["year"] <= 2018)
    test_mask = matches_men["year"] == 2022

    # training data: historic men's World Cups up through 2018
    matches_train = matches_men.loc[train_mask].copy()
    # test data: 2022 men's World Cup only
    matches_test = matches_men.loc[test_mask].copy()

    return matches_train, matches_test


def main() -> None:
    """loading, cleaning, splitting, and writing out CSVs"""
    # Make sure output folder exists
    DATA_CLEAN_DIR.mkdir(exist_ok=True)

    # Step 1: load raw CSVs
    matches_raw, tournaments_raw = load_raw_data()

    # Step 2: restrict to mens World Cups and attach the tournament year
    matches_men = filter_mens_world_cup(matches_raw, tournaments_raw)

    # Step 3: do light type cleaning (dates, booleans, scores)
    matches_men = basic_type_cleaning(matches_men)

    # Step 4: split into train (1930–2018) and test (2022)
    matches_train, matches_test = split_train_test(matches_men)

    train_path = DATA_CLEAN_DIR / "matches_train.csv"
    test_path = DATA_CLEAN_DIR / "matches_test.csv"

    matches_train.to_csv(train_path, index=False)
    matches_test.to_csv(test_path, index=False)

    print(f"Wrote train set to: {train_path}")
    print(f"Wrote test set to:  {test_path}")


if __name__ == "__main__":
    main()

