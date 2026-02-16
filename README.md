## World Cup Match Subset for Milestone 1

### 1. Source

- **Original database**: *The Fjelstul World Cup Database v.1.2.0* (R package `worldcup`).
- **Author**: Joshua C. Fjelstul, Ph.D.
- **Repository**: https://www.github.com/jfjelstul/worldcup
- **License**: CC-BY-SA 4.0.

### 2. Files and folders in this project

- `files_needed/matches.csv`: all World Cup matches (men's and women's).
- `files_needed/tournaments.csv`: all tournaments (men's and women's), used to identify men's World Cups and the tournament year.

- `data_clean/`  
  Cleaned, analysis-ready datasets produced by the Python script in `scripts/`.
  - `data_clean/matches_train.csv`: men's World Cup matches from **1930–2018** (training set), one row per match.
  - `data_clean/matches_test.csv`: men's World Cup matches from **2022** (test set), one row per match.

- `scripts/`  
  Python scripts for data management and cleaning.
  - `scripts/clean_worldcup.py`: loads raw CSVs from `files_needed/`, filters to men's World Cups, performs light cleaning, and writes train/test CSVs into `data_clean/`.

- `docs/`  
  Documentation and metadata for this project.
  - `docs/README_worldcup_subset.md`: this file, describing data sources, processing steps, and project structure.
  - `docs/worldcup_subset_codebook.csv`: data dictionary for key variables in the cleaned match datasets.

- `requirements.txt`  
  Python dependency file (currently `pandas>=2.0.0`) used to recreate the environment with `pip install -r requirements.txt`.

### 3. Units of analysis

- `matches.csv`, `matches_train.csv`, `matches_test.csv`: **one row per match per tournament**.
- `tournaments.csv`: **one row per tournament**.

### 4. Cleaning and transformations

All transformations are scripted in `scripts/clean_worldcup.py`:

1. **Load raw data**
   - Load `files_needed/matches.csv` and `files_needed/tournaments.csv`.

2. **Filter to men's World Cups**
   - Identify tournaments whose `tournament_name` contains `"FIFA Men's World Cup"`.
   - Keep only matches whose `tournament_id` appears in those tournaments.
   - Merge the `year` column from `tournaments.csv` into the match data.

3. **Basic type cleaning**
   - Parse `match_date` as a date.
   - Convert boolean-like columns (`group_stage`, `knockout_stage`, `replayed`, `replay`,
     `extra_time`, `penalty_shootout`, `home_team_win`, `away_team_win`, `draw`)
     into integer 0/1.
   - Ensure score-related columns (`home_team_score`, `away_team_score`, margins, penalties)
     are numeric.

4. **Train / test split**
   - **Training set**: all men's World Cup matches with `year` in `[1930, 2018]`.
   - **Test set**: all men's World Cup matches with `year == 2022`.

The raw CSVs in `files_needed/` are not modified by the script; new cleaned files are written to `data_clean/`.

### 5. Key variables used

- `tournament_id` — ID of the tournament (e.g., `WC-2018`).
- `tournament_name` — Name of the tournament.
- `year` — Year of the tournament (merged from `tournaments.csv`).
- `match_id` — ID of the match (per tournament and year).
- `match_date` — Date of the match.
- `stage_name` — Stage of the tournament (e.g., "group stage", "semi-finals").
- `group_stage`, `knockout_stage` — Indicators for group vs knockout stage.
- `home_team_name`, `away_team_name` — Team names.
- `home_team_code`, `away_team_code` — 3-letter team codes.
- `home_team_score`, `away_team_score` — Goals scored by each team.
- `extra_time` — Indicator for whether the match went to extra time.
- `penalty_shootout` — Indicator for whether there was a penalty shootout.
- `score_penalties`, `home_team_score_penalties`, `away_team_score_penalties` — Penalty shootout result (if applicable).
- `result` — Outcome category (home team win, away team win, draw, replayed).
- `home_team_win`, `away_team_win`, `draw` — Outcome indicators.

### 6. Running

- run from the project root:

  ```bash
  python3 scripts/clean_worldcup.py
  ```
