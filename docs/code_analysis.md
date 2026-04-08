# Detailed Code Analysis

**Project:** FIFA Men's World Cup Match Outcome Prediction  
**Repository:** `datascienceproject/`  
**Analysis date:** 2026-04-07  
**Total codebase:** 2,564 lines across 3 Python scripts

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Script 1: clean_worldcup.py](#2-script-1-clean_worldcuppy)
3. [Script 2: feature_engineering.py](#3-script-2-feature_engineeringpy)
4. [Script 3: feature_engineering_expanded.py](#4-script-3-feature_engineering_expandedpy)
5. [Data Flow](#5-data-flow)
6. [Design Patterns](#6-design-patterns)
7. [Leakage Prevention Analysis](#7-leakage-prevention-analysis)
8. [Performance Characteristics](#8-performance-characteristics)
9. [External Data Integration](#9-external-data-integration)
10. [Experimental Results Summary](#10-experimental-results-summary)

---

## 1. Repository Structure

```
datascienceproject/
├── scripts/
│   ├── clean_worldcup.py                  (173 lines)  Phase 1: data cleaning
│   ├── feature_engineering.py            (1,985 lines)  Core pipeline + all features
│   └── feature_engineering_expanded.py     (406 lines)  Phase 2: expanded training
│
├── data_clean/
│   ├── matches_train.csv                  900 rows x 38 cols (WC 1930-2018)
│   ├── matches_test.csv                    64 rows x 38 cols (WC 2022)
│   ├── features_train.csv                 900 rows x 80 cols (all features)
│   ├── features_test.csv                   64 rows x 80 cols
│   ├── features_expanded_train.csv     28,227 rows x 37 cols (Phase 2)
│   ├── features_expanded_test.csv          64 rows x 37 cols
│   ├── international_results.csv       49,287 rows (Kaggle, 1872-2026)
│   ├── fifa_rankings.csv               67,894 rows (FIFA, 1992-2024)
│   ├── players.csv                     47,702 rows (Transfermarkt)
│   ├── player_valuations.csv          616,377 rows (Transfermarkt, 2004-2026)
│   └── wc_squads.csv                   13,843 rows (WC squad rosters, 1930-2022)
│
├── files_needed/
│   ├── matches.csv                        Raw source: all WC matches (men's + women's)
│   └── tournaments.csv                    Raw source: tournament metadata
│
├── docs/
│   ├── summary.md                         Project overview
│   ├── improvement_plan.md                3-phase improvement strategy
│   ├── phase2_results.md                  Expanded training results (negative)
│   ├── phase3_results.md                  Feature addition results
│   ├── phase3_summary.md                  Detailed 3a/3b/3c/3d analysis
│   ├── code_analysis.md                   This document
│   └── worldcup_subset_codebook.csv       Data dictionary
│
├── figures/                               11 PNG visualizations
├── Data Science Report.ipynb              Part 1: EDA
├── Part2_Models_and_Results.ipynb         Part 2: Modeling
├── requirements.txt                       Dependencies
└── README.md                              Project documentation
```

---

## 2. Script 1: `clean_worldcup.py`

**Lines:** 173  
**Purpose:** Load raw data, filter to men's WC, clean types, split train/test.

### Function Map

| Function | Lines | Purpose |
|----------|:-----:|---------|
| `load_raw_data()` | 27-41 | Read `matches.csv` and `tournaments.csv` from `files_needed/` |
| `filter_mens_world_cup()` | 44-69 | Keep only men's WC matches via tournament name matching |
| `basic_type_cleaning()` | 72-118 | Parse dates, cast booleans to int8, ensure numeric scores |
| `split_train_test()` | 121-141 | Year-based split: train (1930-2018), test (2022) |
| `main()` | 144-172 | Orchestrate the pipeline |

### Design Decisions

**Path handling:** Uses `pathlib.Path(__file__).resolve().parents[1]` to locate the project root, making the script runnable from any working directory.

**Defensive type casting:** The `basic_type_cleaning()` function uses `pd.to_numeric(col, errors='coerce').fillna(0).astype('int8')` for boolean columns. The `errors='coerce'` safely handles any non-numeric values by converting them to NaN, which is then filled with 0. This is more robust than a bare `.astype(int)` which would crash on malformed data.

**Copy semantics:** Every function operates on `df.copy()`, preventing mutation of input DataFrames. This is a defensive pattern that prevents subtle bugs when functions are called in sequence.

**Minimal cleaning:** The script deliberately does NO feature engineering. Its only job is to produce clean, properly-typed DataFrames. This separation of concerns makes the pipeline modular.

---

## 3. Script 2: `feature_engineering.py`

**Lines:** 1,985  
**Purpose:** Compute all features for WC match prediction. This is the core analytical code.

### Architecture Overview

The script has five major sections:

```
Lines    1-255    Constants, mappings, helper functions
Lines  256-1115   Feature computation functions (14 functions)
Lines 1116-1532   International feature functions (8 functions)
Lines 1533-1716   NaN fill constants and fill_missing()
Lines 1717-1985   Output selection, summary, and main()
```

### Constants and Mappings (Lines 1-255)

The script defines extensive mapping dictionaries for normalizing team names across different data sources:

| Constant | Entries | Purpose |
|----------|:-------:|---------|
| `POST_MATCH_COLS` | 13 | Score/outcome columns excluded from output (prevents leakage) |
| `STAGE_ORDINAL` | 7 | Maps stage names to ordinal values (group=0 through final=6) |
| `HOST_TEAM_ALIASES` | 1 | Handles "South Korea" vs "Korea Republic" for host matching |
| `INTL_TEAM_NAME_MAP` | 19 | Normalizes Kaggle international dataset names to WC names |
| `WC_TO_INTL_LOOKUP` | 1 | Maps "West Germany" to "Germany" for international lookups |
| `TEAM_CONFEDERATION` | 85 | Every WC team -> FIFA confederation (UEFA/CONMEBOL/etc.) |
| `HOST_CONFEDERATION` | 22 | WC year -> host confederation |
| `FIFA_RANK_NAME_MAP` | 6 | FIFA ranking names -> WC names |
| `TM_COUNTRY_MAP` | 7 | Transfermarkt country names -> WC names |
| `INTL_ELO_K_FACTORS` | 14 | K-factor by tournament type for variable ELO |
| `WC_START_DATES` | 5 | WC tournament start dates (2006-2022) |

**Why so many mappings?** Four external data sources (international results, FIFA rankings, Transfermarkt, WC squads) each use different naming conventions. The mapping layer ensures all team names resolve to the canonical form used in the WC dataset. Without this, lookups would silently fail for teams like "South Korea" (which appears as "Korea Republic", "Korea, South", or "South Korea" depending on the source).

### Core Feature Functions (Lines 256-1115)

#### `load_data()` (L261-272, 12 lines)

Concatenates train and test CSVs, parses dates, sorts chronologically. The concatenation is critical: features must be computed on the combined dataset so that test rows get features derived from train-period data.

#### `load_international_data()` (L275-318, 44 lines)

Loads the 49K international results CSV, normalizes team names, derives result columns (`home team win`/`away team win`/`draw` from scores). Returns `None` if the file is missing, enabling graceful degradation.

**Key detail:** Uses `np.where()` for vectorized result derivation rather than `apply()`, which is ~100x faster on 49K rows.

#### `compute_team_history()` (L324-429, 106 lines)

The most complex WC-only feature function. Computes expanding-window historical stats (win rate, draw rate, goals per game, goals conceded) for each team.

**Algorithm:**
1. Transforms wide-form match data (one row per match) into long-form (two rows per match -- one per team)
2. Aggregates stats per (team, date) to handle same-day matches
3. Computes cumulative sums within each team, shifted by 1 date-slot
4. Pivots back to wide-form and merges onto the match DataFrame

**The date-level shift:** Instead of shifting by row (which would leak same-day information), the function uses `.groupby('team').cumsum().groupby('team').shift(1)` at the date level. This means if Brazil plays at 3pm and 7pm on the same day, the 7pm match does NOT see the 3pm result -- both see only pre-day data. This is more conservative than necessary (WC teams don't play twice on the same day) but ensures correctness.

#### `compute_match_context()` (L435-441, 7 lines)

Maps `group_stage`, `knockout_stage`, and `stage_name` to numeric features. The `stage_ordinal` encoding (group=0 through final=6) captures tournament progression as a continuous variable.

#### `compute_host_advantage()` (L447-464, 18 lines)

Flags whether each team is the host nation. Uses a `HOST_TEAM_ALIASES` dictionary to handle name mismatches (e.g., "Korea Republic" in the WC data vs "South Korea" as a host country name).

#### `compute_wc_experience()` (L470-500, 31 lines)

Counts distinct prior WC tournaments each team appeared in. Uses both home and away team appearances to build a complete team -> years mapping, then counts years strictly before the current tournament.

#### `compute_head_to_head()` (L506-574, 69 lines)

Tracks all prior WC meetings between two teams. Uses a `frozenset` key to normalize pair order (France vs Brazil = Brazil vs France), then an iterative approach with a `history` accumulator to process matches chronologically.

**Subtle correctness detail:** When recording who won a prior meeting, the function correctly handles perspective reversal. If Team A beat Team B when A was "home", and now B is "home", that prior result counts as an away win for the current home team.

#### `compute_elo_ratings()` (L580-633, 54 lines)

Standard ELO implementation with K=32. All teams start at 1500. For each match (chronologically sorted), pre-match ratings are recorded as features, then ratings are updated.

**Why K=32?** This is the standard chess ELO K-factor, commonly used in football analytics. Higher K means ratings react more to recent results; lower K means more stable ratings. K=32 is a moderate choice that balances responsiveness and stability.

#### `compute_rolling_form()` (L639-733, 95 lines)

Last 5 WC matches' win rate and goals per game. Uses the same long-form transformation as `compute_team_history()` but with a rolling window instead of expanding window.

**Cold-start fallback hierarchy:**
1. Rolling 5-match window (if >= 1 prior match)
2. All-time team average (if team exists but < 5 matches)
3. Default: 0.33 win rate, 0.0 goals (complete cold-start)

#### `compute_rest_days()` (L739-798, 60 lines)

Days since each team's previous WC match. Capped at 365 to prevent extreme outliers (some teams have 4+ year gaps between WCs).

#### `compute_continent_advantage()` (L804-829, 26 lines) -- Phase 3a

Binary flags for whether each team is playing on their home continent. Uses the `TEAM_CONFEDERATION` and `HOST_CONFEDERATION` dictionaries.

**Why this feature works when others don't:** It's a binary geographic flag with near-zero correlation to existing continuous features (ELO, win rates). It captures a distinct signal: the "continental familiarity" effect (similar climate, food, travel fatigue, fan support from diaspora communities).

#### `compute_fifa_rankings()` (L880-912, 33 lines) -- Phase 3b

Looks up each team's FIFA ranking points at the most recent ranking date before the match. Uses `_build_ranking_lookup()` to create a per-team chronological list, then `_lookup_ranking_before()` with `bisect.bisect_right()` for O(log n) lookup.

**Coverage note:** Returns NaN for pre-1993 matches (FIFA rankings didn't exist). The `fill_missing()` function fills these with the median of existing values.

#### `compute_qualifying_record()` (L918-999, 82 lines) -- Phase 3c

For each team at each WC, computes their win rate in the qualifying campaign. Maps qualifying matches to target WC years using a date-window heuristic (qualifiers in the 4 years before each WC).

**Edge cases handled:**
- Host nations (don't qualify) -> NaN, filled with 0.33
- Pre-qualifying-era WCs (1930) -> NaN, filled with 0.33
- Uses `_intl_name()` for team lookup (handles "West Germany" -> "Germany")

#### `compute_squad_market_value()` (L1071-1115, 45 lines) -- Phase 3d

Computes squad market value from Transfermarkt data. The heavy lifting is in `_build_squad_value_lookup()` (L1005-1068, 64 lines):

1. Joins player valuations with nationality from the players table
2. For each WC year (2006-2022), filters to a 2-year window before the WC
3. Takes the latest valuation per player within that window
4. For each country, sums the top 23 players' values
5. Requires minimum 11 valued players (to avoid unreliable estimates from sparse coverage)

**Values stored in millions EUR** for readability (€1,433M for England 2022 instead of €1,433,000,000).

#### `compute_interactions()` (L1121-1162, 42 lines)

Creates three WC-only interaction features:
- `home_attack_x_away_defense`: home goals/game * away goals conceded/game
- `away_attack_x_home_defense`: mirror
- `elo_x_form_diff`: ELO difference * rolling form difference (strength * momentum)

Conditionally adds three international interaction features if intl data is loaded:
- `intl_elo_x_form_diff`, `home_intl_attack_x_away_defense`, `away_intl_attack_x_home_defense`

### International Feature Functions (Lines 1116-1532)

These functions compute features from the 49K international match history. They follow a different architecture than the WC-only functions: instead of operating on the combined WC DataFrame directly, they build **lookup structures** from all international matches, then probe those structures for each WC match row.

#### Lookup-Probe Architecture

```
Step 1: Build lookup from all 49K international matches
        (process chronologically, accumulate per-team stats)

Step 2: For each WC match row, probe the lookup
        (binary search by date to find pre-match value)
```

This is more efficient than the WC-only approach (which uses pandas groupby/cumsum/shift) because:
- The lookup is built once from 49K rows
- Probing is O(log n) per row via `bisect`
- Total: O(49K) build + O(964 * log(49K)) probe ≈ O(49K + 15K) ≈ O(64K)

#### `_build_intl_elo_history()` (L1168-1214, 47 lines)

Processes all 49K international matches chronologically. For each match:
1. Looks up K-factor by tournament type (WC=60, Euro=50, qualifiers=40, friendlies=20)
2. Computes expected scores using standard ELO formula
3. Updates ratings
4. Records `(date_ordinal, new_elo)` in per-team lists

Returns `dict[str, list[tuple[float, float]]]` -- team name to chronological ELO history.

**Variable K-factor:** Unlike the WC-only ELO (flat K=32), international ELO uses tournament-weighted K-factors. WC results move ratings 2x more than friendlies. This is a well-established improvement in football ELO systems (FIFA's own rating system does this).

#### `_lookup_elo_before()` (L1217-1234, 18 lines)

Binary search for the most recent ELO before a given date. Uses `bisect.bisect_left(dates, date_ord) - 1` to find the latest entry strictly before the query date. Returns 1500.0 if no prior data exists.

#### `compute_intl_elo()` (L1237-1267, 31 lines)

Orchestrates the build-and-probe. Calls `_build_intl_elo_history()` once, then probes for each WC match. Uses `_intl_name()` to map WC team names (e.g., "West Germany") to international team names (e.g., "Germany") for the lookup.

#### `_build_team_match_log()` (L1273-1293, 21 lines)

Builds per-team chronological match log for rolling form computation. Each entry is `(date_ordinal, win_flag, goals_for)`.

#### `compute_intl_rolling_form()` (L1296-1342, 47 lines)

For each WC match, finds each team's last 5 international matches (any tournament) using binary search, then computes win rate and goals per game. Falls back to 0.33/0.0 for teams with no prior international matches.

#### `compute_intl_h2h()` (L1348-1419, 72 lines)

Builds a `frozenset`-keyed dictionary of all prior meetings between every pair of teams. For each WC match, binary-searches the pair's history to find all meetings before the match date, then counts wins/draws from the current home team's perspective.

**Performance detail:** The pair dictionary is built once from 49K matches. For each WC match, the inner loop iterates only over prior meetings between the specific pair (typically 10-30 matches), not all 49K. With `bisect` limiting the search, total runtime is ~5 seconds.

#### `compute_intl_history()` (L1425-1532, 108 lines)

The most complex international function. Builds per-team cumulative stats from all international matches, using an accumulator pattern:

```python
cum[team] = [wins, draws, goals_for, goals_against, matches]
```

Updated after each match, with the current state appended to a per-team chronological list. Probed via binary search for each WC match.

**Subtle correctness:** The accumulator uses separate `cum` dictionaries (not the same as the output `team_cumulative` dict). After processing each match, the current cumulative state is *copied* to the output list. This ensures that lookup queries return the state at a specific point in time, not the final state.

### NaN Handling (Lines 1533-1716)

#### Fill Strategy Constants

19 constant lists categorize features by their appropriate NaN fill value:

| Category | Fill Value | Rationale | Example Features |
|----------|:----------:|-----------|-----------------|
| Rate features | 0.33 | Uniform 3-class prior | `home_hist_win_rate` |
| Count features | 0 | No prior history | `home_hist_matches_played` |
| Goals rate features | 0.0 | Conservative cold-start | `home_hist_goals_per_game` |
| ELO base | 1500.0 | Starting ELO | `home_elo` |
| ELO diff | 0.0 | Equal teams | `elo_diff` |
| Rolling win rate | 0.33 | Uniform prior | `home_rolling5_win_rate` |
| Rolling goals | 0.0 | Conservative | `home_rolling5_goals_pg` |
| Rest days | Median | Central tendency | `home_rest_days` |
| Interactions | 0.0 | Product of zero-filled | `elo_x_form_diff` |
| Continent | 0 | Not on home continent | `home_on_home_continent` |
| FIFA points | Median | Central tendency | `home_fifa_points` |
| Qualifying rate | 0.33 | Uniform prior | `home_qual_win_rate` |
| Squad value | Median | Central tendency | `home_squad_value` |

#### `fill_missing()` (L1644-1716, 73 lines)

Uses a `_safe_fill()` helper to only fill columns that exist (enabling graceful degradation when optional data sources are missing). The function is idempotent -- running it twice produces the same result.

### Output and Pipeline (Lines 1717-1985)

#### Feature Column Lists

The script defines three feature lists:

| List | Count | Contents |
|------|:-----:|---------|
| `FEATURE_COLS_BASE` | 48 | WC-only (37) + Phase 3 features (11) |
| `FEATURE_COLS_INTL` | 27 | International features |
| `FEATURE_COLS` | Dynamic | `BASE + INTL` if intl data available, else `BASE` |

#### `main()` (L1876-1981, 106 lines)

The pipeline orchestrator. Execution flow:

```
1. load_data()                    → 964 WC matches
2. load_international_data()      → 49,287 intl matches (or None)
3. Validate team name coverage
4. compute_team_history()         → 12 WC expanding-window features
5. compute_match_context()        →  3 stage features
6. compute_host_advantage()       →  2 host flags
7. compute_wc_experience()        →  3 WC appearance features
8. compute_head_to_head()         →  5 WC H2H features
9. compute_elo_ratings()          →  3 WC ELO features
10. compute_rolling_form()        →  4 WC rolling features
11. compute_rest_days()           →  2 rest day features
12. compute_continent_advantage() →  2 continent flags          (Phase 3a)
13. compute_fifa_rankings()       →  3 FIFA ranking features    (Phase 3b)
14. compute_qualifying_record()   →  3 qualifying features      (Phase 3c)
15. compute_squad_market_value()  →  3 squad value features     (Phase 3d)
16. compute_intl_elo()            →  3 intl ELO features        (if intl data)
17. compute_intl_rolling_form()   →  4 intl rolling features    (if intl data)
18. compute_intl_h2h()            →  5 intl H2H features        (if intl data)
19. compute_intl_history()        → 12 intl history features    (if intl data)
20. compute_interactions()        →  3-6 interaction features
21. fill_missing()                → NaN cleanup
22. build_output() + split + save → CSV output
```

Steps 12-15 (Phase 3) and 16-19 (international) are conditional: they run if the corresponding data files exist, and gracefully produce NaN columns if not.

---

## 4. Script 3: `feature_engineering_expanded.py`

**Lines:** 406  
**Purpose:** Phase 2 experiment -- train on 28K competitive international matches instead of 900 WC matches. Imports functions from `feature_engineering.py`.

### Key Design Differences from Main Pipeline

| Aspect | Main Pipeline | Expanded Pipeline |
|--------|:------------:|:-----------------:|
| Training data | 900 WC matches | 28,227 competitive intl matches |
| Test data | WC 2022 from WC dataset | WC 2022 from WC dataset |
| Features used | 75 (base + intl) | 31 (intl only + context) |
| WC-only features | Yes | No (not applicable to non-WC matches) |
| Sample weighting | None | Tournament-type weights (WC=3x, continental=2x) |
| Feature computation | Embedded functions | Imports from main pipeline |

**Why a separate script?** The expanded pipeline has fundamentally different training data (international matches, not WC matches). Merging this into the main script would have added conditional logic throughout, making both pipelines harder to understand and maintain.

**Import mechanism:** Uses `sys.path.insert(0, str(_SCRIPTS_DIR))` to import from `feature_engineering.py`. The imported functions (`compute_intl_elo`, `compute_intl_rolling_form`, etc.) work correctly because they take `(df, intl)` where `df` can be any DataFrame of matches to annotate.

**Result:** This experiment showed that expanding training to non-WC matches hurts performance due to domain mismatch (details in `docs/phase2_results.md`).

---

## 5. Data Flow

```
files_needed/matches.csv + tournaments.csv
        │
        ▼  clean_worldcup.py
data_clean/matches_train.csv (900 x 38) + matches_test.csv (64 x 38)
        │
        │   ┌── data_clean/international_results.csv (49K, Kaggle)
        │   ├── data_clean/fifa_rankings.csv (68K, Dato-Futbol)
        │   ├── data_clean/players.csv (48K, Transfermarkt)
        │   └── data_clean/player_valuations.csv (616K, Transfermarkt)
        │
        ▼  feature_engineering.py
data_clean/features_train.csv (900 x 80) + features_test.csv (64 x 80)
        │
        ▼  Part2_Models_and_Results.ipynb
        7 models → temporal CV → test evaluation → results
```

---

## 6. Design Patterns

### Pattern 1: Concatenate-Compute-Split

The main pipeline concatenates train and test before computing features, then splits afterward:

```python
df = pd.concat([train, test])       # 964 rows
df = compute_features(df)            # Features use only prior data
train_out = df[df["_split"] == "train"]
test_out = df[df["_split"] == "test"]
```

**Why:** This ensures test rows get features computed from the full training period. If features were computed separately on train and test, the test set would need access to the same historical data structures.

### Pattern 2: Build-Lookup-Probe (International Features)

```python
lookup = _build_history(all_49K_matches)    # O(n) build
for _, row in wc_matches.iterrows():        # O(m * log n) probe
    value = _lookup_before(lookup, team, date)
```

**Why:** Separating build and probe makes the code cleaner and enables binary search for temporal queries. The alternative (concatenating 49K + 964 rows and using pandas groupby) would mix WC and international data and make leakage harder to verify.

### Pattern 3: Graceful Degradation

Every external data function returns NaN or is skipped when data files are missing:

```python
intl = load_international_data()
if intl is None:
    print("  File not found — skipping")
    return df_with_nan_columns
```

**Why:** The pipeline must work even if a user hasn't downloaded the optional datasets. The base 37 WC-only features require only the original WC data.

### Pattern 4: Name Normalization Layers

Three separate normalization pathways for the three directions of lookup:

| Direction | Function | Example |
|-----------|----------|---------|
| Intl dataset names → WC names | `normalize_team_name()` | "Korea Republic" → "South Korea" |
| WC names → Intl lookup keys | `_intl_name()` | "West Germany" → "Germany" |
| FIFA ranking names → WC names | `FIFA_RANK_NAME_MAP` | "IR Iran" → "Iran" |
| Transfermarkt names → WC names | `TM_COUNTRY_MAP` | "Korea, South" → "South Korea" |

---

## 7. Leakage Prevention Analysis

### WC-Only Features

| Feature Type | Anti-Leakage Mechanism |
|-------------|----------------------|
| Team history | Date-level cumulative shift: `.groupby('team').cumsum().groupby('team').shift(1)` -- shifts by date-slot, not row |
| ELO ratings | Pre-match ratings recorded before update; sequential processing |
| Rolling form | `shift(1).rolling(5, min_periods=1)` -- excludes current match |
| Head-to-head | Iterative accumulator: records current match for future rows AFTER computing features |
| Rest days | `groupby('team').shift(1)` on match dates |
| WC experience | `_prior_appearances()` counts years strictly before current year |
| Post-match columns | Explicit `POST_MATCH_COLS` list excluded from output |

### International Features

| Feature Type | Anti-Leakage Mechanism |
|-------------|----------------------|
| All lookups | `bisect.bisect_left(dates, date_ord) - 1` -- returns only entries strictly before query date |
| Same-day isolation | `bisect_left` treats same-day entries as "not before", so matches on the same day don't see each other's results |

### Phase 3 Features

| Feature Type | Anti-Leakage Mechanism |
|-------------|----------------------|
| FIFA rankings | `bisect.bisect_right(dates, date_ord) - 1` -- most recent ranking before match |
| Qualifying record | Only uses qualifying matches from the cycle BEFORE the current WC year |
| Squad value | 2-year valuation window ends before WC start date |
| Continent advantage | Static geographic mapping -- no temporal component |

### Train/Test Separation

The `_split` column tags each row. Features are computed on the combined dataset (ensuring test rows benefit from train-period history), but the train/test split is strictly enforced at output time.

---

## 8. Performance Characteristics

### Runtime (full pipeline with all data sources)

| Step | Time | Bottleneck |
|------|:----:|-----------|
| Load + clean data | <1s | I/O |
| WC-only features (9 functions) | ~3s | `compute_head_to_head` (Python for-loop over 964 rows) |
| Phase 3 features (4 functions) | ~2s | `_build_squad_value_lookup` (joins 616K valuations) |
| International features (4 functions) | ~5s | `_build_intl_elo_history` (for-loop over 49K rows) |
| Fill + output | <1s | I/O |
| **Total** | **~11s** | |

### Memory

- Peak: ~500 MB (when loading Transfermarkt valuations: 616K rows x 6 columns joined with 48K players)
- Steady-state: ~200 MB (lookup dictionaries for ~300 unique teams)

### Scalability Limitations

Two functions use Python-level for-loops that would be slow on larger datasets:

| Function | Loop Over | Rows | Would Need Vectorization At |
|----------|-----------|:----:|:---------------------------:|
| `compute_head_to_head()` | WC matches | 964 | ~10K rows |
| `compute_elo_ratings()` | WC matches | 964 | ~10K rows |
| `_build_intl_elo_history()` | All intl matches | 49K | ~500K rows |

At current scale (964 WC matches, 49K intl matches), performance is acceptable.

---

## 9. External Data Integration

### Data Sources Summary

| Source | File | Rows | Date Range | Used For |
|--------|------|:----:|:----------:|----------|
| Fjelstul WC Database | `matches.csv`, `tournaments.csv` | ~1,100 | 1930-2022 | Base WC data |
| Kaggle (martj42) | `international_results.csv` | 49,287 | 1872-2026 | Intl ELO, form, H2H, history |
| Dato-Futbol (GitHub) | `fifa_rankings.csv` | 67,894 | 1992-2024 | FIFA ranking points |
| Transfermarkt (R2 CDN) | `players.csv`, `player_valuations.csv` | 48K + 616K | 2004-2026 | Squad market value |
| Fjelstul (GitHub) | `wc_squads.csv` | 13,843 | 1930-2022 | Reference (not used in pipeline) |

### Download Commands

```bash
# International results (automatically downloaded by earlier work)
curl -sL "https://raw.githubusercontent.com/martj42/international_results/master/results.csv" \
  -o data_clean/international_results.csv

# FIFA rankings
curl -sL "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv" \
  -o data_clean/fifa_rankings.csv

# Transfermarkt players + valuations
curl -so data_clean/players.csv.gz \
  "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz" && gunzip -f data_clean/players.csv.gz
curl -so data_clean/player_valuations.csv.gz \
  "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/player_valuations.csv.gz" && gunzip -f data_clean/player_valuations.csv.gz

# WC squad rosters (reference)
curl -sL "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv/squads.csv" \
  -o data_clean/wc_squads.csv
```

---

## 10. Experimental Results Summary

### What Was Tried

| Experiment | Feature Count | Training Rows | CV F1 | Outcome |
|------------|:------------:|:-------------:|:-----:|:-------:|
| Original pipeline | 37 | 900 | 0.488 | Baseline |
| + RF tuning (min_leaf=5) | 37 | 900 | 0.531 | **+4.3 pp** |
| + continent advantage | 39 | 900 | **0.541** | **+1.0 pp** |
| + intl H2H | 39 | 900 | 0.534 | +0.3 pp |
| + FIFA rankings | 38-42 | 900 | 0.483-0.505 | Negative |
| + qualifying record | 38-42 | 900 | 0.510-0.522 | Neutral |
| + squad market value | 38-42 | 900 | 0.509-0.530 | Neutral |
| + all Phase 3 combined | 47 | 900 | 0.460 | Negative |
| + all 75 features | 75 | 900 | ~0.42 | Negative |
| Expanded training (Phase 2) | 31 | 28,227 | N/A | Negative (domain mismatch) |

### Why Most Features Didn't Help

On 900 training rows with a 3-class target (57% / 24% / 19% imbalance):

1. **Regularization > features.** `min_samples_leaf=5` delivered 4.3 pp F1 gain. The best feature addition (continent advantage) delivered 1.0 pp.

2. **Precision kills draws.** Features that precisely measure team strength (intl ELO, FIFA rankings, squad value) make the model overconfident, eliminating draw predictions entirely. WC-only ELO's 4-year update gaps create "productive imprecision" that enables draw prediction.

3. **900 rows ≈ 40 feature ceiling.** Every feature beyond ~40 dilutes Random Forest split quality. The model wastes splits on low-information features instead of the 5-10 most predictive ones.

4. **Domain specificity matters.** WC-only features outperform international features because WC matches have unique dynamics (neutral venue, "tournament DNA", tactical draws). Training on non-WC data introduces domain mismatch that no weighting scheme can overcome.

### Best Model

```
Features: 39 (37 WC-only + 2 continent advantage)
Model:    RandomForestClassifier(
              n_estimators=300,
              max_features='sqrt',
              min_samples_leaf=5,
              class_weight='balanced',
              random_state=42,
          )
CV F1:    0.541
CV Draw:  0.44
```
