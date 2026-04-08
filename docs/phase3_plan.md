# Phase 3 Plan: Additional Feature Sources

**Date:** 2026-04-07  
**Starting point:** Phase 1 result -- 39 features, acc=0.625, F1=0.578, draw_recall=0.40

---

## Lessons from Phase 1 and 2 that constrain Phase 3

1. **900 rows cannot absorb many more features.** Going from 37 to 39 features helped (with regularization). Going to 64 features hurt badly. The ceiling is roughly 40-45 features.
2. **Only low-correlation features are safe to add.** The +2 intl H2H features worked because they correlated at only r=0.287 with existing H2H. High-correlation additions (intl ELO vs WC ELO at r=0.606) caused draw prediction collapse.
3. **`min_samples_leaf=5` is essential.** This regularization prevents overfitting when adding features. Any new feature must be tested with this setting.
4. **Temporal CV is the honest evaluation.** Test set results (64 matches) have high variance. Always validate with 4-fold temporal CV before trusting test metrics.

**Strategy for Phase 3:** Add 1-3 features at a time. Validate each addition with temporal CV. If CV F1 doesn't improve, drop it. Never exceed ~45 total features.

---

## 3a. Home Continent Advantage (highest priority)

**What:** Flag whether each team is playing on their home continent.

**Why it's promising:**
- European teams historically win 60%+ of WC matches played in Europe vs ~50% elsewhere
- South American teams overperform at South American WCs (Uruguay 1930, Brazil 1950/2014, Argentina 1978)
- This signal is NOT captured by any existing feature -- `home_is_host` only flags the specific host nation (1-2 matches), not the entire continent
- Only adds 2 features (won't trigger feature bloat)
- Low correlation with existing features (binary flag vs continuous ELO/rates)

**New features (2):**
- `home_on_home_continent` -- 1 if the home team's confederation matches the host country's confederation
- `away_on_home_continent` -- same for away team

**Implementation:**
1. Create a team -> confederation mapping. Sources:
   - Derive from the Kaggle intl dataset: teams that play in "UEFA Euro qualification" are UEFA members, teams in "African Cup of Nations qualification" are CAF members, etc.
   - Or hardcode from FIFA's confederation list (only ~210 teams total)
2. Create a host_country -> confederation mapping for each WC year (17 host countries)
3. For each match, check if each team's confederation matches the host's confederation

**Data needed:** No external download required. The confederation can be inferred from the intl results dataset already in `data_clean/`.

**Effort:** ~1 hour. Add 1 function to `feature_engineering.py`.

**Risk:** Low. Only 2 binary features. Cannot cause feature bloat.

---

## 3b. FIFA World Rankings (medium priority)

**What:** Official FIFA ranking points for each team before each WC match.

**Source:** Kaggle dataset `cashncarry/fifaworldranking` -- ~63K rows, monthly rankings 1992-2024.

**Why it's promising:**
- FIFA ranking points use a different formula than ELO (they weight tournament importance, goal margin, opponent strength differently)
- Ranking points are more widely recognized and may capture "reputation effects" that ELO misses
- Published research shows FIFA rank + ELO together outperform either alone

**Why it's risky:**
- Only available from 1993 onward -- covers WCs 1994-2022 (8 tournaments, ~512 training matches out of 900)
- Pre-1994 matches get NaN, filled with defaults -- effectively 40% of training data has no ranking signal
- Correlation with ELO is likely moderate-to-high (both measure team strength)

**New features (2-4):**
- `home_fifa_points`, `away_fifa_points` -- ranking points at most recent ranking date before the match
- `fifa_points_diff` -- difference
- Optionally: `home_fifa_rank`, `away_fifa_rank` (ordinal rank position)

**Implementation:**
1. Download the rankings CSV
2. For each WC match, find the most recent ranking date before the match date
3. Look up each team's ranking points at that date
4. Join to the feature matrix

**Data needed:** ~1.8 MB CSV download from Kaggle.

**Effort:** ~2 hours. New function in `feature_engineering.py` + download.

**Risk:** Medium. Moderate correlation with ELO may not add value. Missing data for pre-1994 matches. Must validate with temporal CV before committing.

**Testing strategy:** Add ONLY `fifa_points_diff` first (1 feature). If CV improves, add the base features.

---

## 3c. Qualifying Path Strength (medium priority)

**What:** How dominant was each team in their qualifying campaign?

**Source:** Already available in `data_clean/international_results.csv` -- WC qualification matches are tagged as `tournament = "FIFA World Cup qualification"`.

**Why it's promising:**
- A team that topped their qualifying group with 10 wins is different from one that barely scraped through a playoff
- Captures recent competitive form against opponents of known strength
- Different signal than rolling form (qualifying is a multi-match campaign, not just last 5 games)

**Why it's risky:**
- Qualifying didn't exist before 1934. Even after, formats changed dramatically over decades
- Different confederations have vastly different qualifying difficulty (UEFA vs CONCACAF vs AFC)
- Host nations don't qualify -- they get NaN for qualifying features
- Adds complexity to handle cross-confederation normalization

**New features (2-3):**
- `home_qual_win_rate` -- win rate in the qualifying campaign for the current WC
- `away_qual_win_rate`
- `qual_win_rate_diff`

**Implementation:**
1. From the intl dataset, filter to WC qualification matches
2. Group by the qualifying cycle (e.g., 2020-2022 qualifiers for 2022 WC)
3. For each team, compute their qualifying record
4. Join to WC match data

**Data needed:** Already available in `international_results.csv`.

**Effort:** ~3 hours. Requires careful handling of qualifying cycles and host-nation exceptions.

**Risk:** Medium-high. Many edge cases. Pre-1970s data is thin. Host nations need special handling.

---

## 3d. Squad Market Value (low priority, high effort)

**What:** Total squad market value from Transfermarkt data.

**Source:** Kaggle `davidcariboo/player-scores` (~205 MB).

**Why it's promising:** Published research consistently shows squad market value is one of the strongest single predictors of WC success.

**Why it's impractical for this project:**
- Requires mapping ~700 players per WC to their national team squads
- Need external squad list data (Wikipedia, FIFA archives)
- Only covers 1998-2022 (6 WCs, ~384 training matches)
- ~1-2 days of effort for a feature that applies to <half the training data

**Recommendation:** Skip unless specifically needed for the report.

---

## Priority Ordering

| Feature | New Cols | CV Improvement (est.) | Effort | Risk | Priority |
|---------|:--------:|:---------------------:|:------:|:----:|:--------:|
| 3a. Continent advantage | 2 | +0.5-1.5 pp F1 | 1 hour | Low | **Do first** |
| 3b. FIFA rankings | 1-2 | +0-2 pp F1 | 2 hours | Medium | Do second |
| 3c. Qualifying path | 2-3 | +0-1 pp F1 | 3 hours | Med-high | Optional |
| 3d. Squad market value | 1-2 | +1-3 pp F1 | 1-2 days | High | Skip |

**Total if doing 3a + 3b:** 39 -> 42-43 features. Within the safe range.

---

## Implementation Checklist

For each new feature:

- [ ] Add computation function to `feature_engineering.py`
- [ ] Add to `FEATURE_COLS_BASE` list
- [ ] Add NaN fill strategy to `fill_missing()`
- [ ] Run `python3 scripts/feature_engineering.py` to regenerate CSVs
- [ ] Run 4-fold temporal CV with Phase 1 tuned RF settings (300 trees, min_leaf=5)
- [ ] Compare CV F1 with current best (0.534)
- [ ] If improved: keep. If not: revert.
- [ ] Run on test set only AFTER CV validation

---

## Expected Final State

After Phase 3a + 3b (if both help):

| Metric | Original | Phase 1 | Phase 3 (est.) |
|--------|:--------:|:-------:|:--------------:|
| Features | 37 | 39 | 41-43 |
| CV Accuracy | 0.562 | 0.574 | ~0.58 |
| CV Macro F1 | 0.488 | 0.534 | ~0.54-0.55 |
| Test Accuracy | 0.609 | 0.625 | ~0.63-0.65 |
| Test Macro F1 | 0.556 | 0.578 | ~0.58-0.60 |
