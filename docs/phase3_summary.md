# Phase 3 Summary: Additional Feature Sources (3a, 3b, 3c, 3d)

**Date:** 2026-04-07  
**Starting point:** 37 WC-only features, RF(200 trees, default), acc=0.609, F1=0.556  
**After tuning:** 37 WC-only features, RF(300 trees, min_leaf=5), CV F1=0.531

---

## Overview

Phase 3 tested four additional feature sources on top of the original 37 WC-only features. Each was evaluated incrementally using 4-fold temporal walk-forward cross-validation with the tuned RF hyperparameters (n_estimators=300, max_features='sqrt', min_samples_leaf=5, class_weight='balanced').

The constraint from Phase 1 and Phase 2 experiments: 900 training rows cannot absorb more than ~2-3 additional features without draw prediction collapse.

---

## 3a. Home Continent Advantage -- SMALL WIN

**What it captures:** Whether each team is playing on their home continent (e.g., European teams at a European WC).

**Features added (2):**
- `home_on_home_continent` -- 1 if home team's FIFA confederation matches the host country's confederation
- `away_on_home_continent` -- same for away team

**Implementation:** Hardcoded team-to-confederation mapping (85 WC teams -> UEFA/CONMEBOL/CAF/AFC/CONCACAF/OFC) and host-country-to-confederation mapping (22 WC host entries).

**Why it works:**
- Captures a real effect: European teams historically win 60%+ of WC matches played in Europe vs ~50% elsewhere. South American teams dominate at South American WCs.
- Only 2 binary features -- minimal bloat risk
- Low correlation with existing features (binary geographic flag vs continuous ELO/rates)
- Not captured by `home_is_host` (which only flags the specific host nation, 1-2 matches per WC)

**Result:**
| Metric | Before (37 feat) | After (39 feat) |
|--------|:-:|:-:|
| CV F1 | 0.531 | **0.541** (+1.0 pp) |
| CV Draw Recall | 0.41 | **0.44** |
| Test Accuracy | 0.656 | 0.609 |
| Test F1 | 0.636 | 0.583 |

Test set results appear worse, but the 64-match test set has high variance. The CV metric (4 folds, ~250 validation matches total) is the reliable signal, and it shows a clear improvement.

**Verdict:** Keep. Only Phase 3 feature that consistently improves CV F1.

---

## 3b. FIFA World Rankings -- HURTS

**What it captures:** Official FIFA ranking points (a different formula than ELO, weighing tournament importance, goal margins, opponent strength).

**Data source:** Historical FIFA rankings from Dato-Futbol/fifa-ranking on GitHub (67,894 records, 1992-12 to 2024-09, 235 countries, 335 ranking dates).

**Features tested (1-3):**
- `home_fifa_points`, `away_fifa_points` -- ranking points at most recent ranking date before match
- `fifa_points_diff` -- difference

**Why it fails:**
1. **High correlation with ELO.** Both measure team strength. FIFA points add little beyond what ELO already captures. On 900 rows, the model can't learn the subtle differences between the two signals.
2. **Overconfidence effect.** Same pattern as international ELO (Phase 1 experiments): precise team-strength features make the model stop predicting draws. When the model "knows" France has 1845 points vs Australia's 1488, it always predicts a winner. But WC draws happen even between mismatched teams.
3. **Coverage gap.** Rankings start in 1993 -- 40% of training matches (pre-1994) get median-filled values, which are uninformative.

**Result:**
| Config | CV F1 | CV Draw Recall |
|--------|:-----:|:--------------:|
| Base (37) | 0.531 | 0.41 |
| + fifa_points_diff (38) | 0.497 | 0.29 |
| + all FIFA (40) | 0.483 | 0.23 |

Draw recall drops from 0.41 to 0.23. The model becomes a binary home-win/away-win classifier.

**Verdict:** Do not use. FIFA ranking points are redundant with ELO and destroy draw prediction on this dataset size.

---

## 3c. Qualifying Path Strength -- NEUTRAL

**What it captures:** Each team's win rate in their World Cup qualifying campaign (e.g., did they dominate qualifying or barely scrape through?).

**Data source:** Already available in `international_results.csv` -- 8,771 WC qualification matches tagged by tournament type. Grouped by qualifying cycle mapped to target WC year.

**Features tested (1-3):**
- `home_qual_win_rate`, `away_qual_win_rate` -- win rate in qualifying for the current WC
- `qual_win_rate_diff` -- difference

**Implementation notes:**
- Qualifying cycles mapped by date windows (4 years before each WC)
- Host nations don't qualify and get NaN (filled with 0.33 uniform prior)
- Uses `_intl_name()` for team name lookup (handles West Germany -> Germany)
- Pre-1934 WCs had no qualifying and get NaN

**Why it's neutral:**
- The signal is real (a team that dominated qualifying IS stronger), but it's already captured by ELO and historical win rates
- Host nations get NaN, removing the signal for 1-2 teams per WC
- Qualifying difficulty varies by confederation (UEFA qualifying is harder than CONCACAF), adding noise
- Adding 1-3 features to 900 rows: marginal signal diluted by the overhead

**Result:**
| Config | CV F1 | CV Draw Recall |
|--------|:-----:|:--------------:|
| Base (37) | 0.531 | 0.41 |
| + qual_win_rate_diff (38) | 0.522 | 0.40 |
| + all qualifying (40) | 0.510 | 0.40 |

Draw recall is preserved (unlike FIFA rankings), but F1 doesn't improve.

**Verdict:** Does not help, but also does not harm draw prediction. Could be kept for completeness in the feature set without risk, but adds no value to the model.

---

## 3d. Squad Market Value -- DOES NOT HELP

**What it captures:** Total Transfermarkt market value of a country's top 23 most valuable football players, measured at the most recent valuation within 2 years before each World Cup.

**Data sources:**
- `players.csv` -- 47,702 Transfermarkt players with nationality and position (from dcaribou/transfermarkt-datasets)
- `player_valuations.csv` -- 616,377 historical market value records spanning 2004-2026
- `wc_squads.csv` -- 13,843 WC squad roster entries (from jfjelstul/worldcup, used for reference but not for player matching)

**Features tested (1-3):**
- `home_squad_value`, `away_squad_value` -- squad value in millions EUR
- `squad_value_diff` -- difference

**Implementation approach:** For each WC year (2006-2022), joined player valuations with nationality, took the latest valuation per player within a 2-year window before the WC, then summed the top 23 by value for each country. Required a 6-entry name mapping (Transfermarkt uses "Korea, South", "Cote d'Ivoire", etc. vs our "South Korea", "Ivory Coast"). Set minimum threshold of 11 valued players per country to avoid unreliable estimates.

**Sanity-check values (2022 WC):**
| Team | Squad Value | Plausibility |
|------|:-----------:|:------------:|
| England | ~1,433M | Correct range |
| Brazil | ~1,255M | Correct range |
| USA | ~326M | Reasonable |
| Wales | ~180M | Reasonable |
| Iran | ~62M | Reasonable |
| Qatar | ~15M (only 9 players in TM) | Underestimated due to limited coverage |

**Why it fails on this dataset:**

1. **70% of training data is uninformative.** Valuations only exist for 2006-2022 WCs (294 out of 964 matches = 30%). The remaining 670 matches get median-filled constant values, contributing zero signal. The model sees the same squad value for France 1938 and France 2022 (both = median), which is nonsensical.

2. **Redundant with ELO.** Squad market value correlates strongly with ELO: wealthy nations (France, Brazil, England) also have high ELOs. The marginal signal -- "this team's squad is worth more than their ELO suggests" -- is too subtle for 294 informative rows.

3. **Coverage bias.** Transfermarkt coverage is concentrated on European club leagues. Teams from smaller leagues (Qatar, Saudi Arabia, some African teams) have fewer valued players, systematically undervaluing them. This creates a confound: the model partially learns "teams in European leagues do well" rather than "expensive squads do well."

**Result:**
| Config | CV F1 | CV Draw Recall |
|--------|:-----:|:--------------:|
| Base (37) | 0.531 | 0.41 |
| + squad_value_diff (38) | 0.530 | 0.39 |
| + all squad (40) | 0.509 | 0.33 |
| Base + continent + squad_diff (40) | 0.504 | 0.35 |

Adding squad value to the best configuration (Base + continent) actually degrades it from 0.541 to 0.504. The squad value features cancel out the continent advantage benefit.

**Verdict:** Do not use on this dataset. Squad market value is a powerful predictor in the literature but requires (a) larger datasets where every row has real values and (b) sufficient examples for the model to learn non-linear value-outcome relationships.

---

## Cross-Cutting Findings

### The "Overconfidence" Pattern

Three of the four feature sources (FIFA rankings, squad value, and intl ELO from earlier experiments) share the same failure mode:

1. They provide a more precise measure of team strength than WC-only ELO
2. This makes the model confident that the stronger team will win
3. The model stops predicting draws
4. Macro F1 drops because draw F1 goes to zero

WC-only ELO avoids this because its 4-year update gaps create productive imprecision -- many teams enter a tournament with similar ELOs, making draws plausible.

### Feature Budget on 900 Rows

| Features | CV F1 | Draw Recall | Assessment |
|:--------:|:-----:|:-----------:|:----------:|
| 37 | 0.531 | 0.41 | Baseline (tuned) |
| 39 | 0.541 | 0.44 | Sweet spot |
| 40-42 | 0.504-0.530 | 0.33-0.39 | Diminishing returns |
| 45+ | <0.490 | <0.25 | Overfitting |

The 900-row dataset supports approximately 37-39 features before overfitting begins. Every feature beyond that range must provide very strong, uncorrelated signal to justify its slot.

### What Actually Helped (Ranked by Impact)

| Improvement | CV F1 Gain | Type |
|-------------|:----------:|:----:|
| `min_samples_leaf=5` | +4.3 pp | Regularization |
| `n_estimators=300` | ~+0.5 pp | Model capacity |
| Continent advantage (+2 feat) | +1.0 pp | New feature |
| All other feature additions | 0 or negative | -- |

**Regularization delivered 4x more improvement than the best feature addition.** The model was overfitting, not under-informed.

---

## 3e. StatsBomb Within-Tournament xG -- DOES NOT HELP (but most interesting data)

**What it captures:** Expected goals (xG), shots, pass completion from event-level match data. Within-tournament rolling averages from prior matches in the same World Cup.

**Data source:** StatsBomb open data via `statsbombpy` Python package (free, no API key). Covers WC 2018 (64 matches) and WC 2022 (64 matches) with ~3,000 events per match including every pass, shot, tackle, and pressure event.

**Features tested:**
- `home_rolling_xg`, `away_rolling_xg` -- average xG from prior matches in the same WC
- `rolling_xg_diff` -- difference

**Why this feature had the best theoretical case:**
- Lowest correlation with ELO of any feature tested: r=0.35-0.43 (vs r=0.61 for intl ELO, r=0.50 for pass completion)
- Captures match quality and playing style, not team strength
- Major rank discrepancies reveal genuinely different information: France ranked 18th by xG but 4th by ELO at WC 2018 (won the tournament through clinical finishing, not xG dominance)
- Spearman rank correlation between xG and ELO: 0.601 (moderate, substantial disagreement)

**Why it still failed:**

| Config | CV F1 | Draw Recall |
|--------|:-----:|:-----------:|
| Base (37) | 0.531 | 0.41 |
| Base + xG_diff (38) | 0.510 | 0.39 |
| Base + continent + xG_diff (40) | 0.520 | 0.40 |
| Best: Base + continent (39) | 0.541 | 0.44 |

Even on Fold 4 alone (where StatsBomb data exists in both train and val):
- Base alone: F1=0.636
- Base + xG_diff: F1=0.577 (worse)

**Root cause:** Within-tournament rolling xG from 1-3 prior matches is extremely noisy. A team's xG against their group-stage opponents depends on opponent quality, match context, and tactical approach -- it's not a stable team metric. With 10% training coverage (96/964), the median-filled 90% drowns out the signal.

**The paradox:** xG had the best theoretical properties of any feature (lowest ELO correlation, genuinely novel signal) but the worst coverage. If StatsBomb data existed for all 22 World Cups (not just 2), this could be the breakthrough feature.

**Value for the report:** The data is saved at `data_clean/statsbomb_wc_stats.csv` (256 team-match records). Excellent for analytical visualizations:
- xG created vs xG conceded per team
- "Overperformers" (scored more than xG predicted -- e.g., France 2018)
- Model predictions vs xG-based expectations

---

## Final Summary Across All Features Tested

| Feature | ELO Corr | Coverage | CV F1 Change | Verdict |
|---------|:--------:|:--------:|:------------:|:-------:|
| Continent advantage (3a) | ~0.0 | 100% | **+1.0 pp** | **KEEP** |
| FIFA rankings (3b) | ~0.50 | 60% | -2.8 to -4.8 pp | Do not use |
| Qualifying record (3c) | ~0.30 | 70% | -0.9 to -2.1 pp | Neutral |
| Squad market value (3d) | ~0.50 | 30% | -0.1 to -2.2 pp | Do not use |
| StatsBomb xG (3e) | **0.35** | 10% | -1.1 to -2.1 pp | Do not use |

**The irony:** The feature with the lowest ELO correlation (xG at r=0.35) and most novel signal had the worst coverage. The feature that worked (continent advantage at 100% coverage) was the simplest.

**Lesson:** On 900 training rows, a feature needs THREE things to help: (1) novel signal uncorrelated with existing features, (2) full or near-full training coverage, and (3) minimal feature count. Continent advantage is the only feature tested that had all three.
