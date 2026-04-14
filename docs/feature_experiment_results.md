# Additional Feature Sources: Experiment Results

**Date:** 2026-04-07  
**RF config:** 300 trees, max_features=sqrt, min_samples_leaf=5, balanced

---

## Summary

| Config | Features | CV F1 | CV Draw Recall | Test Acc | Test F1 | Test Draw Recall |
|--------|:--------:|:-----:|:--------------:|:--------:|:-------:|:----------------:|
| Original WC-only (default RF) | 37 | 0.488 | 0.24 | 0.609 | 0.556 | 0.30 |
| **WC-only + tuned RF** | **37** | **0.531** | **0.41** | **0.656** | **0.636** | **0.60** |
| **+ continent advantage (3a)** | **39** | **0.541** | **0.44** | **0.609** | **0.583** | **0.50** |
| + intl H2H | 39 | 0.534 | 0.40 | 0.625 | 0.578 | 0.40 |
| + FIFA rankings (3b) | 38-42 | 0.483-0.505 | 0.23-0.29 | -- | -- | -- |
| + qualifying record (3c) | 38-42 | 0.510-0.522 | 0.39-0.45 | -- | -- | -- |
| + all features combined | 47 | 0.460 | 0.14 | 0.578 | 0.469 | 0.10 |

## Continent Advantage -- HELPS (marginally)

Adding `home_on_home_continent` and `away_on_home_continent` to the base 37 WC features:

- **CV F1: 0.531 -> 0.541** (+1.0 pp, best CV F1 observed)
- **CV draw recall: 0.41 -> 0.44**
- Only 2 binary features, no bloat risk
- Captures that European teams overperform at European WCs, etc.

However: does NOT combine well with intl H2H. Adding continent to intl H2H drops CV F1 to 0.506. The two features compete for similar geographic/relational signal.

## FIFA Rankings -- HURTS

Every configuration with FIFA ranking points reduces CV F1 and kills draw recall:

- Best case: `fifa_points_diff` alone drops CV F1 from 0.534 to 0.505
- Draw recall drops from 0.40 to 0.29

**Same pattern as intl ELO**: precise team-strength features make the model overconfident, destroying draw prediction. FIFA points and ELO are moderately correlated -- adding both is redundant, and either one on its own replaces the "productive uncertainty" of the WC-only ELO that helps predict draws.

## Qualifying Record -- NEUTRAL

- Preserves draw recall (0.39-0.45) unlike FIFA rankings
- But doesn't improve CV F1 (0.510-0.522 vs baseline 0.531-0.534)
- `qual_win_rate_diff` is the most useful single feature from this group

## Key Insight

The biggest improvement came not from new features but from **hyperparameter tuning**: `min_samples_leaf=5` (from default 1). This single change improved CV F1 from 0.488 to 0.531 on the original 37 features -- a larger gain than any feature addition.

## Recommended Final Configuration

**Option A (best CV F1):** WC-only 37 features + continent advantage (2) = 39 features  
- CV F1: 0.541, CV draw recall: 0.44

**Option B (simplest):** WC-only 37 features, tuned RF  
- CV F1: 0.531, CV draw recall: 0.41

Both use: RF(n_estimators=300, max_features='sqrt', min_samples_leaf=5, class_weight='balanced')

Note: the 64-match test set has very high variance (1 match = 1.5 pp accuracy). CV metrics are more reliable. On CV, Option A edges Option B by 1 pp F1.

---

## Squad Market Value -- DOES NOT HELP

**Data:** Transfermarkt player valuations (616K records, 39K players, 2004-2026). For each WC team, summed the top 23 most valuable players of that nationality using their most recent valuation within 2 years before the WC.

**Coverage:** 30% of training matches (294/964 -- WCs 2006-2022 only). Pre-2006 matches get median-filled values.

| Config | Features | CV F1 |
|--------|:--------:|:-----:|
| Base WC-only | 37 | 0.531 |
| Base + squad_value_diff | 38 | 0.530 |
| Base + squad values (3) | 40 | 0.509 |
| Base + continent + squad_diff | 40 | 0.504 |

**Same pattern as FIFA rankings and intl ELO**: precise team-strength features hurt on 900 rows. Additionally, 70% median-filled values mean the feature is uninformative for most training data.

**Files produced:**
- `data_clean/players.csv` -- 47,702 Transfermarkt players
- `data_clean/player_valuations.csv` -- 616,377 historical market values
- `data_clean/wc_squads.csv` -- 13,843 WC squad roster entries (1930-2022)

---

## Final Results: All Experiments Combined

| Config | Features | CV F1 | CV Draw R | Test Acc | Test F1 |
|--------|:--------:|:-----:|:---------:|:--------:|:-------:|
| Original (default RF) | 37 | 0.488 | 0.24 | 0.609 | 0.556 |
| **+ tuned RF (min_leaf=5)** | **37** | **0.531** | **0.41** | **0.656** | **0.636** |
| **+ continent advantage** | **39** | **0.541** | **0.44** | **0.609** | **0.583** |
| + intl H2H | 39 | 0.534 | 0.40 | 0.625 | 0.578 |
| + FIFA rankings | 38+ | 0.483-0.505 | 0.23-0.29 | -- | -- |
| + qualifying record | 38+ | 0.510-0.522 | 0.39-0.45 | -- | -- |
| + squad market value | 38+ | 0.509-0.530 | 0.33-0.39 | -- | -- |
| + all features combined | 47 | 0.460 | 0.14 | 0.578 | 0.469 |

### Best model: WC-only (37) + continent advantage (2) + tuned RF

```
RandomForestClassifier(n_estimators=300, max_features='sqrt',
                       min_samples_leaf=5, class_weight='balanced')
```

**What delivered the most improvement:**
1. `min_samples_leaf=5` tuning: +4.3 pp CV F1 (biggest single gain)
2. Continent advantage features: +1.0 pp CV F1
3. Everything else: neutral or negative

**Key lesson:** On 900 training rows, the model is regularization-limited, not feature-limited. The only features worth adding are those with genuinely novel signal AND low correlation with existing features.
