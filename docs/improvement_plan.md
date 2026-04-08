# Improvement Plan: FIFA World Cup Match Prediction

**Based on:** Experimental results from international data integration (April 2026)

---

## Current State

| Metric | Value |
|--------|-------|
| Best model | Random Forest (200 trees, balanced weights) |
| Test accuracy | 0.609 (vs 0.500 baseline) |
| Test macro F1 | 0.556 |
| Draw recall | 0.30 (3/10 draws detected) |
| Training rows | 900 (WC matches 1930-2018) |
| Features | 37 (WC-only) |
| Temporal CV accuracy | 0.562 +/- 0.048 |

## What We Learned

We added 27 international features (intl ELO, intl rolling form, intl H2H, intl history) computed from 49,287 international matches. Results:

| Finding | Detail |
|---------|--------|
| Intl features carry real signal | `away_intl_elo` ranked #1 in RF feature importance |
| Adding all 64 features hurts | F1 dropped 0.556 -> 0.420 due to feature bloat on 900 rows |
| Draw prediction is the casualty | More precise ELO makes the model overconfident; draw recall -> 0% |
| Only intl H2H is safely additive | Adding 2 H2H features: F1 = 0.489, draw recall = 0.21 |
| Replacing WC features with intl also hurts | WC-specific features are more relevant to WC prediction than general intl features |

**Root cause:** 900 training rows cannot support 64 features. The model overfits to majority classes and abandons draw prediction.

---

## Improvement Plan (3 Phases)

### Phase 1: Quick Wins (feature selection + tuning)

**Goal:** Get the best possible result from the current 900-row training set.

**1a. Adopt the +2 H2H configuration**

Add only `intl_h2h_total` and `intl_h2h_home_win_rate` to the 37 WC features. This was the only configuration that matched or improved the original across all temporal CV folds while maintaining draw recall.

- Temporal CV: acc=0.566, f1=0.489, draw_recall=0.21
- Low correlation with existing features (r=0.287 for `h2h_total` vs `intl_h2h_total`)
- No draw prediction collapse

**1b. Hyperparameter tuning with temporal CV**

The current RF uses default `max_features='sqrt'`. Our experiments showed `max_features=0.3` recovered accuracy on the 64-feature set. Run a grid search over:

- `n_estimators`: [100, 200, 300, 500]
- `max_features`: [0.2, 0.3, 'sqrt', 'log2']
- `min_samples_leaf`: [1, 3, 5]
- `class_weight`: ['balanced', {home: 1, away: 1.2, draw: 2.0}]

Use 4-fold temporal CV (not stratified) for selection.

**1c. Draw-specific class weighting**

The current `class_weight='balanced'` assigns weights inversely proportional to frequency. Since draws are 18.8% of training but the hardest class, experiment with manual weights that upweight draws further:

- `{home team win: 1.0, away team win: 1.5, draw: 3.0}`
- `{home team win: 1.0, away team win: 1.2, draw: 2.5}`

**Expected impact:** +1-2 pp F1 from tuning, maintained or improved draw recall.

---

### Phase 2: Train on Competitive Internationals (the big unlock)

**Goal:** Expand training data from 900 to ~10,000+ rows.

This is the highest-impact change. The reason intl features didn't help as *features* is that they introduced more dimensions than 900 rows can handle. The solution is to use those matches as *training data* instead.

**2a. Expand the training set**

Currently: train on 900 WC matches, test on 64 WC matches.

Proposed: train on all competitive international matches before 2022, test on 2022 WC.

Available competitive matches before 2022:

| Tournament Type | Matches |
|----------------|---------|
| WC qualification | 7,770 |
| Continental championships (Euro, Copa, AFCON, Asian Cup, Gold Cup) | 2,233 |
| FIFA World Cup | 900 |
| Other competitive | ~16,783 |
| **Total competitive** | **~27,686** |

A 31x training data increase means:
- XGBoost can finally outperform RF (boosting needs data volume)
- 64 features become viable (64 features on 27K rows = healthy ratio)
- Draw prediction improves from more draw examples (~5,800 draws at ~21% rate)

**2b. Add a tournament-weight feature**

Not all matches are equally relevant to WC prediction. Add `sample_weight` during training:

| Tournament Type | Sample Weight |
|----------------|:------------:|
| FIFA World Cup | 3.0 |
| Continental championships | 2.0 |
| WC qualification | 1.5 |
| Nations League | 1.0 |
| Other competitive | 0.5 |

This tells the model "WC matches matter most" without throwing away the other data.

**2c. Add tournament-type features**

When training on mixed tournament types, add:
- `is_world_cup`: 1/0
- `is_continental_championship`: 1/0
- `is_qualifying`: 1/0
- `is_neutral_venue`: from the Kaggle dataset's `neutral` column

**2d. Recompute features for the expanded training set**

The existing `feature_engineering.py` already computes intl features from 49K matches. For Phase 2, each training row (whether WC or qualifier) gets the same feature vector: intl ELO, intl rolling form, intl H2H, etc.

The WC-only features (`home_elo`, `home_hist_win_rate`, etc.) should be dropped for the expanded model since they only apply to WC matches. Use the full intl versions instead.

**Implementation:**

Create `scripts/feature_engineering_expanded.py`:
1. Load `international_results.csv` as training data (filter to competitive, pre-2022)
2. Load WC 2022 as test data
3. Compute intl features for ALL rows (same functions already written)
4. Add tournament-type features
5. Output `features_expanded_train.csv` (~27K rows) and `features_expanded_test.csv` (64 rows)

**Expected impact:** +5-10 pp accuracy, +5-8 pp F1. XGBoost should become the best model. This is the single highest-impact improvement available.

---

### Phase 3: Additional Feature Sources

**Goal:** Add features from external datasets that provide genuinely new signal dimensions.

These are only worth pursuing AFTER Phase 2, because the current 900-row training set can't absorb more features without overfitting.

**3a. FIFA World Rankings (high priority)**

Source: Kaggle `cashncarry/fifaworldranking` (1992-2024)

Features to add:
- `home_fifa_rank`, `away_fifa_rank` (pre-match ranking position)
- `home_fifa_points`, `away_fifa_points` (pre-match ranking points)
- `fifa_rank_diff`, `fifa_points_diff`

Why it helps: FIFA ranking points use a different formula than ELO (weights tournament importance, match result margin, opponent strength). They provide a complementary strength signal. Only covers 1994-2022 WCs (~512 WC matches), but with Phase 2's expanded training set, this covers thousands of matches.

**3b. Home continent advantage (medium priority)**

From the existing data, derive whether each team is playing on their home *continent*:

- Map each team to their confederation (UEFA, CONMEBOL, AFC, CAF, CONCACAF, OFC)
- Map each host country to its confederation
- Feature: `home_on_home_continent`, `away_on_home_continent`

This captures that European teams historically perform better in European WCs, South American teams in South American WCs -- a more granular signal than just `home_is_host`.

**3c. Qualifying path strength (medium priority)**

From WC qualification data in the intl results:

- Average ELO of opponents faced in qualifying
- Win rate in qualifying campaign
- Goals scored/conceded ratio in qualifying

These capture "how dominant was this team on the road to the World Cup" -- a team that scraped through playoffs is different from one that topped their group.

**3d. Squad market value (lower priority, higher effort)**

Source: Kaggle `davidcariboo/player-scores` (Transfermarkt data)

Requires mapping players to national team squads (cross-referencing Wikipedia squad lists). Only feasible for 1998-2022 WCs. Squad market value is one of the strongest single predictors in published research, but the integration effort is substantial.

---

## Priority Matrix

| Improvement | Impact | Effort | Priority |
|-------------|:------:|:------:|:--------:|
| 1a. Add intl H2H (+2 features) | Low | 10 min | Do now |
| 1b. Hyperparameter tuning | Low-Med | 1 hour | Do now |
| 1c. Draw class weighting | Low | 30 min | Do now |
| **2a-d. Train on competitive intl** | **High** | **4-6 hours** | **Do next** |
| 3a. FIFA rankings | Medium | 2 hours | After Phase 2 |
| 3b. Continent advantage | Low-Med | 1 hour | After Phase 2 |
| 3c. Qualifying path | Medium | 3 hours | After Phase 2 |
| 3d. Squad market value | Med-High | 1-2 days | Optional |

---

## Expected Outcomes

| Phase | Accuracy | Macro F1 | Draw Recall |
|-------|:--------:|:--------:|:-----------:|
| Current | 0.609 | 0.556 | 0.30 |
| After Phase 1 | ~0.61-0.63 | ~0.55-0.57 | ~0.25-0.35 |
| After Phase 2 | ~0.65-0.70 | ~0.58-0.64 | ~0.30-0.40 |
| After Phase 3 | ~0.67-0.72 | ~0.60-0.66 | ~0.30-0.40 |

These estimates are based on published research benchmarks (Hvattum & Arntzen 2010; 2023 Soccer Prediction Challenge) and the scaling behavior we observed in our experiments.

---

## Key Lesson

More features on the same small dataset makes things worse. More *training data* is what enables more features. The international results dataset is valuable -- not as a feature source for 900 rows, but as 27,000 additional training rows that make the full feature set viable.
