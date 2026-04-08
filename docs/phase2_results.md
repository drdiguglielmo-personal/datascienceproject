# Phase 2 Results: Expanded Training on International Matches

**Date:** 2026-04-07  
**Outcome:** Did not improve over original pipeline. Documented as a negative result.

---

## What We Tried

Expanded training from 900 World Cup matches to ~28,000 competitive international matches (qualifiers, continental championships, etc.) using 31 features computed from 49,287 international match records.

## Configurations Tested

| Config | Training Data | Rows | Acc | F1 | Draw Recall |
|--------|--------------|:----:|:---:|:--:|:-----------:|
| **Original (baseline)** | **WC only, WC features** | **900** | **0.609** | **0.556** | **0.30** |
| WC only, intl features | WC matches | 900 | 0.516 | 0.372 | 0.00 |
| WC + WC qualifiers | Focused | 8,770 | 0.531 | 0.377 | 0.00 |
| WC + quals + continental | Major tournaments | 11,055 | 0.531 | 0.374 | 0.00 |
| All competitive | Everything non-friendly | 28,227 | 0.516 | 0.359 | 0.00 |
| WC + quals, WC 10x weight | Aggressive weighting | 8,770 | 0.531 | 0.378 | 0.00 |
| WC + quals, WC 20x weight | Very aggressive | 8,770 | 0.484 | 0.343 | 0.00 |
| All comp, WC 20x + continental 3x | Best weighting | 28,227 | 0.547 | 0.392 | 0.00 |

None beat the original. None predicted even one draw correctly.

## Why It Failed: Domain Mismatch

### 1. "Home team" means different things

In qualifiers/friendlies (97% of expanded training), the home team plays at their actual stadium with ~60% win rate. In WC matches, "home team" is a FIFA administrative label at a neutral venue with ~50% win rate. The model learns a home-win bias from qualifiers that doesn't apply to the WC.

### 2. WC-only features encode tournament-specific signal

The original pipeline's WC-only features (WC appearances, WC-only ELO, WC-only H2H, WC host flag) capture "tournament DNA" -- how teams perform under World Cup conditions specifically. A team like Costa Rica may be mediocre internationally but consistently overperform at World Cups. International features miss this.

### 3. Draw dynamics don't transfer

WC group-stage draws are often tactical (both teams need a point). In qualifiers, home advantage suppresses draws. The model never learns WC draw patterns from qualifier data.

## Technical Issues Discovered

### Test set label mismatch

The Kaggle international results dataset and the Fjelstul WC dataset have different match ordering for same-day matches. Naive positional matching produced 37/64 home/away swaps. Fixed by using the original WC test set directly.

Additionally, the Kaggle dataset records penalty-shootout matches as draws (e.g., Argentina vs France 3-3), while the WC dataset records who advanced. This affected 5+ knockout-stage labels.

### Feature correlation

WC-only and international feature pairs are correlated (r=0.27 to 0.63) but not identical. Adding both creates redundancy that dilutes Random Forest splits, particularly hurting the minority draw class.

| Feature Pair | Correlation |
|-------------|:-----------:|
| `elo_diff` vs `intl_elo_diff` | 0.606 |
| `home_elo` vs `home_intl_elo` | 0.628 |
| `rolling5_win_rate` (home) | 0.550 |
| `h2h_total` vs `intl_h2h_total` | 0.287 |
| `hist_win_rate` (home) | 0.347 |

## What DID Work (From Earlier Experiments)

The only safe addition from international data: **+2 intl H2H features** (`intl_h2h_total`, `intl_h2h_home_win_rate`).

| Config | Features | CV Acc | CV F1 | Draw Recall |
|--------|:--------:|:------:|:-----:|:-----------:|
| Original WC-only | 37 | 0.562 | 0.488 | 0.24 |
| WC + intl H2H (+2) | 39 | 0.566 | 0.489 | 0.21 |

Low correlation with existing features (r=0.287) means they add genuinely new signal without disrupting draw prediction.

## Key Takeaway

**Domain-specific features on domain-specific data outperform general features on large datasets** for specialized prediction tasks. More data is not always better -- the right data matters more.

## Files Produced

- `scripts/feature_engineering_expanded.py` -- Phase 2 pipeline (preserved for reference)
- `data_clean/features_expanded_train.csv` -- 28,227 rows x 37 cols
- `data_clean/features_expanded_test.csv` -- 64 rows x 37 cols
- `data_clean/international_results.csv` -- 49,287 international matches (source data)
