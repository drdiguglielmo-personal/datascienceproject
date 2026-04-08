# Research Report: Additional Datasets for FIFA World Cup Prediction Project

**Date:** 2026-03-30  
**Depth:** Deep  
**Persona:** Data Scientist  

---

## Executive Summary

Your current project predicts FIFA World Cup match outcomes using **~900 training matches** (1930-2018) with 37 engineered features. The main limitation is **small sample size** and **World Cup-only data**. This report identifies **7 high-value datasets** that can expand your data by **40-50x** and add powerful new feature dimensions (FIFA rankings, squad market value, player ratings, economic indicators). The most impactful addition is the **International Football Results dataset** (~49,000 matches), which would let you train on qualifiers, friendlies, and continental tournaments — dramatically improving ELO estimates and rolling form features.

---

## Current State Assessment

| Metric | Current Value | Limitation |
|--------|--------------|------------|
| Training rows | 900 matches | Too small for complex models (XGBoost underperforms RF) |
| Test rows | 64 matches | High variance — one upset shifts accuracy by ~1.5% |
| Feature sources | Match history only | No external quality signals (rankings, squad value, economy) |
| ELO computation | World Cup matches only | ~4-year gaps between tournaments create noisy updates |
| Best model accuracy | 60.9% (Random Forest) | Ceiling likely limited by data volume and feature richness |

---

## Recommended Datasets (Priority Order)

### 1. International Football Results 1872-2026 (HIGHEST IMPACT)

| Property | Value |
|----------|-------|
| **Source** | [Kaggle (Mart Jurisoo)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| **GitHub** | [martj42/international_results](https://github.com/martj42/international_results) |
| **Size** | ~49,016 matches x 9 columns (1.26 MB) |
| **Coverage** | 1872-2025, updated Feb 2026 |
| **License** | CC0 Public Domain |
| **Downloads** | 129,743 |

**Files:**
| File | Rows | Description |
|------|------|-------------|
| `results.csv` | ~49,016 | `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `city`, `country`, `neutral` |
| `goalscorers.csv` | Large | Goal-level detail: scorer, own_goal, penalty flags |
| `shootouts.csv` | Varies | Penalty shootout outcomes |
| `former_names.csv` | Small | Historical team name mappings |

**Why this matters:**
- **40x more data** for computing ELO ratings — your current ELO updates only every 4 years (between World Cups). With qualifiers, friendlies, and continental tournaments, ELO updates are near-continuous, yielding much more accurate pre-WC strength estimates.
- **Better rolling form** — "last 5 WC matches" can span 8-12 years. "Last 5 international matches" captures actual recent form.
- **Richer head-to-head records** — most team pairs have 0-3 WC meetings but 10-30 total international meetings.
- **Tournament type as a feature** — performance in qualifiers vs. friendlies vs. competitive matches carries signal.
- Covers all tournament types: World Cup, qualifiers, Euro, Copa America, AFCON, Asian Cup, Nations League, friendlies.

**Alternative:** [International Football Results: Daily Updates (Clement Bravo)](https://www.kaggle.com/datasets/patateriedata/all-international-football-results) — 50,243 matches, auto-updated daily, but lacks city and goalscorer data.

**Integration approach:**
```
1. Download results.csv
2. Recompute ELO using ALL international matches (not just WC)
3. Recompute rolling form using recent international matches
4. Recompute head-to-head using all meetings
5. Add tournament-type features (competitive vs. friendly form)
6. Keep WC matches as your prediction target, but use all matches for feature computation
```

---

### 2. FIFA World Rankings 1992-2024

| Property | Value |
|----------|-------|
| **Source** | [Kaggle (cashncarry)](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) |
| **Size** | ~63,000 rows x 8 columns (1.77 MB) |
| **Coverage** | December 1992 - June 2024 (monthly) |
| **License** | CC0 Public Domain |

**Columns:** `country_full`, `country_abrv`, `rank`, `total_points`, `previous_points`, `rank_change`, `confederation`, `rank_date`

**Why this matters:**
- **Official FIFA ranking points** are a complementary signal to your computed ELO — they use a different formula and weigh tournament importance differently.
- **Rank trajectory** (improving vs. declining teams) heading into a World Cup is predictive.
- **Confederation** membership enables confederation-strength features.
- The `total_points` column is more granular than rank position (rank 5 vs 6 may differ by 1 point, while rank 50 vs 51 may differ by 100 points).

**Limitation:** Only available from 1993 onward, so covers World Cups 1994-2022 (8 tournaments, ~512 training matches).

**Alternative scraper:** [GitHub (cnc8)](https://github.com/cnc8/fifa-world-ranking) — can scrape latest rankings from fifa.com.

---

### 3. Transfermarkt Player Market Values

| Property | Value |
|----------|-------|
| **Source** | [Kaggle (davidcariboo)](https://www.kaggle.com/datasets/davidcariboo/player-scores) / [GitHub](https://github.com/dcaribou/transfermarkt-datasets) |
| **Size** | ~205 MB across 10 CSV tables |
| **Coverage** | Ongoing, updated weekly (last: March 2026) |
| **License** | CC0 Public Domain |

**Key tables:**
- **Players:** 30,000+ players with nationality, position, market value
- **Valuations:** 400,000+ historical market value records
- **Appearances:** 1,200,000+ match appearances

**Why this matters:**
- **Squad market value is one of the strongest single predictors** of World Cup success (published research confirms this).
- Enables features like: total squad value, median player value, positional depth (avg GK value, avg striker value), squad age profile.
- Market value is a crowdsourced expert assessment that integrates player ability, form, injury status, and league quality.

**Integration challenge:** Requires mapping players to their national team World Cup squads. Cross-reference with Wikipedia or FIFA squad lists. Realistically covers 1998-2022 World Cups.

---

### 4. FIFA Video Game Player Ratings (FIFA 15-23)

| Property | Value |
|----------|-------|
| **Source** | [Kaggle (stefanoleone992)](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset) |
| **Size** | ~1.7 GB, 19,000+ players x 110 attributes per game version |
| **Coverage** | FIFA 15 through FIFA 23 (2015-2023) |
| **License** | CC0 Public Domain |

**Why this matters:**
- **110 attributes per player** including Overall, Pace, Shooting, Passing, Defending, Physical, Potential.
- EA Sports ratings are expert-curated and widely used in football analytics research.
- Easier to aggregate than Transfermarkt: just average the `overall` rating per national team squad.
- Can compute **positional strength**: avg GK rating, avg defender rating, avg midfielder rating, avg forward rating.

**Limitation:** Only covers 2015-2023, so applicable to 2018 and 2022 World Cups (2 tournaments, ~128 matches).

---

### 5. FiveThirtyEight Soccer Power Index (SPI)

| Property | Value |
|----------|-------|
| **Source** | [GitHub (fivethirtyeight/data)](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) |
| **Coverage** | 2016-present |
| **License** | Open (FiveThirtyEight data repository) |

**Files:**
- `spi_matches_intl.csv` — match-level SPI ratings + pre-match win/draw/loss probabilities
- `spi_global_rankings_intl.csv` — current SPI ratings per team

**Why this matters:**
- Provides **pre-match win probabilities from a sophisticated model** — serves as both a powerful feature and a benchmark to compare your model against.
- **Offensive and defensive rating decomposition** is richer than simple rankings.
- Can use SPI probabilities as a feature to see if your model adds value beyond FiveThirtyEight's.

**Limitation:** Only covers 2016+, so limited to 2018 and 2022 World Cups. Better as a benchmark than a training feature.

**Note:** FiveThirtyEight was acquired by ABC News; verify data freshness before use.

---

### 6. World Bank Economic Indicators

| Property | Value |
|----------|-------|
| **Source** | [World Bank Open Data](https://data.worldbank.org/) |
| **Size** | 217 countries x 60+ years per indicator |
| **Coverage** | 1960-present, annual |
| **License** | Creative Commons Attribution 4.0 |
| **Access** | CSV download or Python API (`wbgapi` package) |

**Key indicators:**
| Indicator | Code | Why Useful |
|-----------|------|-----------|
| GDP (current USD) | `NY.GDP.MKTP.CD` | Larger economies invest more in football infrastructure |
| GDP per capita | `NY.GDP.PCAP.CD` | Controls for population; wealthy small nations can punch above weight |
| Population | `SP.POP.TOTL` | Larger talent pools |

**Python access:**
```python
import wbgapi as wb
df_gdp = wb.data.DataFrame('NY.GDP.MKTP.CD')
df_pop = wb.data.DataFrame('SP.POP.TOTL')
df_gdp_pc = wb.data.DataFrame('NY.GDP.PCAP.CD')
```

**Why this matters:**
- GDP and population explain **structural advantages** — why Brazil, Germany, and France consistently qualify and perform well.
- GDP per capita captures small wealthy footballing nations (e.g., Netherlands, Belgium, Uruguay).
- Covers the full 1930-2022 period (World Bank data from 1960; pre-1960 matches may need imputation).
- Very low-effort integration: just join by country + year.

**Alternative:** [Maven Analytics World Economic Indicators](https://mavenanalytics.io/data-playground/world-economic-indicators) — 216 countries, 62 variables, single pre-cleaned CSV (108 KB), 1960-2018.

---

### 7. Pre-Computed International ELO Ratings

| Property | Value |
|----------|-------|
| **Source** | [Kaggle (saifalnimri)](https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings) / [eloratings.net](https://www.eloratings.net/) |
| **Size** | ~44,060 rows x 9 columns |
| **Coverage** | 1872-2025 |
| **License** | Available on Kaggle |

**Why this matters:**
- Provides **externally validated ELO ratings** computed from ALL international matches (not just World Cup).
- Can cross-check against your internally computed ELO to validate your implementation.
- If you don't want to recompute ELO from the full 49K match dataset, this gives you pre-computed values.

---

## Impact Analysis: What Changes With These Datasets

### Feature Engineering Improvements

| Current Feature | Problem | Improvement with New Data |
|----------------|---------|--------------------------|
| `home_elo` / `away_elo` | Computed from WC-only (4-year gaps) | Recompute from ALL 49K international matches — near-continuous updates |
| `rolling5_win_rate` | Last 5 WC matches can span 8-12 years | Last 5 international matches = actual recent form (weeks/months) |
| `h2h_total` | Most pairs have 0-3 WC meetings | 10-30 total international meetings per pair |
| `hist_win_rate` | WC-only history is thin for many teams | Full international history with thousands of data points |
| *(new)* `fifa_rank_diff` | N/A — not in current data | Official ranking points difference at time of match |
| *(new)* `squad_market_value` | N/A | Total Transfermarkt value — one of strongest predictors |
| *(new)* `avg_player_overall` | N/A | EA Sports ratings aggregated to squad level |
| *(new)* `gdp_per_capita_ratio` | N/A | Economic structural advantage |
| *(new)* `population_ratio` | N/A | Talent pool size difference |
| *(new)* `competitive_form` | N/A | Win rate in competitive (non-friendly) recent matches |
| *(new)* `qualifying_performance` | N/A | How dominant was the team in their qualifying group |

### Expected Model Performance Gains

| Change | Expected Impact | Confidence |
|--------|----------------|------------|
| ELO from all international matches | +3-5% accuracy | High — published research consistently shows this |
| FIFA ranking features | +1-2% accuracy | Medium — partially correlated with ELO |
| Squad market value | +2-4% accuracy | High — strongest single predictor in literature |
| Economic indicators | +0.5-1% accuracy | Low-Medium — mostly captured by ELO already |
| XGBoost with larger dataset | +2-3% over RF | High — boosting benefits from more data |
| Combined | **65-72% accuracy** (from 60.9%) | Medium — feature correlation limits additive gains |

### Training Data Expansion

| Approach | Training Rows | Description |
|----------|--------------|-------------|
| Current | 900 | World Cup matches only (1930-2018) |
| Option A: WC prediction, all-match features | 900 | Same target, but features computed from 49K matches |
| Option B: Train on all competitive internationals | ~15,000 | Train on all competitive matches, test on WC 2022 |
| Option C: Train on all internationals | ~49,000 | Train on everything including friendlies |

**Recommendation:** Start with **Option A** (biggest bang for least effort), then experiment with **Option B**.

---

## Interesting Analytical Angles These Datasets Enable

### 1. "Money Ball" Analysis
- Plot squad market value vs. World Cup finish position
- Identify overperformers (e.g., Morocco 2022, Croatia 2018) and underperformers (e.g., England historically)
- Feature: `value_rank_gap` = (squad value rank) - (FIFA ranking) — teams undervalued by the market

### 2. Economic Development and Football Success
- Visualize GDP per capita vs. World Cup appearances over decades
- Track how newly wealthy nations (Qatar, Saudi Arabia) translate economic growth into football investment
- Show the "football poverty trap" — very poor nations rarely qualify regardless of population

### 3. Friendly vs. Competitive Performance Gap
- Some teams dominate friendlies but choke in tournaments (and vice versa)
- Feature: `competitive_form_ratio` = competitive win rate / friendly win rate
- Could help predict "tournament DNA" — teams that elevate their game under pressure

### 4. The "Dark Horse" Detector
- Combine ELO trajectory (rapidly rising), squad value (undervalued), and recent competitive form
- Identify teams whose underlying quality exceeds their ranking/reputation
- Retroactively validate: would this have flagged Morocco 2022? Croatia 2018? South Korea 2002?

### 5. Home Continent Advantage (Beyond Host Nation)
- Your current features capture host country advantage
- With confederation data, test whether playing on your home continent matters (e.g., European teams in European WCs, South American teams in South American WCs)

### 6. Qualifying Path Difficulty
- Teams from UEFA face stiffer qualifying competition than CONCACAF
- Feature: average ELO of teams in qualifying group
- A team that barely qualified from UEFA may be stronger than a team that dominated CONCACAF qualifying

---

## Quick-Start Integration Guide

### Phase 1: Immediate (1-2 hours)
```bash
# Download the international results dataset
kaggle datasets download martj42/international-football-results-from-1872-to-2017
# Download FIFA rankings
kaggle datasets download cashncarry/fifaworldranking
```
- Recompute ELO from all 49K international matches
- Add FIFA ranking points as features
- Re-run your existing model pipeline

### Phase 2: Medium effort (3-5 hours)
```bash
# Download World Bank data
pip install wbgapi
```
- Add GDP per capita and population features
- Add confederation and qualifying performance features
- Experiment with training on all competitive internationals

### Phase 3: Higher effort (1-2 days)
```bash
# Download Transfermarkt data
kaggle datasets download davidcariboo/player-scores
# Download FIFA video game ratings
kaggle datasets download stefanoleone992/fifa-23-complete-player-dataset
```
- Map players to national team squads
- Compute squad-level market value and rating aggregates
- Build the "Dark Horse" detector

---

## Sources

1. [International Football Results 1872-2026 — Kaggle (Mart Jurisoo)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
2. [GitHub: martj42/international_results](https://github.com/martj42/international_results)
3. [International Football Results: Daily Updates — Kaggle (Clement Bravo)](https://www.kaggle.com/datasets/patateriedata/all-international-football-results)
4. [FIFA World Ranking 1992-2024 — Kaggle](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)
5. [Transfermarkt Datasets — Kaggle / GitHub](https://www.kaggle.com/datasets/davidcariboo/player-scores)
6. [FIFA 15-23 Player Stats — Kaggle](https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset)
7. [FiveThirtyEight Soccer SPI — GitHub](https://github.com/fivethirtyeight/data/tree/master/soccer-spi)
8. [World Bank Open Data](https://data.worldbank.org/)
9. [International Football ELO Ratings — Kaggle](https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings)
10. [FIFA World Ranking Scraper — GitHub](https://github.com/cnc8/fifa-world-ranking)
11. [Maven Analytics World Economic Indicators](https://mavenanalytics.io/data-playground/world-economic-indicators)
12. [FIFA World Cup 1930-2022 (piterfm) — Kaggle](https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup)

---

*Research complete. No code was written or modified. Next steps are at user's discretion.*
