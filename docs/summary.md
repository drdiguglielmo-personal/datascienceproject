# Project Summary: FIFA Men's World Cup Match Outcome Prediction

**Course:** CSE 40467 -- Data Science  
**Repository:** `datascienceproject/`  
**Analysis date:** 2026-04-07  

---

## 1. Project Overview and Motivation

This project tackles a classic sports analytics problem: predicting the outcome of FIFA Men's World Cup matches before they are played. Specifically, the research question is a **three-class classification task** -- given two teams and the context of a World Cup match, will the result be a *home team win*, *away team win*, or *draw*?

The project was developed for CSE 40467 (Data Science) and is structured across two major deliverables aligned with the course curriculum:

- **Part 1** (40% weight): Data description, exploratory data analysis, cleaning, and preprocessing. Delivered through `Data Science Report.ipynb` and `scripts/clean_worldcup.py`.
- **Part 2** (60% weight): Feature engineering, supervised modeling, evaluation, unsupervised analysis, and critical interpretation. Delivered through `Part2_Models_and_Results.ipynb` and `scripts/feature_engineering.py`.

The problem is non-trivial for several reasons. Football is an inherently low-scoring, high-variance sport where upsets are common. Published academic research (Hvattum & Arntzen, 2010) reports approximately 53--55% accuracy for three-class World Cup prediction using large datasets. The project's best model achieves 60.9%, which is competitive with the literature -- though the small test set (64 matches from a single tournament) introduces significant variance into that estimate.

---

## 2. Dataset Description

### Source

The data comes from the **Fjelstul World Cup Database v1.2.0** (Joshua C. Fjelstul, Ph.D.), a comprehensive open-source dataset of all FIFA World Cup matches. Two raw CSV files are used:

| File | Description |
|------|-------------|
| `files_needed/matches.csv` | All World Cup matches (men's and women's combined) |
| `files_needed/tournaments.csv` | Tournament metadata including year |

### Scope After Cleaning

After filtering to men's World Cups only, the dataset contains **964 matches** across 22 tournaments (1930--2022), with 84 distinct national teams.

| Split | Matches | Tournaments | Years |
|-------|---------|-------------|-------|
| Training | 900 | 21 | 1930--2018 |
| Test | 64 | 1 | 2022 |

The test split is a single tournament -- the 2022 Qatar World Cup. This is a deliberate design choice: it simulates the real-world use case where a model trained on all historical data is used to predict the next World Cup. However, it means all test-set metrics carry high variance; a single upset shifts accuracy by approximately 1.5 percentage points.

### Raw Schema (38 columns)

The raw match data includes: tournament and match identifiers (`tournament_id`, `match_id`), stage information (`group_stage`, `knockout_stage`, `stage_name`), team identifiers (`home_team_name`, `away_team_name`, team codes), scores (`home_team_score`, `away_team_score`, margins), extra-time and penalty indicators, venue details (`stadium_name`, `city_name`, `country_name`), and outcome labels (`result`, `home_team_win`, `away_team_win`, `draw`).

A full codebook is maintained in `docs/worldcup_subset_codebook.csv`.

### Target Variable

The target is the `result` column with three classes:

| Class | Train Count | Train % | Test Count | Test % |
|-------|-------------|---------|------------|--------|
| Home team win | 513 | 57.0% | 32 | 50.0% |
| Away team win | 218 | 24.2% | 16 | 25.0% |
| Draw | 169 | 18.8% | 16 | 25.0% |

The training set has substantial **class imbalance**: home wins dominate at 57%, while draws are the minority at 19%. Notably, the 2022 test set is more balanced (50/25/25), reflecting that Qatar 2022 had an unusually even distribution of outcomes compared to historical norms. This distributional shift between train and test is itself an interesting analytical detail -- it means models that heavily exploit the home-win prior will be penalized on the test set.

The "home team" designation in World Cup matches is a FIFA administrative assignment, not a true home advantage as in domestic leagues. This is a conceptual subtlety that the project handles partly through the `home_is_host` feature.

---

## 3. Project Structure and Data Flow

```
datascienceproject/
├── files_needed/               # Raw source data (untouched)
│   ├── matches.csv
│   └── tournaments.csv
│
├── scripts/
│   ├── clean_worldcup.py       # Part 1: cleaning + train/test split
│   └── feature_engineering.py  # Part 2: 37-feature pipeline
│
├── data_clean/
│   ├── matches_train.csv       # 900 rows x 38 cols
│   ├── matches_test.csv        # 64 rows x 38 cols
│   ├── features_train.csv      # 900 rows x 42 cols (37 features + 4 meta + target)
│   └── features_test.csv       # 64 rows x 42 cols
│
├── docs/
│   ├── worldcup_subset_codebook.csv
│   └── summary.md              # This file
│
├── figures/                    # 11 generated visualizations
├── claudedocs/                 # Research notes on additional datasets
├── Data Science Report.ipynb   # Part 1 notebook (EDA)
├── Part2_Models_and_Results.ipynb  # Part 2 notebook (modeling)
├── requirements.txt
└── README.md
```

### Pipeline

The data flows through three stages:

1. **Cleaning** (`clean_worldcup.py`): Loads raw CSVs, filters to men's World Cups via tournament name matching, parses dates, casts boolean-like columns to 0/1 integers, ensures numeric score columns, merges in tournament year, and splits into train (1930--2018) and test (2022).

2. **Feature Engineering** (`feature_engineering.py`): Loads cleaned data, concatenates train + test for chronologically consistent feature computation, engineers 37 features across 9 categories, fills NaN values with principled defaults, then splits back into train/test CSVs.

3. **Modeling** (`Part2_Models_and_Results.ipynb`): Loads engineered features, scales with `StandardScaler`, evaluates 7 supervised models with temporal and stratified cross-validation, performs class imbalance experiments, runs unsupervised analysis (PCA, t-SNE, K-Means), and reports results.

The Part 2 notebook auto-runs `feature_engineering.py` if the feature CSVs are missing, making the pipeline self-contained from step 2 onward.

---

## 4. Data Cleaning (Part 1) -- Detailed Analysis

### `scripts/clean_worldcup.py` (173 lines)

The cleaning script is well-organized into four functions with a `main()` driver:

**`load_raw_data()`**: Loads the two raw CSVs using `pathlib`-based paths relative to the project root. Uses `pathlib.Path(__file__).resolve().parents[1]` to reliably locate the project root regardless of working directory -- a good practice.

**`filter_mens_world_cup()`**: Filters tournaments by name match on "FIFA Men's World Cup", then joins the resulting tournament IDs back to the matches table. Also merges in the `year` column for temporal splitting.

**`basic_type_cleaning()`**: Applies three categories of type conversion:
- Dates: `pd.to_datetime` with `errors='coerce'` to safely handle malformed values.
- Boolean columns (9 total): Converted to `int8` via `pd.to_numeric` + `fillna(0)` -- a defensive approach that handles edge cases in the source data.
- Score columns (6 total): Ensured numeric via `pd.to_numeric` with coercion.

**`split_train_test()`**: Simple year-based split with a guard clause for missing `year` column.

**Notable design decisions:**
- The script operates on copies (`df.copy()`) throughout, avoiding mutation of the input DataFrames. This is defensive and good practice.
- The cleaning is deliberately minimal ("light type cleaning") -- no feature engineering happens here, maintaining a clean separation of concerns.
- Output directory creation uses `mkdir(exist_ok=True)`, making the script idempotent.

---

## 5. Feature Engineering (Part 2) -- Detailed Analysis

### `scripts/feature_engineering.py` (856 lines)

This is the most substantial code artifact in the project. It implements a **leakage-safe** feature engineering pipeline producing 37 features across 9 categories. The "leakage-safe" property is critical and well-executed: all features use only information available *before* the current match.

#### Design Philosophy

The pipeline works by concatenating train and test data, sorting chronologically, computing features using expanding windows with `shift(1)` (to exclude the current match), and then splitting back. This ensures that:
- Test set features are computed using only data from the training period.
- Within the training set, each row's features use only strictly earlier matches.
- No same-day leakage occurs (the shift is by date-slot, not by row).

#### Feature Categories

**A. Team Historical Performance (12 features)**

The core historical features are computed via a long-form transformation: each match generates two rows (one per team), enabling team-level expanding-window aggregation. The implementation uses `groupby().cumsum().shift(1)` at the date level to prevent same-day leakage -- this is more careful than a simple row-level shift, since multiple matches can occur on the same date.

Features include cumulative win rate, draw rate, goals per game, goals conceded per game, and total matches played for both home and away teams, plus two difference features.

**B. Match Context (3 features)**

Binary indicators for group/knockout stage plus an ordinal encoding of stage depth (group=0 through final=6). The ordinal mapping covers 7 stages including the "second group stage" format used in early World Cups.

**C. Host Advantage (2 features)**

Binary flags for whether home/away team is the host nation. Includes a `HOST_TEAM_ALIASES` dictionary to handle known name mismatches (e.g., "South Korea" vs "Korea Republic") -- a nice defensive touch even though the data currently has matching names.

**D. World Cup Experience (3 features)**

Counts of distinct prior World Cup tournaments each team appeared in, plus their difference. Computed by building a team-to-years mapping and counting years strictly before the current tournament.

**E. Head-to-Head Record (5 features)**

Prior meetings between the two teams, regardless of who was home/away in those prior matches. The implementation correctly handles the asymmetry: if Team A beat Team B in a prior meeting where Team B was "home", that counts as a win for Team A from Team A's current perspective.

The implementation uses an iterative approach with a `history` dictionary accumulating results as it processes matches chronologically. The default `h2h_home_win_rate` for first-time meetings is 0.33 (uniform prior for 3 classes) -- a principled choice.

**F. ELO Ratings (3 features)**

Standard ELO with K=32, all teams starting at 1500. Pre-match ratings are recorded as features, then updated post-match. The ELO formula uses:
- Expected score: `E = 1 / (1 + 10^((R_opponent - R_self) / 400))`
- Update: `R_new = R_old + K * (S - E)` where S is 1/0.5/0 for win/draw/loss.

**A critical limitation acknowledged by the project**: ELO is computed from World Cup matches only, meaning ratings update only every 4 years between tournaments. This creates noisy, infrequently-updated estimates compared to what's achievable with all international matches.

**G. Rolling Form -- Last 5 Matches (4 features)**

Win rate and goals per game over each team's last 5 World Cup matches, with shift(1) to exclude the current match. Uses a hierarchical cold-start fallback: rolling window -> all-time team average -> default (0.33 for win rate, 0.0 for goals).

**H. Rest Days (2 features)**

Days since each team's previous World Cup match, capped at 365 to avoid extreme outliers from 4+ year gaps between tournaments. First-match NaNs filled with the median rest value.

**I. Interaction Features (3 features)**

- `home_attack_x_away_defense`: Home goals per game * away goals conceded per game (attack vs. leaky defense).
- `away_attack_x_home_defense`: Mirror of the above.
- `elo_x_form_diff`: ELO difference * rolling form difference (strength * momentum interaction).

These interactions capture non-linear relationships that linear models can't learn and that tree models might miss in small datasets.

#### Missing Value Strategy

The fill strategy is principled and consistent:

| Feature Type | Fill Value | Rationale |
|-------------|------------|-----------|
| Rate features | 0.33 | Uniform prior for 3-class problem |
| Count features | 0 | No prior history |
| Goals rate features | 0.0 | Conservative cold-start |
| ELO ratings | 1500.0 | Standard starting ELO |
| Rest days | Median | Central tendency fallback |
| Interaction features | 0.0 | Product of zero-filled components |

#### Code Quality Observations

- Functions are well-documented with docstrings explaining inputs, outputs, and methodology.
- Each feature category is implemented as a separate function, making the pipeline modular and testable.
- The pipeline correctly handles edge cases: cold-start teams, same-day matches, first-time head-to-head meetings.
- Post-match columns (scores, outcomes) are explicitly listed in `POST_MATCH_COLS` and excluded from the final output, preventing accidental leakage.
- The `print_summary()` function provides a NaN audit after filling, which is valuable for verifying the pipeline's integrity.

---

## 6. Exploratory Data Analysis (Part 1 Notebook)

### `Data Science Report.ipynb`

This notebook was originally developed in Google Colab (it contains `google.colab.drive` mounts). It performs standard EDA:

1. **Shape inspection**: Train (900 x 38), Test (64 x 38).
2. **Column types**: 17 integer, 21 object columns. No nulls in either split.
3. **Numerical summaries**: `describe()` reveals home teams score 1.78 goals on average vs. 1.05 for away teams -- a substantial home advantage in the FIFA-designated sense.
4. **Categorical summaries**: 84 unique teams, 8 stage names, 17 host countries. Most frequent match-up: Brazil vs Sweden (6 meetings).
5. **Class distribution**: Plotted as bar chart, confirming the 57/24/19 split.
6. **Score distributions**: Histograms showing right-skewed distributions for goals, heavy zero-mass for penalty columns.
7. **Correlation heatmap**: Among score columns and outcome indicators.

The EDA is solid but relatively brief -- it establishes the data quality and class imbalance that inform Part 2's modeling decisions.

---

## 7. Modeling and Evaluation (Part 2 Notebook)

### `Part2_Models_and_Results.ipynb`

This notebook is the analytical core of the project. It implements a rigorous evaluation framework and compares seven models.

### 7.1 Preprocessing

- **Feature matrix**: 37 numeric features (metadata and target dropped).
- **Scaling**: `StandardScaler` fit on training data only, applied to both sets (no test leakage).
- **Target encoding**: `LabelEncoder` maps result strings to integers.

### 7.2 Evaluation Framework

The notebook implements two cross-validation schemes, which is one of the project's strongest methodological contributions:

**Temporal Walk-Forward CV (3 folds):**

| Fold | Training | Validation | ~Train Size | ~Val Size |
|------|----------|------------|-------------|-----------|
| 1 | 1930--2006 | 2010 | ~770 | ~64 |
| 2 | 1930--2010 | 2014 | ~834 | ~64 |
| 3 | 1930--2014 | 2018 | ~898 | ~64 |

This respects the temporal ordering of the data and is the honest evaluation protocol. Standard k-fold would allow models to train on 2018 data and validate on 2010 data, creating temporal leakage.

**Stratified 5-Fold CV:** Reported alongside temporal CV for comparison. Every model shows 4--9 percentage points higher accuracy under stratified CV than temporal CV, confirming that random CV overestimates performance on temporal sports data.

**Metrics reported:**
- Accuracy
- Macro F1 (penalizes poor minority-class performance)
- Ranked Probability Score (RPS) -- measures quality of predicted probabilities assuming ordinal outcome ordering (away win < draw < home win). Lower is better.
- Confusion matrix
- Classification report
- Calibration curves (for best model)

### 7.3 Models Evaluated

**KNN**: Tuned k in {3, 5, 7, 9} via temporal CV. Distance-based, so benefits from StandardScaler.

**Decision Tree**: Tuned `max_depth` in {3, 5, 7, 10, None}. Feature importance analysis included.

**Naive Bayes (GaussianNB)**: No hyperparameters. Assumes conditional independence of features given class -- a strong assumption violated by the correlated features in this dataset.

**SVM (RBF)**: Tuned C in {0.1, 1, 10, 100} with `class_weight='balanced'`. Probability calibration enabled for RPS computation.

**Random Forest**: 200 trees with `class_weight='balanced'`. Feature importance analysis included.

**Neural Network (MLP)**: Two hidden layers (64, 32 neurons), early stopping, max 500 iterations.

**XGBoost**: Three configurations tested via temporal CV, varying `max_depth` (3/4/5), `n_estimators` (100/150/200), and `learning_rate` (0.05/0.08/0.1). Uses L1/L2 regularization (`reg_alpha=0.5`, `reg_lambda=2.0`). Draw upweighting at 1.5x sample weight.

### 7.4 Results

| Model | Temporal CV Acc | Temporal CV F1 | Test Acc | Test F1 | Test RPS |
|-------|:-:|:-:|:-:|:-:|:-:|
| KNN | 0.479 | 0.445 | 0.531 | 0.463 | 0.158 |
| Decision Tree | 0.536 | 0.419 | 0.516 | 0.469 | 0.165 |
| Naive Bayes | 0.359 | 0.356 | 0.328 | 0.316 | 0.253 |
| SVM (RBF) | 0.479 | 0.450 | 0.516 | 0.392 | 0.139 |
| Random Forest | 0.547 | 0.465 | **0.609** | **0.556** | **0.133** |
| Neural Network | 0.411 | 0.242 | 0.516 | 0.415 | 0.148 |
| XGBoost | 0.521 | 0.434 | 0.594 | 0.514 | 0.146 |

**Best model: Random Forest** with 60.9% test accuracy, 0.556 macro F1, and 0.133 RPS.

### 7.5 Key Findings

1. **ELO features dominate**: Both Decision Tree and Random Forest rank `elo_diff`, `home_elo`, and `away_elo` as top features. The interaction term `elo_x_form_diff` also ranks highly, suggesting that combining long-term strength (ELO) with short-term momentum (rolling form) captures meaningful signal.

2. **Temporal CV is essential**: Every model shows inflated performance under stratified CV (4--9 pp higher). This is one of the project's most valuable empirical contributions -- it quantifies exactly how much random CV lies in temporal prediction contexts.

3. **Draws are nearly unpredictable**: Best draw recall is approximately 30%. This is consistent with the literature and reflects a fundamental property of football -- draws arise from close, balanced matchups where the outcome is essentially a coin flip.

4. **XGBoost did not beat Random Forest**: This is likely a sample-size effect. With ~900 training rows, RF's bagging provides better variance reduction than boosting. Gradient boosting methods typically need larger datasets to outperform bagging ensembles.

5. **Naive Bayes performs worst**: The conditional independence assumption is badly violated by the highly correlated feature set (e.g., `home_elo` and `home_hist_win_rate` are strongly correlated).

6. **Neural Network shows high variance**: 41.1% temporal CV accuracy but 51.6% test accuracy suggests the MLP is unstable on this small dataset, even with early stopping.

### 7.6 Class Imbalance Experiments

Three strategies compared for RF, SVM, and XGBoost:
- **Default**: No reweighting.
- **Balanced**: `class_weight='balanced'` (inversely proportional to class frequency).
- **Draw 1.5x**: Manual sample weights giving draws 1.5x, others 1.0x.

Additionally, **SMOTE** was tested for synthetic minority oversampling.

The balanced weighting used in the final Random Forest model was selected based on these experiments.

---

## 8. Unsupervised Analysis

Three unsupervised techniques explore the structure of match data:

### PCA
- Approximately 15 components explain ~90% of variance.
- 2D PCA scatter shows some separation between outcome classes but significant overlap.
- The high dimensionality required for 90% variance (15 out of 37 features) suggests the features capture diverse, non-redundant information.

### t-SNE
- Applied with perplexity=30 and 1000 iterations on combined train+test.
- Reveals local clustering structure but no clean separation by outcome class.
- This is expected -- match outcomes are determined by subtle feature interactions, not by gross distributional differences.

### K-Means (k=3)
- Adjusted Rand Index (ARI) of 0.022 -- essentially random agreement with actual labels.
- Confirms that unsupervised distance-based methods cannot recover the outcome structure. The "clusters" in match data correspond to other latent factors (e.g., era of play, tournament stage) rather than outcome categories.

---

## 9. Visualizations Produced

The project generates 11 publication-quality figures saved to `figures/`:

| Figure | Description |
|--------|-------------|
| `01_class_distribution.png` | Bar chart of target variable distribution |
| `02_correlation_heatmap.png` | Correlation matrix of all 37 engineered features |
| `03_model_comparison.png` | Grouped bar chart comparing test accuracy and F1 across 7 models |
| `04_temporal_vs_stratified_cv.png` | Side-by-side comparison showing CV inflation |
| `05_confusion_matrix_best.png` | Confusion matrix for the best model (Random Forest) |
| `06_feature_importance_rf.png` | Random Forest feature importances (top features) |
| `07_pca_explained_variance.png` | Cumulative variance explained by PCA components |
| `08_pca_scatter.png` | 2D PCA scatter colored by outcome class |
| `09_tsne_scatter.png` | 2D t-SNE scatter colored by outcome class |
| `10_kmeans_clusters.png` | K-Means cluster assignments vs. actual labels |
| `11_elo_distribution.png` | ELO rating distributions by match outcome |

---

## 10. Code Quality Assessment

### Strengths

- **Separation of concerns**: Cleaning, feature engineering, and modeling are in separate files/notebooks, making the pipeline modular.
- **Leakage prevention**: The feature engineering pipeline is meticulously designed to avoid information leakage, using date-level shifts and expanding windows.
- **Reproducibility**: The pipeline is self-contained -- `feature_engineering.py` auto-runs from the notebook if CSVs are missing. `requirements.txt` lists the key dependency.
- **Defensive coding**: Copy semantics throughout, `errors='coerce'` for type conversions, `mkdir(exist_ok=True)` for directories, alias dictionaries for name mismatches.
- **Documentation**: Comprehensive README, docstrings on all functions, inline comments explaining non-obvious choices.
- **Methodological rigor**: Temporal CV alongside stratified CV, multiple metrics (accuracy, F1, RPS), calibration analysis, and honest discussion of limitations.

### Areas for Improvement

- **`requirements.txt` is incomplete**: Only lists `pandas>=2.0.0`. Missing: `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, and optionally `imblearn`. The README lists them manually, but `pip install -r requirements.txt` would not install a working environment.
- **Google Colab coupling in Part 1 notebook**: The EDA notebook uses `google.colab.drive` mounts with hardcoded Colab paths (`/content/drive/MyDrive/DS_project/`). This breaks local execution without modification. The Part 2 notebook uses relative paths correctly.
- **No automated tests**: The feature engineering pipeline is complex enough to warrant unit tests (e.g., verifying that ELO updates are correct, that shift(1) prevents leakage, that cold-start defaults are applied).
- **Iterative loops in `compute_head_to_head()` and `compute_elo_ratings()`**: These use Python-level `for` loops over all rows, which is slow for larger datasets. Acceptable for ~964 rows but would need vectorization for the 49K-match expansion discussed in the research notes.
- **No random seed documentation**: `RANDOM_STATE` is used in the Part 2 notebook but not explicitly shown in what was visible; documenting the seed value aids reproducibility.

---

## 11. Limitations and Honest Self-Assessment

The project is commendably transparent about its limitations:

| Limitation | Impact | How Addressed |
|------------|--------|---------------|
| Small test set (64 matches, 1 tournament) | High variance in test metrics | Temporal CV reported alongside |
| Temporal shift (92 years of data) | Early data may not represent modern football | Expanding windows naturally downweight old data |
| ELO from WC matches only | 4-year gaps produce noisy ratings | Identified as future work |
| No FIFA rankings, betting odds, squad data | Missing strong predictive signals | Researched in `claudedocs/` |
| "Home team" is a FIFA designation | No true home advantage at neutral venues | `home_is_host` partially addresses |
| Class imbalance (19% draws) | Draws are hardest to predict | Balanced weights, SMOTE tested |
| No stacking or ensembling | Could potentially improve accuracy | Risk of overfitting with ~900 samples |
| Historical/geographic bias | Model less calibrated for underrepresented confederations | Acknowledged in bias discussion |

---

## 12. Research and Future Directions

The `claudedocs/research_additional_datasets_20260330.md` file documents a thorough investigation into datasets that could improve the project. The highest-impact additions identified are:

1. **International Football Results (49K matches)**: Would allow ELO computation from all international matches (not just WC), dramatically improving rating accuracy and rolling form features.
2. **FIFA World Rankings (1992--2024)**: Official ranking points as a complementary strength signal.
3. **Transfermarkt Player Market Values**: Squad market value is one of the strongest single predictors in published research.
4. **FIFA Video Game Ratings**: EA Sports' 110-attribute player profiles, aggregatable to squad level.
5. **World Bank Economic Indicators**: GDP and population as structural advantage features.

The research estimates that integrating these sources could push accuracy from 60.9% to approximately 65--72%, with the largest gains from all-match ELO computation and squad market value features.

---

## 13. Course Curriculum Alignment

The project demonstrates strong alignment with course content:

| Course Unit | Methods Applied |
|-------------|----------------|
| Unit 1: Data Management | Data description, preprocessing, transformation, visualization, feature engineering |
| Unit 2: Supervised Learning | KNN, Decision Tree, Naive Bayes, SVM, Random Forest, Neural Network, XGBoost |
| Week 7: Model Evaluation | Temporal CV, stratified CV, confusion matrices, classification reports, balanced weights, SMOTE |
| Unit 3: Unsupervised Learning | PCA (dimensionality reduction), t-SNE (visualization), K-Means clustering (ARI evaluation) |

---

## 14. Post-Baseline Experiments: External Data and Feature Engineering

After establishing the baseline (60.9% accuracy, 0.556 F1), we conducted an extensive series of experiments integrating 5 external datasets and testing multiple modeling strategies. The full details are in separate documents under `docs/`; this section summarizes the key findings.

### 14.1 External Data Sources Integrated

| Source | File | Records | Coverage |
|--------|------|--------:|:--------:|
| [Kaggle (martj42)](https://github.com/martj42/international_results) | `international_results.csv` | 49,287 | 1872--2026, all international matches |
| [Dato-Futbol (GitHub)](https://github.com/Dato-Futbol/fifa-ranking) | `fifa_rankings.csv` | 67,894 | 1992--2024, monthly FIFA rankings |
| [Transfermarkt (R2 CDN)](https://github.com/dcaribou/transfermarkt-datasets) | `players.csv` + `player_valuations.csv` | 47,702 + 616,377 | 2004--2026, player market values |
| [Fjelstul (GitHub)](https://github.com/jfjelstul/worldcup) | `wc_squads.csv` | 13,843 | 1930--2022, WC squad rosters |
| [StatsBomb (open data)](https://github.com/statsbomb/open-data) | `statsbomb_wc_stats.csv` | 256 | WC 2018 + 2022, event-level match data |

Total new data: ~780K records from 5 sources, all freely available without authentication.

### 14.2 Phase 1: Hyperparameter Tuning (biggest impact)

A grid search over RF hyperparameters using 4-fold temporal CV found that **`min_samples_leaf=5`** (up from default 1) was the single largest improvement in the entire project:

| Config | CV F1 | Test Acc | Test F1 | Draw Recall |
|--------|:-----:|:--------:|:-------:|:-----------:|
| Original RF (default) | 0.488 | 0.609 | 0.556 | 0.30 |
| **Tuned RF (min_leaf=5, 300 trees)** | **0.531** | **0.656** | **0.636** | **0.60** |

This single change improved CV F1 by **+4.3 pp** -- more than any feature addition. The regularization prevents the model from memorizing noise in 900-row training data.

### 14.3 Phase 2: Expanded Training on International Matches (negative result)

**Hypothesis:** Training on ~28K competitive international matches (instead of 900 WC matches) would enable more features.

**Result:** Every configuration underperformed the original baseline. None predicted even one draw correctly.

| Config | Training Rows | Test Acc | Test F1 |
|--------|:------------:|:--------:|:-------:|
| Original (WC only) | 900 | 0.609 | 0.556 |
| All competitive intl | 28,227 | 0.562 | 0.400 |
| WC + qualifiers only | 8,770 | 0.531 | 0.377 |
| All comp, WC 20x weighted | 28,227 | 0.547 | 0.392 |

**Root cause:** Domain mismatch. In qualifiers and friendlies, the home team plays at their actual stadium (~60% win rate). In WC matches, "home team" is a FIFA administrative label at a neutral venue (~50%). The model learned a home-win bias that doesn't apply to the WC. WC-specific features (WC appearances, WC-only ELO, WC host advantage) capture "tournament DNA" that international features miss.

Full details: `docs/phase2_results.md`

### 14.4 Phase 3: Feature Additions (5 sources tested)

Each feature was tested incrementally with 4-fold temporal CV using the tuned RF.

#### 3a. Home Continent Advantage -- THE ONLY IMPROVEMENT

Two binary flags: whether each team plays on their home continent (e.g., European team at a European WC).

| Config | Features | CV F1 | CV Draw Recall |
|--------|:--------:|:-----:|:--------------:|
| Base WC-only (tuned) | 37 | 0.531 | 0.41 |
| **+ continent advantage** | **39** | **0.541** | **0.44** |

**Why it works:** 100% training coverage, near-zero correlation with existing features (binary geographic flag vs continuous ELO/rates), captures a real effect (European teams win 60%+ at European WCs).

#### 3b. FIFA World Rankings -- HURTS

FIFA ranking points (1992--2024). Every configuration reduced F1 and killed draw recall (0.41 -> 0.23). Same "overconfidence" pattern as international ELO: precise team-strength features make the model stop predicting draws.

#### 3c. Qualifying Path Strength -- NEUTRAL

Each team's win rate in their WC qualifying campaign. Preserved draw recall but didn't improve F1.

#### 3d. Squad Market Value -- DOES NOT HELP

Transfermarkt top-23 squad values (2006--2022). Only 30% training coverage; 70% median-filled. The signal is real (England €1,433M vs Iran €62M) but redundant with ELO and too sparse for 900 rows.

#### 3e. StatsBomb xG -- DOES NOT HELP (but best theoretical case)

Within-tournament rolling expected goals from event-level match data. Had the **lowest ELO correlation** of any feature tested (r=0.35, vs r=0.50-0.61 for others) -- genuinely novel signal capturing match quality rather than team strength. But only 10% training coverage (WC 2018 + 2022) killed it.

**The paradox:** The most theoretically promising feature (lowest correlation, most novel signal) had the worst coverage. The feature that worked (continent advantage) was the simplest.

| Feature | ELO Corr | Coverage | CV F1 Change | Verdict |
|---------|:--------:|:--------:|:------------:|:-------:|
| **Continent advantage** | **~0.0** | **100%** | **+1.0 pp** | **Keep** |
| FIFA rankings | ~0.50 | 60% | -2.8 to -4.8 pp | Hurts |
| Qualifying record | ~0.30 | 70% | -0.9 to -2.1 pp | Neutral |
| Squad market value | ~0.50 | 30% | -0.1 to -2.2 pp | Neutral |
| StatsBomb xG | 0.35 | 10% | -1.1 to -2.1 pp | Noise |

Full details: `docs/phase3_summary.md`, `docs/phase3_results.md`

### 14.5 Key Finding: The 900-Row Constraint

The central finding across all experiments is that **900 training rows is the binding constraint**, not feature richness. On 900 rows:

- **Regularization helps more than features.** `min_samples_leaf=5` gained +4.3 pp F1; the best feature addition gained +1.0 pp.
- **~40 features is the ceiling.** Beyond that, Random Forest overfitting begins, draw prediction collapses first.
- **Precise team-strength features are counterproductive.** They make the model overconfident, eliminating draw predictions. The WC-only ELO's 4-year update gaps create "productive imprecision" that helps predict draws.
- **A feature needs three properties to help:** (1) low correlation with existing features, (2) 100% training coverage, (3) minimal column count. Only continent advantage met all three.

---

## 15. Best Model (Final)

```
Features: 39 (37 WC-only + 2 continent advantage)
Model:    RandomForestClassifier(
              n_estimators=300,
              max_features='sqrt',
              min_samples_leaf=5,
              class_weight='balanced',
          )
```

| Metric | Original | Final |
|--------|:--------:|:-----:|
| Temporal CV F1 | 0.488 | **0.541** |
| Temporal CV Draw Recall | 0.24 | **0.44** |
| Test Accuracy | 0.609 | **0.625** |
| Test Macro F1 | 0.556 | **0.578** |

---

## 16. Summary of Key Numbers

| Metric | Value |
|--------|-------|
| Total matches in dataset | 964 |
| Training matches | 900 |
| Test matches | 64 |
| Distinct national teams | 85 |
| Tournament span | 1930--2022 (92 years) |
| Raw features per match | 38 columns |
| Engineered features (base) | 37 WC-only + 2 continent = 39 |
| Engineered features (all available) | 78 (incl. international + Phase 3) |
| Feature categories | 14 |
| External data sources integrated | 5 |
| External data records | ~780K |
| Models evaluated | 7 |
| Best model | Random Forest (tuned) |
| Best CV F1 (honest metric) | 0.541 |
| Best test accuracy | 0.625 (vs. 50.0% baseline) |
| Best test macro F1 | 0.578 |
| Most important feature | `elo_diff` |
| Draw recall (best) | 0.44 (CV), 0.40 (test) |
| CV inflation (stratified vs. temporal) | 4--9 pp |
| Python scripts | 3 (2,564 lines total) |
| Jupyter notebooks | 2 |
| Generated figures | 11 |
| Documentation files | 7 markdown files in `docs/` |

---

## 17. Documentation Index

| Document | Purpose |
|----------|---------|
| `docs/summary.md` | This file -- full project overview |
| `docs/code_analysis.md` | Detailed code walkthrough (every function, design patterns, leakage audit) |
| `docs/improvement_plan.md` | Original 3-phase improvement strategy |
| `docs/phase2_results.md` | Expanded training experiment (negative result) |
| `docs/phase3_plan.md` | Feature addition planning document |
| `docs/phase3_results.md` | Phase 3 initial results (continent, FIFA rankings, qualifying) |
| `docs/phase3_summary.md` | Complete Phase 3 analysis (3a-3e with all experimental data) |

---

## 18. Final Reflection

This project demonstrates a complete data science pipeline from raw data to model evaluation, with an extensive post-baseline experimental phase that tested 5 external data sources and multiple modeling strategies.

**The strongest contributions are:**

1. **Leakage-safe feature engineering** -- the expanding-window and binary-search lookup architectures ensure no future information contaminates predictions.

2. **Temporal evaluation** -- quantifying that stratified CV overestimates performance by 4-9 pp on temporal sports data.

3. **Systematic negative results** -- the Phase 2 and Phase 3 experiments are documented failures that reveal important principles:
   - Domain-specific data beats general data (Phase 2)
   - Regularization beats feature engineering on small datasets (Phase 1)
   - Feature coverage matters more than feature novelty (Phase 3)
   - Precise strength features destroy draw prediction (the "overconfidence" pattern)

4. **ELO as the dominant predictor** -- confirmed across all experiments. Even with 5 external data sources offering squad values, FIFA rankings, xG, and international match history, the WC-only ELO difference remains the single most important feature.

The inherent randomness of football imposes a hard performance ceiling. Our best model (CV F1 = 0.541) is competitive with published research, and the gap between what we achieved and the theoretical maximum is dominated by the sport's irreducible unpredictability rather than by methodological limitations.
