# Final Report: FIFA Men’s World Cup Match Outcome Prediction

**Course:** CSE 40467, Data Science  
**Project:** Course project (Parts 1 and 2)

---

## 1. Introduction

### 1.1 Executive Summary

This project predicts **FIFA Men’s World Cup** match outcomes before kickoff using only **leakage-safe** pre-match information. The task is **three-class classification**: home team win, away team win, or draw. The work spans **964 matches** across **22 tournaments (1930–2022)**, with **900 training matches (1930–2018)** and **64 test matches (2022)**.

We engineered **39 features** from World Cup history (rates, ELO, rolling form, head-to-head, stage, host, rest, interactions, and home-continent context). Models were compared under **temporal walk-forward cross-validation** (chronological folds) so that validation years are never seen during training for that fold—avoiding the optimistic bias of shuffled k-fold on time-ordered sports data. We also report **stratified k-fold** for contrast.

Among seven supervised learners, a **tuned Random Forest** (300 trees, `max_features='sqrt'`, `min_samples_leaf=5`, `class_weight='balanced'`) achieved the strongest results on honest temporal validation and on the held-out 2022 tournament: approximately **62.5% test accuracy** and **0.578 macro F1**, beating a **50% majority-class baseline** on 2022 labels and published ballpark benchmarks (~53–55% for similar three-class football tasks). **ELO-based features** dominated importance; **regularization** (`min_samples_leaf`) improved generalization more than most **extra data sources** tested later. **XGBoost** did not beat Random Forest on this sample size, consistent with **variance reduction from bagging** helping more than boosting when \(n \approx 900\)).

**Recommendations:** keep **temporal CV** for any future sports work; treat **draw prediction** and **class imbalance** explicitly; prefer **simple, regularized ensembles** on small tabular tournament data unless domain-matched data volume increases substantially.

---

### 1.2 Problem definition

**What the problem is.** Each row is one World Cup match. Given pre-match features (team strength proxies, form, context), the model outputs one of three **mutually exclusive outcomes** for administrative “home” vs “away” sides: home win, away win, or draw (after normal time / defined dataset rules).

**Why study it.**

1. **Decision value.** Forecasting supports planning, media analytics, and research on what pre-tournament signals actually carry information.
2. **Methodological lessons.** Sports outcomes are **non-IID over time**; naive cross-validation **leaks future tournaments** into training. The project shows how evaluation design changes conclusions.
3. **Class and label complexity.** Training data are **home-win heavy (~57%)** while draws are **~19%**; “home” in World Cups is often **not true home advantage**, so models must not blindly encode league-style home priors.
4. **Small-\(n\) learning.** ~900 rows with dozens of features makes **overfitting** and **feature redundancy** central issues—relevant to many real analytics settings, not only football.

---

## 2. Related Work

### 2.1 Related work with this data source

The primary data come from the **Fjelstul World Cup Database** (open repository of World Cup matches and tournaments). Related academic and applied work relevant to **this kind of source** (international / tournament football, pre-match features, no in-play leakage) includes:

- **Poisson and score models** (e.g., Maher, 1982; Dixon & Coles, 1997): foundational goal-count modeling; informs why separate attack/defense style signals appear in features.
- **ELO for match prediction** (e.g., Hvattum & Arntzen, 2010): ELO differences are strong predictors for three-class results; our feature importance aligns with that finding.
- **Machine learning on football** (e.g., Tax & Joustra, 2015): compared Naive Bayes, logistic regression, and random forests; **ensembles** often beat simple classifiers when combining heterogeneous inputs.
- **Hybrid / ensemble tournament methods** (e.g., Groll et al., 2019): World Cup prediction tournaments suggest **no single algorithm dominates**; careful features and evaluation matter as much as model complexity.
- **Ordinal / market-aware approaches** (e.g., Hubacek et al., 2019): three outcomes have structure (strength ordering); we used standard multiclass models—ordinal extensions are future work.

Our contribution relative to this thread is **strict temporal evaluation**, **documented leakage-safe feature construction**, and **empirical comparison** of multiple algorithms on **World-Cup-only** rows with external sources tested and mostly **rejected** when harmful to draw recall or F1.

---

## 3. Data description

### 3.1 Data collection

**Origin and purpose.** Raw tables were obtained from the **Fjelstul World Cup Database (v1.2.0)** and companion tournament metadata, distributed as CSVs for research and reproducibility. The database aggregates **official World Cup match records** (teams, stage, scores, venue fields, indicators for extra time and penalties, etc.). The original compiler likely **assembled** these from **historical FIFA / official competition records** and structured them into relational tables (`matches`, `tournaments`).

**How we used it.** We used:

- `files_needed/matches.csv` — match-level rows (men’s and women’s combined in raw form).
- `files_needed/tournaments.csv` — tournament identifiers and **year**.

**Collection in our pipeline.** We did not scrape the web at runtime; we **ingested** the provided CSVs from the project’s `files_needed/` directory. Additional experimental sources (international results, FIFA rankings, squads, StatsBomb summaries, etc.) were integrated in later phases per `README.md` and `docs/`; the **final modeling feature matrix** for the best configuration uses **World-Cup-derived columns plus continent flags**, with experimental columns **excluded** in the Part 2 notebook when they hurt or did not help temporal CV.

**Context.** Rows are **only teams that qualified** for a World Cup (selection into the tournament). Coverage spans many **eras of football** (rule changes, expansion), which affects stationarity.

---

### 3.2 Data preprocessing

**Cleaning (`scripts/clean_worldcup.py`).**

1. **Restrict to men’s World Cups** by filtering tournaments whose name contains **“FIFA Men’s World Cup”**, then restrict matches to those `tournament_id` values.
2. **Merge** tournament **year** onto matches.
3. **Train/test split by year:** train **1930–2018** (900 matches), test **2022** (64 matches)—simulates deploying after historical cups.
4. **Type cleaning:** parse `match_date` as datetime; coerce boolean-like fields (`group_stage`, `knockout_stage`, `extra_time`, `penalty_shootout`, win/draw indicators, etc.) to numeric 0/1 with safe handling of missing text.

**Feature engineering (`scripts/feature_engineering.py`).**

1. **Chronological processing** of combined train+test (sorted by date) so rolling statistics respect time.
2. **Leakage prevention:** features use **expanding / shifted windows** (`shift(1)` at **date** granularity so same-day matches do not leak across rows).
3. **Exclusion of post-match columns** from model inputs (scores, margins, win flags, penalty strings, etc.); they may inform **target construction** or historical updates only, not the current row’s inputs.
4. **Cold-start imputations** for rates, counts, ELO starts, rest-day medians, etc., as documented in `README.md` Section 4.3.

**Modeling preprocessing (`Part2_Models_and_Results.ipynb`).**

1. **Column selection:** drop identifiers and labels (`match_date`, `year`, `home_team_name`, `away_team_name`, `result`) from **\(X\)**; exclude experimental / rejected feature groups via `EXCLUDE_FEATURES` (FIFA points, qualifying rates, squad value, xG, international-history block, etc.) so the reported baseline+continent model uses **39** inputs.
2. **`StandardScaler`:** fit on **training** features only; transform train and test (prevents scaler leakage from future tournament).
3. **`LabelEncoder`** on string labels `result` for sklearn compatibility.

**Quality notes.** The dataset is **not a random sample of all football**; it is **elite tournament** matches with **longitudinal drift**. **“Home”** is an administrative label at often **neutral** venues. External files can have **partial coverage** (rankings, market values, xG)—those columns were imputed or dropped from the final feature set when they **hurt** temporal metrics.

---

### 3.3 Data documentation

| Property | Value |
|----------|------:|
| Total matches | 964 |
| Training | 900 (21 tournaments, 1930–2018) |
| Test | 64 (1 tournament, 2022) |
| Distinct teams | 84 |
| Final feature count (modeling) | 39 |

**Target distribution (from project documentation).**

| Class | Train count | Train % | Test (2022) count | Test % |
|-------|------------:|--------:|------------------:|-------:|
| Home win | 513 | 57.0 | 32 | 50.0 |
| Away win | 218 | 24.2 | 16 | 25.0 |
| Draw | 169 | 18.8 | 16 | 25.0 |

**Assumptions.**

- Pre-match information is **truthfully recorded** in the database.
- **World-Cup-only ELO** (updated only on World Cup matches) is a deliberate modeling choice: it trades **precision** for behavior that can **preserve draw plausibility** compared to very sharp strength measures (see discussion in `README.md` on “overconfidence”).
- **Train/test split by year** assumes the deployment setting is “predict the next cup,” not random matches from the same year.

**Limitations.**

- **Tiny test set (64)** → high variance in test accuracy.
- **92-year span** → oldest matches may be weakly comparable to modern ones.
- **Qualification selection** → no non-qualified national teams.
- **Label semantics** for “home” vs real home advantage.

**Reproducibility.** Run `python3 scripts/clean_worldcup.py`, then `python3 scripts/feature_engineering.py`, then execute `Part2_Models_and_Results.ipynb`. The codebook `docs/worldcup_subset_codebook.csv` documents raw match columns; `README.md` documents engineered features and experiment outcomes.

---

### 3.4 Variables

**Dependent variable (target).**

- **`result`** (string, then label-encoded): **`home team win`**, **`away team win`**, **`draw`**.

**Independent variables (features used in the final 39-column model).**

Metadata **excluded** from \(X\): `match_date`, `year`, `home_team_name`, `away_team_name`, and `result`.

The **39 features** are the engineered columns retained after dropping `EXCLUDE_FEATURES` in the notebook. They correspond to the categories in `README.md` Section 4.2, including (illustrative, not exhaustive):

- **Historical performance:** e.g. `home_hist_win_rate`, `away_hist_draw_rate`, `hist_win_rate_diff`, goals per game and conceded, matches played.
- **Context:** `is_group_stage`, `is_knockout`, `stage_ordinal`.
- **Host:** `home_is_host`, `away_is_host`.
- **Experience:** `home_wc_appearances`, `away_wc_appearances`, `wc_appearances_diff`.
- **Head-to-head:** `h2h_home_wins`, `h2h_away_wins`, `h2h_draws`, `h2h_total`, `h2h_home_win_rate`.
- **ELO:** `home_elo`, `away_elo`, `elo_diff`.
- **Rolling form:** `home_rolling5_win_rate`, `away_rolling5_win_rate`, `home_rolling5_goals_pg`, `away_rolling5_goals_pg`.
- **Rest:** `home_rest_days`, `away_rest_days`.
- **Interactions:** `home_attack_x_away_defense`, `away_attack_x_home_defense`, `elo_x_form_diff`.
- **Geography:** `home_on_home_continent`, `away_on_home_continent`.

**Excluded from final \(X\)** (still may exist in CSV): FIFA ranking points, qualifying win rates, squad valuations, StatsBomb rolling xG, and the full **international-match** feature block—see `EXCLUDE_FEATURES` in `Part2_Models_and_Results.ipynb`.

---

## 4. Modeling

All models use the same **scaled 39-dimensional** feature matrix and **label-encoded** target. **Hyperparameter tuning** for the models below uses **temporal walk-forward** validation on the training years (folds: validate 2010, 2014, 2018, 2022 while training on all prior tournament years in train). **Random state** 42 is used where applicable.

### 4.1 Algorithm 1: Random Forest (tuned)

**Description.** An ensemble of **decision trees** trained on **bootstrap** samples of rows and **random feature subsets** at each split (**bagging** + feature randomness). Predictions aggregate tree votes (or mean probabilities). Suited to **mixed feature types**, **nonlinearities**, and **interaction** effects without explicit specification.

**Key parameters (final reported configuration).**

| Parameter | Value | Role |
|-----------|-------|------|
| `n_estimators` | 300 | More trees reduce variance of the ensemble. |
| `max_features` | `'sqrt'` | Subsample features per split → decorrelates trees. |
| `min_samples_leaf` | 5 | **Regularization**: avoids tiny leaves fit to noise (tuned). |
| `class_weight` | `'balanced'` | Reweights classes inversely to frequency → better minority classes. |
| `random_state` | 42 | Reproducibility. |

**Fine-tuning.** Yes—grid search over tree count, `max_features`, and **minimum leaf size** under temporal CV (`README.md` Section 7.1). **`min_samples_leaf=5`** was among the most impactful choices for F1 and draw recall.

---

### 4.2 Algorithm 2: XGBoost (gradient boosted trees)

**Description.** **Gradient boosting** fits shallow trees **sequentially**, each correcting residual errors of the ensemble, with **regularization** on loss (L1/L2 style penalties in XGBoost). Often strong on tabular data when enough data exist to avoid overfitting.

**Key parameters (documented experiment range / typical setting).**

| Parameter | Typical / searched values | Notes |
|-----------|---------------------------|-------|
| `max_depth` | 3, 4, 5 | Tree depth cap. |
| `n_estimators` | 100, 150, 200 | Number of boosting rounds. |
| `learning_rate` | 0.05, 0.08, 0.1 | Shrinkage per step. |
| `reg_alpha` | 0.5 | L1 regularization. |
| `reg_lambda` | 2.0 | L2 regularization. |
| Sample weights | draws ×1.5 (variant) | Addresses imbalance in a boosting-specific way. |

**Fine-tuning.** Yes—**temporal CV** over the above grids (`README.md` Section 5.2.7). Despite tuning, **XGBoost did not exceed** the tuned Random Forest on reported test and temporal CV metrics—interpreted as a **small-sample** effect where **bagging** wins over **boosting**.

*Note:* If `xgboost` is not installed in an environment, the notebook may skip this block; full reproduction requires `pip install xgboost`.

---

### 4.3 Algorithm 3: Support Vector Machine (RBF kernel)

**Description.** **SVC** with **RBF kernel** maps inputs implicitly to a high-dimensional space and separates classes with a **maximum-margin** hyperplane. Can capture **nonlinear** boundaries; **scaled** inputs are important for distance-based kernels.

**Key parameters.**

| Parameter | Values / setting | Notes |
|-----------|------------------|------|
| `kernel` | `'rbf'` | Nonlinear decision surface. |
| `C` | Tuned over **{0.1, 1, 10, 100}** | Trade-off margin vs. misclassification. |
| `class_weight` | `'balanced'` | Mitigates home-win majority. |
| `probability` | Enabled where needed | For probability-based metrics (e.g., RPS in notebook). |

**Fine-tuning.** Yes—**\(C\)** selected via **temporal CV**. SVM **underperformed** Random Forest on the main table but provides a **kernel-based** contrast to tree ensembles.

---

## 5. Evaluation

### 5.1 Baseline

**Majority-class baseline:** always predict the **most frequent** training label, **home team win** (57% of training rows; 50% of 2022 test rows). Any useful model must beat **~50–57% accuracy** depending on which distribution you compare to; the README emphasizes beating the **50%** reference on the **2022** test distribution.

**Additional baselines implicit in the work:** stratified k-fold scores (optimistic vs. temporal reality), and **Naive Bayes** / **KNN** as weak or distance baselines.

---

### 5.2 Metrics

**Metrics used.**

| Metric | Definition / intent |
|--------|---------------------|
| **Accuracy** | Fraction of correct class labels. |
| **Macro F1** | Average of class-wise F1 **unweighted** → penalizes neglecting draws. |
| **Ranked Probability Score (RPS)** | Assesses **probability vectors** with respect to an **ordinal** outcome ordering (away–draw–home); lower is better. |
| **Confusion matrix / per-class report** | Where errors concentrate (especially draws). |
| **Temporal walk-forward CV** | Honest **time-ordered** performance; primary for model selection. |
| **Stratified k-fold CV** | Shown to demonstrate **optimism** when time is ignored. |

**Reported results (from project summary table, `README.md` Section 6.2).** Values below are **test-set** performance on **2022** unless labeled as CV.

| Model | Temporal CV accuracy | Temporal CV macro F1 | Test accuracy | Test macro F1 | Test RPS |
|-------|---------------------:|---------------------:|---------------:|---------------:|---------:|
| KNN | 0.479 | 0.445 | 0.531 | 0.463 | 0.158 |
| Decision Tree | 0.536 | 0.419 | 0.516 | 0.469 | 0.165 |
| Naive Bayes | 0.359 | 0.356 | 0.328 | 0.316 | 0.253 |
| SVM (RBF) | 0.479 | 0.450 | 0.516 | 0.392 | 0.139 |
| **Random Forest (tuned)** | **0.547** | **0.541** | **0.625** | **0.578** | **0.133** |
| Neural Network | 0.411 | 0.242 | 0.516 | 0.415 | 0.148 |
| XGBoost | 0.521 | 0.434 | 0.594 | 0.514 | 0.146 |

**Interpretation highlights.**

- Random Forest is **best on all listed metrics**, including **best (lowest) RPS** among compared models.
- **Temporal CV** scores are **materially lower** than naive stratified CV for the same models (README reports ~4–9 accuracy points gap), confirming **evaluation design matters**.
- **Draw** remains the hardest class (moderate draw recall vs. strong home/away behavior).

---

## 6. Discussion

### 6.1 Discussion of the results

**Alignment with prior literature.** Published three-class international football accuracy often clusters around **53–55%** when using serious pre-match structure (e.g., ELO-focused studies). Our **temporal CV macro F1 ~0.54** and **test accuracy ~62.5%** suggest **meaningful signal**, but the **single-tournament test** implies **uncertainty**: a few upsets move percentages noticeably.

**Why Random Forest won here.** With **~900 rows** and **dozens of correlated strength proxies**, **variance control** matters. **Bagging** stabilizes predictions; **`min_samples_leaf`** curbs **overfitting** in deep trees. **XGBoost**, despite regularization and tuning, **did not surpass** RF—consistent with boosting needing more **effective sample size** or cleaner signal-to-noise per step than we have.

**Why SVM and others lagged.** **SVM** can struggle with **class overlap** and **high-dimensional** noisy tabular sports data without very careful tuning and calibration; tree ensembles **match feature interactions** locally. **Naive Bayes** assumes conditional independence badly violated by correlated ELO and rate features, and performed worst.

**Feature story.** **ELO difference** and related ELO features rank at the top of importance analyses, agreeing with **Hvattum & Arntzen (2010)** and much applied work. **Interaction** (`elo_x_form_diff`) also ranks highly, supporting that **strength × momentum** is useful.

**Negative experimental lessons (from README).** Adding **highly precise** strength proxies with **partial coverage** (FIFA points, squad value, sparse xG) often **hurt draw recall** and macro F1—an **“overconfidence”** pattern: the model becomes sure the stronger side wins, **suppressing draws** that are structurally hard and minority. **Expanded training on tens of thousands of non–World Cup** matches failed due to **domain mismatch** (true home in qualifiers vs. administrative “home” in neutral World Cup games).

---

### 6.2 Recommendations and implications

1. **Keep temporal / walk-forward validation** for any match-level or season-level model. **Do not** shuffle World Cup rows across decades for final claims.
2. **Prioritize regularization and model simplicity** on **sub-1k row** tabular sports tasks before chasing **more columns**; this project showed **tuning `min_samples_leaf`** can dominate marginal gains from new sources.
3. **Monitor draw calibration explicitly** (macro F1, draw recall, RPS). **Accuracy alone** can hide **home-win majority** exploitation.
4. **Be cautious merging foreign domains** (friendlies, qualifiers) without **rebuilding labels and features** so that **“home”** and **strength** mean the same thing as in the deployment tournament.
5. **Future modeling directions:** ordinal models that respect away–draw–home ordering; **stacking** with caution on small \(n\); **market odds** as features or calibration anchors if ethically and legally appropriate; richer **event data** if coverage spans more years than 2018/2022-only proxies.

**Practical implication.** For **World Cup–only prediction** with **public structural data**, a **regularized Random Forest** with **careful leakage control** and **honest temporal validation** is a **strong, interpretable default**; gains from **complex boosting** or **many extra sources** are **not automatic** and can **backfire** when data are small or mismatched.

---

## References (key sources cited in README)

- Fjelstul, J. C. (2021). *The Fjelstul World Cup Database v1.2.0*. https://github.com/jfjelstul/worldcup  
- Hvattum, L. M., & Arntzen, H. (2010). Using ELO ratings for match result prediction in association football. *International Journal of Forecasting*, 26(3), 460–470.  
- Tax, N., & Joustra, Y. (2015). Predicting the Dutch football competition using public data: A machine learning approach.  
- Groll, A., Ley, C., Schauberger, G., & Lock, H. (2019). A hybrid random forest to predict soccer matches in national and international tournaments. *Journal of Quantitative Analysis in Sports*, 15(4), 271–287.  
- Hubacek, O., Sourek, G., & Zelezny, F. (2019). Exploiting sports-betting market using machine learning. *International Journal of Forecasting*, 35(2), 783–796.  
- Pedregosa, F., et al. (2011). scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.  
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*, 785–794.

---

*This report synthesizes `README.md`, `Part2_Models_and_Results.ipynb`, and project scripts. For full reproduction steps, see `README.md` Section 10.*
