# MIB Bot Detection — Operativo Plan

## Overview

**Goal:** Build a reproducible, benchmark-ready bot-detection pipeline on `Mib_final.parquet` (dataset MIB, Cresci et al.) that compares supervised, probabilistic, and anomaly-based models using a unified ROC/PR-AUC output surface.

**Scope:**
- Single Python project, structured as a set of modular scripts (one per phase).
- No external ML platform required — pure scikit-learn / xgboost / hdbscan stack.
- Deliverable: a final benchmark plot (ROC + PR curves) and a model-comparison table.

**Approach:**
The pipeline follows five sequential phases: ingest → split → feature engineering → model training → evaluation. Each phase is a self-contained script. Data leakage is prevented by fitting all transformers exclusively on the train set.

**Schema Assumptions (MIB standard, Cresci et al.):**

> ⚠️ These assumptions were made because the parquet file could not be inspected at plan time. Verify and adjust column names in `config.py` before running any phase.

| Assumed Column | Type | Notes |
|---|---|---|
| `user_id` | str/int | Unique user identifier |
| `text` | str | Message content |
| `created_at` | datetime/str | Publication timestamp |
| `is_bot` | int (0/1) | Ground-truth label |
| `urls` | str/list | URL field (may be absent — derive ratio from `text`) |
| `in_reply_to_user_id` | str/int | Interaction metadata |
| Additional metadata | various | Follower counts, retweet flags, etc. |

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffolding and Configuration

**Status:** `[ ] pending`

**Intent:**
Create the project skeleton so that every subsequent script has a consistent, importable foundation. A single `config.py` acts as the single source of truth for file paths, column names, hyperparameters, and random seeds — making adjustments from schema assumptions to real schema trivial without touching pipeline code.

**Expected Outcomes:**
- Directory structure created with dedicated folders for data, features, models, and output.
- `config.py` defines all column-name constants, file paths, split ratio, and model hyperparameters.
- `requirements.txt` lists all dependencies with pinned major versions.
- `README.md` explains how to run each phase.

**Todo List:**
1. Create directories: `data/`, `features/`, `models/`, `output/`, `src/`.
2. Move (or symlink) `Mib_final.parquet` into `data/`.
3. Write `src/config.py` with:
   - `PARQUET_PATH`, `FEATURES_PATH`, `MODELS_DIR`, `OUTPUT_DIR`
   - Column constants: `COL_USER_ID`, `COL_TEXT`, `COL_TIMESTAMP`, `COL_LABEL`, `COL_URLS`
   - Pipeline constants: `RANDOM_STATE = 42`, `TEST_SIZE = 0.30`, `MIN_MESSAGES_PER_USER = 20`
   - Model hyperparameter dicts (one per model)
4. Write `requirements.txt` (pandas, pyarrow, scikit-learn, xgboost, hdbscan, nltk/spacy for tokenisation, matplotlib, seaborn).
5. Write `README.md` with a numbered run-order guide.

**Relevant Context:**
- `Mib_final.parquet` is the sole input file.
- Schema assumption constants in `config.py` make it easy to patch column names once the real schema is confirmed.

---

### Sub-Task 2 — Ingest, Validate, and Aggregate (`01_ingest.py`)

**Status:** `[ ] pending`

**Intent:**
Load `Mib_final.parquet`, verify data integrity, and aggregate row-level messages into per-user profiles. This is the critical transformation described by Cresci et al.: classifying single messages is noisy and unreliable; classification must happen at the user level over a temporal sequence of at least N=20 messages.

**Expected Outcomes:**
- Script produces `data/users_raw.parquet`: one row per `user_id` with:
  - `messages`: list of texts sorted by `created_at`
  - `timestamps`: list of ISO timestamps sorted ascending
  - `is_bot`: single label per user (majority vote if inconsistent, log warning)
  - `url_count`, `mention_count`, `total_messages` raw counts
- Users with fewer than `MIN_MESSAGES_PER_USER` messages are dropped and logged.
- Null handling: numeric nulls imputed with median, empty text flagged.
- Schema mismatch raises a clear `ValueError` pointing to `config.py` constants.

**Todo List:**
1. Load parquet with `pandas.read_parquet`.
2. Assert required columns exist (raise with instructions to update `config.py` if not).
3. Parse `created_at` to `pd.Timestamp`; coerce errors to NaT and log count.
4. Sort each user's messages by timestamp ascending.
5. `groupby(COL_USER_ID)` to aggregate texts into list, timestamps into list, compute raw counts.
6. Resolve `is_bot` per user (assert uniqueness; if not, use majority vote + warning).
7. Filter users with `total_messages < MIN_MESSAGES_PER_USER` and log dropped count.
8. Save to `data/users_raw.parquet`.
9. Print a summary report: total users, bot/human ratio, median messages per user, null stats.

**Relevant Context:**
- All constants from `src/config.py`.
- If `urls` column does not exist, derive `url_count` by counting `http` occurrences in `text`.
- If `in_reply_to_user_id` does not exist, set `mention_count` from `@` occurrences in `text`.

---

### Sub-Task 3 — Stratified Split (`02_split.py`)

**Status:** `[ ] pending`

**Intent:**
Perform a 70/30 stratified split at the user level before any feature engineering. This is the anti-leakage boundary: all transformers (scalers, TF-IDF vectorisers, embedding models) will be fit exclusively on the train set and only applied (transform) on the test set.

**Expected Outcomes:**
- `data/train_users.parquet` and `data/test_users.parquet` saved, both containing the `is_bot` column.
- Class distribution printed before and after split to confirm stratification.
- No feature columns yet — only raw aggregated user profiles.

**Todo List:**
1. Load `data/users_raw.parquet`.
2. Extract `y = df['is_bot']`.
3. Call `sklearn.model_selection.train_test_split` with `stratify=y`, `test_size=TEST_SIZE`, `random_state=RANDOM_STATE`.
4. Save train and test splits to parquet.
5. Print class-balance report for both splits.

**Relevant Context:**
- `is_bot` is preserved in both splits for evaluation; it will be withheld from unsupervised models in Sub-Task 5.
- Using `user_id` as the splitting unit prevents the same user's messages leaking into both splits.

---

### Sub-Task 4 — Feature Engineering (`03_features.py`)

**Status:** `[ ] pending`

**Intent:**
Compute the six high-discriminativity indices defined in the plan. All transformers (TF-IDF vectoriser, scaler) are fit on train data only and serialised for reuse on the test set. This is the mathematical core of the pipeline, directly implementing the invariant-variance fingerprinting approach from Cresci et al.

**Expected Outcomes:**
- `features/X_train.parquet` and `features/X_test.parquet`: numerical feature matrices with the six index columns.
- `features/y_train.npy` and `features/y_test.npy`: label vectors aligned with the feature matrices.
- `models/tfidf_vectorizer.pkl` and `models/robust_scaler.pkl`: serialised transformers.
- A feature-statistics summary printed (mean, std, min, max per feature, split by bot/human) to give a first sanity check on discriminativity.

**Todo List:**

**4a — Shannon Entropy (`entropy_shannon`):**
1. For each user's message list, concatenate into one string and compute character 2-gram frequency distribution.
2. Apply the Shannon entropy formula: H = -sum(p * log2(p)) over the k-gram probability distribution.
3. Users with a single message or empty text receive `entropy_shannon = 0` (log warning).

**4b — CV of Inter-Arrival Times (`cv_iat`):**
1. Compute `delta_t` as differences between consecutive timestamps in seconds.
2. Compute `cv_iat = std(delta_t) / mean(delta_t)`. Users with fewer than 2 timestamps get `cv_iat = 0`.
3. Clip extreme outliers at 99th percentile to reduce scaler sensitivity.

**4c — Mean Self-Cosine Similarity (`self_cosine_sim`):**
1. Fit a `TfidfVectorizer` (max_features=5000, sublinear_tf=True) on all train messages (flattened list).
2. For each user, transform their messages into TF-IDF vectors and compute the mean pairwise cosine similarity using `sklearn.metrics.pairwise.cosine_similarity`.
3. Users with a single message receive `self_cosine_sim = 1.0`.
4. Serialise the fitted vectoriser to `models/tfidf_vectorizer.pkl`; apply (transform only) to test users.

**4d — MATTR (Moving Average Type-Token Ratio) (`mattr_50`):**
1. Concatenate all words from a user's messages into a flat list.
2. Slide a window of W=50 words; compute TTR = unique_words / 50 for each window.
3. `mattr_50` = mean of all window TTRs. Users with fewer than 50 words total receive raw TTR.

**4e — Circular Time Entropy (`circular_time_entropy`):**
1. Extract the hour-of-day (0–23) for each timestamp.
2. Bin into 24 buckets; compute a probability distribution.
3. Apply Shannon entropy formula; divide by log2(24) to normalise to [0,1].

**4f — Interaction Ratios (`url_ratio`, `mention_ratio`):**
1. `url_ratio = url_count / total_messages`.
2. `mention_ratio = mention_count / total_messages`.
3. Clamp to [0, 1].

**4g — Scaling:**
1. Stack the six features into a DataFrame.
2. Fit `RobustScaler` on `X_train` only; serialise to `models/robust_scaler.pkl`.
3. Transform both `X_train` and `X_test`.
4. Save scaled feature matrices and label vectors.

**Relevant Context:**
- The TF-IDF vectoriser must be fit on train messages only. Test messages are transformed using the vocabulary learned from train.
- `RobustScaler` is preferred over `StandardScaler` because several indices (especially `cv_iat`) have heavy-tailed distributions in bot populations.
- `self_cosine_sim` computation is O(N²) per user — consider using a random subsample of 50 messages per user if total corpus is large.

---

### Sub-Task 5 — Model Training (`04_train.py`)

**Status:** `[ ] pending`

**Intent:**
Train all seven models on `X_train`. Supervised models use `y_train`; unsupervised models (GMM, Isolation Forest, HDBSCAN) receive only `X_train` without labels, simulating pure anomaly detection. Each trained model is serialised to `models/`.

**Expected Outcomes:**
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/xgboost.pkl`
- `models/knn.pkl`
- `models/gmm.pkl`
- `models/isolation_forest.pkl`
- `models/hdbscan.pkl` (HDBSCAN does not have a standard sklearn predict; save the fitted object and GLOSH scores separately)
- Console log of training completion per model.

**Todo List:**

**5a — Logistic Regression:**
- `LogisticRegression(C=1.0, solver='lbfgs', class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)`
- Fit on `(X_train, y_train)`. Serialise.

**5b — Random Forest:**
- `RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced_subsample', random_state=RANDOM_STATE)`
- Fit on `(X_train, y_train)`. Extract and log feature importances. Serialise.

**5c — XGBoost:**
- `XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)`
- Compute `scale_pos_weight = count(y==0) / count(y==1)` to handle class imbalance.
- Fit on `(X_train, y_train)`. Serialise.

**5d — k-NN+:**
- `KNeighborsClassifier(n_neighbors=15, weights='distance', metric='euclidean')`
- Fit on `(X_train, y_train)`. Serialise.

**5e — GMM:**
- `GaussianMixture(n_components=2, covariance_type='full', random_state=RANDOM_STATE)`
- Fit on `X_train` only (no labels). Select `n_components` via BIC on a grid [2, 3, 4] if computation allows.
- The component associated with bots will be identified in evaluation (Sub-Task 6) by checking which component has higher mean posterior probability for known train bots.
- Serialise.

**5f — Isolation Forest:**
- `IsolationForest(n_estimators=200, contamination='auto', random_state=RANDOM_STATE)`
- Fit on `X_train` only (no labels). Serialise.

**5g — HDBSCAN:**
- `hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)`
- Fit on `X_train` only (no labels).
- Extract `clusterer.outlier_scores_` (GLOSH scores) for train set diagnostics.
- Serialise the fitted object; GLOSH scores on test will be produced in Sub-Task 6 via `hdbscan.approximate_predict`.

**Relevant Context:**
- Unsupervised models (GMM, Isolation Forest, HDBSCAN) intentionally never see `y_train` during fitting.
- For HDBSCAN, `prediction_data=True` must be set at fit time to enable `approximate_predict` on the test set.
- XGBoost: if the installed version requires `enable_categorical` flag, add it.

---

### Sub-Task 6 — Score Standardisation and ROC Benchmark (`05_evaluate.py`)

**Status:** `[ ] pending`

**Intent:**
Generate a single continuous bot-probability score S ∈ [0,1] per model per test user, then produce unified ROC and Precision-Recall curves. This is the comparative output that answers which approach best captures the variance-invariance signal described in the domain literature.

**Expected Outcomes:**
- `output/scores.parquet`: table with columns `[user_id, y_true, score_logreg, score_rf, score_xgb, score_knn, score_gmm, score_isoforest, score_hdbscan]`.
- `output/roc_curves.png`: single figure with all 7 ROC curves + AUC legend, diagonal baseline.
- `output/pr_curves.png`: single figure with all 7 PR curves + AP legend.
- `output/model_comparison.csv`: table with columns `[Model, ROC-AUC, PR-AUC, F1@Youden, Threshold@Youden, Precision@Youden, Recall@Youden]`.
- `output/confusion_matrices.png`: grid of 7 confusion matrices at the Youden-optimal threshold.
- `output/feature_importances.png`: bar chart from Random Forest and XGBoost feature importances.
- Console printout of the full comparison table.

**Todo List:**

**6a — Supervised score extraction:**
1. For Logistic Regression, Random Forest, XGBoost, k-NN: call `.predict_proba(X_test)[:, 1]` → `score ∈ [0,1]`.

**6b — Isolation Forest score normalisation:**
1. Call `model.score_samples(X_test)` → raw scores (more negative = more anomalous).
2. Negate and min-max normalise to [0,1]: `score = (s_neg - min) / (max - min)`.

**6c — GMM score extraction:**
1. Call `model.predict_proba(X_test)` → posterior probabilities per component.
2. Identify the bot-associated component using train-set posteriors: the component where known bots have higher posterior probability is the bot component.
3. Use that component's posterior column as `score`.

**6d — HDBSCAN score extraction:**
1. Use `hdbscan.approximate_predict(clusterer, X_test)` to obtain soft cluster memberships.
2. Use `clusterer.outlier_scores_` pattern; for test points use GLOSH approximation or the membership strength of the noise cluster.
3. Min-max normalise to [0,1].

**6e — ROC curves:**
1. For each model, compute `fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, score)`.
2. Compute `roc_auc = sklearn.metrics.auc(fpr, tpr)`.
3. Plot all curves on one figure with distinct colors and AUC in legend.

**6f — PR curves:**
1. For each model, compute `precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, score)`.
2. Compute `pr_auc = sklearn.metrics.average_precision_score(y_test, score)`.
3. Plot all curves on one figure.

**6g — Youden's J threshold and per-model metrics:**
1. For each model, compute `J = TPR - FPR` across all thresholds from the ROC curve.
2. Select `threshold* = argmax(J)`.
3. Compute `y_pred = (score >= threshold*).astype(int)`.
4. Compute F1, Precision, Recall at `threshold*`.
5. Populate `model_comparison.csv`.

**6h — Confusion matrices and feature importances:**
1. Plot a 2×4 subplot grid of confusion matrices (one per model, unnormalised).
2. Plot side-by-side horizontal bar charts for RF and XGBoost feature importances.

**Relevant Context:**
- For HDBSCAN test-set GLOSH scores: `hdbscan` does not natively provide GLOSH for new points. The recommended workaround is to fit HDBSCAN on `X_train` and use `approximate_predict` for soft cluster membership, treating low membership strength as high anomaly score. Document this limitation in output.
- All plots use `matplotlib`; use consistent color palette and figure size (12×8 at 150 dpi) for report-quality output.

---

## Dependency Graph

```
Sub-Task 1 (scaffold)
    └─► Sub-Task 2 (ingest) — produces data/users_raw.parquet
            └─► Sub-Task 3 (split) — produces train/test splits
                    └─► Sub-Task 4 (features) — produces X_train, X_test
                            └─► Sub-Task 5 (train) — produces serialised models
                                    └─► Sub-Task 6 (evaluate) — produces all output artefacts
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Aggregate to user level before split | Prevents message-level leakage; matches Cresci et al. methodology |
| Fit TF-IDF and RobustScaler on train only | Anti-leakage: test set must be invisible during transformer fitting |
| Youden's J for threshold selection | Balances TPR and FPR without assuming a cost function |
| GLOSH via approximate_predict for HDBSCAN | HDBSCAN does not support direct test-set GLOSH; soft membership is the closest approximation |
| `contamination='auto'` on Isolation Forest | Avoids assuming a fixed bot prevalence which may not match the real dataset distribution |
| `class_weight='balanced'` for supervised models | Compensates for class imbalance present in MIB without oversampling (SMOTE) which would require fitting on train only and complicates the pipeline |

---

## Files Produced (Summary)

| Phase | Output Files |
|---|---|
| Sub-Task 1 | `src/config.py`, `requirements.txt`, `README.md` |
| Sub-Task 2 | `data/users_raw.parquet` |
| Sub-Task 3 | `data/train_users.parquet`, `data/test_users.parquet` |
| Sub-Task 4 | `features/X_train.parquet`, `features/X_test.parquet`, `features/y_train.npy`, `features/y_test.npy`, `models/tfidf_vectorizer.pkl`, `models/robust_scaler.pkl` |
| Sub-Task 5 | `models/logistic_regression.pkl`, `models/random_forest.pkl`, `models/xgboost.pkl`, `models/knn.pkl`, `models/gmm.pkl`, `models/isolation_forest.pkl`, `models/hdbscan.pkl` |
| Sub-Task 6 | `output/scores.parquet`, `output/roc_curves.png`, `output/pr_curves.png`, `output/model_comparison.csv`, `output/confusion_matrices.png`, `output/feature_importances.png` |
