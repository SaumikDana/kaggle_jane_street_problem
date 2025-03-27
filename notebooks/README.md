## 📓 Notebooks Overview

### `interrogate_models.ipynb`

**Goal:** Train and evaluate models on a single symbol using PCA-reduced features.

**Workflow:**

- **Imports and Setup:**
  - Brings in utility functions from `src/` (like data engineering, PCA, model training).
  - Sets up paths to access data.

- **Load Parquet Data:**
  - Reads a small sample (`n_partitions = 1`) from the training set.

- **Loop Over Symbol(s):**
  - For each `symbol_id`, it:
    - Combines data across partitions.
    - Extracts features, responders, and target via `create_timeseries_for_symbol`.
    - Cleans NaNs using `clean_data`.
    - Prepares regression data with `prepare_regression_data`.
    - Reduces dimensions with PCA (default: 25 components).
    - Trains and compares multiple models (`train_and_evaluate_multiple_models`).

---

### `interrogate_test_data.ipynb`

**Goal:** Explore the test data (`test.parquet` and `lags.parquet`).

**Workflow:**

- **Load Files:**
  - Reads `test.parquet` and `lags.parquet` from the data directory.

- **Exploratory Checks:**
  - Uses `.info()` to get schema.
  - Checks for missing values in both datasets.

> 📌 *No modeling or transformations here—just inspection.*

---

### `submission.ipynb`

**Goal:** Train final model and make predictions for submission.

**Workflow:**

- **Imports and Setup:**
  - Loads standard packages plus XGBoost, PCA, and preprocessing.
  - Reads all required training/test data.

- **Train Across All Symbols:**
  - Loops over each `symbol_id`, and for each:
    - Prepares features, responders, and targets.
    - Cleans NaNs.
    - Applies PCA (25 components).
    - Samples training data.
    - Stores reduced features and targets for later concatenation.

- **Final Model Training:**
  - Concatenates all sampled features/targets across symbols.
  - Trains a single global XGBoost model.

- **Prediction Loop:**
  - For each symbol in the test data:
    - Prepares test features using the same columns and PCA.
    - Generates prediction for `responder_6`.
    - Appends to a final prediction DataFrame.

✅ **Returns** `row_id` and predicted `responder_6` for submission.

---

### `tune_best_model.ipynb`

**Goal:** Hyperparameter tuning using `GridSearchCV` on reduced features.

**Workflow:**

- **Setup:**
  - Imports model evaluation, PCA, and tuning functions from `src/`.

- **Data Load and Preparation:**
  - Loads partitioned data.
  - Combines for a specific symbol.
  - Cleans and processes data like before.

- **Dimensionality Reduction:**
  - Uses PCA to reduce feature size.

- **Model Tuning:**
  - Uses `tune_xgboost()` from `tuning.py` to grid search XGBoost hyperparameters.

- **Model Evaluation:**
  - Trains best model on train split.
  - Evaluates on test split using R² and MSE.

---
