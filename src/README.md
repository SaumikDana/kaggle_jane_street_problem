## 🔍 Python Script Details (`src/`)

---

### `data_engineering.py`

Functions:

- **`clean_data(features, responders)`**
  - Removes columns with any NaN values.
  - Returns cleaned features and responders.
  
- **`create_timeseries_for_symbol(df, symbol_id)`**
  - Filters data for a specific `symbol_id`.
  - Sorts chronologically and builds time series for features, responders, and target (`responder_6`).

- **`prepare_regression_data(features, responders, target=None)`**
  - Merges features and responders (and target if available) for regression modeling.

- **`sample_training_data(X_train, y_train, sample_fraction=1/10, n_bins=10)`**
  - Uses binning (quantile or equal-width) to sample balanced data across target value ranges.
  - Falls back to random sampling if needed.

- **`plot_separate_timeseries(features, responders, target)`**
  - Generates three plots: top 5 clean features, all responders, and the target time series.

- **`prepare_regression_data_responders_only(features, responders, target)`**
  - Creates regression input using only responders (ignores features).

- **`prepare_prediction_data(features_df, lags_df)`**
  - Prepares test input data by combining cleaned features with lagged responders.

---

### `models.py`

Functions:

- **`train_model(X, y)`**
  - Trains a linear regression model with a train/test split.

- **`plot_performance(y_test, y_test_pred)`**
  - Plots actual vs predicted values and residuals.

- **`evaluate_model(model, X_train, X_test, y_train, y_test)`**
  - Calculates MSE and R² for train and test sets.

- **`train_xgboost_model(X, y)`**
  - Trains an XGBoost model with basic hyperparameters.

- **`train_and_evaluate_multiple_models(...)`**
  - Compares 7 models: Linear, Ridge, Lasso, Random Forest, XGBoost, LightGBM, CatBoost.
  - Uses smart sampling for tree-based models.
  - Evaluates each model and plots results.

- **`train_best_xgboost_model(X, y)`**
  - Trains a tuned XGBoost model using sampled data.
  - Configured with stronger regularization.

- **`evaluate_best_model(model, X_train, X_test, y_train, y_test)`**
  - Same as `evaluate_model`, specifically for the best tuned model.

---

### `pca.py`

Function:

- **`reduce_dimensions_pca(X, n_components=None, variance_threshold=None)`**
  - Standardizes input features.
  - Performs PCA based on fixed component count or explained variance.
  - Returns transformed data, PCA model, and scaler.

---

### `tuning.py`

Functions:

- **`tune_xgboost(X, y)`**
  - GridSearchCV with 32 XGBoost parameter combinations.
  - Optimizes for R² score.

- **`tune_lightgbm(X, y)`**
  - Similar grid search setup for LightGBM.

- **`tune_catboost(X, y)`**
  - Similar grid search for CatBoost with fewer params.

> All tuning functions use a sample of the dataset via `sample_training_data()`.

---

### `dump_to_csv.py`

Main Logic (runs on script execution):

- Creates `symbol_data/` folder.
- Iterates over all `symbol_id`s found in `train.parquet/` partitions.
- For each symbol:
  - Aggregates data across partitions.
  - Saves as a CSV file (skips if already exists).

---

### `setup_path.py`

- Adds the repository root directory to `sys.path` to allow importing from `src/` in notebooks and scripts.

---
