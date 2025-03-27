
![pipeline](https://github.com/user-attachments/assets/2b4a6fbd-16c0-46f3-b0ac-8e484441a261)


## 📓 `submission.ipynb` – Final Prediction Pipeline

**Goal:** Train a global model across all symbols and generate predictions for the `responder_6` target using test data.

---

### 🧩 1. Imports and Setup

- Loads essential packages:
  - `pandas`, `polars` for data handling
  - `sklearn` for PCA and preprocessing
  - `xgboost` for modeling
  - `os`, `numpy` for utilities

- Sets up `script_directory` and `data_directory` to point to the root `data/` folder.

---

### 📥 2. Load Data

- Reads:
  - `test.parquet` → test features
  - `lags.parquet` → lagged responders for test data
  - Training partitions (`train_0.parquet` assumed, can scale to 10)

---

### 🧼 3. Utility Functions

#### `clean_data(features, responders)`
- Drops columns with NaNs in features/responders and returns cleaned versions.

#### `create_timeseries_for_symbol(df, symbol_id)`
- Builds a time-aligned set of:
  - Features (excluding the first date)
  - Responders (lags aligned)
  - Target (`responder_6`)
- Used to extract symbol-specific time series data.

#### `prepare_regression_data(features, responders, target=None)`
- Merges features + responders as `X`, and optionally includes `target` as `y`.

#### `sample_training_data(...)`
- Uses binning to evenly sample training data from `X_train` and `y_train`.
- Falls back to random sampling if binning fails.

#### `reduce_dimensions_pca(X, n_components=25)`
- Applies standard scaling and PCA.
- Returns transformed features, PCA object, and scaler.

#### `prepare_prediction_data(features_df, lags_df)`
- Combines clean test features with lagged responders for model input.

---

### 🏋️‍♂️ 4. Train on All Symbols

For each `symbol_id`:
- **Data Aggregation:**
  - Combines all partitioned rows matching the symbol.
- **Preprocessing:**
  - Extracts features, responders, and target
  - Cleans NaNs
  - Combines into training data
- **Dimensionality Reduction:**
  - Applies PCA with 25 components
- **Sampling:**
  - Uses `sample_training_data` to reduce volume
- **Storage:**
  - Collects reduced feature matrices and targets in lists (`all_X_reduced`, `all_y`)
  - Stores PCA, scaler, and feature columns used per symbol in dictionaries for reuse

---

### 🤖 5. Final Model Training

- Concatenates all sampled data across symbols
- Splits into train/test sets
- Trains a single `XGBRegressor` on all combined training data
- Evaluation is minimal here—the model is assumed ready for inference after training

---

### 📈 6. Prediction Logic – `predict()` Function

- Loops through each `symbol_id` present in the test data
- For each symbol:
  - Extracts test rows and matching lagged responders
  - Prepares model input using the same cleaned columns
  - Applies saved `scaler` and `pca` for that symbol
  - Makes prediction using the trained global model
  - Appends predicted `responder_6` value with corresponding `row_id` to output DataFrame

---

### ✅ 7. Final Output

- The predictions DataFrame is sorted by `row_id`
- Ensures structure:
  - Columns: `['row_id', 'responder_6']`
  - Format: `pandas` or `polars` DataFrame
- Returned by the notebook (can be exported to CSV)

---
