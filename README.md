
![problem](https://github.com/user-attachments/assets/8a5d2d12-9441-4cf7-8bf8-4ed3875c4acb)

## 📓 `submission.ipynb` – Final Prediction Pipeline

**Goal:** Train **a separate model for each symbol** and generate predictions for the `responder_6` target using test data.

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

### 🧠 4. Train Per-Symbol Models

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
- **Model Training:**
  - Trains a **dedicated `XGBRegressor`** for that symbol using its own data
- **Storage:**
  - Stores PCA, scaler, model, and selected feature columns per symbol for reuse during prediction

---

### 📈 5. Prediction Logic – `predict()` Function

- Loops through each `symbol_id` in the test data
- For each symbol:
  - Extracts test rows and matching lagged responders
  - Prepares model input using cleaned features and responders
  - Applies the corresponding `scaler` and `pca`
  - Uses the **symbol-specific model** to predict `responder_6`
  - Appends the prediction and `row_id` to the output DataFrame

---

### ✅ 6. Final Output

- The predictions DataFrame is sorted by `row_id`
- Ensures structure:
  - Columns: `['row_id', 'responder_6']`
  - Format: `pandas` or `polars` DataFrame
- Can be exported to CSV for submission

---

