## 🧠 Python Scripts (`src/`)

### `data_engineering.py`

**Purpose:** Data cleaning, time series preparation, regression dataset setup, and visualizations.

**Highlights:**
- Removes columns with NaNs.
- Creates time series per `symbol_id`.
- Merges features and responders for modeling.
- Samples training data intelligently using binning.
- Generates plots for features, responders, and target.

---

### `models.py`

**Purpose:** Model training, evaluation, and comparison across various algorithms.

**Models Used:**
- Linear Regression, Ridge, Lasso
- Random Forest, XGBoost, LightGBM, CatBoost

**Utilities:**
- Performance metrics (MSE, R²), visualizations
- Best model selection based on sampling and scoring

---

### `pca.py`

**Purpose:** Dimensionality reduction via PCA.

**Options:**
- Reduce using fixed number of components or variance threshold.

---

### `tuning.py`

**Purpose:** Hyperparameter tuning via `GridSearchCV`.

**Supported Models:**
- XGBoost, LightGBM, CatBoost

**Feature:**
- Uses sampled data for efficient tuning.

---

### `dump_to_csv.py`

**Purpose:** Extracts and saves per-symbol data from parquet files into CSVs.

**Scans:**
- All partitions (0–9) and saves each `symbol_id` data separately.

---

### `setup_path.py`

**Purpose:** Adds project root to `sys.path` for consistent module importing.
