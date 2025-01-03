import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def clean_data(features, responders):

    # Print columns before cleaning
    print(f"\nTotal feature columns before cleaning: {len(features.columns)}")
    print(f"Total responder columns before cleaning: {len(responders.columns)}")
    
    # Get clean features (no NaN)
    features_with_nan = features.columns[features.isna().any()].tolist()
    clean_features = features.loc[:, ~features.isna().any()].reset_index(drop=True)
    
    # Get clean responders (no NaN)
    responders_with_nan = responders.columns[responders.isna().any()].tolist()
    clean_responders = responders.loc[:, ~responders.isna().any()].reset_index(drop=True)
        
    # Print dropped columns
    if features_with_nan:
        print("\nFeature columns dropped due to NaN values:")
        for col in features_with_nan:
            print(f"- {col}")
    
    if responders_with_nan:
        print("\nResponder columns dropped due to NaN values:")
        for col in responders_with_nan:
            print(f"- {col}")
    
    print(f"\nNumber of clean features: {len(clean_features.columns)}")
    print(f"Number of clean responders: {len(clean_responders.columns)}")
    
    return clean_features, clean_responders

def create_timeseries_for_symbol(df, symbol_id):
    """
    Create feature and responder time series for a given symbol
    Args:
        df: Input dataframe
        symbol_id: Symbol to process
    Returns:
        tuple: (feature_series, responder_series, target_series)
    """
    # Filter for our symbol, then sort by date_id and time_id 
    df_for_symbol = df[df['symbol_id'] == symbol_id].copy()
    symbol_data = df_for_symbol.sort_values(['date_id', 'time_id'])
    
    # Get column names
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    responder_cols = [col for col in df.columns if col.startswith('responder_') and col != 'responder_6']
    target_col = 'responder_6'
    
    # Get first date and its last time for responders
    first_date = symbol_data['date_id'].min()
    first_date_last_time = symbol_data[symbol_data['date_id'] == first_date]['time_id'].max()
    first_date_last_responders = symbol_data[
        (symbol_data['date_id'] == first_date) &
        (symbol_data['time_id'] == first_date_last_time)
    ][responder_cols]
    
    # Get all data after first date (for features)
    feature_series = symbol_data[symbol_data['date_id'] > first_date][feature_cols].copy()
    
    # Get all data after first date (for target)
    target_series = symbol_data[symbol_data['date_id'] > first_date][target_col].copy()
    target_series = target_series.reset_index(drop=True)
    
    # Get all data after first date except the last row (for responders)
    responder_data = symbol_data[symbol_data['date_id'] > first_date][responder_cols].iloc[:-1]
    
    # Add first date's last responders at the start
    responder_series = pd.concat([first_date_last_responders, responder_data])
    
    # Print verification
    print(f"\nFeature series shape: {feature_series.shape}")
    print(f"\nResponder series shape: {responder_series.shape}")
    print(f"\nTarget series shape: {target_series.shape}")
    
    # clean_features, clean_responders = clean_data(feature_series, responder_series)
    clean_features, clean_responders = feature_series, responder_series

    return clean_features, clean_responders, target_series

def plot_separate_timeseries(features, responders, target):
    """
    Create separate subplots for first 5 features and all 8 responders
    """
    plt.figure(figsize=(20, 5))
    clean_features_plotted = 0
    all_features = features.columns
    
    # Keep plotting until we get 5 clean features
    for feature_name in all_features:
        # Check if this feature has any NaN
        if not features[feature_name].isna().any():
            ax = plt.subplot(1, 5, clean_features_plotted + 1)
            timeseries = features[feature_name]
            ax.plot(timeseries, color='blue', alpha=0.7)
            ax.set_title(feature_name)
            ax.grid(True)
            if clean_features_plotted == 0:
                ax.set_ylabel('Value')
            ax.set_xlabel('Time Steps')
            
            clean_features_plotted += 1
            if clean_features_plotted == 5:  # Stop after 5 clean features
                break
    
    plt.suptitle('First 5 Non-NaN Features', y=1.05)
    plt.tight_layout()
    plt.show()

    # Second plot - Responders
    plt.figure(figsize=(20, 8))
    for i, col in enumerate(responders.columns):
        ax = plt.subplot(2, 4, i+1)  # 2 rows, 4 columns
        ax.plot(responders[col], color='red', alpha=0.7)
        ax.set_title(f'Responder {col[-1]}')
        ax.grid(True)
        if i % 4 == 0:  # Add y-label for leftmost plots
            ax.set_ylabel('Value')
        ax.set_xlabel('Time Steps')
    plt.suptitle('All 8 Responders', y=1.05)
    plt.tight_layout()
    plt.show()

    # Third plot - Target
    plt.figure(figsize=(10, 6))
    plt.plot(target, color='green', alpha=0.7)
    plt.title('Target (responder_6)')
    plt.tight_layout()
    plt.show()

def prepare_regression_data(features, responders, target):
    common_indices = features.index.intersection(responders.index).intersection(target.index)
    X = pd.concat([features.loc[common_indices], responders.loc[common_indices]], axis=1)
    y = target.loc[common_indices]
    print(f"\nRegression data shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y

def train_model(X, y):
    """
    Train a linear regression model
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train/Test split sizes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate its performance
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nModel Performance:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Plot actual vs predicted for test set
    plt.figure(figsize=(10, 5))
    
    # First subplot: Actual vs Predicted scatter
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted (Test Set)')
    
    # Second subplot: Residuals
    plt.subplot(1, 2, 2)
    residuals = y_test - y_test_pred
    plt.hist(residuals, bins=50)
    plt.xlabel('Residual')
    plt.ylabel('Count')
    plt.title('Residuals Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return model

def prepare_regression_data_responders_only(features, responders, target):
    """
    Prepare X and y for regression using only responders
    """
    # Reset indices
    responders = responders.reset_index(drop=True)
    target = target.reset_index(drop=True)
    
    # Make sure all have same length
    min_len = min(len(responders), len(target))
    responders = responders.iloc[:min_len]
    target = target.iloc[:min_len]
    
    # X is just responders
    X = responders
    y = target
    
    print("\nRegression data shapes:")
    print(f"X shape: {X.shape} (samples, responders)")
    print(f"y shape: {y.shape}")
    print("\nResponder columns:")
    print("Responders:", X.columns.tolist())
    
    return X, y

def train_xgboost_model(X, y):
    """
    Train an XGBoost regression model with basic hyperparameters
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train/Test split sizes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Initialize and train model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    # Train the model with eval set for validation
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    return model, X_train, X_test, y_train, y_test

def sample_training_data(X_train, y_train, sample_fraction=1/10, n_bins=10):

    # Handle RF and XGB with binning
    sample_size = int(len(X_train) * sample_fraction)

    # Initialize sampling variables
    sampling_succeeded = False
    X_train_sampled = None
    y_train_sampled = None

    # Try different binning approaches
    y_train_df = pd.DataFrame(y_train, columns=['target'])

    try:
        # Try quantile-based bins first
        y_train_df['bins'] = pd.qcut(y_train_df['target'], q=n_bins, labels=False, duplicates='drop')
        sampling_succeeded = True
    except ValueError:
        try:
            # If quantile binning fails, try equal-width bins
            y_train_df['bins'] = pd.cut(y_train_df['target'], bins=n_bins, labels=False)
            sampling_succeeded = True
        except ValueError:
            print("Warning: Binning methods failed. Falling back to random sampling.")
            sampling_succeeded = False

    if sampling_succeeded:
        # Proceed with bin-based sampling
        bin_sample_size = max(1, sample_size // n_bins)  # Ensure at least 1 sample per bin
        sampled_indices = []
        
        # Get unique bins that actually exist in the data
        existing_bins = y_train_df['bins'].dropna().unique()
        
        for bin_idx in existing_bins:
            bin_indices = y_train_df[y_train_df['bins'] == bin_idx].index.values  # Get numpy array
            if len(bin_indices) > 0:
                n_samples = min(len(bin_indices), bin_sample_size)
                sampled_bin_indices = np.random.choice(bin_indices, n_samples, replace=False)
                sampled_indices.extend(sampled_bin_indices)
        
        # Convert to numpy array for indexing
        sampled_indices = np.array(sampled_indices)
        
        # Verify we got some samples
        if len(sampled_indices) > 0:
            X_train_sampled = X_train[sampled_indices]
            y_train_sampled = y_train[sampled_indices]
        else:
            sampling_succeeded = False

    # If all sampling methods failed or got no samples, fall back to random sampling
    if not sampling_succeeded or X_train_sampled is None or len(X_train_sampled) == 0:
        print("Falling back to random sampling...")
        sampled_indices = np.random.choice(
            len(X_train), 
            size=min(sample_size, len(X_train)), 
            replace=False
        )
        X_train_sampled = X_train[sampled_indices]
        y_train_sampled = y_train[sampled_indices]

    # Verify final samples
    if len(X_train_sampled) == 0 or len(y_train_sampled) == 0:
        raise ValueError("Failed to create valid samples for tree-based models")

    print("\nSampled data shapes:")
    print(f"X shape: {X_train_sampled.shape} (samples, features+responders)")
    print(f"y shape: {y_train_sampled.shape}")

    return X_train_sampled, y_train_sampled

def train_and_evaluate_multiple_models(X, y, n_bins=10, sample_fraction=1/10, random_state=42, show_plots=True):
   """
   Train and evaluate multiple regression models on the same data
   
   Parameters:
       X: Features matrix
       y: Target variable
       n_bins: Number of bins for sampling tree-based models (default: 10)
       sample_fraction: Fraction of data to use for tree-based models (default: 1/3)
       random_state: Random seed for reproducibility (default: 42)
       show_plots: Whether to show comparison plots (default: True)
   
   Returns:
       dict: Dictionary containing for each model:
           - model: Trained model object
           - test_pred: Predictions on test set
           - test_actual: Actual test values 
           - metrics: Dictionary of metrics (train/test MSE and R²)
   """
   # Convert inputs to numpy arrays if they're pandas DataFrames
   if isinstance(X, pd.DataFrame):
       X = X.to_numpy()
   if isinstance(y, pd.Series):
       y = y.to_numpy()

   # Input validation
   if len(X) < n_bins:
       raise ValueError("Number of samples is less than number of bins")
   
   # Regular train-test split for linear models
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

   # Define models
   models = {
       'Linear Regression': LinearRegression(),
       'Ridge': Ridge(alpha=1.0),
       'Lasso': Lasso(alpha=1.0)
   }

   # Train and evaluate linear models first
   results = {}
   for name, model in models.items():
       print(f"\nTraining {name}...")
       try:
           model.fit(X_train, y_train)
           y_train_pred = model.predict(X_train)
           y_test_pred = model.predict(X_test)
           
           # Calculate metrics
           train_mse = mean_squared_error(y_train, y_train_pred)
           test_mse = mean_squared_error(y_test, y_test_pred)
           train_r2 = r2_score(y_train, y_train_pred)
           test_r2 = r2_score(y_test, y_test_pred)
           
           print(f"{name} Performance:")
           print(f"Train MSE: {train_mse:.4f}")
           print(f"Test MSE: {test_mse:.4f}")
           print(f"Train R²: {train_r2:.4f}")
           print(f"Test R²: {test_r2:.4f}")
           
           results[name] = {
               'model': model,
               'test_pred': y_test_pred,
               'test_actual': y_test,
               'metrics': {
                   'train_mse': train_mse,
                   'test_mse': test_mse,
                   'train_r2': train_r2,
                   'test_r2': test_r2
               }
           }
       except Exception as e:
           print(f"Error training {name}: {str(e)}")
           continue

   X_train_sampled, y_train_sampled = sample_training_data(X_train, y_train, sample_fraction=sample_fraction, n_bins=n_bins)

   tree_models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=50,           # Fewer trees
            max_depth=10,              # Limit tree depth
            min_samples_leaf=20,       # Require more samples per leaf
            n_jobs=-1,                 # Use all CPU cores
            random_state=random_state
        ),
        'XGBoost': XGBRegressor(
            n_estimators=50,           # Fewer trees
            learning_rate=0.1,
            max_depth=6,
            n_jobs=-1,                 # Use all CPU cores
            tree_method='hist',        # Faster histogram-based algorithm
            random_state=random_state
        )
   }

   for name, model in tree_models.items():
       print(f"\nTraining {name} with sampled data...")
       try:
           model.fit(X_train_sampled, y_train_sampled)
           
           # Predict on full datasets
           y_train_pred = model.predict(X_train)
           y_test_pred = model.predict(X_test)
           
           # Calculate metrics
           train_mse = mean_squared_error(y_train, y_train_pred)
           test_mse = mean_squared_error(y_test, y_test_pred)
           train_r2 = r2_score(y_train, y_train_pred)
           test_r2 = r2_score(y_test, y_test_pred)
           
           print(f"{name} Performance:")
           print(f"Train MSE: {train_mse:.4f}")
           print(f"Test MSE: {test_mse:.4f}")
           print(f"Train R²: {train_r2:.4f}")
           print(f"Test R²: {test_r2:.4f}")
           
           results[name] = {
               'model': model,
               'test_pred': y_test_pred,
               'test_actual': y_test,
               'metrics': {
                   'train_mse': train_mse,
                   'test_mse': test_mse,
                   'train_r2': train_r2,
                   'test_r2': test_r2
               }
           }
       except Exception as e:
           print(f"Error training {name}: {str(e)}")
           continue

   # Plot comparison of actual vs predicted for all models
   if show_plots and results:
       plt.figure(figsize=(15, 10))
       for i, (name, result) in enumerate(results.items(), 1):
           plt.subplot(2, 3, i)
           plt.scatter(result['test_actual'], result['test_pred'], alpha=0.5)
           plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
           plt.xlabel('Actual')
           plt.ylabel('Predicted')
           plt.title(f'{name}\nTest R²: {result["metrics"]["test_r2"]:.4f}')
       
       plt.tight_layout()
       plt.show()
   
   return results

def tune_xgboost(X, y):
    """
    Tune XGBoost hyperparameters using a focused parameter grid
    """
    # Define smaller parameter grid
    param_grid = {
        'n_estimators': [100, 200],          # removed 300
        'max_depth': [4, 6],                 # just 2 values
        'learning_rate': [0.01, 0.1],        # removed middle value
        'subsample': [0.8, 1.0],             # removed 0.9
        'min_child_weight': [1, 3]           # removed 5
    }
    
    # Calculate total combinations
    total_combinations = 2 * 2 * 2 * 2 * 2  # = 32 combinations
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total fits with 5-fold CV: {total_combinations * 5}")
    
    # Initialize XGBoost model
    xgb = XGBRegressor(random_state=42)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    X_sampled, y_sampled = sample_training_data(X, y)

    # Fit GridSearchCV
    grid_search.fit(X_sampled, y_sampled)
    
    # Print results
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print(f"\nBest cross-validation R²: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_best_xgboost_model(X, y):
    """
    Train an XGBoost regression model with basic hyperparameters
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train/Test split sizes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Initialize and train model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=3,     # Higher value provides better regularization
        subsample=0.8,         # More common than 1.0 and provides regularization
        random_state=42
    )
    
    X_train_sampled, y_train_sampled = sample_training_data(X_train, y_train)

    # Train the model with eval set for validation
    model.fit(
        X_train_sampled, y_train_sampled,
        eval_set=[(X_train_sampled, y_train_sampled), (X_test, y_test)],
        verbose=False
    )
    
    return model, X_train, X_test, y_train, y_test

def evaluate_best_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate its performance
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nModel Performance:")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
        
    return model

def prepare_prediction_data(features_df, lags_df):
    """
    Prepare data for prediction by:
    1. Getting clean features (no NaN)
    2. Combining with lagged responders (excluding responder_6)
    """
    # Get clean features
    clean_features = features_df.loc[:, ~features_df.isna().any()]
    feature_cols = [col for col in clean_features.columns if col.startswith('feature_')]
    clean_features = clean_features[feature_cols]
    
    # Get lagged responders (excluding responder_6)
    lag_cols = [col for col in lags_df.columns if col.startswith('responder_') and not col.startswith('responder_6')]
    responder_lags = lags_df[lag_cols]
    
    # Combine features and responders
    X = pd.concat([clean_features, responder_lags], axis=1)
    
    print("\nPrediction data preparation:")
    print(f"Number of clean features: {len(feature_cols)}")
    print(f"Number of lagged responders: {len(lag_cols)}")
    print(f"Final X shape: {X.shape}")
    
    return X

def make_predictions(model, features_df, lags_df):
    """
    Use trained model to predict responder_6
    """
    # Prepare prediction data
    X = prepare_prediction_data(features_df, lags_df)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Create output DataFrame
    results = pd.DataFrame({
        'symbol_id': features_df['symbol_id'],
        'predicted_responder_6': predictions
    })
    
    if 'responder_6_lag_1' in lags_df.columns:
        results['actual_lag'] = lags_df['responder_6_lag_1']
        
    if 'weight' in features_df.columns:
        results['weight'] = features_df['weight']
    
    print("\nPrediction Results:")
    print(results)
    
    return results
