import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from data_engineering import prepare_prediction_data, sample_training_data
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import math


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


def plot_performance(y_test, y_test_pred):
    """
    Plot performance
    """
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
    
    return


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

    return model


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
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=50,           # Fewer trees
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,             # Maximum tree leaves
            min_child_samples=20,      # Minimum samples per leaf
            n_jobs=-1,                 # Use all CPU cores
            random_state=random_state,
            verbose=-1                 # Suppress verbose output
        ),
        'CatBoost': CatBoostRegressor(
            iterations=50,             # Number of trees
            learning_rate=0.1,
            depth=6,                   # Tree depth
            min_data_in_leaf=20,       # Minimum samples per leaf
            thread_count=-1,           # Use all CPU cores
            random_seed=random_state,
            verbose=False              # Suppress verbose output
        )
    }

    for name, model in tree_models.items():
        print(f"\nTraining {name} with sampled data...")
        try:
            # Special handling for CatBoost to suppress additional output
            if name == 'CatBoost':
                model.fit(X_train_sampled, y_train_sampled, verbose=False)
            else:
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
        # Calculate required grid dimensions
        n_models = len(results)
        n_cols = 3  # Keep 3 columns
        n_rows = math.ceil(n_models / n_cols)
        
        plt.figure(figsize=(15, 5*n_rows))
        
        for i, (name, result) in enumerate(results.items(), 1):
            plt.subplot(n_rows, n_cols, i)
            plt.scatter(result['test_actual'], result['test_pred'], alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'{name}\nTest R²: {result["metrics"]["test_r2"]:.4f}')
        
        plt.tight_layout()
        plt.show()

    return results


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
