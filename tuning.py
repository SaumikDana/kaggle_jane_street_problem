from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from methods import sample_training_data
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


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


def tune_lightgbm(X, y):
    """
    Tune LightGBM hyperparameters using a focused parameter grid
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'num_leaves': [31, 63],  # Specific to LightGBM
    }
    
    # Calculate total combinations
    total_combinations = 2 * 2 * 2 * 2 * 2  # = 32 combinations
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total fits with 5-fold CV: {total_combinations * 5}")
    
    # Initialize LightGBM model
    lgbm = LGBMRegressor(random_state=42)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=lgbm,
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


def tune_catboost(X, y):
    """
    Tune CatBoost hyperparameters using a focused parameter grid
    """
    param_grid = {
        'iterations': [100, 200],
        'depth': [4, 6],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
    }
    
    # Initialize CatBoost model
    cb = CatBoostRegressor(random_state=42, silent=True)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=cb,
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
