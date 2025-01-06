import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    
    return feature_series, responder_series, target_series


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


def prepare_regression_data(features, responders, target=None):
    if target is None:
        common_indices = features.index.intersection(responders.index)
    else:
        common_indices = features.index.intersection(responders.index).intersection(target.index)
    X = pd.concat([features.loc[common_indices], responders.loc[common_indices]], axis=1)
    if target is not None:
        y = target.loc[common_indices]
    if target is not None:
        print(f"\nRegression data shapes:")
    print(f"X shape: {X.shape}")
    if target is not None:
        print(f"y shape: {y.shape}")
    return X, y if target is not None else X


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
