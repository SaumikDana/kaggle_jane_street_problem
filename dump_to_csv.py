import pandas as pd
from methods import create_timeseries_for_symbol
import os

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = "symbol_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # First get all unique symbols across all partitions
    all_symbols = set()
    for partition in range(10):
        partition_path = f"train.parquet/partition_id={partition}/part-0.parquet"
        df_partition = pd.read_parquet(partition_path)
        all_symbols.update(df_partition['symbol_id'].unique())
    
    list_of_symbols = list(all_symbols)
    print(f"\nTotal number of symbols (financial instruments) is {len(list_of_symbols)}")
    
    # Process each symbol
    for symbol_id in list_of_symbols:
        # Check if CSV already exists
        csv_path = os.path.join(output_dir, f"symbol_{symbol_id}.csv")
        if os.path.exists(csv_path):
            print(f"\nSkipping symbol {symbol_id} - CSV already exists")
            continue
            
        print(f"\nProcessing symbol {symbol_id} ...")
        
        # Read and combine data for this symbol from all partitions
        symbol_data = []
        for partition in range(10):
            partition_path = f"train.parquet/partition_id={partition}/part-0.parquet"
            df_partition = pd.read_parquet(partition_path)
            symbol_partition = df_partition[df_partition['symbol_id'] == symbol_id]
            if not symbol_partition.empty:
                symbol_data.append(symbol_partition)
        
        if symbol_data:
            df_symbol = pd.concat(symbol_data, ignore_index=True)
            
            # Save to CSV
            df_symbol.to_csv(csv_path, index=False)
            print(f"Saved data for symbol {symbol_id} to {csv_path}")
            