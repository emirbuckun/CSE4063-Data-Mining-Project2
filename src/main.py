import pandas as pd
import numpy as np
from preprocessing import merge_datasets, preprocess_data
from models import create_models
from evaluate import evaluate_models
import time

def load_and_analyze_data(athlete_events_path='src/athlete_events.csv', 
                         noc_regions_path='src/noc_regions.csv',
                         sample_size=10000):  # Örneklem boyutu eklendi
    """
    Load and analyze the Olympic Games dataset
    """
    try:
        # Load datasets
        print("Loading datasets...")
        athlete_events = pd.read_csv(athlete_events_path)
        noc_regions = pd.read_csv(noc_regions_path)
        
        # Print full dataset information
        print("\nFull Dataset Information:")
        print("-" * 50)
        print(f"Total number of rows: {len(athlete_events)}")
        print(f"Total number of columns: {len(athlete_events.columns)}")
        
        # Take a sample for analysis
        if sample_size and sample_size < len(athlete_events):
            print(f"\nTaking a sample of {sample_size} rows for analysis...")
            athlete_events = athlete_events.sample(n=sample_size, random_state=42)
            print(f"Sample shape: {athlete_events.shape}")
        
        # Print dataset information for sample
        print("\nSample Dataset Information:")
        print("-" * 50)
        print("\nColumns and their data types:")
        print(athlete_events.dtypes)
        
        print("\nMissing values:")
        print(athlete_events.isnull().sum())
        
        print("\nSample of unique values in key columns:")
        for col in ['Sport', 'Event', 'Medal', 'NOC']:
            unique_vals = athlete_events[col].dropna().unique()
            print(f"\n{col} unique values: {len(unique_vals)}")
            print(f"Sample values: {unique_vals[:5]}")
            
        # Memory usage
        memory_usage = athlete_events.memory_usage(deep=True).sum() / 1024**2
        print(f"\nMemory usage: {memory_usage:.2f} MB")
        
        # Merge datasets
        print("\nMerging datasets...")
        start_time = time.time()
        merged_df = merge_datasets(athlete_events, noc_regions)
        merge_time = time.time() - start_time
        print(f"Merge completed in {merge_time:.2f} seconds")
        
        # Basic statistics of numeric columns
        numeric_cols = ['Age', 'Height', 'Weight']
        print("\nNumeric Columns Statistics:")
        print(merged_df[numeric_cols].describe())
        
        # Sport statistics
        print("\nTop 10 Sports by number of events:")
        print(merged_df['Sport'].value_counts().head(10))
        
        # Medal statistics
        print("\nMedal distribution:")
        print(merged_df['Medal'].value_counts(dropna=False))
        
        return merged_df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def main():
    try:
        print("Starting Olympic Games Data Mining Analysis...")
        
        # Sample size için kullanıcıdan input alabiliriz
        sample_size = 10000  # Örnek boyutu
        
        # Load and analyze data with sampling
        merged_df = load_and_analyze_data(sample_size=sample_size)
            
        print(f"\nPreprocessing data (sample size: {sample_size})...")
        start_time = time.time()
        preprocessed_df = preprocess_data(merged_df)
        preprocess_time = time.time() - start_time
        print(f"Preprocessing completed in {preprocess_time:.2f} seconds")
        
        # Create and evaluate models
        print("\nCreating and evaluating models...")
        start_time = time.time()
        models = create_models(preprocessed_df)
        model_time = time.time() - start_time
        print(f"Model creation completed in {model_time:.2f} seconds")
        
        # Evaluate models
        print("\nEvaluating models...")
        start_time = time.time()
        evaluate_models(models, preprocessed_df)
        evaluate_time = time.time() - start_time
        print(f"Evaluation completed in {evaluate_time:.2f} seconds")
        
        # Print total runtime
        total_time = preprocess_time + model_time + evaluate_time
        print(f"\nTotal analysis time: {total_time:.2f} seconds")

    except Exception as e:
        print(f"\nError: An issue occurred while running the program:")
        print(str(e))
        raise

if __name__ == "__main__":
    main()
    print("\nProgram completed successfully.")