import time
import pandas as pd
from preprocessing import merge_datasets

def load_datasets(athlete_events_path, noc_regions_path):
    print("Loading datasets...")
    athlete_events = pd.read_csv(athlete_events_path)
    noc_regions = pd.read_csv(noc_regions_path)
    return athlete_events, noc_regions

def print_dataset_info(athlete_events):
    print("\nFull Dataset Information:")
    print("-" * 50)
    print(f"Total number of rows: {len(athlete_events)}")
    print(f"Total number of columns: {len(athlete_events.columns)}")

def take_stratified_sample(athlete_events, sample_size):
    print(f"\nTaking a stratified sample of {sample_size} rows...")
    sampled_indices = []
    for medal in athlete_events['Medal'].unique():
        if pd.isna(medal):
            temp_df = athlete_events[athlete_events['Medal'].isna()]
        else:
            temp_df = athlete_events[athlete_events['Medal'] == medal]
        prop = len(temp_df) / len(athlete_events)
        n_samples = int(sample_size * prop)
        if len(temp_df) > n_samples:
            sampled_indices.extend(temp_df.sample(n=n_samples, random_state=42).index)
        else:
            sampled_indices.extend(temp_df.index)
    return athlete_events.loc[sampled_indices]

def print_sample_info(athlete_events):
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
    memory_usage = athlete_events.memory_usage(deep=True).sum() / 1024**2
    print(f"\nMemory usage: {memory_usage:.2f} MB")

def load_and_analyze_data(athlete_events_path='data/athlete_events.csv', 
                        noc_regions_path='data/noc_regions.csv',
                        sample_size=5000):
    """
    Load and analyze the Olympic Games dataset
    """
    try:
        athlete_events, noc_regions = load_datasets(athlete_events_path, noc_regions_path)
        print_dataset_info(athlete_events)
        if sample_size and sample_size < len(athlete_events):
            athlete_events = take_stratified_sample(athlete_events, sample_size)
            print(f"Sample shape: {athlete_events.shape}")
        print_sample_info(athlete_events)

        print("\nMerging datasets...")
        start_time = time.time()
        merged_df = merge_datasets(athlete_events, noc_regions)
        merge_time = time.time() - start_time
        print(f"Merge completed in {merge_time:.2f} seconds")
        return merged_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
