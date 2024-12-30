import pandas as pd
from preprocessing import merge_datasets, preprocess_data

def main():
    try:
        # Load and merge datasets
        print("Loading and merging datasets...")
        merged_df = merge_datasets('data/athlete_events.csv', 'data/noc_regions.csv')
        
        # Print a few lines of the merged data
        print("Merged Data (first 5 rows):")
        print(merged_df.head())
        
        # Preprocess the merged data
        print("Preprocessing merged data...")
        preprocessed_df = preprocess_data(merged_df)
        
        # Print a few lines of the preprocessed data
        print("Preprocessed Data (first 5 rows):")
        print(preprocessed_df.head())

    except Exception as e:
        print("Error: An issue occurred while running the program:")
        print(str(e))

if __name__ == "__main__":
    main()
    print("\nProgram completed successfully.")