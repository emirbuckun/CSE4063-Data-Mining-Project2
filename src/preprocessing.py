import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """
    Preprocess the Olympic Games dataset while keeping original columns
    """
    print("Starting preprocessing...")
    
    # Select only the most important features
    selected_features = ['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Medal']
    df = df[selected_features].copy()
    
    print(f"\nSelected features: {selected_features}")
    
    # Create a copy for encoded features
    encoded_df = df.copy()
    
    # Handle missing values
    print("\nHandling missing values...")
    # For numeric columns
    numeric_cols = ['Age', 'Height', 'Weight']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        encoded_df[col] = encoded_df[col].fillna(encoded_df[col].median())
    
    # For categorical columns
    categorical_cols = ['Sex', 'Sport', 'Medal']
    for col in categorical_cols:
        mode_val = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_val)
        encoded_df[col] = encoded_df[col].fillna(mode_val)
    
    print("\nMissing values after imputation:")
    print(df.isnull().sum())
    
    # Encode categorical variables for analysis
    print("\nEncoding categorical variables...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_data = encoder.fit_transform(encoded_df[categorical_cols])
    categorical_columns = encoder.get_feature_names_out(categorical_cols)
    
    # Scale numeric features
    print("\nScaling numeric features...")
    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(encoded_df[numeric_cols])
    
    # Create encoded features DataFrame
    encoded_features = pd.DataFrame(
        np.hstack([numeric_data, categorical_data]),
        columns=list(numeric_cols) + list(categorical_columns),
        index=df.index
    )
    
    # Combine encoded features with original columns
    final_df = pd.concat([
        encoded_features,
        df[['Medal', 'Sport']]  # Keep original Medal and Sport columns for pattern mining
    ], axis=1)
    
    print("\nPreprocessing completed.")
    print(f"Final dataset shape: {final_df.shape}")
    
    return final_df

def merge_datasets(athlete_events, noc_regions):
    """
    Merge athlete_events and noc_regions datasets
    """
    return pd.merge(athlete_events, noc_regions, on='NOC', how='left')