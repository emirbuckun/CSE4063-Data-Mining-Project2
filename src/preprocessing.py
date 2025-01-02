import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """
    Preprocess the Olympic Games dataset
    """
    print("Starting preprocessing...")
    
    # Select features that exist in both test and real dataset
    selected_features = ['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event', 
                        'Medal', 'Team', 'Games', 'region']
    
    # Check which features are actually available
    available_features = [f for f in selected_features if f in df.columns]
    df = df[available_features].copy()
    
    print(f"\nSelected features: {available_features}")
    
    # Handle missing values
    print("\nHandling missing values...")
    # For numeric columns
    numeric_cols = ['Age', 'Height', 'Weight']
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    # For categorical columns
    categorical_cols = [col for col in ['Sex', 'Sport', 'Event', 'Medal', 'Team', 'Games', 'region'] 
                       if col in df.columns]
    for col in categorical_cols:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
    
    print("\nMissing values after imputation:")
    print(df.isnull().sum())
    
    # Create binary columns for pattern mining
    print("\nEncoding categorical variables...")
    # Combine Sport and Medal for pattern mining
    df['Sport_Medal'] = df['Sport'] + '_' + df['Medal'].fillna('No_Medal')
    
    # Scale numeric features
    print("\nScaling numeric features...")
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Encode categorical variables
    categorical_cols = [col for col in ['Sex', 'Sport', 'Event', 'region'] 
                       if col in df.columns]
    if categorical_cols:
        try:
            # Try new sklearn version
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            # Fall back to old sklearn version
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
        encoded_data = encoder.fit_transform(df[categorical_cols])
        encoded_cols = []
        for i, col in enumerate(categorical_cols):
            feature_names = [f"{col}_{val}" for val in encoder.categories_[i]]
            encoded_cols.extend(feature_names)
        
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)
        
        # Combine numeric and encoded features
        keep_original = [col for col in ['Games', 'Team', 'Sport', 'Medal'] 
                        if col in df.columns]
        final_df = pd.concat([
            df[numeric_cols] if numeric_cols else pd.DataFrame(),
            encoded_df,
            df[keep_original] if keep_original else pd.DataFrame()
        ], axis=1)
    else:
        final_df = df
    
    print("\nPreprocessing completed.")
    print(f"Final dataset shape: {final_df.shape}")
    
    return final_df

def merge_datasets(athlete_events, noc_regions):
    """
    Merge athlete_events and noc_regions datasets
    """
    return pd.merge(athlete_events, noc_regions, on='NOC', how='left')

if __name__ == "__main__":
    # Test preprocessing
    df = pd.DataFrame({
        'Age': [25, 30, np.nan],
        'Height': [175, np.nan, 180],
        'Weight': [70, 75, np.nan],
        'Sex': ['M', 'F', 'M'],
        'Sport': ['Athletics', 'Swimming', 'Athletics'],
        'Medal': ['Gold', None, 'Silver']
    })
    preprocessed_df = preprocess_data(df)
    print("\nPreprocessed data sample:")
    print(preprocessed_df.head())