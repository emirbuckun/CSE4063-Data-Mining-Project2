import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_dataset(filepath):
    # Load the dataset from the given filepath
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Select reasonable features for the models
    selected_features = ['Age', 'Height', 'Weight', 'Sex', 'Sport', 'Event', 'region']
    df = df[selected_features]
    
    # Separate numeric and string features
    numeric_features = df.select_dtypes(include=['int64', 'float64'])
    string_features = df.select_dtypes(include=['object'])
    
    # Handle missing values
    numeric_features.ffill(inplace=True)
    string_features.ffill(inplace=True)
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_features_scaled = scaler.fit_transform(numeric_features)
    
    # Encode string features
    encoder = OneHotEncoder(sparse_output=False)
    string_features_encoded = encoder.fit_transform(string_features)
    
    # Combine numeric and encoded string features
    preprocessed_df = pd.concat([
        pd.DataFrame(numeric_features_scaled, columns=numeric_features.columns),
        pd.DataFrame(string_features_encoded, columns=encoder.get_feature_names_out(string_features.columns))
    ], axis=1)
    
    return preprocessed_df

def merge_datasets(athlete_events_path, noc_regions_path):
    # Load datasets
    athlete_events = pd.read_csv(athlete_events_path)
    noc_regions = pd.read_csv(noc_regions_path)
    
    # Merge datasets on 'NOC' column
    merged_df = pd.merge(athlete_events, noc_regions, on='NOC', how='left')
    
    return merged_df
