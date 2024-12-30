import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_explore_data(filepath):
    # Load the dataset from the given filepath
    return pd.read_csv(filepath)

def preprocess_data():
    print("Loading datasets...")
    athlete_events = pd.read_csv('data/athlete_events.csv')
    noc_regions = pd.read_csv('data/noc_regions.csv')
    print(f"Datasets loaded: {len(athlete_events)} athlete events, {len(noc_regions)} NOC regions.\n")

    print("Merging datasets...")
    athlete_events = athlete_events.merge(noc_regions, on='NOC', how='left')
    print("Datasets merged.\n")

    print("Handling missing values...")
    athlete_events['Age'] = athlete_events['Age'].fillna(athlete_events['Age'].mean())
    athlete_events['Height'] = athlete_events['Height'].fillna(athlete_events['Height'].mean())
    athlete_events['Weight'] = athlete_events['Weight'].fillna(athlete_events['Weight'].mean())
    athlete_events['Medal'] = athlete_events['Medal'].fillna('NA')
    print("Missing values handled.\n")

    print("Encoding categorical columns...")
    athlete_events_encoded = pd.get_dummies(athlete_events, columns=['Sex', 'Season', 'Sport', 'Medal'])
    print("Encoding completed.\n")

    print("Scaling features...")
    features = athlete_events_encoded[['Age', 'Height', 'Weight']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print("Features scaled.\n")

    basket = athlete_events_encoded.iloc[:, -10:]
    return athlete_events, scaled_features, basket
