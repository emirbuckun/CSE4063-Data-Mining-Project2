import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from eclat import ECLAT

def balance_dataset(df):
    """Balance dataset using downsampling"""
    df = df.copy()
    df['Medal'] = df['Medal'].fillna('No Medal')
    medal_counts = df[df['Medal'] != 'No Medal']['Medal'].value_counts()
    min_count = medal_counts.min()
    
    balanced_dfs = []
    for medal in medal_counts.index:
        medal_df = df[df['Medal'] == medal]
        balanced_df = resample(medal_df, 
                             replace=False,
                             n_samples=min_count,
                             random_state=42)
        balanced_dfs.append(balanced_df)
    
    return pd.concat(balanced_dfs)

def create_feature_groups(df):
    """Create meaningful feature groups for clustering"""
    # Sport groups
    endurance_sports = ['Athletics', 'Swimming', 'Cycling', 'Rowing']
    team_sports = ['Basketball', 'Football', 'Volleyball', 'Hockey', 'Ice Hockey']
    combat_sports = ['Boxing', 'Wrestling', 'Judo', 'Taekwondo']
    technical_sports = ['Gymnastics', 'Diving', 'Figure Skating']
    precision_sports = ['Shooting', 'Archery', 'Golf']
    individual_sports = ['Swimming', 'Athletics', 'Gymnastics', 'Boxing', 
                        'Wrestling', 'Judo', 'Taekwondo', 'Tennis', 'Golf']
    
    df = df.copy()
    
    # Create sport type features
    df['is_endurance'] = df['Sport'].isin(endurance_sports)
    df['is_team'] = df['Sport'].isin(team_sports)
    df['is_combat'] = df['Sport'].isin(combat_sports)
    df['is_technical'] = df['Sport'].isin(technical_sports)
    df['is_precision'] = df['Sport'].isin(precision_sports)
    df['is_individual'] = df['Sport'].isin(individual_sports)
    
    # Create derived features
    df['bmi'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    # Yaş kategorileri
    age_bins = [0, 20, 25, 30, 35, 100]
    age_labels = ['Genç', 'Genç Yetişkin', 'Yetişkin', 'Tecrübeli', 'Usta']
    df['age_category'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    # Boy ve kilo grupları için manuel aralıklar
    weight_bins = [0, 50, 65, 80, 95, float('inf')]
    weight_labels = ['Very Light', 'Light', 'Medium', 'Heavy', 'Very Heavy']
    df['weight_group'] = pd.cut(df['Weight'], bins=weight_bins, labels=weight_labels)
    
    height_bins = [0, 165, 175, 185, 195, float('inf')]
    height_labels = ['Very Short', 'Short', 'Average', 'Tall', 'Very Tall']
    df['height_group'] = pd.cut(df['Height'], bins=height_bins, labels=height_labels)
    
    # BMI kategorileri
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels)
    
    # Sporcu başarı metrikleri
    df['medal_rate'] = df.groupby('Name')['Medal'].transform(lambda x: x.notna().mean())
    df['gold_rate'] = df.groupby('Name')['Medal'].transform(lambda x: (x == 'Gold').mean())
    
    # Deneyim hesaplama
    df['experience'] = df.groupby('Name')['Year'].transform(lambda x: x.max() - x.min())
    
    # Ülke bazlı metrikler
    df['country_medal_rate'] = df.groupby('NOC')['Medal'].transform(lambda x: x.notna().mean())
    df['country_gold_rate'] = df.groupby('NOC')['Medal'].transform(lambda x: (x == 'Gold').mean())
    
    # Sport difficulty ve performans metrikleri
    df['sport_difficulty'] = df.groupby('Sport')['Medal'].transform(
        lambda x: 1 / (x.notna().mean() + 0.01))
    
    df['age_efficiency'] = df['medal_rate'] / (df['Age'] + 1)
    
    df['fitness_score'] = (df['Height'] * df['Weight']) / \
                         ((df['Age'] + 1) * df['sport_difficulty'])
    
    return df

def prepare_clustering_data(df):
    """Prepare data for clustering with improved preprocessing"""
    # Add feature groups
    df = create_feature_groups(df)
    
    # Select base features
    numeric_features = ['Age', 'Height', 'Weight', 'bmi', 'medal_rate', 
                       'experience', 'gold_rate', 'country_medal_rate', 
                       'country_gold_rate', 'sport_difficulty', 
                       'age_efficiency', 'fitness_score']
    binary_features = ['is_endurance', 'is_team', 'is_combat', 
                      'is_technical', 'is_precision', 'is_individual']
    categorical_features = ['Sex_F', 'Sex_M', 'age_category', 
                          'height_group', 'weight_group', 'bmi_category']
    
    # Prepare feature matrix
    feature_matrix = []
    
    # Add numeric features (scaled)
    scaler = RobustScaler()
    numeric_data = scaler.fit_transform(df[numeric_features])
    feature_matrix.append(numeric_data)
    
    # Add binary features
    binary_data = df[binary_features].values
    feature_matrix.append(binary_data)
    
    # Add encoded categorical features
    categorical_data = pd.get_dummies(df[categorical_features])
    feature_matrix.append(categorical_data.values)
    
    # Combine all features
    combined_features = np.hstack(feature_matrix)
    
    # Apply MinMaxScaler to all features
    scaler_all = MinMaxScaler()
    combined_features = scaler_all.fit_transform(combined_features)
    
    # Apply PCA with variance ratio
    pca = PCA(n_components=0.85)
    data_reduced = pca.fit_transform(combined_features)
    
    print("\nClustering data:")
    print(f"Original features: {combined_features.shape[1]}")
    print(f"Selected features:")
    print(f"- Numeric: {len(numeric_features)}")
    print(f"- Binary: {len(binary_features)}")
    print(f"- Categorical: {len(categorical_features)}")
    print(f"Reduced dimensions: {data_reduced.shape[1]}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    return data_reduced

def find_optimal_dbscan_params(data):
    """Find optimal DBSCAN parameters using nearest neighbors"""
    nbrs = NearestNeighbors(n_neighbors=5).fit(data)
    distances, _ = nbrs.kneighbors(data)
    eps = np.percentile(distances[:, -1], 75)  # 75. percentil
    min_samples = 4  # Decreased min_samples
    return eps*0.7, min_samples

def create_models(df):
    """Create models with improved parameters"""
    print("\nCreating models...")
    
    # Balance dataset
    balanced_df = balance_dataset(df)
    print(f"\nDataset balance:")
    print(f"Original size: {len(df)}")
    print(f"Balanced size: {len(balanced_df)}")
    print("\nMedal distribution:")
    print(balanced_df['Medal'].value_counts())
    
    # Prepare clustering data
    clustering_data = prepare_clustering_data(balanced_df)
    
    # Find optimal DBSCAN parameters
    eps, min_samples = find_optimal_dbscan_params(clustering_data)
    
    print(f"\nClustering parameters:")
    print(f"K-means clusters: 5")
    print(f"DBSCAN eps: {eps:.3f}")
    print(f"DBSCAN min_samples: {min_samples}")
    
    # Prepare pattern mining data
    pattern_data = pd.DataFrame()
    
    # Add Sport columns
    for sport in balanced_df['Sport'].unique():
        pattern_data[f'Sport_{sport}'] = (balanced_df['Sport'] == sport)
        
    # Add Medal columns
    for medal in balanced_df['Medal'].unique():
        pattern_data[f'Medal_{medal}'] = (balanced_df['Medal'] == medal)
        
    # Create combined features
    for sport in balanced_df['Sport'].unique():
        for medal in balanced_df['Medal'].unique():
            pattern_data[f'Sport_{sport}_Medal_{medal}'] = (
                (balanced_df['Sport'] == sport) & 
                (balanced_df['Medal'] == medal)
            )
    
    # Convert to boolean type
    pattern_data = pattern_data.astype(bool)
    
    # Create models dictionary
    try:
        models = {
            'data': {
                'clustering': clustering_data,
                'original_df': df,
                'balanced_df': balanced_df
            },
            'clustering': {
                'K-Means': KMeans(
                    n_clusters=5,
                    n_init=200,
                    random_state=42,
                    max_iter=2000
                ).fit(clustering_data),
                'AGNES': AgglomerativeClustering(
                    n_clusters=5,
                    linkage='ward'
                ).fit(clustering_data),
                'DBSCAN': DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric='euclidean'
                ).fit(clustering_data)
            },
            'pattern_mining': {
                'Apriori': apriori(
                    pattern_data, 
                    min_support=0.002,
                    use_colnames=True,
                    max_len=6,
                    low_memory=True,
                    verbose=0
                ),
                'FP-Growth': fpgrowth(
                    pattern_data, 
                    min_support=0.002,
                    use_colnames=True,
                    max_len=6
                ),
                'ECLAT': ECLAT(min_support=0.002).fit(pattern_data)
            }
        }
        print("\nModels created successfully")
        
    except Exception as e:
        print(f"\nError creating models: {str(e)}")
        raise
    
    return models