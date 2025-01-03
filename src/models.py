import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample

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
    
    df = df.copy()
    
    # Create sport type features
    df['is_endurance'] = df['Sport'].isin(endurance_sports)
    df['is_team'] = df['Sport'].isin(team_sports)
    df['is_combat'] = df['Sport'].isin(combat_sports)
    df['is_technical'] = df['Sport'].isin(technical_sports)
    df['is_precision'] = df['Sport'].isin(precision_sports)
    
    # Create derived features
    df['bmi'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['age_group'] = pd.qcut(df['Age'], q=5, labels=['Very Young', 'Young', 'Middle', 'Mature', 'Senior'])
    df['height_group'] = pd.qcut(df['Height'], q=5, labels=['Very Short', 'Short', 'Average', 'Tall', 'Very Tall'])
    df['weight_group'] = pd.qcut(df['Weight'], q=5, labels=['Very Light', 'Light', 'Medium', 'Heavy', 'Very Heavy'])
    
    return df

def prepare_clustering_data(df):
    """Prepare data for clustering with improved features"""
    # Add feature groups
    df = create_feature_groups(df)
    
    # Select base features
    numeric_features = ['Age', 'Height', 'Weight', 'bmi']
    binary_features = ['is_endurance', 'is_team', 'is_combat', 'is_technical', 'is_precision']
    categorical_features = ['Sex_F', 'Sex_M', 'age_group', 'height_group', 'weight_group']
    
    # Prepare feature matrix
    feature_matrix = []
    
    # Add numeric features (scaled)
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
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
    
    # Apply PCA with whitening
    pca = PCA(n_components=0.95, whiten=True)
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
    """Daha agresif parametre seçimi"""
    nbrs = NearestNeighbors(n_neighbors=3).fit(data)  # Daha az komşu
    distances, _ = nbrs.kneighbors(data)
    eps = np.percentile(distances[:, -1], 50)  # Medyan kullan
    min_samples = max(3, len(data) // 30)  # Daha küçük min_samples
    return eps, min_samples

def create_models(df):
    """Create models with improved clustering"""
    print("\nCreating models...")
    
    # Balance dataset
    balanced_df = balance_dataset(df)
    print(f"\nDataset balance:")
    print(f"Original size: {len(df)}")
    print(f"Balanced size: {len(balanced_df)}")
    print("\nMedal distribution:")
    print(balanced_df['Medal'].value_counts())
    
    # Prepare data
    clustering_data = prepare_clustering_data(balanced_df)
    pattern_data = pd.get_dummies(balanced_df[['Sport', 'Medal']])
    
    # Find optimal DBSCAN parameters
    eps, min_samples = find_optimal_dbscan_params(clustering_data)
    optimal_k = min(5, len(clustering_data) // 20)  # Reasonable number of clusters
    
    print(f"\nClustering parameters:")
    print(f"K-means clusters: {optimal_k}")
    print(f"DBSCAN eps: {eps:.3f}")
    print(f"DBSCAN min_samples: {min_samples}")
    
    # Create models dictionary
    models = {
        'data': {
            'clustering': clustering_data,
            'original_df': df,
            'balanced_df': balanced_df
        },
        'clustering': {
            'K-Means': KMeans(
                n_clusters=optimal_k,
                n_init=20,  # More initializations
                random_state=42
            ).fit(clustering_data),
            'AGNES': AgglomerativeClustering(
                n_clusters=optimal_k,
                linkage='complete'  # Changed from 'ward' for better separation
            ).fit(clustering_data),
            'DBSCAN': DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='euclidean'
            ).fit(clustering_data)
        },
        'pattern_mining': {
            'Apriori': apriori(pattern_data, min_support=0.1, use_colnames=True),
            'FP-Growth': fpgrowth(pattern_data, min_support=0.1, use_colnames=True)
        }
    }
    
    print("\nModels created successfully")
    return models