import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from eclat import ECLAT, prepare_olympic_data_for_eclat

def prepare_data_for_pattern_mining(df):
    """
    Prepare data for pattern mining algorithms
    """
    transactions = df[['Games', 'Team', 'Sport', 'Medal']].copy()
    sport_dummies = pd.get_dummies(transactions['Sport'], prefix='Sport').astype(bool)
    medal_dummies = pd.get_dummies(transactions['Medal'], prefix='Medal').astype(bool)
    pattern_mining_data = pd.concat([sport_dummies, medal_dummies], axis=1)
    return pattern_mining_data

def prepare_data_for_clustering(df):
    """
    Prepare data for clustering with enhanced preprocessing
    """
    # Feature weights
    feature_weights = {
        'Age': 2.0,
        'Height': 1.5,
        'Weight': 1.5,
        'Sex': 2.0,
        'Sport': 2.5,
        'Event': 1.0,
        'region': 1.0
    }
    
    # Select and prepare features
    numeric_cols = ['Age', 'Height', 'Weight']
    encoded_cols = [col for col in df.columns if any(col.startswith(prefix) 
                   for prefix in ['Sex_', 'Sport_', 'Event_', 'region_'])]
    
    # Create a copy and scale numeric features
    data = df[numeric_cols + encoded_cols].copy()
    
    # Scale numeric features
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    # Apply feature weights
    for col in numeric_cols:
        data[col] *= feature_weights.get(col, 1.0)
    
    for col in encoded_cols:
        prefix = col.split('_')[0]
        data[col] *= feature_weights.get(prefix, 1.0)
    
    # Apply PCA with whitening
    pca = PCA(n_components=0.95, whiten=True)
    data_reduced = pca.fit_transform(data)
    
    # Print detailed information
    print(f"\nPreprocessing Information:")
    print(f"Original features: {len(numeric_cols + encoded_cols)}")
    print(f"After PCA: {data_reduced.shape[1]} components")
    print(f"Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.2%}")
    print("\nFeature Weights Used:")
    for feature, weight in feature_weights.items():
        print(f"{feature}: {weight}")
    
    return data_reduced

def find_optimal_params(X):
    """
    Find optimal clustering parameters using enhanced methods
    """
    # Calculate various distance statistics
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Calculate multiple eps candidates
    eps_candidates = {
        'median': np.median(distances[:, 1]),
        'percentile_75': np.percentile(distances[:, 1], 75),
        'mean': np.mean(distances[:, 1])
    }
    
    # Choose eps based on data size
    if len(X) < 15:  # Small dataset
        eps = eps_candidates['percentile_75']
        min_samples = 2
        n_clusters = 3
    else:  # Larger dataset
        eps = eps_candidates['median']
        min_samples = max(3, int(len(X) * 0.15))
        n_clusters = min(5, len(X) // 3)
    
    print("\nClustering Parameters:")
    print(f"Dataset size: {len(X)}")
    print(f"Eps candidates: {eps_candidates}")
    print(f"Chosen eps: {eps:.3f}")
    print(f"Min samples: {min_samples}")
    print(f"Number of clusters: {n_clusters}")
    
    return {
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': n_clusters
    }

def create_models(df):
    """Create all models with optimized parameters"""
    # Prepare data
    pattern_mining_data = prepare_data_for_pattern_mining(df)
    clustering_data = prepare_data_for_clustering(df)
    
    # Get optimal parameters
    params = find_optimal_params(clustering_data)
    
    models = {}
    
    # Pattern Mining Models
    try:
        models['Apriori'] = apriori(pattern_mining_data, min_support=0.15, use_colnames=True)
        models['FP-Growth'] = fpgrowth(pattern_mining_data, min_support=0.15, use_colnames=True)
        models['ECLAT'] = ECLAT(min_support=0.15).fit(pattern_mining_data)
    except Exception as e:
        print(f"Error in pattern mining: {str(e)}")
    
    # Clustering Models
    try:
        # K-Means
        models['K-Means'] = KMeans(
            n_clusters=params['n_clusters'],
            random_state=42,
            n_init=30,
            max_iter=500
        ).fit(clustering_data)
        
        # AGNES
        models['AGNES'] = AgglomerativeClustering(
            n_clusters=params['n_clusters'],
            linkage='ward'
        ).fit(clustering_data)
        
        # DBSCAN
        models['DBSCAN'] = DBSCAN(
            eps=params['eps'],
            min_samples=params['min_samples'],
            metric='euclidean'
        ).fit(clustering_data)
        
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
    
    return models