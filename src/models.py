from mlxtend.frequent_patterns import apriori, fpgrowth
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def create_models(df):
    """Create all models"""
    models = {
        'Apriori': apriori(df, min_support=0.1, use_colnames=True),
        'FP-Growth': fpgrowth(df, min_support=0.1, use_colnames=True),
        # 'ECLAT': Construct ECLAT model (custom implementation),
        'K-Means': KMeans(n_clusters=3),
        'AGNES': AgglomerativeClustering(n_clusters=3),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    }
    return models

