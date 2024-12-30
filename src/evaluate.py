from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def evaluate_models(models, df):
    # Evaluate and compare models
    for model in models:
        if isinstance(model, (KMeans, AgglomerativeClustering, DBSCAN)):
            labels = model.labels_
            score = silhouette_score(df, labels)
            print(f'{model.__class__.__name__} Silhouette Score: {score}')
        else:
            print(f'{model.__class__.__name__} Model: {model}')
