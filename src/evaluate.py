import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from models import prepare_data_for_clustering

def evaluate_models(models, df):
    """
    Evaluate clustering models using multiple metrics
    """
    print("\nClustering Evaluation Metrics:")
    print("-" * 50)
    
    # Prepare data for evaluation
    clustering_data = prepare_data_for_clustering(df)
    
    for name, model in models.items():
        if isinstance(model, (KMeans, AgglomerativeClustering, DBSCAN)):
            try:
                labels = model.labels_
                unique_labels = np.unique(labels)
                n_clusters = len(unique_labels[unique_labels != -1])
                
                # Create cluster distribution before checking n_clusters
                cluster_sizes = pd.Series(labels).value_counts()
                cluster_distribution = {f"Cluster {k}" if k != -1 else "Noise": int(v) 
                                     for k, v in cluster_sizes.items()}
                
                print(f"\n{name} Results:")
                
                if n_clusters < 2 and name == 'DBSCAN':
                    print(f"Warning: Found {n_clusters} clusters")
                    print(f"Label distribution: {cluster_distribution}")
                    continue
                elif n_clusters < 2:
                    print(f"Warning: Found {n_clusters} clusters")
                    continue
                
                # Calculate metrics
                silhouette = silhouette_score(clustering_data, labels)
                calinski = calinski_harabasz_score(clustering_data, labels)
                davies = davies_bouldin_score(clustering_data, labels)
                
                # Print results
                print(f"Number of clusters: {n_clusters}")
                print(f"Silhouette Score: {silhouette:.4f}")
                print(f"Calinski-Harabasz Score: {calinski:.4f}")
                print(f"Davies-Bouldin Score: {davies:.4f}")
                print("Cluster distribution:", cluster_distribution)
                
                if name == 'DBSCAN':
                    noise_points = len(labels[labels == -1])
                    if noise_points > 0:
                        print(f"Number of noise points: {noise_points}")
                        print(f"Noise percentage: {(noise_points/len(labels))*100:.2f}%")
                
                # Print cluster quality
                print("\nCluster Quality:")
                if silhouette > 0.5:
                    print("Good cluster separation")
                elif silhouette > 0.25:
                    print("Moderate cluster separation")
                else:
                    print("Poor cluster separation\n")
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")