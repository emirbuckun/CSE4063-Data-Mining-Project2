import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def evaluate_pattern_mining(results, data_dict):
    """
    Evaluate pattern mining results
    """
    print("\nPattern Mining Results:")
    print("-" * 50)
    
    df = data_dict['balanced_df']
    
    for name, model in results.items():
        print(f"\n{name} Results:")
        print("Top frequent patterns:")
        patterns = model.sort_values('support', ascending=False)
        
        for _, row in patterns.head(10).iterrows():
            items = list(row['itemsets'])
            support = row['support'] * 100
            
            # Only show meaningful patterns
            if len(items) > 0:
                print(f"- {items}: {support:.2f}%")
                
                # Show related patterns
                if any('Sport_' in str(i) for i in items):
                    related = patterns[patterns['itemsets'].apply(
                        lambda x: any('Medal_' in str(i) for i in x)
                    )].head(3)
                    if not related.empty:
                        print("  Related medal patterns:")
                        for _, r in related.iterrows():
                            print(f"    {list(r['itemsets'])}: {r['support']*100:.2f}%")

def evaluate_clustering(results, data_dict):
    """
    Evaluate clustering results
    """
    print("\nClustering Results:")
    print("-" * 50)
    
    clustering_data = data_dict['clustering']
    df = data_dict['balanced_df']
    
    for name, model in results.items():
        print(f"\n{name} Results:")
        labels = model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters < 2:
            print(f"Warning: Only {n_clusters} cluster(s) found")
            continue
            
        # Calculate metrics
        silhouette = silhouette_score(clustering_data, labels)
        calinski = calinski_harabasz_score(clustering_data, labels)
        davies = davies_bouldin_score(clustering_data, labels)
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Score: {calinski:.4f}")
        print(f"Davies-Bouldin Score: {davies:.4f}")
        
        # Cluster distribution
        print("\nCluster Distribution:")
        for i in range(n_clusters):
            cluster_size = np.sum(labels == i)
            print(f"Cluster {i}: {cluster_size} samples ({cluster_size/len(labels)*100:.1f}%)")
            
            # Analyze cluster characteristics
            cluster_df = df[labels == i]
            print("  Top sports:")
            print(cluster_df['Sport'].value_counts().head(3).to_string())
            print("  Medal distribution:")
            print(cluster_df['Medal'].value_counts().to_string())
            print()
            
        # Quality assessment
        print("Cluster Quality:")
        if silhouette > 0.5:
            print("- Good separation")
        elif silhouette > 0.25:
            print("- Moderate separation")
        else:
            print("- Poor separation")

def evaluate_models(models):
    """
    Evaluate all models
    """
    try:
        # Evaluate pattern mining
        evaluate_pattern_mining(models['pattern_mining'], models['data'])
        
        # Evaluate clustering
        evaluate_clustering(models['clustering'], models['data'])
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")