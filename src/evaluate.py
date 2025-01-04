import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def evaluate_pattern_mining(name, model, df):
    """
    Evaluate pattern mining results with improved details
    """
    print(f"\n{name} Results:")
    print("Top 10 frequent patterns:")
    
    if name == 'ECLAT':
        # ECLAT sonuçlarını işle
        for itemset, support in list(model)[:10]:
            support_pct = (support / len(df)) * 100
            items = list(itemset)
            
            # Spor ve madalya kombinasyonlarını ayır
            sports = [item for item in items if 'Sport_' in item]
            medals = [item for item in items if 'Medal_' in item]
            
            print(f"- Pattern:")
            if sports:
                print(f"  Sports: {[s.replace('Sport_', '') for s in sports]}")
            if medals:
                print(f"  Medals: {[m.replace('Medal_', '') for m in medals]}")
            print(f"  Support: {support_pct:.2f}%")
            print()
    else:
        # Apriori ve FP-Growth sonuçlarını işle
        patterns = model.sort_values('support', ascending=False).head(10)
        for _, row in patterns.iterrows():
            support_pct = row['support'] * 100
            items = list(row['itemsets'])
            
            # Spor ve madalya kombinasyonlarını ayır
            sports = [item for item in items if 'Sport_' in item]
            medals = [item for item in items if 'Medal_' in item]
            
            print(f"- Pattern:")
            if sports:
                print(f"  Sports: {[s.replace('Sport_', '') for s in sports]}")
            if medals:
                print(f"  Medals: {[m.replace('Medal_', '') for m in medals]}")
            print(f"  Support: {support_pct:.2f}%")
            print()

def evaluate_clustering(model, data, name, original_df):
    """
    Evaluate clustering results with detailed analysis
    """
    try:
        labels = model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        print(f"\n{name} Results:")
        if n_clusters < 2:
            print(f"Warning: Only {n_clusters} cluster(s) found")
            return
        
        # Calculate metrics
        try:
            silhouette = silhouette_score(data, labels)
            calinski = calinski_harabasz_score(data, labels)
            davies = davies_bouldin_score(data, labels)
            
            print(f"Number of clusters: {n_clusters}")
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Calinski-Harabasz Score: {calinski:.4f}")
            print(f"Davies-Bouldin Score: {davies:.4f}")
            
            # Cluster distribution
            unique, counts = np.unique(labels, return_counts=True)
            cluster_sizes = dict(zip(range(n_clusters), [0] * n_clusters))
            for label, count in zip(unique, counts):
                if label != -1:  # Ignore noise points
                    cluster_sizes[label] = count
            
            print("\nCluster Distribution:")
            for cluster, size in cluster_sizes.items():
                percentage = (size / len(labels)) * 100
                print(f"Cluster {cluster}: {size} samples ({percentage:.1f}%)")
                
                # Get samples in this cluster
                cluster_mask = labels == cluster
                cluster_df = original_df[cluster_mask]
                
                # Show sport distribution
                print("  Top sports:")
                print(cluster_df['Sport'].value_counts().head(3).to_string())
                
                # Show medal distribution
                print("  Medal distribution:")
                print(cluster_df['Medal'].value_counts().to_string())
                print()
            
            # Cluster quality assessment
            print("Cluster Quality:")
            if silhouette > 0.5:
                print("- Good separation")
            elif silhouette > 0.25:
                print("- Moderate separation")
            else:
                print("- Poor separation")
                
        except ValueError as ve:
            print(f"Error calculating metrics: {str(ve)}")
            
    except Exception as e:
        print(f"Error evaluating {name}: {str(e)}")

def evaluate_models(models):
    """
    Evaluate all models with improved output
    """
    print("\nPattern Mining Results:")
    print("-" * 50)
    
    # Evaluate pattern mining
    pattern_mining = models['pattern_mining']
    for name, model in pattern_mining.items():
        evaluate_pattern_mining(name, model, models['data']['balanced_df'])
    
    print("\nClustering Results:")
    print("-" * 50)
    
    # Evaluate clustering
    clustering = models['clustering']
    for name, model in clustering.items():
        evaluate_clustering(
            model, 
            models['data']['clustering'],
            name,
            models['data']['balanced_df']
        )