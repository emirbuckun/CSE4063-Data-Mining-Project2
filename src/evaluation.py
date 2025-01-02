import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import os
import pandas as pd

def evaluate_models(frequent_pattern_results, clustering_results, scaled_features):
    # Ensure the images directory exists
    if not os.path.exists('images'):
        os.makedirs('images')

    frequent_itemsets_apriori, frequent_itemsets_fpgrowth, frequent_itemsets_eclat = frequent_pattern_results
    kmeans, kmeans_labels, agnes_labels, dbscan_labels = clustering_results

    print("Evaluating Apriori patterns...")
    print(f"Top Apriori patterns:\n{frequent_itemsets_apriori.head()}\n")

    print("Evaluating FP-Growth patterns...")
    print(f"Top FP-Growth patterns:\n{frequent_itemsets_fpgrowth.head()}\n")

    print("Evaluating ECLAT patterns...")
    eclat_df = pd.DataFrame(list(frequent_itemsets_eclat.items()), columns=['itemsets', 'support'])
    eclat_df = eclat_df.sort_values(by='support', ascending=False).reset_index(drop=True)
    print(f"Top ECLAT patterns:\n{eclat_df.head()}\n")

    print("Evaluating K-Means clustering...")
    print(f"K-Means cluster centers:\n{kmeans.cluster_centers_}\n")

    print("Evaluating AGNES clustering...")
    agnes_labels_unique = np.unique(agnes_labels)
    print(f"AGNES cluster labels: {agnes_labels_unique}\n")

    print("Evaluating DBSCAN clustering...")
    dbscan_labels_unique = np.unique(dbscan_labels)
    print(f"DBSCAN cluster labels: {dbscan_labels_unique}\n")

    print("Plotting K-Means clustering...")
    plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=kmeans_labels, cmap='viridis', label='K-Means')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig('images/kmeans_clustering.png')
    plt.close()

    print("Plotting Hierarchical Clustering Dendrogram...")
    sample_size = min(1000, len(scaled_features))  # Limit to 1000 samples or less
    sample_indices = np.random.choice(len(scaled_features), sample_size, replace=False)
    sampled_features = scaled_features[sample_indices]
    linkage_matrix = linkage(sampled_features, method='ward')
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.savefig('images/hierarchical_clustering_dendrogram.png')
    plt.close()