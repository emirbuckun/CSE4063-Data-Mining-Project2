import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def evaluate_models(frequent_pattern_results, clustering_results, scaled_features):
    frequent_itemsets_apriori, frequent_itemsets_fpgrowth, frequent_itemsets_eclat = frequent_pattern_results
    kmeans, kmeans_labels, agnes_labels, dbscan_labels = clustering_results

    print("Evaluating Apriori patterns...")
    print(f"Top Apriori patterns:\n{frequent_itemsets_apriori.head()}\n")

    print("Evaluating FP-Growth patterns...")
    print(f"Top FP-Growth patterns:\n{frequent_itemsets_fpgrowth.head()}\n")

    print("Evaluating ECLAT patterns...")
    print(f"Top ECLAT patterns:\n{list(frequent_itemsets_eclat.items())[:5]}\n")

    print("Evaluating K-Means clustering...")
    print(f"K-Means cluster centers:\n{kmeans.cluster_centers_}\n")

    print("Evaluating AGNES clustering...")
    print(f"AGNES cluster labels:\n{set(agnes_labels)}\n")

    print("Evaluating DBSCAN clustering...")
    print(f"DBSCAN cluster labels:\n{set(dbscan_labels)}\n")

    print("Plotting K-Means clustering...")
    plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=kmeans_labels, cmap='viridis', label='K-Means')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    print("Plotting Hierarchical Clustering Dendrogram...")
    linkage_matrix = linkage(scaled_features, method='ward')
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()