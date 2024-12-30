from mlxtend.frequent_patterns import apriori, fpgrowth
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from itertools import combinations
import numpy as np

def construct_frequent_pattern_models(basket):
    print("Running Apriori algorithm...")
    frequent_itemsets_apriori = apriori(basket, min_support=0.01, use_colnames=True)
    print(f"Apriori generated {len(frequent_itemsets_apriori)} patterns.\n")

    print("Running FP-Growth algorithm...")
    frequent_itemsets_fpgrowth = fpgrowth(basket, min_support=0.01, use_colnames=True)
    print(f"FP-Growth generated {len(frequent_itemsets_fpgrowth)} patterns.\n")

    print("Running ECLAT algorithm...")
    item_support = {}
    for col in basket.columns:
        item_support[frozenset([col])] = basket[col].sum() / len(basket)

    min_support = 0.01
    frequent_itemsets_eclat = {k: v for k, v in item_support.items() if v >= min_support}
    for k in range(2, len(basket.columns)):
        for comb in combinations(basket.columns, k):
            support = basket[list(comb)].all(axis=1).mean()
            if support >= min_support:
                frequent_itemsets_eclat[frozenset(comb)] = support
    print(f"ECLAT generated {len(frequent_itemsets_eclat)} patterns.\n")

    return frequent_itemsets_apriori, frequent_itemsets_fpgrowth, frequent_itemsets_eclat

def perform_clustering(scaled_features):
    print("Performing K-Means clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(scaled_features)
    print("K-Means clustering completed.\n")

    print("Performing AGNES clustering...")
    sample_size = min(1000, len(scaled_features))  # Limit to 1000 samples or less
    sample_indices = np.random.choice(len(scaled_features), sample_size, replace=False)
    sampled_features = scaled_features[sample_indices]
    agnes = AgglomerativeClustering(n_clusters=3, linkage='average')
    agnes_labels = agnes.fit_predict(sampled_features)
    print("AGNES clustering completed.\n")

    print("Performing DBSCAN clustering...")
    sample_size = min(1000, len(scaled_features))  # Limit to 1000 samples or less
    sample_indices = np.random.choice(len(scaled_features), sample_size, replace=False)
    sampled_features = scaled_features[sample_indices]
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(sampled_features)
    print("DBSCAN clustering completed.\n")

    return kmeans, kmeans_labels, agnes_labels, dbscan_labels