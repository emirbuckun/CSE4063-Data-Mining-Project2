import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def plot_clustering_results(models, data, save_path=None):
    """
    Görselleştir kümeleme sonuçlarını
    """
    # PCA ile 2 boyuta indir
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    
    # Plot ayarları
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Clustering Results Comparison')
    
    for idx, (name, model) in enumerate(models.items()):
        if hasattr(model, 'labels_'):
            # Kümeleri çiz
            scatter = axes[idx].scatter(data_2d[:, 0], data_2d[:, 1], 
                                      c=model.labels_, cmap='viridis')
            axes[idx].set_title(f'{name} Clusters')
            axes[idx].set_xlabel('First Principal Component')
            axes[idx].set_ylabel('Second Principal Component')
            
            # Colorbar ekle
            plt.colorbar(scatter, ax=axes[idx])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_pattern_mining_results(models, save_path=None):
    """
    Görselleştir pattern mining sonuçları
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Pattern Mining Results Comparison')
    
    for idx, (name, results) in enumerate(models.items()):
        if name == 'ECLAT':
            # ECLAT sonuçlarını DataFrame'e dönüştür
            df = pd.DataFrame([(list(itemset), support) for itemset, support in results[:5]],
                            columns=['itemsets', 'support'])
        else:
            df = results.head()
        
        # Bar plot
        df.plot(kind='bar', x='itemsets', y='support', ax=axes[idx], title=name)
        axes[idx].set_xticklabels(range(len(df)), rotation=45)
        axes[idx].set_xlabel('Itemsets')
        axes[idx].set_ylabel('Support')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_feature_distributions(df, save_path=None):
    """
    Görselleştir özellik dağılımları
    """
    numeric_cols = ['Age', 'Height', 'Weight']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Feature Distributions')
    
    for idx, col in enumerate(numeric_cols):
        sns.histplot(data=df, x=col, ax=axes[idx])
        axes[idx].set_title(f'{col} Distribution')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_correlation_matrix(df, save_path=None):
    """
    Görselleştir korelasyon matrisi
    """
    numeric_cols = ['Age', 'Height', 'Weight']
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def generate_analysis_report(models, df):
    """
    Analiz raporu oluştur
    """
    report = []
    
    # Pattern Mining Analizi
    report.append("Pattern Mining Analysis:")
    report.append("-" * 30)
    
    # Her algoritma için en sık örüntüleri bul
    for name, results in models.items():
        if name in ['Apriori', 'FP-Growth', 'ECLAT']:
            report.append(f"\n{name} Most Frequent Patterns:")
            if name == 'ECLAT':
                for itemset, support in results[:5]:
                    report.append(f"- {list(itemset)}: {support/len(df):.2%}")
            else:
                for _, row in results.head().iterrows():
                    report.append(f"- {row['itemsets']}: {row['support']:.2%}")
    
    # Clustering Analizi
    report.append("\nClustering Analysis:")
    report.append("-" * 30)
    
    for name, model in models.items():
        if hasattr(model, 'labels_'):
            labels = model.labels_
            unique_labels = np.unique(labels)
            report.append(f"\n{name} Results:")
            report.append(f"Number of clusters: {len(unique_labels[unique_labels != -1])}")
            
            # Küme boyutları
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            report.append("Cluster sizes:")
            for cluster, size in cluster_sizes.items():
                if cluster != -1:
                    report.append(f"- Cluster {cluster}: {size} samples")
                else:
                    report.append(f"- Noise points: {size} samples")
    
    return "\n".join(report)