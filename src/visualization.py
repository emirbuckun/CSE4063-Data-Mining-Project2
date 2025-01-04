import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def plot_dataset_analysis(df, save_path=None):
    """Dataset dağılımları için görselleştirmeler"""
    plt.figure(figsize=(15, 10))
    
    # Yaş dağılımı
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='Age', bins=30)
    plt.title('Age Distribution')
    
    # Spor bazlı madalya dağılımı (top 10)
    plt.subplot(2, 2, 2)
    sport_medals = df.groupby('Sport')['Medal'].value_counts().unstack()
    sport_medals.head(10).plot(kind='bar', stacked=True)
    plt.title('Medal Distribution by Sport (Top 10)')
    plt.xticks(rotation=45)
    
    # Boy-Kilo ilişkisi
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='Height', y='Weight', hue='Medal')
    plt.title('Height vs Weight by Medal')
    
    # Yıllara göre madalya dağılımı
    plt.subplot(2, 2, 4)
    year_medals = df.groupby('Year')['Medal'].value_counts().unstack()
    year_medals.plot(kind='line')
    plt.title('Medal Distribution Over Years')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

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
    plt.show()

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
    plt.show()

def plot_feature_distributions(df, save_path=None):
    """
    Görselleştir özellik dağılımları
    """
    # Önce hangi kolonların var olduğunu kontrol et
    available_cols = ['Age', 'Height', 'Weight']
    extra_cols = ['bmi', 'medal_rate', 'experience']
    
    numeric_cols = available_cols + [col for col in extra_cols if col in df.columns]
    
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    fig.suptitle('Feature Distributions')
    
    for idx, col in enumerate(numeric_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        if n_rows > 1:
            ax = axes[row, col_idx]
        else:
            ax = axes[col_idx]
        sns.histplot(data=df, x=col, ax=ax)
        ax.set_title(f'{col} Distribution')
    
    # Boş subplot'ları gizle
    for idx in range(len(numeric_cols), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        if n_rows > 1:
            axes[row, col_idx].set_visible(False)
        else:
            axes[col_idx].set_visible(False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_correlation_matrix(df, save_path=None):
    """
    Görselleştir korelasyon matrisi
    """
    # Önce hangi kolonların var olduğunu kontrol et
    available_cols = ['Age', 'Height', 'Weight']
    extra_cols = ['bmi', 'medal_rate', 'experience', 'gold_rate', 'sport_difficulty']
    
    numeric_cols = available_cols + [col for col in extra_cols if col in df.columns]
    
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_pattern_analysis(models, save_path=None):
    """Pattern mining sonuçlarının detaylı görselleştirmesi"""
    plt.figure(figsize=(15, 5))
    
    for idx, (name, results) in enumerate(models.items(), 1):
        plt.subplot(1, 3, idx)
        if name == 'ECLAT':
            pattern_lengths = [len(itemset) for itemset, _ in results]
        else:
            pattern_lengths = results['itemsets'].apply(len)
        
        sns.histplot(pattern_lengths, bins=10)
        plt.title(f'{name} Pattern Length Distribution')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

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