import numpy as np
import pandas as pd
from preprocessing import preprocess_data
from models import create_models, prepare_data_for_clustering
from evaluate import evaluate_models
from visualization import (plot_clustering_results, plot_pattern_mining_results,
                            plot_feature_distributions, plot_correlation_matrix,
                            generate_analysis_report)

def create_sample_data():
    """Create sample dataset for testing"""
    data = {
        'ID': range(1, 11),
        'Name': ['Athlete' + str(i) for i in range(1, 11)],
        'Sex': np.random.choice(['M', 'F'], 10),
        'Age': np.random.randint(18, 35, 10),
        'Height': np.random.uniform(160, 190, 10),
        'Weight': np.random.uniform(50, 100, 10),
        'Team': np.random.choice(['USA', 'GBR', 'FRA', 'GER'], 10),
        'NOC': np.random.choice(['USA', 'GBR', 'FRA', 'GER'], 10),
        'Games': np.random.choice(['2012 Summer', '2016 Summer'], 10),
        'Sport': np.random.choice(['Athletics', 'Swimming', 'Gymnastics'], 10),
        'Event': np.random.choice(['100m', '200m', 'Freestyle'], 10),
        'Medal': np.random.choice(['Gold', 'Silver', 'Bronze', None], 10),
    }
    
    df = pd.DataFrame(data)
    regions = {
        'USA': 'United States',
        'GBR': 'United Kingdom',
        'FRA': 'France',
        'GER': 'Germany'
    }
    df['region'] = df['NOC'].map(regions)
    
    return df

def test_pipeline():
    """Test the entire data mining pipeline"""
    print("Starting tests...\n")
    
    # Create sample data
    df = create_sample_data()
    print("Sample data (first 5 rows):")
    print(df.head())
    
    # Preprocess data
    preprocessed_df = preprocess_data(df)
    print("\nPreprocessed data (first 5 rows):")
    print(preprocessed_df.head())
    
    # Create models
    models = create_models(preprocessed_df)
    
    # Plot feature distributions
    plot_feature_distributions(df, 'feature_distributions.png')
    
    # Plot correlation matrix
    plot_correlation_matrix(df, 'correlation_matrix.png')
    
    # Plot clustering results
    clustering_data = prepare_data_for_clustering(preprocessed_df)
    plot_clustering_results(
        {k: v for k, v in models.items() if hasattr(v, 'labels_')},
        clustering_data,
        'clustering_results.png'
    )
    
    # Plot pattern mining results
    pattern_mining_models = {
        k: v for k, v in models.items() 
        if k in ['Apriori', 'FP-Growth', 'ECLAT']
    }
    plot_pattern_mining_results(pattern_mining_models, 'pattern_mining_results.png')
    
    # Generate and print analysis report
    report = generate_analysis_report(models, df)
    print("\nDetailed Analysis Report:")
    print(report)
    
    # Evaluate models
    evaluate_models(models, preprocessed_df)

if __name__ == "__main__":
    test_pipeline()
    print("\nAll tests completed!")