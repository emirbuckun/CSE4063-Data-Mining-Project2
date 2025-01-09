import os
import time
from models import create_models
from evaluate import evaluate_models
from preprocessing import preprocess_data
from load_analyze import load_and_analyze_data
from visualization import (plot_dataset_analysis, plot_clustering_results,
                        plot_pattern_mining_results, plot_feature_distributions,
                        plot_correlation_matrix, plot_pattern_analysis)
from logger import set_logger

# Set logger
set_logger("logs")

def main():
    try:
        print("Starting Olympic Games Data Mining Analysis...")
        figures_dir = 'results/figures'
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        merged_df = load_and_analyze_data()

        print("\nGenerating initial dataset visualizations...")
        plot_dataset_analysis(merged_df, save_path=f'{figures_dir}/dataset_analysis.png')
        
        print("\nPreprocessing data...")
        start_time = time.time()
        preprocessed_df = preprocess_data(merged_df)
        preprocess_time = time.time() - start_time
        print(f"Preprocessing completed in {preprocess_time:.2f} seconds")

        print("\nGenerating feature distributions and correlations...")
        plot_feature_distributions(preprocessed_df, save_path=f'{figures_dir}/feature_distributions.png')
        plot_correlation_matrix(preprocessed_df, save_path=f'{figures_dir}/correlation_matrix.png')
        
        print("\nCreating and evaluating models...")
        start_time = time.time()
        models = create_models(preprocessed_df)
        model_time = time.time() - start_time
        print(f"Model creation completed in {model_time:.2f} seconds")
        
        print("\nGenerating model visualizations...")
        plot_clustering_results(models['clustering'], models['data']['clustering'], 
                                save_path=f'{figures_dir}/clustering_results.png')
        plot_pattern_mining_results(models['pattern_mining'], 
                                save_path=f'{figures_dir}/pattern_mining_results.png')
        plot_pattern_analysis(models['pattern_mining'], 
                                save_path=f'{figures_dir}/pattern_analysis.png')
        
        print("\nEvaluating models...")
        start_time = time.time()
        evaluate_models(models)
        evaluate_time = time.time() - start_time
        print(f"Evaluation completed in {evaluate_time:.2f} seconds")
        
        total_time = preprocess_time + model_time + evaluate_time
        print(f"\nTotal analysis time: {total_time:.2f} seconds")
    except Exception as e:
        print(f"\nError: An issue occurred while running the program:")
        print(str(e))
        raise

if __name__ == "__main__":
    main()
    print("\nProgram completed successfully.")