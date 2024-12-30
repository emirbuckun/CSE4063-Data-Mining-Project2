from preprocessing import preprocess_data
from models import construct_frequent_pattern_models, perform_clustering
from evaluation import evaluate_models
from logger import set_logger
# Set logger
set_logger("logs")

def main():
    try:
        print("Starting preprocessing...")
        athlete_events, scaled_features, basket = preprocess_data()
        print("Preprocessing completed.\n")

        print("Constructing frequent pattern mining models...")
        frequent_pattern_results = construct_frequent_pattern_models(basket)
        print("Frequent pattern mining completed.\n")

        print("Performing clustering...")
        clustering_results = perform_clustering(scaled_features)
        print("Clustering completed.\n")

        print("Evaluating models...")
        evaluate_models(frequent_pattern_results, clustering_results, scaled_features)
        print("Evaluation completed.")

    except Exception as e:
        print("\nError: An issue occurred while running the program:")
        print(str(e))

if __name__ == "__main__":
    main()
    print("\nProgram completed successfully.")