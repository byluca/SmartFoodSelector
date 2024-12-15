# main.py

from src.bayes_net import build_and_query_bayes_net
from src.dataset_preprocessing import load_and_preprocess_data
from src.supervised_models import train_and_evaluate_supervised
from src.unsupervised_clustering import find_optimal_k, assign_clusters


def main():
    # 1) Preprocessing
    load_and_preprocess_data("data/en.openfoodfacts.org.products.tsv", "data/newDataset.csv")

    # 2) Clustering
    elbow = find_optimal_k("data/newDataset.csv", max_k=10)
    assign_clusters("data/newDataset.csv", elbow, "data/clustered_dataset.csv")

    # 3) Supervised
    train_and_evaluate_supervised("data/clustered_dataset.csv")

    # 4) Bayesian Network
    build_and_query_bayes_net("data/clustered_dataset.csv")




if __name__ == "__main__":
    main()
