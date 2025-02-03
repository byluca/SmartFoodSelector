# main.py (Aggiornato per la nuova struttura)
import pandas as pd
import matplotlib.pyplot as plt
import time
from src.dataset_preprocessing import DataPreprocessor
from src.unsupervised_clustering import ClusteringManager, plot_cluster_pie_chart
from src.supervised_models import SupervisedTrainer
from src.bayes_net import BayesianNetworkBuilder

# Percorsi aggiornati secondo la nuova struttura
RAW_DATA = "data/raw/en.openfoodfacts.org.products.tsv"
PREPROCESSED_DATA = "data/processed/Dataset.csv"
CLUSTERED_DATA = "data/results/clustered_dataset.csv"
PLOTS_DIR = "data/results/plots/"


def preprocess_data(input_file, output_file):
    """..."""  # Mantieni la docstring originale
    print("1) Preprocessing dei dati...")
    start = time.time()
    DataPreprocessor(input_file, output_file).load_data().preprocess().save()
    print(f"Preprocessing completato in {time.time() - start:.2f} secondi.\n")


def perform_clustering(input_file, output_file, elbow=3):
    """..."""  # Mantieni la docstring originale
    print("2) Clustering dei dati...")
    start = time.time()
    cluster_mgr = ClusteringManager(input_file)
    cluster_mgr.assign_clusters(elbow, output_file)
    print(f"Clustering completato in {time.time() - start:.2f} secondi.\n")


def plot_clustering_results(file_path, cluster_column='cluster'):
    """..."""  # Mantieni la docstring originale
    print("Generazione del grafico a torta per il clustering...")
    clustered_df = pd.read_csv(file_path)
    plot_cluster_pie_chart(clustered_df, cluster_column=cluster_column)


def train_and_evaluate_models(input_file):
    """..."""  # Mantieni la docstring originale
    print("3) Training e valutazione dei modelli...")
    start = time.time()
    trainer = SupervisedTrainer(input_file)
    trainer.load_data(sample_fraction=0.1).train_test_split().resample_data()
    models = trainer.train_models()

    results = trainer.evaluate_models_with_metrics(models)
    trainer.plot_model_performance(results)

    for name, model in models:
        print(f"Cross-validation per il modello {name}...")
        trainer.cross_validate_model(model)
        trainer.plot_learning_curve(model, title=name)

    print(f"Training e valutazione completati in {time.time() - start:.2f} secondi.\n")


def build_bayesian_network(input_file):
    """..."""  # Mantieni la docstring originale
    print("4) Creazione della rete bayesiana...")
    print("Creazione della rete bayesiana con valori continui...")
    bayesian_builder = BayesianNetworkBuilder(input_file)

    # Rete continua
    bayesian_builder.load_and_discretize(strategy=None).learn_structure_continuous()
    bayesian_builder.visualize_network_continuous(
        output_file=f"{PLOTS_DIR}bayesian_network_continuous.png"
    )

    print("Creazione della rete bayesiana con valori discreti...")
    # Rete discreta
    bayesian_builder.load_and_discretize(n_bins=5, strategy='uniform').learn_structure_discrete()
    bayesian_builder.visualize_network_discrete(
        output_file=f"{PLOTS_DIR}bayesian_network_discrete.png"
    )


def main():
    """..."""  # Mantieni la docstring originale
    print("Inizio del programma...")
    start_program = time.time()

    # Utilizza i percorsi costanti definiti sopra
    preprocess_data(RAW_DATA, PREPROCESSED_DATA)
    perform_clustering(PREPROCESSED_DATA, CLUSTERED_DATA)
    plot_clustering_results(CLUSTERED_DATA)
    train_and_evaluate_models(CLUSTERED_DATA)
    build_bayesian_network(CLUSTERED_DATA)

    print(f"Programma completato in {time.time() - start_program:.2f} secondi.")


if __name__ == "__main__":
    main()