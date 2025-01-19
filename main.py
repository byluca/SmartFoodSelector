# main.py (Esempio di "script" principale che coordina i vari step)

import pandas as pd
import matplotlib.pyplot as plt
import time
from src.dataset_preprocessing import DataPreprocessor
from src.unsupervised_clustering import ClusteringManager, plot_cluster_pie_chart
from src.supervised_models import SupervisedTrainer
from src.bayes_net import BayesianNetworkBuilder


def preprocess_data(input_file, output_file):
    """
    Preprocessa i dati caricando il file, selezionando e pulendo le feature,
    e normalizzandole tramite MinMaxScaler.

    Args:
        input_file (str): Percorso del file originale (in formato .tsv, ad es. openfoodfacts).
        output_file (str): Percorso del file CSV su cui salvare il dataset preprocessato.
    """
    print("1) Preprocessing dei dati...")
    start = time.time()
    DataPreprocessor(input_file, output_file).load_data().preprocess().save()
    print(f"Preprocessing completato in {time.time() - start:.2f} secondi.\n")


def perform_clustering(input_file, output_file, elbow=3):
    """
    Esegue il clustering k-Means sul dataset preprocessato, partendo da un valore
    suggerito di k (elbow).

    Args:
        input_file (str): Percorso del file CSV preprocessato.
        output_file (str): Percorso del file CSV dove salvare i dati con i cluster assegnati.
        elbow (int, optional): Numero di cluster da utilizzare, di default 3.
    """
    print("2) Clustering dei dati...")
    start = time.time()
    cluster_mgr = ClusteringManager(input_file)
    cluster_mgr.assign_clusters(elbow, output_file)
    print(f"Clustering completato in {time.time() - start:.2f} secondi.\n")


def plot_clustering_results(file_path, cluster_column='cluster'):
    """
    Genera un grafico a torta per visualizzare la distribuzione dei cluster.

    Args:
        file_path (str): Percorso del file CSV contenente i dati clusterizzati.
        cluster_column (str, optional): Nome della colonna dei cluster (default: 'cluster').
    """
    print("Generazione del grafico a torta per il clustering...")
    clustered_df = pd.read_csv(file_path)
    plot_cluster_pie_chart(clustered_df, cluster_column=cluster_column)


def train_and_evaluate_models(input_file):
    """
    Carica i dati clusterizzati, allena tre modelli di classificazione (Decision Tree,
    Random Forest, Logistic Regression) e ne valuta le prestazioni tramite metriche e
    learning curve.

    Args:
        input_file (str): Percorso del file CSV contenente i dati clusterizzati.
    """
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
    """
    Crea due varianti di Rete Bayesiana (continua e discreta) a partire dal dataset
    clusterizzato, visualizzandole e salvando i relativi grafici su file.

    Args:
        input_file (str): Percorso del file CSV clusterizzato.
    """
    print("4) Creazione della rete bayesiana...")
    print("Creazione della rete bayesiana con valori continui...")
    bayesian_builder = BayesianNetworkBuilder(input_file)
    # Rete con valori continui (strategy=None)
    bayesian_builder.load_and_discretize(strategy=None).learn_structure_continuous()
    bayesian_builder.visualize_network_continuous(output_file="data/bayesian_network_continuous.png")

    print("Creazione della rete bayesiana con valori discreti...")
    # Rete con valori discretizzati
    bayesian_builder.load_and_discretize(n_bins=5, strategy='uniform').learn_structure_discrete()
    bayesian_builder.visualize_network_discrete(output_file="data/bayesian_network_discrete.png")

    # Se necessario, si potrebbe aggiungere la stima dei parametri (CPD) o inferenze.


def main():
    """
    Funzione principale che coordina tutti gli step del flusso:
    1) Preprocessing
    2) Clustering
    3) Visualizzazione cluster
    4) Training e valutazione modelli
    5) Creazione Rete Bayesiana.
    """
    print("Inizio del programma...")
    start_program = time.time()

    input_file = "data/en.openfoodfacts.org.products.tsv"
    preprocessed_file = "data/newDataset.csv"
    clustered_file = "data/clustered_dataset.csv"

    # 1. Preprocessing
    preprocess_data(input_file, preprocessed_file)

    # 2. Clustering
    perform_clustering(preprocessed_file, clustered_file)

    # Visualizzazione dei risultati del clustering
    plot_clustering_results(clustered_file)

    # 3. Training e valutazione modelli supervisionati
    train_and_evaluate_models(clustered_file)

    # 4. Creazione rete bayesiana (continua e discreta)
    build_bayesian_network(clustered_file)

    print(f"Programma completato in {time.time() - start_program:.2f} secondi.")


if __name__ == "__main__":
    main()
