import pandas as pd  # Import necessario per leggere il dataset
import matplotlib
matplotlib.use('TkAgg')  # Backend per mostrare i grafici
import matplotlib.pyplot as plt
import time  # Per monitorare i tempi di esecuzione
from src.dataset_preprocessing import DataPreprocessor
from src.unsupervised_clustering import ClusteringManager, plot_cluster_pie_chart
from src.supervised_models import SupervisedTrainer
from src.bayes_net import BayesianNetworkBuilder
from src.prolog_interface import PrologInterface

def main():
    print("Inizio del programma...")
    start_program = time.time()

    # 1) Preprocessing dei dati
    print("1) Preprocessing dei dati...")
    start = time.time()
    DataPreprocessor("data/en.openfoodfacts.org.products.tsv", "data/newDataset.csv") \
        .load_data() \
        .preprocess() \
        .save()
    print(f"Preprocessing completato in {time.time() - start:.2f} secondi.\n")

    # 2) Clustering
    print("2) Clustering dei dati...")
    start = time.time()
    cluster_mgr = ClusteringManager("data/newDataset.csv")
    elbow = 3  # Numero di cluster fisso per ridurre i tempi
    cluster_mgr.assign_clusters(elbow, "data/clustered_dataset.csv")
    print(f"Clustering completato in {time.time() - start:.2f} secondi.\n")

    # Carica il dataset clusterizzato per visualizzare il grafico a torta
    print("Generazione del grafico a torta per il clustering...")
    clustered_df = pd.read_csv("data/clustered_dataset.csv")
    plot_cluster_pie_chart(clustered_df, cluster_column='cluster')

    # 3) Supervised Learning
    print("3) Training e valutazione dei modelli...")
    start = time.time()
    trainer = SupervisedTrainer("data/clustered_dataset.csv")
    trainer.load_data(sample_fraction=0.05).train_test_split().resample_data()
    models = trainer.train_models()

    # Valutazione dei modelli
    results = trainer.evaluate_models_with_metrics(models)
    trainer.plot_model_performance(results)  # Grafico delle metriche

    # Cross-validation per ogni modello
    for name, model in models:
        print(f"Cross-validation per il modello {name}...")
        trainer.cross_validate_model(model)

    print(f"Training e valutazione completati in {time.time() - start:.2f} secondi.\n")

    # 4) Bayesian Network
    print("4) Creazione della rete bayesiana...")
    start = time.time()
    BayesianNetworkBuilder("data/clustered_dataset.csv") \
        .load_and_discretize() \
        .learn_structure() \
        .query_network()
    print(f"Rete bayesiana completata in {time.time() - start:.2f} secondi.\n")

    # 5) Interfaccia Prolog
    print("5) Query Prolog...")
    start = time.time()
    PrologInterface("prolog/knowledge_base.pl").query_prolog()
    print(f"Query Prolog completata in {time.time() - start:.2f} secondi.\n")

    print(f"Programma completato in {time.time() - start_program:.2f} secondi.")

if __name__ == "__main__":
    main()
