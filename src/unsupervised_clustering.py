# src/unsupervised_clustering.py

import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans


class ClusteringManager:
    """
    Classe che si occupa dell'esecuzione del clustering dei dati utilizzando k-Means
    e dell'identificazione del valore ottimale di k tramite il 'Metodo del Gomito'.

    Attributes:
        data_file (str): Percorso del file CSV contenente i dati da clusterizzare.
        df (pd.DataFrame): DataFrame dei dati caricati da data_file.
    """

    def __init__(self, data_file):
        """
        Inizializza la classe caricando il dataset in un DataFrame.

        Args:
            data_file (str): Percorso del file CSV da utilizzare per il clustering.
        """
        self.data_file = data_file
        self.df = pd.read_csv(data_file)

    def find_optimal_k(self, max_k=10):
        """
        Trova il numero ottimale di cluster k, utilizzando l'inertia di k-Means e
        il 'Knee Locator' (metodo del gomito).

        Args:
            max_k (int, optional): Numero massimo di cluster da testare (default: 10).

        Returns:
            int: Il valore di k che rappresenta il "gomito" (elbow) della curva.
        """
        inertia = []
        Ks = range(1, max_k + 1)
        for k in Ks:
            kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
            kmeans.fit(self.df)
            inertia.append(kmeans.inertia_)

        kl = KneeLocator(Ks, inertia, curve='convex', direction='decreasing')
        elbow = kl.elbow

        # Plot del metodo del gomito (viene salvato in 'data/elbow_plot.png')
        plt.plot(Ks, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Metodo del gomito')
        plt.axvline(x=elbow, color='r', linestyle='--')
        plt.savefig('data/elbow_plot.png')
        plt.close()

        return elbow

    def assign_clusters(self, k, output_file):
        """
        Esegue k-Means con il numero di cluster specificato e salva il risultato su file.

        Args:
            k (int): Numero di cluster da utilizzare.
            output_file (str): Percorso del file CSV in cui salvare i dati con l'assegnazione
                dei cluster.
        """
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = kmeans.fit_predict(self.df)
        self.df['cluster'] = labels
        self.df.to_csv(output_file, index=False)
        print("Assegnazione cluster completata, salvato in:", output_file)


def plot_cluster_pie_chart(df, cluster_column='cluster'):
    """
    Crea un grafico a torta per visualizzare la distribuzione dei cluster in un DataFrame.

    Args:
        df (pd.DataFrame): Il DataFrame con i dati clusterizzati (deve contenere la colonna cluster).
        cluster_column (str, optional): Nome della colonna che contiene i cluster (default: 'cluster').

    Mostra un grafico a torta con le percentuali di ciascun cluster.
    """
    cluster_counts = df[cluster_column].value_counts()
    labels = cluster_counts.index
    sizes = cluster_counts.values
    colors = ['#66b3ff', '#99ff99', '#ff9999']  # Colori personalizzati
    explode = (0.05,) * len(labels)            # Leggera separazione per ogni fetta

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=colors, explode=explode)
    plt.title("Distribuzione dei Cluster")
    plt.legend(title="Cluster", loc="lower left")
    plt.show()
