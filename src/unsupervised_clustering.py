# src/unsupervised_clustering.py
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans


class ClusteringManager:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = pd.read_csv(data_file)

    def find_optimal_k(self, max_k=10):
        inertia = []
        Ks = range(1, max_k + 1)
        for k in Ks:
            kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
            kmeans.fit(self.df)
            inertia.append(kmeans.inertia_)

        kl = KneeLocator(Ks, inertia, curve='convex', direction='decreasing')
        elbow = kl.elbow

        # Plot
        plt.plot(Ks, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Metodo del gomito')
        plt.axvline(x=elbow, color='r', linestyle='--')
        plt.savefig('data/elbow_plot.png')
        plt.close()
        return elbow

    def assign_clusters(self, k, output_file):
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = kmeans.fit_predict(self.df)
        self.df['cluster'] = labels
        self.df.to_csv(output_file, index=False)
        print("Assegnazione cluster completata, salvato in:", output_file)

def plot_cluster_pie_chart(df, cluster_column='cluster'):
    """
    Crea un grafico a torta per visualizzare la distribuzione dei cluster.
    :param df: DataFrame con i dati clusterizzati.
    :param cluster_column: Nome della colonna che contiene i cluster.
    """
    cluster_counts = df[cluster_column].value_counts()
    labels = cluster_counts.index
    sizes = cluster_counts.values
    colors = ['#66b3ff', '#99ff99', '#ff9999']  # Personalizza i colori se desideri
    explode = (0.05, 0.05, 0.05)  # Leggera separazione tra le fette

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=explode)
    plt.title("Distribuzione dei Cluster")
    plt.legend(title="Cluster", loc="lower left")
    plt.show()