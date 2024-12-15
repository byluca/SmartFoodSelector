# src/unsupervised_clustering.py
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator

def find_optimal_k(data_file: str, max_k=10):
    df = pd.read_csv(data_file)
    inertia = []
    Ks = range(1, max_k+1)
    for k in Ks:
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)
    # Individua gomito
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

def assign_clusters(data_file: str, k: int, output_file: str):
    df = pd.read_csv(data_file)
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = kmeans.fit_predict(df)
    df['cluster'] = labels
    df.to_csv(output_file, index=False)
    print("Assegnazione cluster completata, salvato in:", output_file)
