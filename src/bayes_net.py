# -*- coding: utf-8 -*-

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.preprocessing import KBinsDiscretizer


class BayesianNetworkBuilder:
    """
    Questa classe gestisce la creazione di una Rete Bayesiana (Bayesian Network)
    sia in forma continua che discreta, a partire da un dataset.

    Attributes:
        data_file (str): Percorso del file CSV contenente il dataset.
        model (BayesianNetwork): Istanza della rete bayesiana appresa.
        data (pd.DataFrame): Il DataFrame dei dati (eventualmente discretizzati).
    """

    def __init__(self, data_file):
        """
        Inizializza la classe con il percorso del dataset da cui apprendere la rete.

        Args:
            data_file (str): Percorso del file CSV contenente il dataset da utilizzare.
        """
        self.data_file = data_file
        self.model = None
        self.data = None

    def load_and_discretize(self, n_bins=5, strategy='uniform'):
        """
        Carica i dati dal file e, se specificato, li discretizza in base ai parametri forniti.

        Args:
            n_bins (int, optional): Numero di bin per la discretizzazione (default: 5).
            strategy (str or None, optional): Strategia di discretizzazione
                ('uniform', 'quantile', 'kmeans') o None per utilizzare i dati continui
                senza discretizzazione.
                - 'uniform': suddivide i valori in bin di ampiezza uguale.
                - 'quantile': suddivide i valori in base alle quantili.
                - 'kmeans': utilizza l'algoritmo k-means per individuare i bin.
                - None: non viene effettuata alcuna discretizzazione (valori continui).

        Returns:
            BayesianNetworkBuilder: Restituisce l'oggetto stesso per consentire il chaining
            dei metodi (pattern builder).
        """
        print("Caricamento e preparazione dei dati...")
        df = pd.read_csv(self.data_file)
        if strategy:
            # Discretizzazione
            print(f"Discretizzazione dei dati con {n_bins} bin e strategia '{strategy}'...")
            discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
            cols_to_discretize = df.columns
            df[cols_to_discretize] = discretizer.fit_transform(df[cols_to_discretize])
            print("Discretizzazione completata!")
        else:
            print("Utilizzo dei dati continui senza discretizzazione...")

        self.data = df
        return self

    def learn_structure_continuous(self):
        """
        Apprende la struttura della rete bayesiana con dati considerati come continui.

        Utilizza l'algoritmo Hill Climb Search e lo score BIC (Bayesian Information Criterion)
        per cercare la struttura di DAG (Directed Acyclic Graph) che massimizza lo score.

        Returns:
            BayesianNetworkBuilder: Restituisce l'oggetto stesso per concatenare eventuali
            chiamate successive.
        """
        print("Apprendimento della struttura della rete bayesiana (valori continui)...")
        hc = HillClimbSearch(self.data)
        dag = hc.estimate(scoring_method=BicScore(self.data))
        self.model = BayesianNetwork(dag.edges())
        print("Struttura della rete (valori continui) appresa con successo!")
        return self

    def learn_structure_discrete(self):
        """
        Apprende la struttura della rete bayesiana con dati considerati discreti.

        Analogamente al caso continuo, utilizza HillClimbSearch e BicScore per la stima
        della miglior struttura, ma presuppone che le variabili siano già discretizzate.

        Returns:
            BayesianNetworkBuilder: Restituisce l'oggetto stesso per concatenare eventuali
            chiamate successive.
        """
        print("Apprendimento della struttura della rete bayesiana (valori discreti)...")
        hc = HillClimbSearch(self.data)
        dag = hc.estimate(scoring_method=BicScore(self.data))
        self.model = BayesianNetwork(dag.edges())
        print("Struttura della rete (valori discreti) appresa con successo!")
        return self

    def visualize_network_continuous(self, output_file="data/bayesian_network_continuous.png"):
        """
        Genera un grafico della rete bayesiana appresa con valori continui e lo salva su file.

        Args:
            output_file (str, optional): Percorso del file in cui salvare il grafico (default:
                'data/bayesian_network_continuous.png').

        Raises:
            ValueError: Se la rete bayesiana non è stata costruita prima di chiamare
            questa funzione.
        """
        if not self.model:
            raise ValueError("La rete bayesiana non è stata costruita.")
        print("Generazione del grafico della rete bayesiana (valori continui)...")
        G = nx.DiGraph()
        G.add_edges_from(self.model.edges())
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue",
                font_size=10, font_weight="bold", arrowsize=20)
        plt.title("Rete Bayesiana (Valori Continui)")
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Grafico salvato in: {output_file}")
        plt.show()

    def visualize_network_discrete(self, output_file="data/bayesian_network_discrete.png"):
        """
        Genera un grafico della rete bayesiana appresa con valori discreti e lo salva su file.

        Args:
            output_file (str, optional): Percorso del file in cui salvare il grafico (default:
                'data/bayesian_network_discrete.png').

        Raises:
            ValueError: Se la rete bayesiana non è stata costruita prima di chiamare
            questa funzione.
        """
        if not self.model:
            raise ValueError("La rete bayesiana non è stata costruita.")
        print("Generazione del grafico della rete bayesiana (valori discreti)...")
        G = nx.DiGraph()
        G.add_edges_from(self.model.edges())
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightgreen",
                font_size=10, font_weight="bold", arrowsize=20)
        plt.title("Rete Bayesiana (Valori Discreti)")
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Grafico salvato in: {output_file}")
        plt.show()