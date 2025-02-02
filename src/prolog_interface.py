# src/prolog_interface.py

import os
from pyswip import Prolog
from typing import List, Dict


class PrologInterface:
    def __init__(self, kb_file: str):
        self.kb_file = os.path.abspath(kb_file)
        self._validate_kb_file()

    def _validate_kb_file(self):
        """Esegue controlli preliminari sul file della KB"""
        if not os.path.exists(self.kb_file):
            raise FileNotFoundError(f"File {self.kb_file} non trovato")

        if os.path.getsize(self.kb_file) < 100:
            raise ValueError("La knowledge base sembra vuota o non valida")

    def query(self, query_str: str) -> List[Dict]:
        """Esegue una query generica con gestione degli errori"""
        prolog = Prolog()

        try:
            prolog.consult(self.kb_file)
            return list(prolog.query(query_str))
        except Exception as e:
            raise PrologError(f"Errore Prolog: {str(e)}")

    def get_products_in_cluster(self, cluster_id: int) -> List[Dict]:
        """Query specifica per prodotti in un cluster"""
        query = f"product_cluster(E, F, C, S, P, Sa, {cluster_id})"
        return self.query(query)

    def get_cluster_stats(self) -> Dict:
        """Restituisce statistiche sui cluster"""
        stats = {}
        clusters = set(self.query("cluster_assignment(_, _, _, _, _, _, Cluster)"))

        for cluster in clusters:
            count = len(self.get_products_in_cluster(cluster['Cluster']))
            stats[cluster['Cluster']] = {
                'count': count,
                'features': self._get_cluster_features(cluster['Cluster'])
            }
        return stats

    def _get_cluster_features(self, cluster_id: int) -> Dict:
        """Calcola le medie delle features per cluster"""
        # Implementa la logica per aggregare le features
        # usando le query Prolog