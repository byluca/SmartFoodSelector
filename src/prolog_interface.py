# src/prolog_interface.py

import os
from pyswip import Prolog


class PrologInterface:
    """
    Classe che fornisce un'interfaccia per caricare e interrogare la knowledge base di Prolog.
    Utilizza la libreria pyswip.

    Attributes:
        kb_file (str): Percorso assoluto del file Prolog contenente la knowledge base.
    """

    def __init__(self, kb_file):
        """
        Inizializza la classe con il percorso del file di knowledge base Prolog.

        Args:
            kb_file (str): Percorso del file .pl contenente la knowledge base.
        """
        self.kb_file = os.path.abspath(kb_file).replace("\\", "/")
        print(f"Percorso della knowledge base: {self.kb_file}")  # Debug del percorso

    def query_prolog(self):
        """
        Consulta la knowledge base di Prolog e gestisce una query di test per verificarne
        il corretto caricamento.

        Viene utilizzata la query: member(X, [a, b, c]) come esempio.
        In un progetto reale, si potrebbe arricchire la parte di query e l'interfaccia
        di input/output.

        Raises:
            Exception: Se si verifica un errore durante la consultazione del file Prolog.
        """
        prolog = Prolog()
        print(f"Consultazione della knowledge base: {self.kb_file}")
        try:
            prolog.consult(self.kb_file)
            print("Knowledge base consultata con successo.")
        except Exception as e:
            print(f"Errore durante la consultazione della knowledge base: {e}")
            raise

        # Query di test per verificare la consultazione
        try:
            for result in prolog.query("member(X, [a, b, c])"):
                print(f"Risultato query di test: {result}")
        except Exception as e:
            print(f"Errore durante l'esecuzione della query: {e}")