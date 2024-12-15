import os
from pyswip import Prolog

class PrologInterface:
    def __init__(self, kb_file):
        # Normalizza e sostituisce i backslash con forward slash
        self.kb_file = os.path.abspath(kb_file).replace("\\", "/")
        print(f"Percorso della knowledge base: {self.kb_file}")  # Debug del percorso

    def query_prolog(self):
        """
        Consulta la knowledge base di Prolog e gestisce le query.
        """
        prolog = Prolog()
        print(f"Consultazione della knowledge base: {self.kb_file}")
        try:
            # Passaggio del percorso formattato
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
