# ------------------------------------------------------------------------------
# src/dataset_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    """
    Classe responsabile del caricamento, della pulizia e della normalizzazione dei dati
    prima dell'analisi.

    Attributes:
        input_file (str): Percorso del file di input (tipicamente TSV o CSV).
        output_file (str): Percorso del file di output dove salvare il dataset preprocessato.
        df (pd.DataFrame): Il DataFrame caricato e preprocessato.
    """

    def __init__(self, input_file, output_file):
        """
        Inizializza la classe con i percorsi dei file di input e di output.

        Args:
            input_file (str): Percorso del file originale da cui caricare i dati.
            output_file (str): Percorso del file dove salvare i dati preprocessati.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.df = None

    def load_data(self):
        """
        Carica i dati dal file di input in un DataFrame.

        Returns:
            DataPreprocessor: L'istanza corrente, per concatenare i metodi.
        """
        self.df = pd.read_csv(self.input_file, sep='\t', low_memory=False)
        return self

    def preprocess(self):
        """
        Effettua la selezione delle colonne desiderate, la rimozione dei NaN
        e la normalizzazione tramite MinMaxScaler.

        Al momento, vengono mantenute solo le seguenti colonne:
        ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g',
         'proteins_100g', 'salt_100g'].

        Returns:
            DataPreprocessor: L'istanza corrente, per concatenare i metodi.
        """
        desired_cols = ['energy_100g', 'fat_100g', 'carbohydrates_100g',
                        'sugars_100g', 'proteins_100g', 'salt_100g']
        self.df = self.df[desired_cols].dropna()
        scaler = MinMaxScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)
        return self

    def save(self):
        """
        Salva il DataFrame preprocessato su disco nel percorso specificato.

        Returns:
            DataPreprocessor: L'istanza corrente, per concatenare i metodi.
        """
        self.df.to_csv(self.output_file, index=False)
        print("Preprocessing completato, dati salvati in:", self.output_file)
        return self
