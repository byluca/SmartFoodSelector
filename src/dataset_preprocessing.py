# src/dataset_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.input_file, sep='\t', low_memory=False)
        return self

    def preprocess(self):
        desired_cols = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g']
        self.df = self.df[desired_cols].dropna()
        scaler = MinMaxScaler()
        self.df = pd.DataFrame(scaler.fit_transform(self.df), columns=self.df.columns)
        return self

    def save(self):
        self.df.to_csv(self.output_file, index=False)
        print("Preprocessing completato, dati salvati in:", self.output_file)
        return self
