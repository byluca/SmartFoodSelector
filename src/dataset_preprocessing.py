# src/dataset_preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(input_file: str, output_file: str):
    # Carica il dataset TSV con Pandas
    df = pd.read_csv(input_file, sep='\t', low_memory=False)

    # Seleziona un sottoinsieme di feature numeriche per esempio:
    # energy_100g, fat_100g, carbohydrates_100g, sugars_100g, proteins_100g, salt_100g
    # Ci assicuriamo che le colonne esistano
    desired_cols = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g']
    df = df[desired_cols]

    # Rimuoviamo righe con troppi NaN
    df = df.dropna()

    # Normalizzazione tra 0 e 1 con MinMax
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Salviamo il dataset preprocessato
    df_norm.to_csv(output_file, index=False)
    print("Preprocessing completato, dati salvati in:", output_file)
