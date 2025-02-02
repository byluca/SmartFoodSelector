# src/generate_prolog_knowledge_base.py

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def generate_prolog_kb(input_csv, output_pl, n_bins=5):
    """
    Genera una knowledge base Prolog a partire da un dataset clusterizzato,
    con discretizzazione delle features numeriche.

    Args:
        input_csv (str): Percorso del file CSV clusterizzato
        output_pl (str): Percorso del file .pl da generare
        n_bins (int): Numero di categorie per la discretizzazione
    """

    # Carica e discretizza i dati
    df = pd.read_csv(input_csv)
    features = ['energy_100g', 'fat_100g', 'carbohydrates_100g',
                'sugars_100g', 'proteins_100g', 'salt_100g']

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    df[features] = discretizer.fit_transform(df[features])

    # Genera la KB Prolog
    with open(output_pl, "w") as f:
        f.write("%% Knowledge Base generata automaticamente\n\n")
        f.write("%% Regole principali\n")
        f.write("product_cluster(E, F, C, S, P, Sa, Cluster) :-\n")
        f.write("    product(E, F, C, S, P, Sa),\n")
        f.write("    cluster_assignment(E, F, C, S, P, Sa, Cluster).\n\n")

        f.write("%% Fatti prodotti\n")
        for _, row in df.iterrows():
            vals = [str(int(row[col])) for col in features]  # Converti in interi
            f.write(f"product({','.join(vals)}).\n")

        f.write("\n%% Assegnazioni cluster\n")
        for _, row in df.iterrows():
            vals = [str(int(row[col])) for col in features]
            f.write(f"cluster_assignment({','.join(vals)},{int(row['cluster'])}).\n")


if __name__ == "__main__":
    generate_prolog_kb(
        input_csv="data/processed/discrete_dataset.csv",  # Aggiornato al nuovo nome
        output_pl="prolog/knowledge_base.pl",
        n_bins=5
    )
