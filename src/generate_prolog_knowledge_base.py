# src/generate_prolog_knowledge_base.py

import pandas as pd

# Nota: Questo script genera il file knowledge_base.pl da un file CSV con clustering.
#       Vengono definiti fatti (product) e regole (clustered_product) per Prolog.

df = pd.read_csv("data/clustered_dataset.csv")
with open("prolog/knowledge_base.pl", "w") as f:
    f.write("% Prolog knowledge base generata automaticamente\n\n")
    # Scriviamo la definizione della regola
    f.write("product_info(E,F,C,Su,P,Sa,Cl) :-\n")
    f.write("    product(E,F,C,Su,P,Sa),\n")
    f.write("    clustered_product(E,F,C,Su,P,Sa,Cl).\n\n")

    # Iteriamo sulle righe del dataset e scriviamo i fatti 'product(...)'
    for idx, row in df.iterrows():
        E = row['energy_100g']
        F = row['fat_100g']
        C = row['carbohydrates_100g']
        Su = row['sugars_100g']
        P = row['proteins_100g']
        Sa = row['salt_100g']
        Cl = row['cluster']
        f.write(f"product({E},{F},{C},{Su},{P},{Sa}).\n")

    f.write("\n")
    # Iteriamo nuovamente per i fatti 'clustered_product(...)'
    for idx, row in df.iterrows():
        E = row['energy_100g']
        F = row['fat_100g']
        C = row['carbohydrates_100g']
        Su = row['sugars_100g']
        P = row['proteins_100g']
        Sa = row['salt_100g']
        Cl = row['cluster']
        f.write(f"clustered_product({E},{F},{C},{Su},{P},{Sa},{Cl}).\n")
