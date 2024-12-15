# src/generate_prolog_knowledge_base.py
import pandas as pd

df = pd.read_csv("data/clustered_dataset.csv")
with open("prolog/knowledge_base.pl", "w") as f:
    f.write("% Prolog knowledge base generata automaticamente\n\n")
    # Scriviamo la definizione della regola
    f.write("product_info(E,F,C,Su,P,Sa,Cl) :-\n")
    f.write("    product(E,F,C,Su,P,Sa),\n")
    f.write("    clustered_product(E,F,C,Su,P,Sa,Cl).\n\n")

    # Iteriamo sulle righe del dataset
    for idx, row in df.iterrows():
        E = row['energy_100g']
        F = row['fat_100g']
        C = row['carbohydrates_100g']
        Su = row['sugars_100g']
        P = row['proteins_100g']
        Sa = row['salt_100g']
        Cl = row['cluster']

        # Scriviamo product(...)
        f.write(f"product({E},{F},{C},{Su},{P},{Sa}).\n")

    f.write("\n")
    for idx, row in df.iterrows():
        E = row['energy_100g']
        F = row['fat_100g']
        C = row['carbohydrates_100g']
        Su = row['sugars_100g']
        P = row['proteins_100g']
        Sa = row['salt_100g']
        Cl = row['cluster']

        # Scriviamo clustered_product(...)
        f.write(f"clustered_product({E},{F},{C},{Su},{P},{Sa},{Cl}).\n")
