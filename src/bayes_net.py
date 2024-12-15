# src/bayes_net.py
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination


def build_and_query_bayes_net(data_file: str):
    df = pd.read_csv(data_file)
    # Discretizzazione per BN, ad esempio dividiamo ogni feature in 3 bin
    cols = df.columns.drop('cluster')
    disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    df[cols] = disc.fit_transform(df[cols])
    df = df.astype(int)
    df.to_csv("data/discrete_dataset.csv", index=False)

    # Apprendere la struttura (può essere costoso)
    # limitarsi a un sottoinsieme di dati per esempio:
    small_df = df.sample(2000, random_state=42)

    hc = HillClimbSearch(small_df)
    best_model = hc.estimate(scoring_method=BicScore(small_df))

    model = BayesianModel(best_model.edges())
    model.fit(small_df, estimator=MaximumLikelihoodEstimator)

    # Inferenza
    inference = VariableElimination(model)
    # Query di esempio: P(cluster|prima feature=1)
    q = inference.query(variables=['cluster'], evidence={cols[0]: 1})
    print("Distribuzione P(cluster|{}=1):".format(cols[0]), q['cluster'])
