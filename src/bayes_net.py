import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

class BayesianNetworkBuilder:
    def __init__(self, data_file):
        self.data_file = data_file
        self.df = None
        self.model = None

    def load_and_discretize(self):
        df = pd.read_csv(self.data_file)
        cols = df.columns.drop('cluster')
        disc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
        df[cols] = disc.fit_transform(df[cols])
        df = df.astype(int)
        df.to_csv("data/discrete_dataset.csv", index=False)
        self.df = df
        return self

    def learn_structure(self, sample_size=2000):
        small_df = self.df.sample(sample_size, random_state=42)
        hc = HillClimbSearch(small_df)
        best_model = hc.estimate(scoring_method=BicScore(small_df))
        self.model = BayesianNetwork(best_model.edges())
        self.model.fit(small_df, estimator=MaximumLikelihoodEstimator)
        return self

    def query_network(self):
        cols = self.df.columns.drop('cluster')
        inference = VariableElimination(self.model)
        q = inference.query(variables=['cluster'], evidence={cols[0]: 1})
        print("Distribuzione P(cluster|{}=1):".format(cols[0]))
        print(q)
        return self
