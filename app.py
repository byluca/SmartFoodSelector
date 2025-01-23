# app.py (versione semplificata)
from flask import Flask, render_template, session
from src.dataset_preprocessing import DataPreprocessor
from src.unsupervised_clustering import ClusteringManager
from src.supervised_models import SupervisedTrainer
from src.bayes_net import BayesianNetworkBuilder
import os
import time

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['STATIC_FOLDER'] = 'static'

RAW_DATA = "data/raw/en.openfoodfacts.org.products.tsv"
PREPROCESSED_DATA = "data/processed/newDataset.csv"
CLUSTERED_DATA = "data/results/clustered_dataset.csv"
PLOT_PATH = "data/results/plots/"


@app.route('/')
def index():
    # Esegue automaticamente tutto il pipeline
    if not os.path.exists(PREPROCESSED_DATA):
        # Step 1: Preprocessing
        DataPreprocessor(RAW_DATA, PREPROCESSED_DATA).load_data().preprocess().save()
        time.sleep(1)

    if not os.path.exists(CLUSTERED_DATA):
        # Step 2: Clustering
        cluster_mgr = ClusteringManager(PREPROCESSED_DATA)
        cluster_mgr.assign_clusters(3, CLUSTERED_DATA)
        time.sleep(1)

    # Step 3: Addestramento modelli
    trainer = SupervisedTrainer(CLUSTERED_DATA)
    trainer.load_data().train_test_split().resample_data()
    models = trainer.train_models()
    results = trainer.evaluate_models_with_metrics(models)

    # Step 4: Rete Bayesiana
    bayesian_builder = BayesianNetworkBuilder(CLUSTERED_DATA)
    bayesian_builder.load_and_discretize(strategy=None