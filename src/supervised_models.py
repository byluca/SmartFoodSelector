import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


class SupervisedTrainer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, sample_fraction=0.1):
        df = pd.read_csv(self.data_file)
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"Caricati {len(df)} campioni su {len(pd.read_csv(self.data_file))} totali")
        self.X = df.drop('cluster', axis=1)
        self.y = df['cluster']
        return self

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return self

    def resample_data(self):
        sm = SMOTE(random_state=42)
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
        return self

    def train_models(self):
        dt = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)
        rf = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10)
        lr = LogisticRegression(random_state=42, max_iter=200)

        dt.fit(self.X_train, self.y_train)
        rf.fit(self.X_train, self.y_train)
        lr.fit(self.X_train, self.y_train)

        return [("Decision Tree", dt), ("Random Forest", rf), ("Logistic Regression", lr)]

    def evaluate_models_with_metrics(self, models):
        results = []
        for name, model in models:
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='macro')
            precision = precision_score(self.y_test, y_pred, average='macro')
            recall = recall_score(self.y_test, y_pred, average='macro')

            results.append({
                "Modello": name,
                "Accuracy": accuracy,
                "F1": f1,
                "Precision": precision,
                "Recall": recall
            })

        return results

    def plot_model_performance(self, results):
        df = pd.DataFrame(results)
        metrics = ["Accuracy", "F1", "Precision", "Recall"]
        df_melted = df.melt(id_vars="Modello", value_vars=metrics, var_name="Metrica", value_name="Valore")

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Modello", y="Valore", hue="Metrica", data=df_melted)
        plt.title("Confronto delle Metriche dei Modelli")
        plt.xticks(rotation=45)
        plt.ylabel("Valore")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("data/model_performance_comparison.png")
        print("Grafico salvato in: data/model_performance_comparison.png")
        plt.show()
