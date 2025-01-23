import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import optuna
import shap
import plotly.express as px
import json
import os


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

    def optimize_hyperparameters(self, n_trials=10):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
            model = RandomForestClassifier(**params, random_state=42)
            return cross_val_score(model, self.X_train, self.y_train,
                                   scoring='f1_macro', cv=5).mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        trials_df = study.trials_dataframe()
        trials_df.to_csv("data/results/hyperparameter_trials.csv", index=False)

        return study.best_params

    def train_models(self):
        best_params = self.optimize_hyperparameters()

        # Definisci la lista dei modelli
        models = [
            ("Optimized Random Forest", RandomForestClassifier(**best_params, random_state=42)),
            ("Decision Tree", DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5)),
            ("Logistic Regression", LogisticRegression(random_state=42, max_iter=200))
        ]

        # Addestra tutti i modelli
        for name, model in models:
            model.fit(self.X_train, self.y_train)

        return models  # Restituisci la lista dei modelli addestrati

    def evaluate_models_with_metrics(self, models):
        results = []
        for name, model in models:
            y_pred = model.predict(self.X_test)

            self.plot_feature_importance(model, "data/results/plots")
            self.explain_model_shap(model, self.X_test.sample(50, random_state=42))

            results.append({
                "Modello": name,
                "Accuracy": accuracy_score(self.y_test, y_pred),
                "F1": f1_score(self.y_test, y_pred, average='macro'),
                "Precision": precision_score(self.y_test, y_pred, average='macro'),
                "Recall": recall_score(self.y_test, y_pred, average='macro')
            })

        self.save_metrics_to_dvc(results)
        return results

    def plot_feature_importance(self, model, output_path):
        # Crea la directory se non esiste
        os.makedirs(output_path, exist_ok=True)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return

        sorted_idx = np.argsort(importances)[::-1]

        # Static plot
        plt.figure(figsize=(10, 6))
        plt.barh(np.array(self.X.columns)[sorted_idx], importances[sorted_idx])
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{output_path}/feature_importance.png")
        plt.close()

        # Interactive plot
        fig = px.bar(x=importances[sorted_idx],
                     y=np.array(self.X.columns)[sorted_idx],
                     orientation='h',
                     title="Feature Importance",
                     labels={'x': 'Importance', 'y': 'Feature'})
        fig.write_html(f"{output_path}/feature_importance_interactive.html")

    def explain_model_shap(self, model, X_sample):
        # Seleziona l'explainer appropriato in base al tipo di modello
        try:
            if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            elif isinstance(model, LogisticRegression):
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_sample)
                shap_values = explainer.shap_values(X_sample)

            # Genera il plot solo se i valori SHAP sono validi
            plt.figure()
            if isinstance(model, LogisticRegression):
                shap.summary_plot(shap_values, X_sample, feature_names=self.X.columns, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, show=False)

            plt.savefig("data/results/plots/shap_summary.png", bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Errore nella spiegazione SHAP per {type(model).__name__}: {str(e)}")

    def cross_validate_model(self, model, cv=5):
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1_macro')
        print(f"F1 scores: {scores}")
        print(f"Mean F1: {np.mean(scores):.4f}")
        print(f"Std Dev: {np.std(scores):.4f}\n")

    def plot_model_performance(self, results):
        df = pd.DataFrame(results)
        metrics = ["Accuracy", "F1", "Precision", "Recall"]

        plt.figure(figsize=(12, 8))
        sns.barplot(data=df.melt(id_vars="Modello", value_vars=metrics),
                    x="Modello", y="value", hue="variable")
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("data/results/plots/model_performance_comparison.png")
        plt.close()

    def plot_learning_curve(self, model, title, cv=5, scoring="f1_macro",
                            train_sizes=np.linspace(0.1, 1.0, 5)):
        train_sizes, train_scores, test_scores = learning_curve(
            model, self.X_train, self.y_train, cv=cv, scoring=scoring,
            n_jobs=-1, train_sizes=train_sizes
        )

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training Score")
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Validation Score")
        plt.title(f"Learning Curve - {title}")
        plt.xlabel("Training Examples")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"data/results/plots/learning_curve_{title.lower().replace(' ', '_')}.png")
        plt.close()

    def save_metrics_to_dvc(self, metrics):
        with open("data/results/model_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

        os.system