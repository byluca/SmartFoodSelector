# src/supervised_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


def train_and_evaluate_supervised(data_file: str):
    df = pd.read_csv(data_file)
    # Ultima colonna è cluster
    X = df.drop('cluster', axis=1)
    y = df['cluster']

    # Check sbilanciamento
    print("Distribuzione classi prima di SMOTE:", np.bincount(y))

    # Oversampling
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    print("Distribuzione classi dopo SMOTE:", np.bincount(y_res))

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Grid search per DecisionTree
    dt_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    dt = DecisionTreeClassifier(random_state=42)
    dt_gs = GridSearchCV(dt, dt_params, cv=5, scoring='f1_macro')
    dt_gs.fit(X_train, y_train)
    print("Decision Tree best params:", dt_gs.best_params_)
    dt_best = dt_gs.best_estimator_

    # RandomForest
    rf_params = {
        'n_estimators': [50, 100],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10],
    }
    rf = RandomForestClassifier(random_state=42)
    rf_gs = GridSearchCV(rf, rf_params, cv=5, scoring='f1_macro')
    rf_gs.fit(X_train, y_train)
    print("Random Forest best params:", rf_gs.best_params_)
    rf_best = rf_gs.best_estimator_

    # Logistic Regression
    lr_params = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [100, 200]
    }
    lr = LogisticRegression(multi_class='multinomial', random_state=42)
    lr_gs = GridSearchCV(lr, lr_params, cv=5, scoring='f1_macro')
    lr_gs.fit(X_train, y_train)
    print("Logistic Regression best params:", lr_gs.best_params_)
    lr_best = lr_gs.best_estimator_

    # Valutazione finale
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    for model, name in [(dt_best, "Decision Tree"), (rf_best, "Random Forest"), (lr_best, "Logistic Regression")]:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        print(f"{name}: Accuracy={acc:.3f}, F1={f1:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")
