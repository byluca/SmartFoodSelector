import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def explain_model_shap(self, model, sample_data):
    # Check the type of the model to select the right explainer
    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, sample_data)
    else:
        # Fallback to KernelExplainer if model type is unknown
        explainer = shap.KernelExplainer(model.predict_proba, sample_data)

    # Calculate SHAP values
    shap_values = explainer.shap_values(sample_data)

    # Generate plots
    shap.summary_plot(shap_values, sample_data, plot_type="bar")
    shap.summary_plot(shap_values, sample_data)