import matplotlib.pyplot as plt
import shap

def plot_gam_pdp(gam, feature_names):
    """
    Partial dependence plots for GAM
    """
    for i, term in enumerate(gam.terms):
        if term.isintercept:
            continue
        XX = gam.generate_X_grid(term=i)
        plt.figure()
        plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
        plt.title(f'GAM Partial Dependence - {feature_names[i]}')
        plt.xlabel(feature_names[i])
        plt.ylabel('CH4')
        plt.show()

def plot_rf_shap(rf_model, X, feature_names):
    """
    SHAP summary plot for Random Forest
    """
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names)
