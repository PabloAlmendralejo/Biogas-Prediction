import pymc3 as pm
import numpy as np

def train_bayesian_regression(X_train, y_train, samples=2000, tune=1000):
    """
    Train a Bayesian linear regression model using PyMC3.
    """
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        betas = pm.Normal('betas', mu=0, sigma=1, shape=X_train.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        mu = alpha + pm.math.dot(X_train, betas)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
        
        trace = pm.sample(samples, tune=tune, target_accept=0.95, cores=1)
    
    return model, trace

def predict_bayesian(model, trace, X):
    """
    Return predictive mean and std for Bayesian model
    """
    import pymc3 as pm
    posterior_pred = pm.sample_posterior_predictive(trace, model=model, var_names=['y_obs'])
    pred_mean = posterior_pred['y_obs'].mean(axis=0)
    pred_std = posterior_pred['y_obs'].std(axis=0)
    return pred_mean, pred_std
