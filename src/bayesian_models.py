import pymc3 as pm
import numpy as np

def train_bayesian_regression(X_train, y_train, samples=2000, tune=1000):

    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        betas = pm.Normal('betas', mu=0, sigma=1, shape=X_train.shape[1])
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Linear model
        mu = alpha + pm.math.dot(X_train, betas)
        
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_train)
        
        # Sample
        idata = pm.sample(
            samples, 
            tune=tune, 
            target_accept=0.95,
            return_inferencedata=True,
            cores=1
        )
    
    return model, idata

def predict_bayesian(model, idata, X_new):

    with model:
        pm.set_data({'X': X_new})  # Requires model to use pm.Data
        ppc = pm.sample_posterior_predictive(idata, var_names=['y_obs'])
    
    pred_mean = ppc.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values
    pred_std = ppc.posterior_predictive['y_obs'].std(dim=['chain', 'draw']).values
    
    return pred_mean, pred_std
