from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import scipy.stats as stats

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def evaluate_bayesian(y_true, pred_mean, pred_std, coverage=0.95):
    """
    Compute coverage rate of Bayesian predictive intervals.
    """
    lower = pred_mean - stats.norm.ppf(0.5 + coverage/2)*pred_std
    upper = pred_mean + stats.norm.ppf(0.5 + coverage/2)*pred_std
    coverage_rate = ((y_true >= lower) & (y_true <= upper)).mean()
    return coverage_rate
