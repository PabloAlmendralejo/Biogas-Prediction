# Methane Production Prediction

## Problem
To estimate the amount of methane that a substrate may produce a methane production experiment has to be performed, that take weeks to complete.  
Goal: Predict methane production from substrate measurements (pH, DQO, AGV, etc.) to save time and lab costs.

## Data
- Features: pH, DQO, degradated DQO , Alcalinity, AGV, Total Solids, Volatile Solids, TRH  
- Target: CH4 production  
- Data Source: AWS S3

## Approach
1. Load data from S3
2. Preprocess: handle missing values, scaling, polynomial features
3. Train models:
   - Baseline (mean predictor)
   - GAM (Generalized Additive Model)
   - Random Forest
   - kNN
   - Bayesian linear regression
4. Evaluate models using MAE, RMSE, R², coverage (Bayesian)
5. Interpret using PDP (GAM) and SHAP (RF)

## Results
- GAM provided best interpretability vs accuracy
- Random Forest overfits due to small data
- Bayesian regression allows risk-aware predictions with predictive intervals
- Mean predictor surprisingly competitive in some cases

## Business Impact
- Reduce experiment duration by predicting methane production
- Save lab resources and cost
- Prioritize experiments with high predicted methane yield

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
