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
2. Preprocess: handle missing values, scale features
3. Baseline: mean predictor
4. ML models: GAM, Random Forest, kNN
5. Evaluation: MAE, RMSE, R²
6. Interpretation: feature importance, partial dependence plots

## Results
- GAM gave the best tradeoff between interpretability and accuracy
- Random Forest overfits due to small data
- Simple baselines surprisingly competitive

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
