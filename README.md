# Methane Production Prediction

## Problem
Metanogenia S.L. performs methane production experiments that take weeks to complete.  
Goal: Predict methane production from inexpensive measurements (pH, DQO, AGV, etc.) to save time and lab costs.

## Data
- Features: pH, DQO, DQO degradada, Alcalinidad, AGV, Solidos Totales, Solidos Volátiles, TRH  
- Target: CH4 production  
- Data Source: AWS S3 (`s3://bucket/methane_data.csv`)  
- Size: <100 samples (small dataset)

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
