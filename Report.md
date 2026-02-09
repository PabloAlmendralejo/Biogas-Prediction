# Methane Production Prediction Model for Biogas Plant Optimization

## Problem Statement
**Business Context:** Consulting engagements require methane yield analysis that takes **weeks-to-months** per substrate through physical experiments, limiting throughput and client scalability.

**Opportunity:** Build an ML-powered prediction pipeline to estimate methane output in **near real-time**, reducing client time-to-insight and increasing engagement capacity.

---

## Architecture & Data Pipeline

S3 Bucket → s3_utils.py → Local Processing → Model Training → Evaluation → notebooks/ (EDA, Modeling, Interpretation)

| Component | Implementation |
|-----------|---------------|
| **Data Source** | S3 bucket via `load_csv_from_s3()` |
| **Preprocessing** | `preprocessing.py` — feature cleaning, imputation, scaling |
| **Modeling** | `models.py` + `bayesian_models.py` (RF, XGBoost, GAM, k-NN, SVR, KRR) |
| **Evaluation** | `evaluation.py` — RMSE, MAE, R² with cross-validation |
| **Visualization** | `visualization.py` — feature importance, residual plots |
| **Analysis** | Jupyter notebooks for EDA, experimentation, interpretation |

---

## Methodology & Model Performance

Benchmarked **6 algorithms** using 80/20 train-test split with 5-fold cross-validation:

| Model | RMSE | MAE | R² | 
|-------|------|-----|----| 
| **Random Forest** | 8.7 | 6.4 | 0.71 |
| **XGBoost** | 9.2 | 6.9 | 0.68 |
| **GAM** | 10.5 | 7.8 | 0.64 |
| **KRR** | 11.2 | 8.3 | 0.61 |
| **SVR** | 12.1 | 9.0 | 0.58 |
| **k-NN** | 13.4 | 10.2 | 0.52 |

**Selected Model:** Random Forest — best accuracy; feature importance enables substrate characterization insights.

---

## Key Results & Business Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Time-to-prediction | 4-8 weeks | ~15 minutes | **~97%** reduction |
| Cost per analysis | €2,500 | €85 | **~96%** reduction |
| Monthly throughput | 12 clients | ~45 clients | **~3.5x** increase |
| Prediction accuracy | N/A (baseline) | 71% R² | Suitable for screening |

**Note:** Model performs best on mid-range methane yields; accuracy drops ~18% on extreme values. Physical validation still recommended for final client deliverables.

---

## Repository Structure

- **repo/**
  - **src/**
    - `s3_utils.py` – S3 data loading
    - `preprocessing.py` – Feature engineering
    - `models.py` – ML models
    - `bayesian_models.py` – Bayesian approaches
    - `evaluation.py` – Metrics & validation
    - `visualization.py` – Plots & charts
  - **notebooks/**
    - `01_EDA.ipynb`
    - `02_Modeling.ipynb`
    - `03_Interpretation.ipynb`
  - `requirements.txt`
  - `.gitignore`
---

## Known Limitations & Risks
- **Feature gaps:** Current 8-feature set provides incomplete substrate characterization
- **Data quality:** ~12% of historical records required imputation
- **Generalization:** Model undertrained on industrial waste substrates

---

## Next Steps
1. **Feature Engineering:** Integrate additional substrate properties to improve R² target >0.80
2. **Data Enrichment:** Partner with 2-3 clients to expand training data
3. **Hybrid Workflow:** Use model for screening; route low-confidence predictions to physical testing
4. **Productionization:** Deploy as API endpoint for real-time inference

---

*Stack: Python | S3 | Pandas | Scikit-learn | XGBoost | PyGAM*
