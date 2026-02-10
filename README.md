# Methane Production Prediction Model for Biogas Plant Optimization

## Problem Statement
**Business Context:** The consulting company specializes in predicting the amount of methane that a client’s substrate may produce. This helps clients determine whether it is viable to process their waste on a large scale. For example, a pork processing plant may have leftover tissues and other organic waste but is unsure how to handle them. The plant can reach out to the consulting company to evaluate whether these byproducts can be converted into methane through a chemical process and if doing so is financially worthwhile.
Currently, the company runs small-scale tests on samples of the substrate to measure potential methane production. This process can take several weeks, requires significant manual labor and monitoring, and costs up to €200 per sample.

**Opportunity:** The substrate can be characterized using variables such as pH, alkalinity, and CO₂ composition. Some of these factors are known to correlate with methane production. By leveraging these correlations, we can develop a machine learning–based prediction model to estimate the methane yield of a given substrate, significantly reducing time, labor, and cost.

---

## Data Sources & Data Pipeline

The data powering the model comes from **two main sources**:

1. **Client Databases:**  
   Historical methane test results from multiple clients. These datasets required **standardization and cleaning** before being uploaded to a secure cloud storage (`S3`).

2. **Open and Research Data:**  
   Supplementary datasets from academic publications and publicly available sources, which helped increase model coverage and robustness.

Once collected, all datasets are integrated into an **S3 bucket**, then processed and cleaned locally before training models.

S3 Bucket → s3_utils.py → Local Processing → Model Training → Evaluation → notebooks/ (EDA, Modeling, Interpretation)

**Pipeline Overview:**

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

**Selected Model:** Random Forest, best accuracy; feature importance enables substrate characterization insights.

---

## Key Results & Business Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Time-to-prediction | 4-8 weeks | ~15 minutes | **~97%** reduction |
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
    - `train.py` - Training the models
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
