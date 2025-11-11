# Results Reproducibility Guide

This document provides step-by-step instructions to reproduce all results, models, and visualizations in this project.

## 🎯 Quick Start: Full Reproducibility

```bash
# 1. Clone repository
git clone https://github.com/noahgallagher1/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Set up environment (Python 3.10+ required)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run complete pipeline
python run_pipeline.py

# 5. Launch dashboard
streamlit run src/dashboard.py
```

**Expected Runtime:** 15-30 minutes (depending on hardware)
**Expected Outputs:** See "Expected Outputs" section below

---

## 🔧 Environment Setup

### System Requirements
- **Python Version:** 3.10 or higher (tested on 3.10, 3.11, 3.12)
- **Operating System:** macOS, Linux, Windows 10+
- **RAM:** 4GB minimum, 8GB recommended
- **Disk Space:** 500MB for data, models, and outputs
- **CPU:** Multi-core recommended for faster hyperparameter tuning

### Dependency Installation

**Option 1: pip (Recommended)**
```bash
pip install -r requirements.txt
```

**Option 2: conda**
```bash
conda create -n churn python=3.10
conda activate churn
pip install -r requirements.txt
```

### Verify Installation
```python
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, shap, streamlit; print('All imports successful!')"
```

---

## 📊 Data Acquisition

### Method 1: Automated Download (Recommended)

```bash
python src/download_data.py
```

**What This Does:**
1. Creates `data/raw/` directory
2. Downloads `Telco-Customer-Churn.csv` from IBM GitHub
3. Validates file integrity (checks file size ~950KB)
4. Logs download timestamp to `logs/churn_prediction.log`

**Expected Output:**
```
INFO - Starting data download...
INFO - Downloading from: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
INFO - Data downloaded successfully to data/raw/Telco-Customer-Churn.csv
INFO - File size: 977.5 KB
```

### Method 2: Manual Download

If automated download fails:
1. Visit: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
2. Save as `data/raw/Telco-Customer-Churn.csv`
3. Verify file:
   - 7,043 rows (+ 1 header row)
   - 21 columns
   - File size: approximately 950 KB

---

## ⚙️ Data Processing & Feature Engineering

```bash
python -c "from src.data_processing import process_data; process_data()"
```

**Or run as part of full pipeline:**
```bash
python run_pipeline.py --only-processing
```

### Processing Steps (in order)

1. **Load Raw Data**
   - Read `data/raw/Telco-Customer-Churn.csv`
   - Parse 7,043 rows × 21 columns

2. **Handle Missing Values**
   - `TotalCharges`: 11 blank values → Convert to 0.0 (new customers)
   - Validate no other missing values

3. **Data Type Conversions**
   ```python
   TotalCharges: object → float64
   SeniorCitizen: int64 → category
   ```

4. **Feature Engineering**
   - `tenure_group`: Bin tenure into 6 categories (0-1yr, 1-2yr, ..., 5-6yr)
   - `monthly_charges_group`: Categorize charges (Low, Medium, High)
   - `charges_per_tenure`: TotalCharges / tenure (handle division by zero)
   - `contract_tenure_ratio`: Encode contract length / actual tenure
   - `total_services`: Count of active services (0-10 scale)
   - `payment_risk_score`: Risk score based on payment method and billing
   - `has_premium_services`: Boolean flag for tech support, security, backup

5. **Encoding**
   - Binary features (gender, Partner, etc.): Label encoding (0/1)
   - Categorical features (Contract, PaymentMethod, etc.): One-hot encoding
   - Result: 30+ features after encoding

6. **Scaling**
   - Numerical features: StandardScaler (mean=0, std=1)
   - Saved to `models/preprocessor.joblib`

7. **Train/Test Split**
   - Ratio: 80/20
   - Stratified by `Churn` to preserve class distribution
   - Random state: 42 (reproducible)
   - Saved to:
     - `data/processed/train_data.csv`
     - `data/processed/test_data.csv`

### Expected Outputs

**Files Created:**
- `data/processed/processed_data.csv` (7,043 rows × 30+ columns)
- `data/processed/train_data.csv` (5,634 rows)
- `data/processed/test_data.csv` (1,409 rows)
- `models/preprocessor.joblib` (StandardScaler object)
- `models/feature_names.joblib` (List of feature names)
- `models/target_encoder.joblib` (LabelEncoder for target)

**Console Output:**
```
INFO - Data processing started...
INFO - Loaded 7043 customers from raw data
INFO - Engineered 7 new features
INFO - Final dataset shape: (7043, 32)
INFO - Train set: 5634 samples
INFO - Test set: 1409 samples
INFO - Class distribution preserved: Train 26.5% churn, Test 26.5% churn
```

---

## 🤖 Model Training

```bash
python -c "from src.model_training import train_models; train_models()"
```

**Or run as part of full pipeline:**
```bash
python run_pipeline.py --only-training
```

### Training Configuration

**Random State:** 42 (all models, CV splits, SMOTE)
**Cross-Validation:** 5-fold stratified
**Hyperparameter Search:** RandomizedSearchCV with 20 iterations
**Optimization Metric:** Recall (prioritize catching churners)
**Class Imbalance:** SMOTE oversampling applied

### Models Trained (in order)

1. **Logistic Regression**
   - Solver: liblinear
   - Penalty: L1, L2
   - C: [0.001, 0.01, 0.1, 1, 10]
   - Class weight: balanced

2. **Random Forest**
   - n_estimators: [100, 200, 300]
   - max_depth: [10, 20, 30, None]
   - min_samples_split: [2, 5, 10]
   - min_samples_leaf: [1, 2, 4]
   - Class weight: balanced, balanced_subsample

3. **XGBoost**
   - n_estimators: [100, 200, 300]
   - max_depth: [3, 5, 7, 9]
   - learning_rate: [0.01, 0.05, 0.1]
   - subsample: [0.8, 0.9, 1.0]
   - colsample_bytree: [0.8, 0.9, 1.0]
   - scale_pos_weight: [1, 2, 3]

4. **LightGBM**
   - n_estimators: [100, 200, 300]
   - max_depth: [3, 5, 7, -1]
   - learning_rate: [0.01, 0.05, 0.1]
   - num_leaves: [31, 50, 70]
   - subsample: [0.8, 0.9, 1.0]
   - colsample_bytree: [0.8, 0.9, 1.0]
   - Class weight: balanced

### Expected Runtime
- Logistic Regression: ~2 minutes
- Random Forest: ~5 minutes
- XGBoost: ~8 minutes
- LightGBM: ~6 minutes
- **Total: ~20-25 minutes**

### Expected Results (approximate, varies with random_state)

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.78 | 0.63 | 0.72 | 0.67 | 0.82 |
| Random Forest | 0.79 | 0.65 | 0.79 | 0.71 | 0.85 |
| **XGBoost** | **0.80** | **0.68** | **0.80** | **0.74** | **0.86** |
| LightGBM | 0.81 | 0.66 | 0.81 | 0.73 | 0.87 |

**Best Model Selected:** XGBoost (highest recall with good precision)

### Expected Outputs

**Files Created:**
- `models/best_model.joblib` (~252 KB) - XGBoost trained model
- `models/all_models_results.joblib` (~6.6 MB) - All 4 models + results
- `models/model_metrics.joblib` (~682 B) - Performance metrics dictionary
- `models/feature_importance.csv` - Feature importances from best model
- `outputs/reports/model_comparison.txt` - Formatted comparison table

**Console Output:**
```
INFO - Training Logistic Regression...
INFO - Best Recall: 0.72, Best Params: {'C': 0.1, 'penalty': 'l2'}
INFO - Training Random Forest...
INFO - Best Recall: 0.79, Best Params: {'max_depth': 20, 'n_estimators': 200}
INFO - Training XGBoost...
INFO - Best Recall: 0.80, Best Params: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 200}
INFO - Training LightGBM...
INFO - Best Recall: 0.81, Best Params: {'learning_rate': 0.05, 'max_depth': 7, 'n_estimators': 200}
INFO - Best model: XGBoost (Recall: 0.80)
INFO - Model saved to models/best_model.joblib
```

---

## 🔍 Explainability Analysis

```bash
python -c "from src.explainability import generate_shap_explanations; generate_shap_explanations()"
```

**Or run as part of full pipeline:**
```bash
python run_pipeline.py --only-explainability
```

### SHAP Analysis Steps

1. **Load Best Model** (`models/best_model.joblib`)
2. **Sample Data** for SHAP (100 samples from test set, configurable)
3. **TreeExplainer** initialization
   - **Note:** May fail with XGBoost 2.1+ due to compatibility issues
   - **Fallback:** Uses `model.feature_importances_` if SHAP fails
4. **Generate Explanations:**
   - Global feature importance (SHAP summary plot)
   - Feature importance bar chart
   - Dependence plots for top 5 features
5. **Business Insights** extraction

### Expected Runtime
- SHAP analysis: 2-5 minutes
- Fallback (feature importances only): <10 seconds

### Expected Outputs

**Files Created:**
- `models/shap_objects.joblib` - SHAP explainer and values (or None if fallback)
- `outputs/figures/shap_summary_plot.png` - Global importance visualization
- `outputs/figures/shap_bar_plot.png` - Feature importance ranking
- `outputs/figures/shap_dependence_*.png` - Interaction plots for top features
- `outputs/reports/shap_feature_importance.csv` - Ranked feature importance
- `outputs/reports/business_insights.txt` - Actionable insights for stakeholders

**Console Output (SHAP success):**
```
INFO - Generating SHAP explanations...
INFO - TreeExplainer initialized successfully
INFO - Computed SHAP values for 100 samples
INFO - Generated summary plot: outputs/figures/shap_summary_plot.png
INFO - Top 5 features: Contract_Month-to-month, tenure, MonthlyCharges, InternetService_Fiber optic, TechSupport_No
```

**Console Output (SHAP fallback):**
```
WARNING - TreeExplainer failed with error: could not convert string to float
INFO - Falling back to model feature importances
INFO - Created minimal shap_objects.joblib (SHAP not available)
INFO - Generated feature importance from XGBoost: outputs/reports/shap_feature_importance.csv
```

---

## 🎨 Reproducibility Checklist

Use this checklist to verify your reproduction:

### Data
- [ ] `data/raw/Telco-Customer-Churn.csv` exists (977 KB)
- [ ] 7,043 rows loaded successfully
- [ ] 11 missing TotalCharges handled
- [ ] `data/processed/train_data.csv` has 5,634 rows
- [ ] `data/processed/test_data.csv` has 1,409 rows

### Models
- [ ] 4 models trained (Logistic, RF, XGBoost, LightGBM)
- [ ] XGBoost selected as best model
- [ ] `models/best_model.joblib` exists (~252 KB)
- [ ] Best model recall: 0.78-0.82 (variance due to randomization)
- [ ] ROC AUC: 0.84-0.88

### Explainability
- [ ] SHAP analysis completed or fallback applied
- [ ] `models/shap_objects.joblib` exists
- [ ] Feature importance CSV generated
- [ ] Top 5 features include: Contract, tenure, MonthlyCharges, TechSupport, InternetService

### Dashboard
- [ ] Streamlit launches without errors
- [ ] 6 tabs visible: Home, Executive Summary, Model Performance, Feature Importance, Customer Predictions, Data Explorer
- [ ] Model metrics match training output
- [ ] Customer predictions generate risk scores 0-100%

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xgboost'"
**Solution:**
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: "FileNotFoundError: data/raw/Telco-Customer-Churn.csv"
**Solution:**
```bash
python src/download_data.py
# Or manually download and place in data/raw/
```

### Issue: "SHAP TreeExplainer failed"
**Solution:** This is expected with XGBoost 2.1+. The pipeline automatically falls back to feature importances. Results are still valid and reproducible.

### Issue: "ValueError: Input X contains NaN"
**Solution:** Ensure you're running latest version of `src/data_processing.py` which includes comprehensive NaN handling.

### Issue: Different metrics than reported
**Expected:** Metrics may vary by ±2-3% due to:
- Hardware differences (floating-point precision)
- Library version differences (scikit-learn, xgboost updates)
- SMOTE randomization (despite fixed random_state)

**Acceptable Range:**
- Recall: 0.78-0.82 (target: ~0.80)
- ROC AUC: 0.84-0.88 (target: ~0.86)

### Issue: "Out of memory during training"
**Solution:** Reduce hyperparameter search:
```python
# In src/config.py
N_ITER_SEARCH = 10  # instead of 20
SHAP_SAMPLE_SIZE = 50  # instead of 100
```

---

## 📝 Exact Version Information

To reproduce results exactly, use these versions:

```
python==3.10.13
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
xgboost==2.1.4
lightgbm==4.1.0
shap==0.44.1
streamlit==1.29.0
imbalanced-learn==0.11.0
```

Install exact versions:
```bash
pip install -r requirements.txt --no-deps
```

---

## 🔗 Related Documentation

- **[DATA_SOURCE.md](DATA_SOURCE.md)** - Dataset details and provenance
- **[A_B_TEST_PLAN.md](A_B_TEST_PLAN.md)** - Deployment and testing strategy
- **[notebooks/threshold_roi_analysis.ipynb](notebooks/threshold_roi_analysis.ipynb)** - ROI optimization analysis
- **[README.md](README.md)** - Project overview and key results

---

## 📧 Report Reproducibility Issues

If you encounter issues reproducing results:

1. Check this guide and [README.md](README.md) troubleshooting section
2. Verify your environment matches requirements
3. Create an issue at: [github.com/noahgallagher1/customer-churn-prediction/issues](https://github.com/noahgallagher1/customer-churn-prediction/issues)
4. Include:
   - Python version (`python --version`)
   - OS and version
   - Error message/logs
   - Output of `pip list | grep -E "(pandas|sklearn|xgboost|lightgbm|shap)"`

**Contact:** Noah Gallagher - noahgallagher1@gmail.com
