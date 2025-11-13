# Customer Churn Prediction with ML Explainability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/noahgal/customer-churn-prediction-with-xgboost-shap-expl)


> End-to-end machine learning solution that identifies at-risk telecom customers with 93% recall, delivering $367K+ in estimated annual savings through targeted retention interventions.

---

## Overview

This project simulates a production-grade ML system for a telecommunications company facing customer churn challenges. The solution combines predictive modeling, explainable AI, and business impact quantification to enable data-driven retention strategies.

### Business Context

Telecommunications companies face 26.5% annual customer churn, with each lost customer representing significant revenue loss. This project addresses the challenge by building a machine learning system that:

1. **Predicts customer churn** with 93% recall using an XGBoost ensemble model
2. **Explains predictions** using SHAP values for interpretable, actionable insights
3. **Prioritizes interventions** by ranking customers by churn probability
4. **Quantifies ROI** for retention campaign scenarios

### Key Results

| Metric | Value | Business Impact |
|--------|-------|-----------------|
| **Model Recall** | 93% | Identifies 348 of 374 customers who will churn |
| **Model Accuracy** | 62.5% | Overall prediction correctness |
| **Customers Saved Annually** | 226 | Assuming 65% intervention success rate |
| **Estimated Annual Savings** | **$367,300** | Net savings after retention program cost |
| **ROI** | **431.6%** | Every dollar spent returns $5.32 |
| **Missed Churners** | 26 | 7% false negative rate |

### Critical Churn Drivers

Through SHAP analysis, identified five critical churn drivers:

1. **Contract Type** - Month-to-month contracts have 42% churn vs. 11% for annual contracts
2. **Tenure** - 50%+ churn rate in first 12 months
3. **Payment Method** - Electronic check users show 45% churn
4. **Tech Support** - Lack of support increases churn by 35%
5. **Monthly Charges** - High charges without perceived value drive attrition

---

## Technical Architecture

### Data Foundation
- **Dataset**: 7,043 telecom customers with 21 features (demographics, services, billing)
- **Target**: Binary churn outcome with 26.5% positive class rate (imbalanced)
- **Processing**: Missing value handling, standardization, validation
- **Documentation**: See [DATA_SOURCE.md](DATA_SOURCE.md) and [RESULTS_REPRODUCIBILITY.md](RESULTS_REPRODUCIBILITY.md)

### Feature Engineering
Transformed raw data into 36 predictive features:
- Tenure segmentation (6-month bins)
- Revenue metrics (charges per tenure month, contract-tenure ratios)
- Service aggregation (total services count, premium service flags)
- Risk scoring based on historical churn correlations
- Interaction features for non-linear relationships

### Model Development
- **Algorithms Evaluated**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Optimization**: RandomizedSearchCV with 5-fold stratified cross-validation (20 iterations)
- **Imbalance Handling**: SMOTE oversampling to address 73.5% / 26.5% class distribution
- **Optimization Target**: Recall (catching churners is 15× more valuable than avoiding false alarms)
- **Selected Model**: XGBoost (93% recall, 62.5% accuracy)

### Explainability Implementation
- **SHAP Values**: TreeExplainer for local and global feature importance
- **Visualization**: Summary plots, dependence plots, waterfall charts
- **Business Translation**: Mapped technical features to business-friendly names
- **Recommendation Engine**: Links predictions to specific retention actions
- **Note**: SHAP values show feature correlation with predictions, not causal relationships

### Deployment
- **Interactive Dashboard**: 6-page Streamlit application with real-time scoring
- **A/B Test Framework**: See [A_B_TEST_PLAN.md](A_B_TEST_PLAN.md) for rollout strategy
- **Production Standards**: Type hints, logging, error handling, modular architecture

### Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Optimize for Recall over Precision** | Missing a churner ($1,500 loss) is 15× more costly than a false alarm ($100 retention cost) |
| **Use SMOTE** | Class imbalance (73.5% / 26.5%) would bias model toward majority class |
| **Tree-based Models** | Non-linear relationships in telecom data; SHAP TreeExplainer compatibility |
| **SHAP for Explainability** | Stakeholder trust requires understanding *why* customers are flagged as high-risk |
| **XGBoost Selection** | Superior recall performance with strong interpretability through SHAP |

---

## Advanced Statistical Evaluation

### 1. Baseline Model Comparison
Compared ML model against simple heuristics to prove value:
- Naive Baseline (always predict "no churn")
- Stratified Random (random predictions matching class distribution)
- Rule-Based Heuristic (month-to-month + tenure <12 months)

**Result**: XGBoost achieves 93% recall vs. 52% for rule-based approach - **79% improvement**

### 2. Statistical Testing
Paired t-tests on 5-fold cross-validation scores:
- XGBoost vs. Logistic Regression: **p<0.001** (statistically significant)
- XGBoost vs. Random Forest: **p=0.044** (statistically significant)
- XGBoost vs. LightGBM: p=0.182 (not significant - comparable performance)

**Conclusion**: XGBoost demonstrates statistically significant superiority over simpler models.

### 3. Confidence Intervals (Bootstrap)
1000-iteration bootstrap for all metrics:
- **Recall: 93.0% (95% CI: [90.2%, 95.4%])**
- Precision: 40.9% (95% CI: [37.6%, 44.3%])
- **ROC-AUC: 0.838 (95% CI: [0.814, 0.861])**

Narrow confidence intervals indicate stable, reliable performance estimates.

### 4. Segment-Level Performance
Model performance varies by customer segment:

| Segment | F1 Score | Performance | Insight |
|---------|----------|-------------|---------|
| Tenure <12 months | **0.74** | Excellent | Highest ROI - focus retention here |
| Tenure 12-24 months | 0.63 | Good | Standard campaigns effective |
| Tenure 24-48 months | 0.50 | Fair | Monitor closely |
| **Tenure >48 months** | **0.33** | Poor | **Different strategy needed** |

**Key Finding**: Model struggles with long-tenure loyal customers (low churn base rate). Recommend separate retention approach for this segment.

### 5. Enhanced ROI Analysis
Full cost accounting including false positives:

```
Campaigns Run: 851 (348 TP + 503 FP)
Campaign Cost: $85,100 (includes false positive waste)
Customers Saved: 226 (348 × 65% success rate)
Revenue Saved: $452,400
Net Benefit: $367,300
ROI: 431.6%
```

**Sensitivity Analysis**:
- Pessimistic (CLV=$1,500, 50% success): ROI = **294%**
- Base Case (CLV=$2,000, 65% success): ROI = **431.6%**
- Optimistic (CLV=$2,500, 75% success): ROI = **569%**

Business case remains strong even under worst-case assumptions.

### Running Advanced Evaluation

```bash
# Execute complete Phase 1 evaluation suite
python src/run_advanced_evaluation.py
```

**Generates**:
- `baseline_comparison.csv` - ML vs. simple heuristics
- `statistical_comparison.csv` - Paired t-test results
- `confidence_intervals.csv` - Bootstrap CIs for all metrics
- `enhanced_roi_analysis.csv` - Full cost accounting
- `roi_sensitivity_analysis.csv` - ROI under different assumptions
- `*_analysis.csv` - Segment-level performance breakdowns
- `advanced_evaluation_summary.txt` - Comprehensive report

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed usage.

---

## Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/                      # Raw dataset (Telco Customer Churn)
│   └── processed/                # Processed train/test splits
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb  # Comprehensive EDA
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration & hyperparameters
│   ├── download_data.py          # Data acquisition module
│   ├── data_processing.py        # Feature engineering pipeline
│   ├── model_training.py         # Multi-model training & evaluation
│   ├── model_evaluation.py       # Advanced evaluation (baselines, stats, segments, CIs, ROI)
│   ├── run_advanced_evaluation.py # Execute Phase 1 evaluation pipeline
│   ├── explainability.py         # SHAP analysis module
│   └── dashboard.py              # Streamlit dashboard (deprecated - see app.py)
│
├── models/                       # Saved models and artifacts
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   ├── feature_names.joblib
│   ├── model_metrics.joblib
│   └── shap_objects.joblib
│
├── outputs/
│   ├── figures/                  # All visualizations (PNG, HTML)
│   └── reports/                  # Analysis reports and insights
│
├── logs/                         # Application logs
│
├── app.py                        # Streamlit dashboard (6 pages)
├── run_pipeline.py               # Main execution script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LIMITATIONS.md                # Comprehensive limitations analysis & future work
├── EVALUATION_GUIDE.md           # Advanced evaluation usage guide
├── GLOSSARY.md                   # Standardized terminology and metrics
├── A_B_TEST_PLAN.md             # A/B test design document
└── DATA_SOURCE.md                # Dataset documentation
```

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- 4GB+ RAM recommended
- ~500MB disk space for data and models

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/noahgallagher1/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Project

#### Full Pipeline (Recommended for First Run)

```bash
python run_pipeline.py
```

This will:
1. Download the Telco Customer Churn dataset
2. Process data and engineer features
3. Train multiple models with hyperparameter tuning
4. Generate SHAP explanations and visualizations
5. Save all artifacts and reports

**Expected runtime:** 15-30 minutes (depending on hardware)

#### Individual Pipeline Steps

```bash
# Download data only
python run_pipeline.py --only-download

# Process data only
python run_pipeline.py --only-processing

# Train models only
python run_pipeline.py --only-training

# Generate explainability analysis only
python run_pipeline.py --only-explainability
```

#### Skip Specific Steps

```bash
# Skip data download (if already downloaded)
python run_pipeline.py --skip-download

# Skip processing (if already processed)
python run_pipeline.py --skip-processing
```

### Launch the Dashboard

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Explore the EDA Notebook

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## Dashboard Features

The Streamlit dashboard includes 6 comprehensive pages:

### 1. Executive Summary
- Key performance metrics (churn rate, model accuracy, savings)
- Top risk factors visualization
- Business recommendations
- ROI analysis
- Model comparison vs. baselines

### 2. Model Performance
- Performance metrics table (accuracy, precision, recall, F1, AUC)
- Confusion matrix heatmap
- ROC and Precision-Recall curves
- Business impact metrics
- Confidence intervals
- Enhanced ROI analysis
- Sensitivity analysis

### 3. Advanced Evaluation
- Baseline model comparison
- Statistical validation (paired t-tests)
- Confidence intervals (bootstrap)
- Segment-level performance
- ROI sensitivity analysis
- Downloadable reports

### 4. Customer Risk Scoring
- Individual customer churn prediction
- Real-time probability calculation
- SHAP-based explanation for each prediction
- Risk-based retention recommendations

### 5. Feature Importance
- SHAP summary plots (global feature importance)
- Interactive feature selection
- SHAP dependence plots
- Feature correlation analysis
- Individual prediction explanations
- Business-friendly interpretations

### 6. About Data
- Dataset overview
- Feature dictionary
- Data quality summary
- Key statistics
- Target variable distribution
- Data privacy and ethics considerations

---

## Model Performance Details

### XGBoost Classifier

**Classification Metrics:**
```
Accuracy:      62.5%
Recall:        93.0%
Precision:     40.9%
F1 Score:      56.8%
ROC AUC:       83.8%
```

**Confusion Matrix:**
```
True Negatives:   532  |  False Positives:  503
False Negatives:   26  |  True Positives:   348
```

**Business Metrics:**
```
Customers Correctly Identified as Churners:  348 (True Positives)
Customers Saved through Intervention:        226 (65% success rate)
Customers Lost (False Negatives):             26

Total Retention Program Cost:                $85,100 (851 campaigns)
Revenue Saved:                               $452,400
Net Savings:                                 $367,300
ROI:                                         431.6%
```

### Model Tradeoffs

The model prioritizes recall (catching churners) over precision, accepting false positives to minimize costly false negatives. This design reflects the business reality that missing a churner ($1,500 loss) is far more expensive than unnecessary retention outreach (~$100 cost).

---

## Key Findings

### High-Risk Segments

- Month-to-month contract customers (42% churn)
- New customers with tenure < 6 months (55% churn)
- Electronic check payment users (45% churn)
- Customers without tech support (41% churn)
- Fiber optic users without premium services (38% churn)

### Low-Risk Segments

- 2-year contract customers (3% churn)
- Customers with 60+ months tenure (7% churn)
- Automatic payment users (15% churn)
- Multiple service subscribers (18% churn)

### Business Recommendations

1. **Early Engagement Program** - Target customers in first 6 months with personalized onboarding
2. **Contract Upgrade Incentives** - Offer discounts for annual/2-year commitments
3. **Service Bundling** - Promote tech support + security packages
4. **Payment Method Migration** - Incentivize switch to automatic payments
5. **Pricing Optimization** - Review high monthly charge customers for loyalty discounts

---

## Configuration

All configuration is centralized in `src/config.py`:

**Key Parameters:**
- `RANDOM_STATE = 42` - Reproducibility seed
- `TEST_SIZE = 0.2` - Train/test split ratio
- `CV_FOLDS = 5` - Cross-validation folds
- `SCORING_METRIC = 'recall'` - Optimization target
- `USE_SMOTE = True` - Apply SMOTE for imbalance
- `N_ITER_SEARCH = 20` - Hyperparameter search iterations

**Business Constants:**
- `CUSTOMER_LIFETIME_VALUE = 2000` - Average CLV ($)
- `RETENTION_COST = 100` - Cost per retention attempt ($)
- `CHURN_COST = 1500` - Cost of customer churn ($)

---

## Dependencies

**Core Libraries:**
- pandas, numpy - Data manipulation
- scikit-learn - ML algorithms and preprocessing
- xgboost, lightgbm - Gradient boosting models
- imbalanced-learn - SMOTE implementation

**Visualization:**
- matplotlib, seaborn - Static plots
- plotly - Interactive visualizations
- streamlit - Dashboard framework

**Explainability:**
- shap - Model interpretability

**Utilities:**
- jupyter - Notebook environment
- joblib - Model serialization
- tqdm - Progress bars

---

## Code Quality Standards

- **Type Hints**: All functions include type annotations
- **Docstrings**: Google-style docstrings throughout
- **Logging**: Comprehensive logging with configurable levels
- **Error Handling**: Try-except blocks for robustness
- **Modularity**: Separate modules for each pipeline stage
- **PEP 8 Compliance**: Following Python style guidelines

---

## Limitations & Future Work

See [LIMITATIONS.md](LIMITATIONS.md) for comprehensive analysis of constraints and future improvements.

**Key Limitations:**
- No temporal validation (dataset has no date columns)
- High false positive rate (50%) - acceptable given business context
- Missing features (NPS, network quality, customer service interactions)
- Static model - no online learning

**Planned Improvements:**
1. Neural network models (MLP, TabNet)
2. Time-series features (seasonality, trends)
3. REST API for predictions
4. Docker containerization
5. CI/CD pipeline
6. Model monitoring and retraining framework
7. Causal inference analysis
8. Uplift modeling

---

## Troubleshooting

### Common Issues

**Import errors after installation**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Data download fails**
```bash
# Download manually from:
# https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
# Place in: data/raw/Telco-Customer-Churn.csv
```

**Out of memory during training**
```python
# In src/config.py, reduce:
N_ITER_SEARCH = 10  # instead of 20
SHAP_SAMPLE_SIZE = 50  # instead of 100
```

**Dashboard doesn't load**
```bash
# Check port availability
streamlit run app.py --server.port 8502

# Clear Streamlit cache
streamlit cache clear
```

---

## Documentation

- **[GLOSSARY.md](GLOSSARY.md)** - Standardized terminology and metric definitions
- **[LIMITATIONS.md](LIMITATIONS.md)** - Comprehensive constraints analysis
- **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)** - Advanced evaluation usage guide
- **[A_B_TEST_PLAN.md](A_B_TEST_PLAN.md)** - A/B test design document
- **[DATA_SOURCE.md](DATA_SOURCE.md)** - Dataset documentation
- **[RESULTS_REPRODUCIBILITY.md](RESULTS_REPRODUCIBILITY.md)** - Reproduction instructions

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

**Noah Gallagher** | Data Scientist

- **Email**: noahgallagher1@gmail.com
- **GitHub**: [github.com/noahgallagher1](https://github.com/noahgallagher1)
- **LinkedIn**: [linkedin.com/in/noahgallagher](https://www.linkedin.com/in/noahgallagher/)
- **Portfolio**: [noahgallagher1.github.io/MySite](https://noahgallagher1.github.io/MySite/)
- **This Project**: [github.com/noahgallagher1/customer-churn-prediction](https://github.com/noahgallagher1/customer-churn-prediction)
