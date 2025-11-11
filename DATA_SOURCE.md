# Data Source Documentation

## Dataset Overview

**Name:** Telco Customer Churn Dataset
**Source:** IBM Sample Data Sets
**Original Repository:** [IBM Telco Customer Churn on ICP4D](https://github.com/IBM/telco-customer-churn-on-icp4d)
**Direct Download:** [CSV File](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
**License:** IBM Sample Data (publicly available for educational and research purposes)
**Last Updated:** 2018 (static dataset)

## Dataset Characteristics

- **Total Records:** 7,043 customers
- **Total Features:** 21 (20 predictive + 1 target)
- **Target Variable:** `Churn` (Yes/No binary classification)
- **Class Distribution:**
  - No Churn: 5,174 customers (73.5%)
  - Churn: 1,869 customers (26.5%)
- **Missing Values:** 11 records with blank `TotalCharges` (handled in preprocessing)
- **File Size:** ~950 KB (CSV format)
- **Data Quality:** High quality, minimal cleaning required

## Feature Description

### Customer Demographics (4 features)
| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `customerID` | String | Unique customer identifier | 7590-VHVEG, 5575-GNVDE |
| `gender` | Binary | Customer gender | Male, Female |
| `SeniorCitizen` | Binary (0/1) | Whether customer is 65+ years old | 0, 1 |
| `Partner` | Binary | Whether customer has a partner | Yes, No |
| `Dependents` | Binary | Whether customer has dependents | Yes, No |

### Customer Account Information (3 features)
| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| `tenure` | Numeric | Months customer has stayed with company | 0-72 months |
| `Contract` | Categorical | Contract type | Month-to-month, One year, Two year |
| `PaperlessBilling` | Binary | Whether customer uses paperless billing | Yes, No |
| `PaymentMethod` | Categorical | Payment method | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `MonthlyCharges` | Numeric | Current monthly charge | $18.25 - $118.75 |
| `TotalCharges` | Numeric | Total charges to date | $18.80 - $8,684.80 |

### Services Subscribed (7 features)
| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `PhoneService` | Binary | Phone service subscription | Yes, No |
| `MultipleLines` | Categorical | Multiple phone lines | Yes, No, No phone service |
| `InternetService` | Categorical | Internet service provider | DSL, Fiber optic, No |
| `OnlineSecurity` | Categorical | Online security add-on | Yes, No, No internet service |
| `OnlineBackup` | Categorical | Online backup add-on | Yes, No, No internet service |
| `DeviceProtection` | Categorical | Device protection add-on | Yes, No, No internet service |
| `TechSupport` | Categorical | Tech support add-on | Yes, No, No internet service |
| `StreamingTV` | Categorical | Streaming TV service | Yes, No, No internet service |
| `StreamingMovies` | Categorical | Streaming movies service | Yes, No, No internet service |

### Target Variable
| Feature | Type | Description | Distribution |
|---------|------|-------------|--------------|
| `Churn` | Binary | Customer churned (left company) within last month | Yes: 26.5%, No: 73.5% |

## Data Collection Context

**Simulated Dataset:** This is a synthetic dataset created by IBM for educational purposes. While not representing a specific real company, it reflects realistic patterns from the telecommunications industry.

**Business Context:**
- Dataset represents a quarterly snapshot of customer base
- "Churn" indicates customers who left in the previous month
- Services and charges reflect typical telecom product offerings
- Demographic and account features are typical CRM data points

**Use Case Alignment:**
This dataset is ideal for:
- Customer churn prediction modeling
- Survival analysis
- Customer segmentation
- Feature importance analysis
- Model explainability demonstrations
- Educational ML projects

## Data Acquisition

### Automated Download (Recommended)
```bash
python src/download_data.py
```

This script:
1. Creates `data/raw/` directory if it doesn't exist
2. Downloads CSV from IBM GitHub repository
3. Validates file size and basic structure
4. Logs download timestamp

### Manual Download
1. Visit: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
2. Save as `data/raw/Telco-Customer-Churn.csv`
3. Verify file size is approximately 950 KB

## Data Processing Pipeline

Our preprocessing handles:

1. **Missing Values**
   - 11 customers have blank `TotalCharges` (new customers with 0 tenure)
   - Filled with 0 or median based on context

2. **Data Type Conversion**
   - `TotalCharges`: String → Float
   - `SeniorCitizen`: Integer (0/1) → Binary label
   - Categorical: String → Label/One-hot encoding

3. **Feature Validation**
   - Tenure range check (0-72 months)
   - Charge consistency (TotalCharges ≈ MonthlyCharges × tenure)
   - Service dependency logic (e.g., MultipleLines requires PhoneService)

4. **Train/Test Split**
   - 80/20 stratified split
   - Random state: 42 (reproducible)
   - Preserves class distribution in both sets

See [RESULTS_REPRODUCIBILITY.md](RESULTS_REPRODUCIBILITY.md) for exact preprocessing steps.

## Data Quality Assessment

### Strengths
✅ No duplicate customer IDs
✅ Minimal missing values (<0.2%)
✅ Realistic feature distributions
✅ Balanced feature representation across demographics
✅ Clear business logic in service features

### Limitations
⚠️ **Synthetic Data:** Not from a real company, may not capture all real-world edge cases
⚠️ **Limited Temporal Info:** Single snapshot, no time-series features
⚠️ **No External Factors:** Missing competitor pricing, market conditions, seasonality
⚠️ **Class Imbalance:** 73.5/26.5 split requires SMOTE or other balancing techniques
⚠️ **Geographic Data Missing:** No location/region information

## Ethical Considerations

### Privacy
- Dataset is synthetic and publicly available
- No real customer PII (Personally Identifiable Information)
- Customer IDs are anonymized alphanumeric codes

### Bias Considerations
- Gender: Approximately balanced (50/50 split)
- Age: SeniorCitizen status may reflect age discrimination concerns in model deployment
- Protected Classes: Dataset lacks race, religion, disability status (good for avoiding direct discrimination)

**Model Deployment Warning:** When deploying churn models in production with real customer data:
- Ensure retention offers don't discriminate based on protected characteristics
- Monitor for disparate impact across demographic groups
- Validate that SHAP explanations don't reveal sensitive attribute inference

## Citation

If using this dataset in academic work:

```bibtex
@misc{ibm_telco_churn,
  author = {IBM},
  title = {Telco Customer Churn Dataset},
  year = {2018},
  publisher = {GitHub},
  journal = {IBM Developer},
  howpublished = {\url{https://github.com/IBM/telco-customer-churn-on-icp4d}}
}
```

## Updates and Versioning

- **Current Version:** v1.0 (static)
- **Last Verified:** November 2025
- **Known Issues:** None
- **Alternative Sources:** Also available on Kaggle (search "Telco Customer Churn")

## Contact for Data Issues

For questions about:
- **Data Source:** Refer to [IBM's repository](https://github.com/IBM/telco-customer-churn-on-icp4d/issues)
- **Our Preprocessing:** Create an issue in [this project's repository](https://github.com/noahgallagher1/customer-churn-prediction/issues)
