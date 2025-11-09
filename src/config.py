"""
Configuration file for Customer Churn Prediction Project.

This module contains all configuration parameters including paths,
model hyperparameters, and business constants.
"""

from pathlib import Path
from typing import Dict, Any

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data Configuration
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_data.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_data.csv"

# Model Configuration
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature Engineering
TENURE_BINS = [0, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5-6yr']

# Columns to encode
BINARY_COLUMNS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                  'PhoneService', 'PaperlessBilling']

CATEGORICAL_COLUMNS = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                      'OnlineBackup', 'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies', 'Contract',
                      'PaymentMethod']

NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Model Hyperparameters
LOGISTIC_REGRESSION_PARAMS: Dict[str, Any] = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10],
    'solver': ['liblinear'],
    'max_iter': [1000],
    'class_weight': ['balanced']
}

RANDOM_FOREST_PARAMS: Dict[str, Any] = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample']
}

XGBOOST_PARAMS: Dict[str, Any] = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'scale_pos_weight': [1, 2, 3]
}

LIGHTGBM_PARAMS: Dict[str, Any] = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 70],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'class_weight': ['balanced']
}

# Model Selection
USE_RANDOMIZED_SEARCH = True
N_ITER_SEARCH = 20  # Number of parameter settings sampled
SCORING_METRIC = 'recall'  # Optimize for catching churners

# SMOTE Configuration
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 'auto'
SMOTE_K_NEIGHBORS = 5

# Business Metrics
CUSTOMER_LIFETIME_VALUE = 2000  # Average customer lifetime value in dollars
RETENTION_COST = 100  # Cost to retain a customer in dollars
CHURN_COST = 1500  # Cost of losing a customer (CLV - acquisition cost)

# Visualization Configuration
PLOT_STYLE = 'whitegrid'
FIGURE_DPI = 300
PLOTLY_TEMPLATE = 'plotly_white'
COLOR_PALETTE = 'Set2'

# Primary colors for dashboard
PRIMARY_COLOR = '#1f77b4'
SECONDARY_COLOR = '#ff7f0e'
CHURN_COLOR = '#d62728'
NO_CHURN_COLOR = '#2ca02c'

# Streamlit Configuration
DASHBOARD_TITLE = "Customer Churn Prediction & Explainability Dashboard"
COMPANY_NAME = "TelcoConnect Analytics"
PAGE_ICON = "📊"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "churn_prediction.log"

# Model Artifacts
MODEL_FILE = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.joblib"
METRICS_FILE = MODELS_DIR / "model_metrics.joblib"

# SHAP Configuration
SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP analysis
SHAP_MAX_DISPLAY = 20  # Maximum features to display in SHAP plots
