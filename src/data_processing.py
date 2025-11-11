"""
Data Processing Module for Customer Churn Prediction.

This module contains functions for loading, cleaning, and preprocessing
the Telco Customer Churn dataset.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.

    Args:
        file_path: Path to the CSV file. If None, uses config.RAW_DATA_FILE

    Returns:
        DataFrame containing the raw data

    Raises:
        FileNotFoundError: If the data file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    file_path = file_path or config.RAW_DATA_FILE

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and data type issues.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning")
    df = df.copy()

    # Drop customerID as it's not useful for prediction
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        logger.info("Dropped customerID column")

    # Handle TotalCharges - convert to numeric and handle spaces
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        missing_charges = df['TotalCharges'].isna().sum()
        if missing_charges > 0:
            logger.info(f"Found {missing_charges} missing values in TotalCharges")
            # Fill missing TotalCharges with MonthlyCharges for new customers
            df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])

    # Convert SeniorCitizen to Yes/No for consistency
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        logger.info("Converted SeniorCitizen to Yes/No format")

    # Handle 'No internet service' and 'No phone service' values
    internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].replace('No internet service', 'No')

    if 'MultipleLines' in df.columns:
        df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

    logger.info(f"Data cleaning complete. Shape: {df.shape}")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing data.

    Args:
        df: Cleaned DataFrame

    Returns:
        DataFrame with additional engineered features
    """
    logger.info("Starting feature engineering")
    df = df.copy()

    # Tenure bins
    if 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=config.TENURE_BINS,
            labels=config.TENURE_LABELS,
            include_lowest=True
        )
        logger.info("Created tenure_group feature")

    # Monthly charges categories
    if 'MonthlyCharges' in df.columns:
        df['monthly_charges_category'] = pd.cut(
            df['MonthlyCharges'],
            bins=[0, 35, 65, 100, 200],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        logger.info("Created monthly_charges_category feature")

    # Revenue per tenure month
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['charges_per_tenure'] = df.apply(
            lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] > 0 else x['MonthlyCharges'],
            axis=1
        )
        logger.info("Created charges_per_tenure feature")

    # Total services count
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies']
    available_service_cols = [col for col in service_cols if col in df.columns]

    if available_service_cols:
        df['total_services'] = df[available_service_cols].apply(
            lambda x: sum(x.isin(['Yes', 'DSL', 'Fiber optic'])), axis=1
        )
        logger.info("Created total_services feature")

    # Contract-Tenure interaction
    if 'Contract' in df.columns and 'tenure' in df.columns:
        df['contract_tenure_ratio'] = df.apply(
            lambda x: x['tenure'] / 12 if x['Contract'] == 'One year'
            else x['tenure'] / 24 if x['Contract'] == 'Two year'
            else x['tenure'],
            axis=1
        )
        logger.info("Created contract_tenure_ratio feature")

    # Payment method risk score
    if 'PaymentMethod' in df.columns:
        payment_risk = {
            'Electronic check': 3,
            'Mailed check': 2,
            'Bank transfer (automatic)': 1,
            'Credit card (automatic)': 1
        }
        df['payment_risk_score'] = df['PaymentMethod'].map(payment_risk)
        logger.info("Created payment_risk_score feature")

    # Has premium services
    premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    available_premium = [col for col in premium_services if col in df.columns]
    if available_premium:
        df['has_premium_services'] = df[available_premium].apply(
            lambda x: 'Yes' if any(x == 'Yes') else 'No', axis=1
        )
        logger.info("Created has_premium_services feature")

    # Handle any remaining NaN values introduced by feature engineering
    nan_counts = df.isna().sum()
    if nan_counts.any():
        logger.warning(f"Found NaN values after feature engineering:\n{nan_counts[nan_counts > 0]}")
        # Fill numerical NaN with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
                logger.info(f"Filled NaN in {col} with median")
        # Fill categorical NaN with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
                logger.info(f"Filled NaN in {col} with mode")

    logger.info(f"Feature engineering complete. New shape: {df.shape}")
    return df


def encode_features(
    df: pd.DataFrame,
    fit_encoder: bool = True,
    encoder_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables.

    Args:
        df: DataFrame with features
        fit_encoder: If True, fit new encoders. If False, load existing
        encoder_path: Path to save/load encoders

    Returns:
        Tuple of (encoded DataFrame, encoders dictionary)
    """
    logger.info("Starting feature encoding")
    df = df.copy()
    encoders = {}

    # Encode target variable
    if config.TARGET_COLUMN in df.columns:
        if fit_encoder:
            le = LabelEncoder()
            df[config.TARGET_COLUMN] = le.fit_transform(df[config.TARGET_COLUMN])
            encoders['target'] = le
            logger.info(f"Encoded target variable. Classes: {le.classes_}")
        else:
            if encoder_path and encoder_path.exists():
                le = joblib.load(encoder_path / 'target_encoder.joblib')
                df[config.TARGET_COLUMN] = le.transform(df[config.TARGET_COLUMN])

    # Binary encoding for binary features
    binary_mapping = {'Yes': 1, 'No': 0}
    binary_cols = [col for col in config.BINARY_COLUMNS if col in df.columns]
    binary_cols += [col for col in ['has_premium_services'] if col in df.columns]

    for col in binary_cols:
        df[col] = df[col].map(binary_mapping)
        logger.info(f"Binary encoded: {col}")

    # One-hot encoding for categorical features
    categorical_cols = [col for col in config.CATEGORICAL_COLUMNS if col in df.columns]
    categorical_cols += [col for col in ['tenure_group', 'monthly_charges_category'] if col in df.columns]

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        logger.info(f"One-hot encoded {len(categorical_cols)} categorical columns")

    # Handle any NaN values introduced during encoding
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found NaN values after encoding:\n{nan_counts[nan_counts > 0]}")
        # Fill all NaN with 0 (safe for encoded features)
        df = df.fillna(0)
        logger.info("Filled all NaN values with 0")

    logger.info(f"Encoding complete. Final shape: {df.shape}")
    return df, encoders


def scale_features(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    fit_scaler: bool = True,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.

    Args:
        df: DataFrame with encoded features
        target_col: Name of target column to exclude from scaling
        fit_scaler: If True, fit new scaler. If False, use provided scaler
        scaler: Pre-fitted scaler (required if fit_scaler=False)

    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    logger.info("Starting feature scaling")
    df = df.copy()

    # Identify numerical columns (excluding target)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)

    if not numerical_cols:
        logger.warning("No numerical columns found to scale")
        return df, StandardScaler()

    if fit_scaler:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logger.info(f"Fitted and transformed {len(numerical_cols)} numerical columns")
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit_scaler=False")
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        logger.info(f"Transformed {len(numerical_cols)} numerical columns using existing scaler")

    return df, scaler


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        df: Preprocessed DataFrame
        target_col: Name of target column
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Splitting data into train and test sets")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Train set size: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    logger.info(f"Test set size: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
    logger.info(f"Train churn rate: {y_train.mean()*100:.2f}%")
    logger.info(f"Test churn rate: {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    encoders: dict,
    output_dir: Optional[Path] = None
) -> None:
    """
    Save processed data and preprocessing objects.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        scaler: Fitted scaler object
        encoders: Dictionary of encoders
        output_dir: Directory to save files (defaults to config.MODELS_DIR)
    """
    output_dir = output_dir or config.MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed data to {output_dir}")

    # Final NaN check before saving
    if X_train.isna().any().any():
        logger.error(f"WARNING: X_train contains NaN values:\n{X_train.isna().sum()[X_train.isna().sum() > 0]}")
        X_train = X_train.fillna(0)
        logger.info("Filled X_train NaN values with 0")

    if X_test.isna().any().any():
        logger.error(f"WARNING: X_test contains NaN values:\n{X_test.isna().sum()[X_test.isna().sum() > 0]}")
        X_test = X_test.fillna(0)
        logger.info("Filled X_test NaN values with 0")

    # Save train/test data
    train_data = X_train.copy()
    train_data[config.TARGET_COLUMN] = y_train
    train_data.to_csv(config.TRAIN_DATA_FILE, index=False)

    test_data = X_test.copy()
    test_data[config.TARGET_COLUMN] = y_test
    test_data.to_csv(config.TEST_DATA_FILE, index=False)

    # Save preprocessing objects
    joblib.dump(scaler, config.PREPROCESSOR_FILE)
    joblib.dump(X_train.columns.tolist(), config.FEATURE_NAMES_FILE)

    # Save encoders
    for name, encoder in encoders.items():
        joblib.dump(encoder, output_dir / f"{name}_encoder.joblib")

    logger.info("✓ All processed data and artifacts saved")


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load processed training and testing data.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Loading processed data")

    train_data = pd.read_csv(config.TRAIN_DATA_FILE)
    test_data = pd.read_csv(config.TEST_DATA_FILE)

    X_train = train_data.drop(config.TARGET_COLUMN, axis=1)
    y_train = train_data[config.TARGET_COLUMN]
    X_test = test_data.drop(config.TARGET_COLUMN, axis=1)
    y_test = test_data[config.TARGET_COLUMN]

    logger.info(f"Loaded train data: {X_train.shape}, test data: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def process_pipeline(force_reprocess: bool = False) -> None:
    """
    Execute the complete data processing pipeline.

    Args:
        force_reprocess: If True, reprocess even if processed data exists
    """
    logger.info("="*60)
    logger.info("Starting Data Processing Pipeline")
    logger.info("="*60)

    # Check if processed data already exists
    if not force_reprocess and config.TRAIN_DATA_FILE.exists() and config.TEST_DATA_FILE.exists():
        logger.info("Processed data already exists. Use force_reprocess=True to reprocess.")
        return

    # Load and clean data
    df = load_raw_data()
    df = clean_data(df)

    # Feature engineering
    df = create_features(df)

    # Encoding
    df, encoders = encode_features(df, fit_encoder=True)

    # Scaling
    df, scaler = scale_features(df, fit_scaler=True)

    # Train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)

    # Save everything
    save_processed_data(X_train, X_test, y_train, y_test, scaler, encoders)

    logger.info("="*60)
    logger.info("✓ Data Processing Pipeline Complete")
    logger.info("="*60)


if __name__ == "__main__":
    process_pipeline(force_reprocess=True)
