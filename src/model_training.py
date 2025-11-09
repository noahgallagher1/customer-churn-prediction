"""
Model Training Module for Customer Churn Prediction.

This module handles training multiple machine learning models,
hyperparameter tuning, and model evaluation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    cross_validate,
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
from tqdm import tqdm
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


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = config.RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to handle class imbalance.

    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (resampled X_train, resampled y_train)
    """
    logger.info("Applying SMOTE to handle class imbalance")

    original_counts = y_train.value_counts()
    logger.info(f"Original class distribution:\n{original_counts}")

    smote = SMOTE(
        sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
        k_neighbors=config.SMOTE_K_NEIGHBORS,
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    new_counts = pd.Series(y_resampled).value_counts()
    logger.info(f"Resampled class distribution:\n{new_counts}")

    return X_resampled, y_resampled


def get_models() -> Dict[str, Any]:
    """
    Get dictionary of models to train.

    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=config.RANDOM_STATE,
            max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'LightGBM': LGBMClassifier(
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    }

    return models


def get_param_grids() -> Dict[str, Dict[str, Any]]:
    """
    Get hyperparameter grids for each model.

    Returns:
        Dictionary mapping model names to parameter grids
    """
    param_grids = {
        'Logistic Regression': config.LOGISTIC_REGRESSION_PARAMS,
        'Random Forest': config.RANDOM_FOREST_PARAMS,
        'XGBoost': config.XGBOOST_PARAMS,
        'LightGBM': config.LIGHTGBM_PARAMS
    }

    return param_grids


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = config.CV_FOLDS,
    scoring: Optional[list] = None
) -> Dict[str, float]:
    """
    Perform cross-validation on a model.

    Args:
        model: Sklearn-compatible model
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        scoring: List of scoring metrics

    Returns:
        Dictionary of mean scores for each metric
    """
    if scoring is None:
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)

    scores = cross_validate(
        model, X, y,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    mean_scores = {
        metric: scores[f'test_{metric}'].mean()
        for metric in scoring
    }

    return mean_scores


def tune_hyperparameters(
    model: Any,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = config.CV_FOLDS,
    use_randomized: bool = config.USE_RANDOMIZED_SEARCH,
    n_iter: int = config.N_ITER_SEARCH
) -> Any:
    """
    Tune model hyperparameters using grid or randomized search.

    Args:
        model: Sklearn-compatible model
        param_grid: Parameter grid for search
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        use_randomized: If True, use RandomizedSearchCV, else GridSearchCV
        n_iter: Number of iterations for RandomizedSearchCV

    Returns:
        Fitted search object with best estimator
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)

    if use_randomized:
        logger.info(f"Using RandomizedSearchCV with {n_iter} iterations")
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring=config.SCORING_METRIC,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
            verbose=1
        )
    else:
        logger.info("Using GridSearchCV")
        search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=config.SCORING_METRIC,
            n_jobs=-1,
            verbose=1
        )

    search.fit(X_train, y_train)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV score ({config.SCORING_METRIC}): {search.best_score_:.4f}")

    return search


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluate model performance on test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging

    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating {model_name}")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    metrics['pr_curve'] = {'precision': precision, 'recall': recall}

    # Log metrics
    logger.info(f"{model_name} Performance:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"  PR AUC:    {metrics['pr_auc']:.4f}")

    return metrics


def calculate_business_metrics(
    metrics: Dict[str, Any],
    n_customers: int
) -> Dict[str, float]:
    """
    Calculate business-relevant metrics.

    Args:
        metrics: Model evaluation metrics
        n_customers: Total number of customers

    Returns:
        Dictionary of business metrics
    """
    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()

    # Calculate costs and savings
    cost_of_false_negatives = fn * config.CHURN_COST
    cost_of_retention_program = tp * config.RETENTION_COST
    cost_of_false_positives = fp * config.RETENTION_COST

    total_cost = cost_of_retention_program + cost_of_false_positives
    potential_loss_prevented = tp * config.CHURN_COST
    net_savings = potential_loss_prevented - total_cost
    cost_without_model = (tp + fn) * config.CHURN_COST

    roi = (net_savings / total_cost * 100) if total_cost > 0 else 0

    business_metrics = {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'cost_of_retention_program': cost_of_retention_program,
        'cost_of_false_positives': cost_of_false_positives,
        'cost_of_false_negatives': cost_of_false_negatives,
        'potential_loss_prevented': potential_loss_prevented,
        'net_savings': net_savings,
        'cost_without_model': cost_without_model,
        'roi_percentage': roi,
        'customers_saved': int(tp),
        'customers_lost': int(fn)
    }

    logger.info("Business Metrics:")
    logger.info(f"  Net Savings: ${net_savings:,.2f}")
    logger.info(f"  ROI: {roi:.2f}%")
    logger.info(f"  Customers Saved: {tp}")
    logger.info(f"  Customers Lost: {fn}")

    return business_metrics


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_smote: bool = config.USE_SMOTE,
    tune_hyperparameters_flag: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all models.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        use_smote: Whether to apply SMOTE
        tune_hyperparameters_flag: Whether to tune hyperparameters

    Returns:
        Dictionary mapping model names to their results
    """
    logger.info("="*60)
    logger.info("Starting Model Training")
    logger.info("="*60)

    # Apply SMOTE if requested
    if use_smote:
        X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    models = get_models()
    param_grids = get_param_grids()
    results = {}

    for model_name, model in tqdm(models.items(), desc="Training models"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*60}")

        try:
            # Tune hyperparameters
            if tune_hyperparameters_flag and model_name in param_grids:
                search = tune_hyperparameters(
                    model,
                    param_grids[model_name],
                    X_train_resampled,
                    y_train_resampled
                )
                best_model = search.best_estimator_
                best_params = search.best_params_
            else:
                logger.info("Training with default parameters")
                best_model = model
                best_model.fit(X_train_resampled, y_train_resampled)
                best_params = {}

            # Evaluate model
            metrics = evaluate_model(best_model, X_test, y_test, model_name)
            business_metrics = calculate_business_metrics(metrics, len(y_test))

            # Store results
            results[model_name] = {
                'model': best_model,
                'params': best_params,
                'metrics': metrics,
                'business_metrics': business_metrics
            }

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue

    return results


def select_best_model(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Select the best model based on the configured scoring metric.

    Args:
        results: Dictionary of model results

    Returns:
        Tuple of (best model name, best model, best metrics)
    """
    logger.info("\n" + "="*60)
    logger.info("Model Comparison")
    logger.info("="*60)

    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': res['metrics']['accuracy'],
            'Precision': res['metrics']['precision'],
            'Recall': res['metrics']['recall'],
            'F1 Score': res['metrics']['f1'],
            'ROC AUC': res['metrics']['roc_auc'],
            'PR AUC': res['metrics']['pr_auc'],
            'Net Savings ($)': res['business_metrics']['net_savings'],
            'ROI (%)': res['business_metrics']['roi_percentage']
        }
        for name, res in results.items()
    }).T

    logger.info(f"\n{comparison_df.round(4)}")

    # Select best model based on recall (to catch churners)
    best_model_name = comparison_df[config.SCORING_METRIC.capitalize()].idxmax()
    best_model = results[best_model_name]['model']
    best_metrics = results[best_model_name]['metrics']
    best_business_metrics = results[best_model_name]['business_metrics']

    logger.info(f"\n{'='*60}")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Best {config.SCORING_METRIC.capitalize()}: {best_metrics[config.SCORING_METRIC]:.4f}")
    logger.info(f"{'='*60}")

    return best_model_name, best_model, {**best_metrics, **best_business_metrics}


def save_model_artifacts(
    model: Any,
    model_name: str,
    metrics: Dict[str, Any],
    feature_names: list,
    output_dir: Optional[Path] = None
) -> None:
    """
    Save trained model and related artifacts.

    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Model evaluation metrics
        feature_names: List of feature names
        output_dir: Directory to save artifacts
    """
    output_dir = output_dir or config.MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model artifacts to {output_dir}")

    # Save model
    joblib.dump(model, config.MODEL_FILE)
    logger.info(f"Saved model to {config.MODEL_FILE}")

    # Save metrics
    metrics_to_save = {k: v for k, v in metrics.items()
                      if k not in ['confusion_matrix', 'classification_report', 'roc_curve', 'pr_curve']}
    metrics_to_save['model_name'] = model_name

    joblib.dump(metrics_to_save, config.METRICS_FILE)
    logger.info(f"Saved metrics to {config.METRICS_FILE}")

    # Save feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_file = output_dir / 'feature_importance.csv'
        feature_importance.to_csv(importance_file, index=False)
        logger.info(f"Saved feature importance to {importance_file}")

    logger.info("✓ All model artifacts saved")


def training_pipeline() -> None:
    """Execute the complete model training pipeline."""
    from data_processing import load_processed_data

    logger.info("="*60)
    logger.info("Starting Model Training Pipeline")
    logger.info("="*60)

    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Train all models
    results = train_all_models(X_train, y_train, X_test, y_test)

    # Select best model
    best_model_name, best_model, best_metrics = select_best_model(results)

    # Save artifacts
    feature_names = X_train.columns.tolist()
    save_model_artifacts(best_model, best_model_name, best_metrics, feature_names)

    # Save all results for comparison
    joblib.dump(results, config.MODELS_DIR / 'all_models_results.joblib')

    logger.info("="*60)
    logger.info("✓ Model Training Pipeline Complete")
    logger.info("="*60)


if __name__ == "__main__":
    training_pipeline()
