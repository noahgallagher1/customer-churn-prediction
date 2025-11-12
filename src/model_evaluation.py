"""
Model Evaluation Module for Advanced Metrics and Analysis.

This module provides:
- Baseline model comparison
- Statistical testing for model comparison
- Segment-level performance analysis
- Confidence intervals via bootstrapping
- Enhanced ROI calculations
"""

import logging
from typing import Dict, Any, Tuple, List, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
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


# ============================================================================
# BASELINE MODELS
# ============================================================================

def create_baseline_models() -> Dict[str, Any]:
    """
    Create baseline models for comparison.

    Baselines:
    1. Naive: Always predict majority class (no churn)
    2. Stratified: Random predictions matching class distribution
    3. Rule-based: Simple heuristic (month-to-month AND tenure < 12 months)

    Returns:
        Dictionary of baseline models
    """
    baselines = {
        'Naive (Majority Class)': DummyClassifier(
            strategy='most_frequent',
            random_state=config.RANDOM_STATE
        ),
        'Stratified Random': DummyClassifier(
            strategy='stratified',
            random_state=config.RANDOM_STATE
        )
    }

    return baselines


def rule_based_baseline(X: pd.DataFrame) -> np.ndarray:
    """
    Rule-based baseline: Predict churn if month-to-month AND tenure < 12 months.

    Business logic: Month-to-month contracts with low tenure are highest risk.

    Args:
        X: Feature dataframe

    Returns:
        Binary predictions (1 = churn, 0 = no churn)
    """
    # Check for month-to-month contract column
    contract_cols = [col for col in X.columns if 'Contract' in col and 'month' in col.lower()]

    if contract_cols:
        # Use one-hot encoded column
        is_month_to_month = X[contract_cols[0]] == 1
    elif 'Contract' in X.columns:
        # Use original categorical column
        is_month_to_month = X['Contract'] == 'Month-to-month'
    else:
        logger.warning("Cannot find Contract column, using tenure only for rule-based baseline")
        is_month_to_month = True

    # Check tenure
    is_new_customer = X['tenure'] < 12

    # Rule: Predict churn if both conditions met
    predictions = (is_month_to_month & is_new_customer).astype(int)

    return predictions.values if isinstance(predictions, pd.Series) else predictions


def evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluate all baseline models.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        DataFrame with baseline comparison results
    """
    logger.info("Evaluating baseline models")

    results = []

    # Evaluate sklearn baselines
    baselines = create_baseline_models()

    for name, model in baselines.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': 0.5 if name == 'Naive (Majority Class)' else roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        }

        results.append(metrics)

        logger.info(f"{name}: Accuracy={metrics['Accuracy']:.3f}, "
                   f"Precision={metrics['Precision']:.3f}, "
                   f"Recall={metrics['Recall']:.3f}, F1={metrics['F1']:.3f}")

    # Evaluate rule-based baseline
    y_pred_rule = rule_based_baseline(X_test)

    rule_metrics = {
        'Model': 'Rule-Based (MTM + <12mo)',
        'Accuracy': accuracy_score(y_test, y_pred_rule),
        'Precision': precision_score(y_test, y_pred_rule, zero_division=0),
        'Recall': recall_score(y_test, y_pred_rule, zero_division=0),
        'F1': f1_score(y_test, y_pred_rule, zero_division=0),
        'ROC-AUC': np.nan  # Rule-based doesn't produce probabilities
    }

    results.append(rule_metrics)

    logger.info(f"Rule-Based: Accuracy={rule_metrics['Accuracy']:.3f}, "
               f"Precision={rule_metrics['Precision']:.3f}, "
               f"Recall={rule_metrics['Recall']:.3f}, F1={rule_metrics['F1']:.3f}")

    df_results = pd.DataFrame(results)

    return df_results


# ============================================================================
# STATISTICAL TESTING
# ============================================================================

def paired_ttest_cv(
    model1: Any,
    model2: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'roc_auc'
) -> Dict[str, float]:
    """
    Perform paired t-test on cross-validation scores.

    Args:
        model1: First model
        model2: Second model
        X: Feature matrix
        y: Target vector
        cv: Number of CV folds
        scoring: Scoring metric

    Returns:
        Dictionary with test statistics
    """
    logger.info(f"Performing paired t-test on {scoring} scores")

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.RANDOM_STATE)

    # Get CV scores for both models
    scores1 = cross_val_score(model1, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
    scores2 = cross_val_score(model2, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    # Effect size (Cohen's d)
    mean_diff = scores1.mean() - scores2.mean()
    pooled_std = np.sqrt((scores1.std()**2 + scores2.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # Interpretation
    if p_value < 0.05:
        if mean_diff > 0:
            interpretation = "Model 1 is significantly better"
        else:
            interpretation = "Model 2 is significantly better"
    else:
        interpretation = "No significant difference"

    results = {
        'model1_mean': scores1.mean(),
        'model1_std': scores1.std(),
        'model2_mean': scores2.mean(),
        'model2_std': scores2.std(),
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'interpretation': interpretation,
        'metric': scoring
    }

    logger.info(f"Model 1: {results['model1_mean']:.4f} ± {results['model1_std']:.4f}")
    logger.info(f"Model 2: {results['model2_mean']:.4f} ± {results['model2_std']:.4f}")
    logger.info(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    logger.info(f"Interpretation: {interpretation}")

    return results


def compare_all_models_statistical(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_model_name: str,
    cv: int = 5,
    scoring: str = 'roc_auc'
) -> pd.DataFrame:
    """
    Compare all models against the best model using statistical tests.

    Args:
        models: Dictionary of trained models
        X_train: Training features
        y_train: Training target
        best_model_name: Name of the best model
        cv: Number of CV folds
        scoring: Scoring metric

    Returns:
        DataFrame with statistical comparison results
    """
    logger.info(f"Comparing all models against {best_model_name}")

    best_model = models[best_model_name]['model']

    comparison_results = []

    for model_name, model_data in models.items():
        if model_name == best_model_name:
            continue

        model = model_data['model']

        # Perform t-test
        result = paired_ttest_cv(best_model, model, X_train, y_train, cv, scoring)

        comparison_results.append({
            'Comparison': f'{best_model_name} vs {model_name}',
            'Best Model Mean': result['model1_mean'],
            'Other Model Mean': result['model2_mean'],
            'Mean Difference': result['mean_difference'],
            't-statistic': result['t_statistic'],
            'p-value': result['p_value'],
            "Cohen's d": result['cohens_d'],
            'Significant?': '✅ Yes' if result['p_value'] < 0.05 else '❌ No',
            'Interpretation': result['interpretation']
        })

    df_comparison = pd.DataFrame(comparison_results)

    return df_comparison


# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_func: Callable,
    n_iterations: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate bootstrapped confidence interval for any metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_func: Metric function (e.g., recall_score)
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (default 95%)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    scores = []
    n = len(y_true)

    for _ in range(n_iterations):
        # Resample with replacement
        indices = resample(range(n), n_samples=n, random_state=None)

        # Calculate metric on bootstrap sample
        try:
            score = metric_func(y_true[indices], y_pred[indices])
            scores.append(score)
        except:
            # Skip if metric calculation fails (e.g., only one class in sample)
            continue

    scores = np.array(scores)

    # Calculate percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    mean_score = np.mean(scores)
    ci_lower = np.percentile(scores, lower_percentile)
    ci_upper = np.percentile(scores, upper_percentile)

    return mean_score, ci_lower, ci_upper


def calculate_all_metrics_with_ci(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    n_iterations: int = 1000
) -> pd.DataFrame:
    """
    Calculate all metrics with confidence intervals.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        n_iterations: Number of bootstrap iterations

    Returns:
        DataFrame with metrics and confidence intervals
    """
    logger.info(f"Calculating metrics with {n_iterations}-iteration bootstrap confidence intervals")

    metrics = {
        'Accuracy': accuracy_score,
        'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score,
        'F1': f1_score
    }

    results = []

    for metric_name, metric_func in metrics.items():
        mean, ci_lower, ci_upper = bootstrap_metric(
            y_test, y_pred, metric_func, n_iterations
        )

        results.append({
            'Metric': metric_name,
            'Mean': mean,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'CI Width': ci_upper - ci_lower
        })

        logger.info(f"{metric_name}: {mean:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")

    # ROC-AUC (using probabilities)
    def roc_auc_func(y_true, y_proba_indices):
        return roc_auc_score(y_true, y_proba_indices)

    scores = []
    n = len(y_test)
    for _ in range(n_iterations):
        indices = resample(range(n), n_samples=n)
        try:
            score = roc_auc_score(y_test[indices], y_pred_proba[indices])
            scores.append(score)
        except:
            continue

    mean_roc = np.mean(scores)
    ci_lower_roc = np.percentile(scores, 2.5)
    ci_upper_roc = np.percentile(scores, 97.5)

    results.append({
        'Metric': 'ROC-AUC',
        'Mean': mean_roc,
        '95% CI Lower': ci_lower_roc,
        '95% CI Upper': ci_upper_roc,
        'CI Width': ci_upper_roc - ci_lower_roc
    })

    logger.info(f"ROC-AUC: {mean_roc:.3f} (95% CI: [{ci_lower_roc:.3f}, {ci_upper_roc:.3f}])")

    df_results = pd.DataFrame(results)

    return df_results


# ============================================================================
# SEGMENT-LEVEL ANALYSIS
# ============================================================================

def analyze_performance_by_segment(
    y_test: pd.Series,
    y_pred: np.ndarray,
    segment_feature: pd.Series,
    segment_name: str
) -> pd.DataFrame:
    """
    Calculate model performance broken down by customer segment.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        segment_feature: Segment values (e.g., Contract type)
        segment_name: Name of the segment

    Returns:
        DataFrame with per-segment metrics
    """
    logger.info(f"Analyzing performance by {segment_name}")

    results = []

    # Reset indices to ensure alignment
    y_test = y_test.reset_index(drop=True)
    segment_feature = segment_feature.reset_index(drop=True)

    # Drop NaN values from segment_feature
    valid_mask = segment_feature.notna()
    if valid_mask.sum() == 0:
        logger.warning(f"All values are NaN in segment feature {segment_name}")
        return pd.DataFrame(columns=['Segment', 'N', 'Churn Rate', 'Precision', 'Recall', 'F1', 'FPR'])

    for segment in segment_feature.dropna().unique():
        mask = segment_feature == segment

        if mask.sum() == 0:
            continue

        y_true_seg = y_test[mask].values
        y_pred_seg = y_pred[mask]

        # Calculate metrics
        precision = precision_score(y_true_seg, y_pred_seg, zero_division=0)
        recall = recall_score(y_true_seg, y_pred_seg, zero_division=0)
        f1 = f1_score(y_true_seg, y_pred_seg, zero_division=0)

        # Confusion matrix for FPR
        cm = confusion_matrix(y_true_seg, y_pred_seg)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fpr = 0

        churn_rate = y_true_seg.mean()

        results.append({
            'Segment': str(segment),
            'N': mask.sum(),
            'Churn Rate': churn_rate,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'FPR': fpr
        })

    # Handle empty results
    if not results:
        logger.warning(f"No valid segments found for {segment_name}")
        return pd.DataFrame(columns=['Segment', 'N', 'Churn Rate', 'Precision', 'Recall', 'F1', 'FPR'])

    df_results = pd.DataFrame(results).sort_values('N', ascending=False)

    logger.info(f"\n{df_results.to_string(index=False)}")

    return df_results


def analyze_all_segments(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    df_test: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Analyze performance across all important segments.

    Args:
        X_test: Test features
        y_test: Test labels
        y_pred: Predictions
        df_test: Original test dataframe with categorical features

    Returns:
        Dictionary of segment analysis dataframes
    """
    logger.info("="*60)
    logger.info("SEGMENT-LEVEL PERFORMANCE ANALYSIS")
    logger.info("="*60)

    segment_analyses = {}

    # Analyze by Contract Type (if exists in original data)
    contract_cols = [col for col in X_test.columns if 'Contract' in col]
    if contract_cols:
        # Reconstruct contract type from one-hot encoding
        contract_types = []
        for idx in range(len(X_test)):
            if 'Contract_One year' in X_test.columns and X_test.iloc[idx]['Contract_One year'] == 1:
                contract_types.append('One year')
            elif 'Contract_Two year' in X_test.columns and X_test.iloc[idx]['Contract_Two year'] == 1:
                contract_types.append('Two year')
            else:
                contract_types.append('Month-to-month')

        contract_series = pd.Series(contract_types, index=X_test.index)
        segment_analyses['Contract Type'] = analyze_performance_by_segment(
            y_test, y_pred, contract_series, 'Contract Type'
        )

    # Analyze by Tenure Group
    tenure = X_test['tenure'] if 'tenure' in X_test.columns else None
    if tenure is not None:
        tenure_groups = pd.cut(
            tenure,
            bins=[0, 12, 24, 36, 48, np.inf],
            labels=['<12 months', '12-24 months', '24-36 months', '36-48 months', '>48 months']
        )
        segment_analyses['Tenure Group'] = analyze_performance_by_segment(
            y_test, y_pred, tenure_groups, 'Tenure Group'
        )

    # Analyze by Monthly Charges tier
    if 'MonthlyCharges' in X_test.columns:
        charge_tiers = pd.cut(
            X_test['MonthlyCharges'],
            bins=[-np.inf, 35, 70, 90, np.inf],
            labels=['Low (<$35)', 'Medium ($35-70)', 'High ($70-90)', 'Very High (>$90)']
        )
        segment_analyses['Monthly Charges'] = analyze_performance_by_segment(
            y_test, y_pred, charge_tiers, 'Monthly Charges'
        )

    # Analyze by Gender (if exists)
    if 'gender' in X_test.columns:
        segment_analyses['Gender'] = analyze_performance_by_segment(
            y_test, y_pred, X_test['gender'], 'Gender'
        )

    # Analyze by Senior Citizen status
    if 'SeniorCitizen' in X_test.columns:
        senior_labels = X_test['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        segment_analyses['Senior Citizen'] = analyze_performance_by_segment(
            y_test, y_pred, senior_labels, 'Senior Citizen'
        )

    return segment_analyses


# ============================================================================
# ENHANCED ROI CALCULATION
# ============================================================================

def calculate_enhanced_roi(
    TP: int,
    FP: int,
    FN: int,
    TN: int,
    clv: float = config.CUSTOMER_LIFETIME_VALUE,
    campaign_cost: float = config.RETENTION_COST,
    success_rate: float = 0.65
) -> Dict[str, Any]:
    """
    Calculate ROI accounting for all confusion matrix components.

    This properly accounts for:
    - True Positives: Successfully identified churners (save some via campaign)
    - False Positives: Wasted campaigns on non-churners (cost only)
    - False Negatives: Missed churners (lost revenue)
    - True Negatives: Correctly identified non-churners (no action needed)

    Args:
        TP: True Positives (correctly identified churners)
        FP: False Positives (incorrectly flagged non-churners)
        FN: False Negatives (missed churners)
        TN: True Negatives (correctly identified non-churners)
        clv: Customer Lifetime Value ($)
        campaign_cost: Cost per retention campaign ($)
        success_rate: Probability that retention campaign works (0-1)

    Returns:
        Dictionary with detailed ROI metrics
    """
    logger.info("Calculating enhanced ROI with full cost accounting")

    # COSTS
    total_campaigns = TP + FP
    campaign_execution_cost = total_campaigns * campaign_cost

    # BENEFITS
    customers_saved = TP * success_rate
    revenue_saved = customers_saved * clv

    # NET BENEFIT
    net_benefit = revenue_saved - campaign_execution_cost

    # ROI
    roi = (net_benefit / campaign_execution_cost * 100) if campaign_execution_cost > 0 else 0

    # COMPARISON TO NO MODEL (baseline)
    # Without model, we'd lose all churners
    total_churners = TP + FN
    baseline_loss = total_churners * clv

    # With model, we lose FN churners + failed campaigns on TP
    customers_lost = FN + (TP * (1 - success_rate))
    actual_loss = customers_lost * clv + campaign_execution_cost

    # Savings vs no model
    savings_vs_baseline = baseline_loss - actual_loss

    results = {
        # Confusion matrix
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN),

        # Campaign metrics
        'total_campaigns': int(total_campaigns),
        'campaign_cost_per_customer': campaign_cost,
        'campaign_execution_cost': campaign_execution_cost,
        'campaign_success_rate': success_rate,

        # Customer impact
        'customers_saved': customers_saved,
        'customers_lost': FN,
        'customers_lost_despite_campaign': TP * (1 - success_rate),

        # Financial metrics
        'clv': clv,
        'revenue_saved': revenue_saved,
        'net_benefit': net_benefit,
        'roi_percentage': roi,
        'cost_per_customer_saved': campaign_execution_cost / customers_saved if customers_saved > 0 else np.inf,

        # Baseline comparison
        'baseline_loss_no_model': baseline_loss,
        'actual_loss_with_model': actual_loss,
        'savings_vs_baseline': savings_vs_baseline,
        'improvement_vs_baseline_pct': (savings_vs_baseline / baseline_loss * 100) if baseline_loss > 0 else 0
    }

    logger.info("\n" + "="*60)
    logger.info("ENHANCED ROI ANALYSIS")
    logger.info("="*60)
    logger.info(f"Campaigns Run: {total_campaigns:,} (TP: {TP}, FP: {FP})")
    logger.info(f"Campaign Cost: ${campaign_execution_cost:,.2f}")
    logger.info(f"Customers Saved: {customers_saved:.0f} (out of {TP} identified)")
    logger.info(f"Revenue Saved: ${revenue_saved:,.2f}")
    logger.info(f"Net Benefit: ${net_benefit:,.2f}")
    logger.info(f"ROI: {roi:.1f}%")
    logger.info(f"Cost per Customer Saved: ${results['cost_per_customer_saved']:,.2f}")
    logger.info(f"\nVs. No Model Baseline:")
    logger.info(f"  Baseline Loss: ${baseline_loss:,.2f}")
    logger.info(f"  With Model Loss: ${actual_loss:,.2f}")
    logger.info(f"  Savings: ${savings_vs_baseline:,.2f} ({results['improvement_vs_baseline_pct']:.1f}% improvement)")
    logger.info("="*60)

    return results


def roi_sensitivity_analysis(
    TP: int,
    FP: int,
    FN: int,
    TN: int
) -> pd.DataFrame:
    """
    Perform sensitivity analysis on ROI calculation.

    Tests how ROI changes when we vary:
    - Customer Lifetime Value (CLV)
    - Campaign cost
    - Campaign success rate

    Args:
        TP, FP, FN, TN: Confusion matrix values

    Returns:
        DataFrame with sensitivity analysis results
    """
    logger.info("Performing ROI sensitivity analysis")

    results = []

    # Base case
    base_clv = config.CUSTOMER_LIFETIME_VALUE
    base_cost = config.RETENTION_COST
    base_success = 0.65

    # Vary CLV
    for clv in [1000, 1500, 2000, 2500, 3000]:
        roi_result = calculate_enhanced_roi(TP, FP, FN, TN, clv=clv, campaign_cost=base_cost, success_rate=base_success)
        results.append({
            'Scenario': 'Vary CLV',
            'CLV': clv,
            'Campaign Cost': base_cost,
            'Success Rate': base_success,
            'ROI (%)': roi_result['roi_percentage'],
            'Net Benefit ($)': roi_result['net_benefit']
        })

    # Vary campaign cost
    for cost in [25, 50, 75, 100, 150]:
        roi_result = calculate_enhanced_roi(TP, FP, FN, TN, clv=base_clv, campaign_cost=cost, success_rate=base_success)
        results.append({
            'Scenario': 'Vary Campaign Cost',
            'CLV': base_clv,
            'Campaign Cost': cost,
            'Success Rate': base_success,
            'ROI (%)': roi_result['roi_percentage'],
            'Net Benefit ($)': roi_result['net_benefit']
        })

    # Vary success rate
    for rate in [0.4, 0.5, 0.65, 0.75, 0.85]:
        roi_result = calculate_enhanced_roi(TP, FP, FN, TN, clv=base_clv, campaign_cost=base_cost, success_rate=rate)
        results.append({
            'Scenario': 'Vary Success Rate',
            'CLV': base_clv,
            'Campaign Cost': base_cost,
            'Success Rate': rate,
            'ROI (%)': roi_result['roi_percentage'],
            'Net Benefit ($)': roi_result['net_benefit']
        })

    df_results = pd.DataFrame(results)

    logger.info("\nROI Sensitivity Analysis Summary:")
    logger.info(f"  CLV Range: ${df_results[df_results['Scenario']=='Vary CLV']['Net Benefit ($)'].min():,.0f} to "
               f"${df_results[df_results['Scenario']=='Vary CLV']['Net Benefit ($)'].max():,.0f}")
    logger.info(f"  Cost Range ROI: {df_results[df_results['Scenario']=='Vary Campaign Cost']['ROI (%)'].min():.0f}% to "
               f"{df_results[df_results['Scenario']=='Vary Campaign Cost']['ROI (%)'].max():.0f}%")
    logger.info(f"  Success Rate Range ROI: {df_results[df_results['Scenario']=='Vary Success Rate']['ROI (%)'].min():.0f}% to "
               f"{df_results[df_results['Scenario']=='Vary Success Rate']['ROI (%)'].max():.0f}%")

    return df_results
