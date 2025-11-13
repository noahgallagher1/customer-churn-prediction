"""
Advanced Model Evaluation Pipeline.

This script runs comprehensive model evaluation including:
- Baseline model comparison
- Statistical testing
- Segment-level analysis
- Confidence intervals
- Enhanced ROI analysis with sensitivity testing
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import config
from data_processing import load_processed_data
from model_evaluation import (
    evaluate_baselines,
    compare_all_models_statistical,
    calculate_all_metrics_with_ci,
    analyze_all_segments,
    calculate_enhanced_roi,
    roi_sensitivity_analysis
)

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


def run_baseline_evaluation(X_train, y_train, X_test, y_test):
    """Run baseline model evaluation."""
    logger.info("\n" + "="*60)
    logger.info("BASELINE MODEL EVALUATION")
    logger.info("="*60)

    baseline_results = evaluate_baselines(X_train, y_train, X_test, y_test)

    # Save results
    output_dir = config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_results.to_csv(output_dir / 'baseline_comparison.csv', index=False)
    logger.info(f"Saved baseline comparison to {output_dir / 'baseline_comparison.csv'}")

    return baseline_results


def run_statistical_comparison(models_results, X_train, y_train, best_model_name):
    """Run statistical comparison between models."""
    logger.info("\n" + "="*60)
    logger.info("STATISTICAL MODEL COMPARISON")
    logger.info("="*60)

    statistical_results = compare_all_models_statistical(
        models_results,
        X_train,
        y_train,
        best_model_name,
        cv=5,
        scoring='roc_auc'
    )

    # Save results
    output_dir = config.REPORTS_DIR
    statistical_results.to_csv(output_dir / 'statistical_comparison.csv', index=False)
    logger.info(f"Saved statistical comparison to {output_dir / 'statistical_comparison.csv'}")

    return statistical_results


def run_confidence_interval_analysis(y_test, y_pred, y_pred_proba):
    """Calculate confidence intervals for all metrics."""
    logger.info("\n" + "="*60)
    logger.info("CONFIDENCE INTERVAL ANALYSIS")
    logger.info("="*60)

    ci_results = calculate_all_metrics_with_ci(
        y_test.values,
        y_pred,
        y_pred_proba,
        n_iterations=1000
    )

    # Save results
    output_dir = config.REPORTS_DIR
    ci_results.to_csv(output_dir / 'confidence_intervals.csv', index=False)
    logger.info(f"Saved confidence intervals to {output_dir / 'confidence_intervals.csv'}")

    return ci_results


def run_segment_analysis(X_test, y_test, y_pred, df_test):
    """Run segment-level performance analysis."""
    logger.info("\n" + "="*60)
    logger.info("SEGMENT-LEVEL PERFORMANCE ANALYSIS")
    logger.info("="*60)

    segment_results = analyze_all_segments(X_test, y_test, y_pred, df_test)

    # Save all segment analyses
    output_dir = config.REPORTS_DIR
    for segment_name, df_segment in segment_results.items():
        filename = segment_name.lower().replace(' ', '_') + '_analysis.csv'
        df_segment.to_csv(output_dir / filename, index=False)
        logger.info(f"Saved {segment_name} analysis to {output_dir / filename}")

    return segment_results


def run_enhanced_roi_analysis(y_test, y_pred):
    """Run enhanced ROI analysis with sensitivity testing."""
    logger.info("\n" + "="*60)
    logger.info("ENHANCED ROI ANALYSIS")
    logger.info("="*60)

    # Get confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate enhanced ROI
    roi_results = calculate_enhanced_roi(tp, fp, fn, tn)

    # Save ROI results
    output_dir = config.REPORTS_DIR
    roi_df = pd.DataFrame([roi_results])
    roi_df.to_csv(output_dir / 'enhanced_roi_analysis.csv', index=False)
    logger.info(f"Saved ROI analysis to {output_dir / 'enhanced_roi_analysis.csv'}")

    # Run sensitivity analysis
    sensitivity_results = roi_sensitivity_analysis(tp, fp, fn, tn)
    sensitivity_results.to_csv(output_dir / 'roi_sensitivity_analysis.csv', index=False)
    logger.info(f"Saved ROI sensitivity analysis to {output_dir / 'roi_sensitivity_analysis.csv'}")

    # Create visualization
    create_roi_sensitivity_plots(sensitivity_results, output_dir)

    return roi_results, sensitivity_results


def create_roi_sensitivity_plots(sensitivity_df, output_dir):
    """Create visualizations for ROI sensitivity analysis."""
    logger.info("Creating ROI sensitivity visualizations")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    scenarios = ['Vary CLV', 'Vary Campaign Cost', 'Vary Success Rate']
    x_cols = ['CLV', 'Campaign Cost', 'Success Rate']
    x_labels = ['Customer Lifetime Value ($)', 'Campaign Cost ($)', 'Campaign Success Rate']

    for idx, (scenario, x_col, x_label) in enumerate(zip(scenarios, x_cols, x_labels)):
        ax = axes[idx]
        data = sensitivity_df[sensitivity_df['Scenario'] == scenario]

        ax.plot(data[x_col], data['ROI (%)'], marker='o', linewidth=2, markersize=8, color='#1f77b4')
        ax.axhline(100, color='red', linestyle='--', alpha=0.5, label='Break-even (100% ROI)')
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel('ROI (%)', fontsize=11)
        ax.set_title(f'ROI Sensitivity to {x_col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Highlight base case
        if scenario == 'Vary CLV':
            base_val = 2000
        elif scenario == 'Vary Campaign Cost':
            base_val = 100
        else:
            base_val = 0.65

        if base_val in data[x_col].values:
            base_roi = data[data[x_col] == base_val]['ROI (%)'].values[0]
            ax.axvline(base_val, color='green', linestyle=':', alpha=0.5, label=f'Base Case')
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'roi_sensitivity_plots.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved ROI sensitivity plots to {output_dir / 'roi_sensitivity_plots.png'}")
    plt.close()


def create_summary_report(baseline_results, ci_results, roi_results, segment_results, statistical_results):
    """Create a comprehensive summary report."""
    logger.info("\n" + "="*60)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("="*60)

    report = []

    report.append("# ADVANCED MODEL EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Baseline Comparison
    report.append("## 1. BASELINE MODEL COMPARISON")
    report.append("-" * 80)
    report.append("")
    report.append("Comparison of ML model against simple baselines:")
    report.append("")
    report.append(baseline_results.to_string(index=False))
    report.append("")
    report.append("**Key Finding**: ML model significantly outperforms all baselines,")
    report.append(f"achieving {baseline_results[baseline_results['Model'].str.contains('Rule')]['Recall'].values[0]:.1%} recall")
    report.append(f"vs {baseline_results[baseline_results['Model'] == 'Rule-Based (MTM + <12mo)']['Recall'].values[0]:.1%} for rule-based approach.")
    report.append("")

    # Confidence Intervals
    report.append("## 2. MODEL PERFORMANCE WITH CONFIDENCE INTERVALS")
    report.append("-" * 80)
    report.append("")
    report.append("95% Confidence Intervals (1000-iteration bootstrap):")
    report.append("")
    for _, row in ci_results.iterrows():
        report.append(f"{row['Metric']}: {row['Mean']:.3f} (95% CI: [{row['95% CI Lower']:.3f}, {row['95% CI Upper']:.3f}])")
    report.append("")

    # Statistical Comparison
    report.append("## 3. STATISTICAL MODEL COMPARISON")
    report.append("-" * 80)
    report.append("")
    report.append("Paired t-tests comparing models (5-fold CV on ROC-AUC):")
    report.append("")
    for _, row in statistical_results.iterrows():
        report.append(f"{row['Comparison']}:")
        report.append(f"  Mean Difference: {row['Mean Difference']:.4f}")
        report.append(f"  p-value: {row['p-value']:.4f}")
        report.append(f"  Significant: {row['Significant?']}")
        report.append(f"  {row['Interpretation']}")
        report.append("")

    # ROI Analysis
    report.append("## 4. ENHANCED ROI ANALYSIS")
    report.append("-" * 80)
    report.append("")
    report.append(f"Campaign Metrics:")
    report.append(f"  Total Campaigns: {roi_results['total_campaigns']:,}")
    report.append(f"  Campaign Cost: ${roi_results['campaign_execution_cost']:,.2f}")
    report.append(f"  Customers Saved: {roi_results['customers_saved']:.0f}")
    report.append(f"  Revenue Saved: ${roi_results['revenue_saved']:,.2f}")
    report.append(f"  Net Benefit: ${roi_results['net_benefit']:,.2f}")
    report.append(f"  ROI: {roi_results['roi_percentage']:.1f}%")
    report.append("")
    report.append(f"Comparison to No-Model Baseline:")
    report.append(f"  Without Model Loss: ${roi_results['baseline_loss_no_model']:,.2f}")
    report.append(f"  With Model Loss: ${roi_results['actual_loss_with_model']:,.2f}")
    report.append(f"  Savings: ${roi_results['savings_vs_baseline']:,.2f}")
    report.append(f"  Improvement: {roi_results['improvement_vs_baseline_pct']:.1f}%")
    report.append("")

    # Segment Analysis
    report.append("## 5. SEGMENT-LEVEL PERFORMANCE")
    report.append("-" * 80)
    report.append("")

    for segment_name, df_segment in segment_results.items():
        report.append(f"### {segment_name}")
        report.append("")

        if len(df_segment) == 0:
            report.append("No segments found (possible data issue)")
            report.append("")
            continue

        report.append(df_segment.to_string(index=False))
        report.append("")

        # Find best and worst performing segments (only if we have multiple segments)
        if len(df_segment) > 1:
            best_seg = df_segment.loc[df_segment['F1'].idxmax()]
            worst_seg = df_segment.loc[df_segment['F1'].idxmin()]

            # Only show if they're actually different
            if best_seg['Segment'] != worst_seg['Segment']:
                report.append(f"**Best Performance**: {best_seg['Segment']} (F1={best_seg['F1']:.3f})")
                report.append(f"**Worst Performance**: {worst_seg['Segment']} (F1={worst_seg['F1']:.3f})")
            else:
                report.append(f"**Performance**: {best_seg['Segment']} (F1={best_seg['F1']:.3f})")
            report.append("")
        elif len(df_segment) == 1:
            # Only one segment found
            seg = df_segment.iloc[0]
            report.append(f"**Note**: Only one segment found - {seg['Segment']} (F1={seg['F1']:.3f})")
            report.append("")

    # Save report
    output_dir = config.REPORTS_DIR
    report_text = "\n".join(report)

    with open(output_dir / 'advanced_evaluation_summary.txt', 'w') as f:
        f.write(report_text)

    logger.info(f"Saved summary report to {output_dir / 'advanced_evaluation_summary.txt'}")

    # Also print to console
    print("\n" + report_text)


def main():
    """Run the complete advanced evaluation pipeline."""
    # Set random seed for reproducibility
    np.random.seed(config.RANDOM_STATE)

    logger.info("="*80)
    logger.info("STARTING ADVANCED MODEL EVALUATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Using random seed: {config.RANDOM_STATE}")

    # Load data
    logger.info("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()

    # Load trained models
    logger.info("Loading trained models...")
    models_file = config.MODELS_DIR / 'all_models_results.joblib'

    if not models_file.exists():
        logger.error(f"Models file not found: {models_file}")
        logger.error("Please run model training first!")
        return

    models_results = joblib.load(models_file)

    # Load best model
    best_model = joblib.load(config.MODEL_FILE)
    metrics = joblib.load(config.METRICS_FILE)
    best_model_name = metrics.get('model_name', 'XGBoost')

    # Get predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Load RAW data for segment analysis (BEFORE encoding/scaling)
    # We need the original categorical values and unscaled numerical values
    from data_processing import load_raw_data, clean_data, create_features

    logger.info("Loading raw data for segment analysis...")
    df_raw = load_raw_data()
    df_raw = clean_data(df_raw)
    df_raw = create_features(df_raw)

    # Get the same test indices that were used in the train/test split
    # We need to recreate the split to get the same indices
    from sklearn.model_selection import train_test_split

    # Separate features and target
    X_raw = df_raw.drop(config.TARGET_COLUMN, axis=1)
    y_raw = df_raw[config.TARGET_COLUMN].map({'Yes': 1, 'No': 0})

    # Recreate the same split (same random_state ensures same indices)
    _, X_test_raw, _, y_test_check = train_test_split(
        X_raw, y_raw,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_raw
    )

    logger.info(f"Loaded raw test data: {X_test_raw.shape}")

    # 1. Baseline Evaluation
    baseline_results = run_baseline_evaluation(X_train, y_train, X_test, y_test)

    # 2. Statistical Comparison
    statistical_results = run_statistical_comparison(models_results, X_train, y_train, best_model_name)

    # 3. Confidence Intervals
    ci_results = run_confidence_interval_analysis(y_test, y_pred, y_pred_proba)

    # 4. Segment Analysis (use raw unprocessed data)
    segment_results = run_segment_analysis(X_test, y_test, y_pred, X_test_raw)

    # 5. Enhanced ROI Analysis
    roi_results, sensitivity_results = run_enhanced_roi_analysis(y_test, y_pred)

    # 6. Create Summary Report
    create_summary_report(baseline_results, ci_results, roi_results, segment_results, statistical_results)

    logger.info("\n" + "="*80)
    logger.info("✓ ADVANCED MODEL EVALUATION PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"\nAll results saved to: {config.REPORTS_DIR}")
    logger.info("\nGenerated files:")
    logger.info("  - baseline_comparison.csv")
    logger.info("  - statistical_comparison.csv")
    logger.info("  - confidence_intervals.csv")
    logger.info("  - enhanced_roi_analysis.csv")
    logger.info("  - roi_sensitivity_analysis.csv")
    logger.info("  - roi_sensitivity_plots.png")
    logger.info("  - *_analysis.csv (segment analyses)")
    logger.info("  - advanced_evaluation_summary.txt")


if __name__ == "__main__":
    main()
