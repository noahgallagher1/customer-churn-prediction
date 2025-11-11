"""
Model Explainability Module using SHAP.

This module provides interpretability for the churn prediction model
using SHAP (SHapley Additive exPlanations) values.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
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

# Set style for plots
sns.set_style(config.PLOT_STYLE)
plt.rcParams['figure.dpi'] = config.FIGURE_DPI


def load_model_and_data() -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load the trained model and test data.

    Returns:
        Tuple of (model, X_test, feature_names, y_test)
    """
    logger.info("Loading model and data for explainability analysis")

    model = joblib.load(config.MODEL_FILE)
    feature_names = joblib.load(config.FEATURE_NAMES_FILE)

    # Load test data
    from data_processing import load_processed_data
    X_train, X_test, y_train, y_test = load_processed_data()

    logger.info(f"Loaded model and {len(X_test)} test samples")

    return model, X_test, feature_names, y_test


def create_shap_explainer(
    model: Any,
    X_background: pd.DataFrame,
    sample_size: int = config.SHAP_SAMPLE_SIZE
) -> shap.Explainer:
    """
    Create a SHAP explainer for the model.

    Args:
        model: Trained model
        X_background: Background dataset for SHAP
        sample_size: Number of samples to use for background

    Returns:
        SHAP explainer object
    """
    logger.info("Creating SHAP explainer")

    # Sample background data if needed
    if len(X_background) > sample_size:
        X_background_sampled = X_background.sample(
            n=sample_size,
            random_state=config.RANDOM_STATE
        )
    else:
        X_background_sampled = X_background

    # Create appropriate explainer based on model type
    model_type = type(model).__name__

    if 'XGB' in model_type or 'LGBM' in model_type or 'RandomForest' in model_type:
        logger.info(f"Using TreeExplainer for {model_type}")
        try:
            explainer = shap.TreeExplainer(model)
        except (ValueError, AttributeError) as e:
            logger.warning(f"TreeExplainer failed with error: {e}")
            logger.info("Falling back to model feature importances instead")
            # Return None to signal that SHAP failed, we'll use feature_importances_ instead
            return None
    else:
        logger.info(f"Using KernelExplainer for {model_type}")
        explainer = shap.KernelExplainer(
            model.predict_proba,
            X_background_sampled
        )

    return explainer


def calculate_shap_values(
    explainer: shap.Explainer,
    X: pd.DataFrame,
    sample_size: Optional[int] = None
) -> np.ndarray:
    """
    Calculate SHAP values for the dataset.

    Args:
        explainer: SHAP explainer object
        X: Feature dataset
        sample_size: Number of samples to explain (None for all)

    Returns:
        SHAP values array
    """
    logger.info("Calculating SHAP values")

    if sample_size and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=config.RANDOM_STATE)
    else:
        X_sample = X

    shap_values = explainer.shap_values(X_sample)

    # Handle multi-output case (e.g., from binary classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Get values for positive class

    logger.info(f"Calculated SHAP values for {len(X_sample)} samples")

    return shap_values, X_sample


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    output_path: Optional[Path] = None,
    max_display: int = config.SHAP_MAX_DISPLAY
) -> None:
    """
    Create SHAP summary plot (beeswarm plot).

    Args:
        shap_values: SHAP values array
        X: Feature dataset
        output_path: Path to save the plot
        max_display: Maximum number of features to display
    """
    logger.info("Creating SHAP summary plot")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()

    if output_path:
        output_path = output_path or config.FIGURES_DIR / 'shap_summary_plot.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        logger.info(f"Saved SHAP summary plot to {output_path}")

    plt.close()


def plot_shap_bar(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    output_path: Optional[Path] = None,
    max_display: int = config.SHAP_MAX_DISPLAY
) -> pd.DataFrame:
    """
    Create SHAP bar plot showing feature importance.

    Args:
        shap_values: SHAP values array
        X: Feature dataset
        output_path: Path to save the plot
        max_display: Maximum number of features to display

    Returns:
        DataFrame with feature importance
    """
    logger.info("Creating SHAP bar plot")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=max_display,
        show=False
    )
    plt.tight_layout()

    if output_path:
        output_path = output_path or config.FIGURES_DIR / 'shap_bar_plot.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        logger.info(f"Saved SHAP bar plot to {output_path}")

    plt.close()

    # Calculate mean absolute SHAP values
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    return feature_importance


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: Optional[str] = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Create SHAP dependence plot for a specific feature.

    Args:
        shap_values: SHAP values array
        X: Feature dataset
        feature: Feature to plot
        interaction_feature: Feature for interaction coloring
        output_path: Path to save the plot
    """
    logger.info(f"Creating SHAP dependence plot for {feature}")

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        shap_values,
        X,
        interaction_index=interaction_feature,
        show=False
    )
    plt.tight_layout()

    if output_path:
        feature_name_clean = feature.replace('/', '_').replace(' ', '_')
        output_path = output_path or config.FIGURES_DIR / f'shap_dependence_{feature_name_clean}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        logger.info(f"Saved SHAP dependence plot to {output_path}")

    plt.close()


def plot_shap_waterfall(
    explainer: shap.Explainer,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    index: int = 0,
    output_path: Optional[Path] = None
) -> None:
    """
    Create SHAP waterfall plot for a single prediction.

    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values array
        X: Feature dataset
        index: Index of the sample to explain
        output_path: Path to save the plot
    """
    logger.info(f"Creating SHAP waterfall plot for sample {index}")

    plt.figure(figsize=(10, 8))

    # Create explanation object
    if hasattr(explainer, 'expected_value'):
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

        shap_exp = shap.Explanation(
            values=shap_values[index],
            base_values=expected_value,
            data=X.iloc[index].values,
            feature_names=X.columns.tolist()
        )

        shap.plots.waterfall(shap_exp, show=False)
    else:
        # Fallback for simpler visualization
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[index],
            X.iloc[index],
            matplotlib=True,
            show=False
        )

    plt.tight_layout()

    if output_path:
        output_path = output_path or config.FIGURES_DIR / f'shap_waterfall_sample_{index}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=config.FIGURE_DPI)
        logger.info(f"Saved SHAP waterfall plot to {output_path}")

    plt.close()


def get_top_features(
    shap_values: np.ndarray,
    feature_names: list,
    n_top: int = 10
) -> pd.DataFrame:
    """
    Get top N most important features based on SHAP values.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        n_top: Number of top features to return

    Returns:
        DataFrame with top features and their importance
    """
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'mean_shap': shap_values.mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    return feature_importance.head(n_top)


def generate_business_insights(
    feature_importance: pd.DataFrame,
    n_top: int = 5
) -> str:
    """
    Generate business-friendly insights from feature importance.

    Args:
        feature_importance: DataFrame with feature importance
        n_top: Number of top features to analyze

    Returns:
        String with business insights
    """
    top_features = feature_importance.head(n_top)

    insights = "KEY CHURN DRIVERS:\n\n"

    feature_insights = {
        'tenure': 'Customer tenure is a critical factor - newer customers are at higher risk',
        'Contract': 'Contract type significantly impacts churn - month-to-month contracts show higher risk',
        'MonthlyCharges': 'Higher monthly charges correlate with increased churn risk',
        'TotalCharges': 'Total amount paid indicates customer value and loyalty',
        'InternetService': 'Internet service type affects churn - fiber optic users may have different expectations',
        'OnlineSecurity': 'Lack of online security service increases churn likelihood',
        'TechSupport': 'Customers without tech support are more likely to churn',
        'PaymentMethod': 'Electronic check users show higher churn rates',
        'PaperlessBilling': 'Paperless billing preference correlates with churn behavior'
    }

    for idx, row in top_features.iterrows():
        feature = row['feature']
        # Handle both SHAP format (mean_abs_shap) and feature importance format (importance)
        importance = row.get('mean_abs_shap', row.get('importance', 0))

        # Find matching insight
        insight = next(
            (v for k, v in feature_insights.items() if k.lower() in feature.lower()),
            f'{feature} is a significant predictor of customer churn'
        )

        insights += f"{idx + 1}. {insight}\n"

    insights += "\nACTIONABLE RECOMMENDATIONS:\n"
    insights += "• Focus retention efforts on month-to-month contract customers\n"
    insights += "• Offer discounts or value-added services to high monthly charge customers\n"
    insights += "• Implement early engagement programs for new customers (low tenure)\n"
    insights += "• Promote tech support and security services to at-risk customers\n"
    insights += "• Encourage contract upgrades with incentives\n"

    return insights


def explainability_pipeline(
    save_plots: bool = True,
    n_samples: int = config.SHAP_SAMPLE_SIZE
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete explainability pipeline.

    Args:
        save_plots: Whether to save plots to disk
        n_samples: Number of samples for SHAP analysis

    Returns:
        Tuple of (shap_values, X_sample, feature_importance)
    """
    logger.info("="*60)
    logger.info("Starting Explainability Pipeline")
    logger.info("="*60)

    # Create output directory
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and data
    model, X_test, feature_names, y_test = load_model_and_data()

    # Create explainer
    explainer = create_shap_explainer(model, X_test, sample_size=100)

    # Check if SHAP explainer was created successfully
    if explainer is None:
        logger.warning("SHAP explainer creation failed. Using model feature importances as fallback.")
        # Use model's feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            # Save feature importance
            config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            feature_importance.to_csv(config.REPORTS_DIR / 'shap_feature_importance.csv', index=False)
            logger.info("Saved feature importances (from model, not SHAP)")

            # Generate basic insights
            insights = generate_business_insights(feature_importance)
            logger.info(f"\n{insights}")

            # Save insights
            insights_file = config.REPORTS_DIR / 'business_insights.txt'
            with open(insights_file, 'w') as f:
                f.write(insights)
            logger.info(f"Saved business insights to {insights_file}")

            # Create a minimal shap_objects file so dashboard doesn't crash
            # Set shap_data to None so dashboard knows SHAP is not available
            joblib.dump({
                'shap_values': None,
                'X_sample': None,
                'explainer': None
            }, config.MODELS_DIR / 'shap_objects.joblib')
            logger.info("Created minimal shap_objects.joblib (SHAP not available)")

            logger.info("="*60)
            logger.info("✓ Explainability Pipeline Complete (Feature Importances Only)")
            logger.info("="*60)

            return None, None, feature_importance
        else:
            logger.error("Model does not have feature_importances_ attribute. Cannot generate explanations.")
            return None, None, None

    # Calculate SHAP values
    shap_values, X_sample = calculate_shap_values(explainer, X_test, sample_size=n_samples)

    # Generate plots
    if save_plots:
        plot_shap_summary(shap_values, X_sample, config.FIGURES_DIR / 'shap_summary_plot.png')
        feature_importance = plot_shap_bar(shap_values, X_sample, config.FIGURES_DIR / 'shap_bar_plot.png')

        # Dependence plots for top 3 features
        top_features = get_top_features(shap_values, X_sample.columns.tolist(), n_top=3)
        for idx, row in top_features.iterrows():
            feature = row['feature']
            plot_shap_dependence(
                shap_values,
                X_sample,
                feature,
                output_path=config.FIGURES_DIR / f'shap_dependence_{idx}.png'
            )

        # Waterfall plots for a few examples
        for i in [0, 1, 2]:
            if i < len(X_sample):
                plot_shap_waterfall(
                    explainer,
                    shap_values,
                    X_sample,
                    index=i,
                    output_path=config.FIGURES_DIR / f'shap_waterfall_{i}.png'
                )

    else:
        feature_importance = plot_shap_bar(shap_values, X_sample)

    # Generate business insights
    insights = generate_business_insights(feature_importance)
    logger.info(f"\n{insights}")

    # Save insights to file
    insights_file = config.REPORTS_DIR / 'business_insights.txt'
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(insights_file, 'w') as f:
        f.write(insights)
    logger.info(f"Saved business insights to {insights_file}")

    # Save feature importance
    feature_importance.to_csv(config.REPORTS_DIR / 'shap_feature_importance.csv', index=False)

    # Save SHAP values and sample data for dashboard
    joblib.dump({
        'shap_values': shap_values,
        'X_sample': X_sample,
        'explainer': explainer
    }, config.MODELS_DIR / 'shap_objects.joblib')

    logger.info("="*60)
    logger.info("✓ Explainability Pipeline Complete")
    logger.info("="*60)

    return shap_values, X_sample, feature_importance


if __name__ == "__main__":
    explainability_pipeline()
