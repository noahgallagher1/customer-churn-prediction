#!/usr/bin/env python3
"""
Main Pipeline Execution Script for Customer Churn Prediction.

This script orchestrates the complete end-to-end pipeline:
1. Data download
2. Data processing and feature engineering
3. Model training and evaluation
4. Explainability analysis (SHAP)
"""

import sys
import logging
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import config
from download_data import download_telco_churn_data
from data_processing import process_pipeline
from model_training import training_pipeline
from explainability import explainability_pipeline

# Configure logging
config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_full_pipeline(skip_download: bool = False, skip_processing: bool = False,
                      skip_training: bool = False, skip_explainability: bool = False):
    """
    Execute the complete machine learning pipeline.

    Args:
        skip_download: Skip data download step
        skip_processing: Skip data processing step
        skip_training: Skip model training step
        skip_explainability: Skip explainability analysis step
    """
    logger.info("="*80)
    logger.info("CUSTOMER CHURN PREDICTION - FULL PIPELINE EXECUTION")
    logger.info("="*80)

    try:
        # Step 1: Download Data
        if not skip_download:
            logger.info("\n" + "="*80)
            logger.info("STEP 1/4: DATA DOWNLOAD")
            logger.info("="*80)
            download_telco_churn_data()
        else:
            logger.info("\n[SKIPPED] Step 1: Data Download")

        # Step 2: Data Processing
        if not skip_processing:
            logger.info("\n" + "="*80)
            logger.info("STEP 2/4: DATA PROCESSING & FEATURE ENGINEERING")
            logger.info("="*80)
            process_pipeline(force_reprocess=True)
        else:
            logger.info("\n[SKIPPED] Step 2: Data Processing")

        # Step 3: Model Training
        if not skip_training:
            logger.info("\n" + "="*80)
            logger.info("STEP 3/4: MODEL TRAINING & EVALUATION")
            logger.info("="*80)
            training_pipeline()
        else:
            logger.info("\n[SKIPPED] Step 3: Model Training")

        # Step 4: Explainability Analysis
        if not skip_explainability:
            logger.info("\n" + "="*80)
            logger.info("STEP 4/4: EXPLAINABILITY ANALYSIS (SHAP)")
            logger.info("="*80)
            explainability_pipeline()
        else:
            logger.info("\n[SKIPPED] Step 4: Explainability Analysis")

        # Success summary
        logger.info("\n" + "="*80)
        logger.info("✓ PIPELINE EXECUTION COMPLETE!")
        logger.info("="*80)
        logger.info("\nGenerated Artifacts:")
        logger.info(f"  • Processed Data: {config.PROCESSED_DATA_DIR}")
        logger.info(f"  • Trained Models: {config.MODELS_DIR}")
        logger.info(f"  • Visualizations: {config.FIGURES_DIR}")
        logger.info(f"  • Reports: {config.REPORTS_DIR}")
        logger.info(f"  • Logs: {config.LOG_FILE}")

        logger.info("\nNext Steps:")
        logger.info("  1. Review EDA notebook: jupyter notebook notebooks/01_exploratory_data_analysis.ipynb")
        logger.info("  2. Launch dashboard: streamlit run src/dashboard.py")
        logger.info("  3. Check model metrics: cat models/model_metrics.joblib")
        logger.info("\n" + "="*80)

        return True

    except Exception as e:
        logger.error(f"\n✗ PIPELINE FAILED: {e}", exc_info=True)
        return False


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py

  # Run specific steps
  python run_pipeline.py --skip-download --skip-processing

  # Run only data processing
  python run_pipeline.py --only-processing

  # Run only training
  python run_pipeline.py --only-training
        """
    )

    # Skip options
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download step')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip data processing step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training step')
    parser.add_argument('--skip-explainability', action='store_true',
                       help='Skip explainability analysis step')

    # Only options (run single step)
    parser.add_argument('--only-download', action='store_true',
                       help='Run only data download')
    parser.add_argument('--only-processing', action='store_true',
                       help='Run only data processing')
    parser.add_argument('--only-training', action='store_true',
                       help='Run only model training')
    parser.add_argument('--only-explainability', action='store_true',
                       help='Run only explainability analysis')

    args = parser.parse_args()

    # Handle "only" options
    if args.only_download:
        download_telco_churn_data()
    elif args.only_processing:
        process_pipeline(force_reprocess=True)
    elif args.only_training:
        training_pipeline()
    elif args.only_explainability:
        explainability_pipeline()
    else:
        # Run full or partial pipeline
        success = run_full_pipeline(
            skip_download=args.skip_download,
            skip_processing=args.skip_processing,
            skip_training=args.skip_training,
            skip_explainability=args.skip_explainability
        )

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
