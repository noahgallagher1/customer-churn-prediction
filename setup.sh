#!/bin/bash
#
# Quick Setup Script for Customer Churn Prediction Project
#
# This script automates the initial setup process

set -e  # Exit on error

echo "=========================================="
echo "Customer Churn Prediction - Quick Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 is required"; exit 1; }
echo "✓ Python found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw data/processed models outputs/figures outputs/reports logs
echo "✓ Directories created"
echo ""

# Summary
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the full pipeline:"
echo "     python run_pipeline.py"
echo ""
echo "  3. Or run individual steps:"
echo "     python run_pipeline.py --only-download"
echo "     python run_pipeline.py --only-processing"
echo "     python run_pipeline.py --only-training"
echo "     python run_pipeline.py --only-explainability"
echo ""
echo "  4. Launch the dashboard:"
echo "     streamlit run src/dashboard.py"
echo ""
echo "  5. Explore the EDA notebook:"
echo "     jupyter notebook notebooks/01_exploratory_data_analysis.ipynb"
echo ""
echo "=========================================="
