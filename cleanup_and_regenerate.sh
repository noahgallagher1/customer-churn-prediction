#!/bin/bash

# Cleanup and Regeneration Script
# This script removes old/stale model artifacts and regenerates everything with correct calculations

set -e  # Exit on error

echo "=================================================="
echo "CLEANUP AND REGENERATION SCRIPT"
echo "=================================================="
echo ""
echo "This script will:"
echo "  1. Remove old model artifacts (potentially incorrect metrics)"
echo "  2. Remove old segment analysis reports (had bugs)"
echo "  3. Regenerate model with CORRECT enhanced ROI calculation"
echo "  4. Regenerate segment analysis with fixed data handling"
echo "  5. Validate all outputs"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Step 1: Backup old files (just in case)
echo ""
echo "Step 1: Creating backup of old files..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
if [ -f "models/model_metrics.joblib" ]; then
    cp models/model_metrics.joblib "$BACKUP_DIR/"
    echo "  ✓ Backed up model_metrics.joblib"
fi
if [ -d "outputs/reports" ]; then
    cp -r outputs/reports "$BACKUP_DIR/"
    echo "  ✓ Backed up outputs/reports/"
fi
echo "  Backup saved to: $BACKUP_DIR/"

# Step 2: Remove old potentially incorrect files
echo ""
echo "Step 2: Removing old files..."

# Remove old model metrics (will be regenerated with correct ROI)
if [ -f "models/model_metrics.joblib" ]; then
    rm models/model_metrics.joblib
    echo "  ✓ Removed models/model_metrics.joblib (will regenerate with enhanced ROI)"
fi

# Remove old segment analysis reports (had bugs using scaled data)
SEGMENT_FILES=(
    "outputs/reports/contract_type_analysis.csv"
    "outputs/reports/tenure_group_analysis.csv"
    "outputs/reports/monthly_charges_analysis.csv"
    "outputs/reports/gender_analysis.csv"
    "outputs/reports/senior_citizen_analysis.csv"
    "outputs/reports/advanced_evaluation_summary.txt"
)

for file in "${SEGMENT_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "  ✓ Removed $file (will regenerate with raw data)"
    fi
done

# Step 3: Regenerate model artifacts with CORRECT ROI calculation
echo ""
echo "Step 3: Regenerating model with enhanced ROI calculation..."
echo "  This will use the fixed calculate_business_metrics() function"
echo "  Expected: customers_saved = 226 (not 348)"
echo ""
python src/model_training.py

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Model training failed!"
    exit 1
fi

echo ""
echo "  ✓ Model training complete with enhanced ROI"

# Step 4: Regenerate segment analysis with FIXED data handling
echo ""
echo "Step 4: Regenerating segment analysis with raw data..."
echo "  This will use unscaled/unencoded data for proper segmentation"
echo "  Expected: Multiple segments per category (not just 1)"
echo ""
python src/run_advanced_evaluation.py

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Advanced evaluation failed!"
    exit 1
fi

echo ""
echo "  ✓ Advanced evaluation complete with fixed segments"

# Step 5: Validate outputs
echo ""
echo "Step 5: Validating outputs..."
echo ""

# Check that enhanced ROI analysis exists
if [ -f "outputs/reports/enhanced_roi_analysis.csv" ]; then
    echo "  ✓ Enhanced ROI analysis generated"

    # Extract key metrics for validation
    CUSTOMERS_SAVED=$(awk -F',' 'NR==2 {print $9}' outputs/reports/enhanced_roi_analysis.csv | cut -d'.' -f1)
    TOTAL_CAMPAIGNS=$(awk -F',' 'NR==2 {print $5}' outputs/reports/enhanced_roi_analysis.csv)
    ROI=$(awk -F',' 'NR==2 {print $15}' outputs/reports/enhanced_roi_analysis.csv | cut -d'.' -f1)

    echo "    - Customers Saved: $CUSTOMERS_SAVED (expected: 226)"
    echo "    - Total Campaigns: $TOTAL_CAMPAIGNS (expected: 851)"
    echo "    - ROI: ${ROI}% (expected: ~432%)"

    if [ "$CUSTOMERS_SAVED" -eq 226 ] 2>/dev/null; then
        echo "    ✅ Customers saved is CORRECT (226)"
    else
        echo "    ⚠️  Customers saved may be incorrect (got $CUSTOMERS_SAVED, expected 226)"
    fi
else
    echo "  ❌ Enhanced ROI analysis NOT found!"
    exit 1
fi

# Check that segment analyses have multiple segments
echo ""
echo "  Checking segment analyses..."

for file in outputs/reports/contract_type_analysis.csv \
           outputs/reports/tenure_group_analysis.csv \
           outputs/reports/monthly_charges_analysis.csv \
           outputs/reports/gender_analysis.csv; do
    if [ -f "$file" ]; then
        SEGMENT_COUNT=$(tail -n +2 "$file" | wc -l)
        FILENAME=$(basename "$file")
        echo "    - $FILENAME: $SEGMENT_COUNT segments"

        # Validate expected segment counts
        case "$FILENAME" in
            "contract_type_analysis.csv")
                [ "$SEGMENT_COUNT" -eq 3 ] && echo "      ✅ CORRECT (expected 3)" || echo "      ⚠️  Expected 3 segments"
                ;;
            "tenure_group_analysis.csv")
                [ "$SEGMENT_COUNT" -eq 5 ] && echo "      ✅ CORRECT (expected 5)" || echo "      ⚠️  Expected 5 segments"
                ;;
            "monthly_charges_analysis.csv")
                [ "$SEGMENT_COUNT" -eq 4 ] && echo "      ✅ CORRECT (expected 4)" || echo "      ⚠️  Expected 4 segments"
                ;;
            "gender_analysis.csv")
                [ "$SEGMENT_COUNT" -eq 2 ] && echo "      ✅ CORRECT (expected 2)" || echo "      ⚠️  Expected 2 segments"
                ;;
        esac
    else
        echo "    ❌ $file NOT found!"
    fi
done

echo ""
echo "=================================================="
echo "✅ CLEANUP AND REGENERATION COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Review the validation results above"
echo "  2. Check the detailed report: outputs/reports/advanced_evaluation_summary.txt"
echo "  3. Review VALIDATION_GUIDE.md for complete validation checklist"
echo "  4. If all validations pass, commit and push:"
echo "     git add -A"
echo "     git commit -m 'Regenerate all artifacts with fixed calculations'"
echo "     git push"
echo ""
echo "Backup of old files saved in: $BACKUP_DIR/"
echo "=================================================="
