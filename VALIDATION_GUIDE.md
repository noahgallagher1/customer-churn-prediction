# Validation Guide - Post Audit Fixes

This guide walks you through validating all the fixes from the Senior DS audit report.

---

## 🎯 Quick Start

**You are here:** All code fixes have been committed to the branch.

**What you need to do:**
1. Pull the latest code
2. Run cleanup and regeneration script
3. Validate outputs match expected values
4. Commit regenerated artifacts

**Time Required:** ~15-20 minutes

---

## 📋 What Was Fixed

### ✅ CRITICAL ISSUE #1: Dual ROI Calculation (FIXED)
- **File:** `src/model_training.py` (lines 290-359)
- **Fix:** Replaced old calculation with call to `calculate_enhanced_roi()`
- **Impact:** Now uses correct formula accounting for 65% success rate and ALL campaigns (TP+FP)

### ✅ CRITICAL ISSUE #2: Segment Analysis Bug (FIXED)
- **Files:** `src/run_advanced_evaluation.py`, `src/model_evaluation.py`
- **Fix:** Load raw unscaled data instead of processed scaled data
- **Impact:** Segments now properly separate customers (e.g., 5 tenure groups instead of 1)

### ✅ HIGH PRIORITY: Dashboard Error Handling (FIXED)
- **File:** `app.py` (line 671-690)
- **Fix:** Fail fast with clear error instead of silent fallback to wrong metrics
- **Impact:** Prevents displaying incorrect metrics to stakeholders

### ✅ MEDIUM PRIORITY: Terminology Standardization (FIXED)
- **File:** `GLOSSARY.md` (NEW)
- **Fix:** Created comprehensive glossary defining all metrics
- **Impact:** Eliminates confusion about "customers saved" vs "customers identified"

---

## 🚀 Step-by-Step Validation Process

### Prerequisites

Make sure you have:
- Python 3.8+ installed
- Git installed
- Cloned repository on your local machine

---

### Step 1: Pull Latest Code

Open your terminal and navigate to the project directory:

```bash
cd /path/to/customer-churn-prediction

# Pull the latest changes from the branch
git pull origin claude/analyze-pro-features-011CV53tkyWNsNoAdgeaiCYa
```

**Verify you have the latest code:**
```bash
# Check that these files exist
ls -l SENIOR_DS_AUDIT_REPORT.md
ls -l SEGMENT_ANALYSIS_FIX.md
ls -l GLOSSARY.md
ls -l cleanup_and_regenerate.sh
ls -l VALIDATION_GUIDE.md  # This file!
```

You should see all 5 files with recent timestamps.

---

### Step 2: Install Dependencies (If Needed)

```bash
# Create virtual environment (if you haven't already)
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import pandas, numpy, sklearn, xgboost, lightgbm, shap; print('✅ All packages installed')"
```

---

### Step 3: Run Cleanup and Regeneration Script

**IMPORTANT:** This will delete old files and regenerate everything. A backup will be created automatically.

```bash
./cleanup_and_regenerate.sh
```

**What this script does:**
1. Creates backup of old files (in `backup_YYYYMMDD_HHMMSS/`)
2. Removes old `model_metrics.joblib` (had incorrect ROI)
3. Removes old segment analysis CSVs (had data bugs)
4. Runs `python src/model_training.py` (generates new model with enhanced ROI)
5. Runs `python src/run_advanced_evaluation.py` (generates segments with raw data)
6. Validates outputs automatically

**Expected output:**
```
==================================================
CLEANUP AND REGENERATION COMPLETE!
==================================================

Validation results:
  ✓ Enhanced ROI analysis generated
    - Customers Saved: 226 (expected: 226)  ✅
    - Total Campaigns: 851 (expected: 851)  ✅
    - ROI: 431% (expected: ~432%)           ✅

  Checking segment analyses...
    - contract_type_analysis.csv: 3 segments  ✅ CORRECT
    - tenure_group_analysis.csv: 5 segments   ✅ CORRECT
    - monthly_charges_analysis.csv: 4 segments ✅ CORRECT
    - gender_analysis.csv: 2 segments          ✅ CORRECT
```

**If you see errors:**
- Check the error message
- Verify you're in the correct directory
- Ensure all dependencies are installed
- Try running the Python commands manually (see Step 4)

---

### Step 4: Manual Regeneration (If Script Fails)

If the cleanup script fails, run commands manually:

```bash
# Step 4a: Backup old files
mkdir -p backup_manual
cp models/model_metrics.joblib backup_manual/ 2>/dev/null || true
cp -r outputs/reports backup_manual/ 2>/dev/null || true

# Step 4b: Remove old files
rm -f models/model_metrics.joblib
rm -f outputs/reports/contract_type_analysis.csv
rm -f outputs/reports/tenure_group_analysis.csv
rm -f outputs/reports/monthly_charges_analysis.csv
rm -f outputs/reports/gender_analysis.csv
rm -f outputs/reports/senior_citizen_analysis.csv
rm -f outputs/reports/advanced_evaluation_summary.txt

# Step 4c: Regenerate model
python src/model_training.py

# Step 4d: Regenerate advanced evaluation
python src/run_advanced_evaluation.py
```

---

### Step 5: Validate Regenerated Files

Run these validation checks:

#### 5.1: Check Enhanced ROI Analysis

```bash
# Display the enhanced ROI metrics
python -c "
import pandas as pd
df = pd.read_csv('outputs/reports/enhanced_roi_analysis.csv')
print('Enhanced ROI Analysis:')
print(f'  Customers Saved: {int(df[\"customers_saved\"].values[0])}')
print(f'  Total Campaigns: {int(df[\"total_campaigns\"].values[0])}')
print(f'  Campaign Cost: \${df[\"campaign_execution_cost\"].values[0]:,.0f}')
print(f'  Revenue Saved: \${df[\"revenue_saved\"].values[0]:,.0f}')
print(f'  Net Benefit: \${df[\"net_benefit\"].values[0]:,.0f}')
print(f'  ROI: {df[\"roi_percentage\"].values[0]:.1f}%')
"
```

**Expected output:**
```
Enhanced ROI Analysis:
  Customers Saved: 226
  Total Campaigns: 851
  Campaign Cost: $85,100
  Revenue Saved: $452,400
  Net Benefit: $367,300
  ROI: 431.6%
```

✅ **Pass Criteria:**
- Customers Saved = 226 (NOT 348)
- Total Campaigns = 851 (TP + FP)
- ROI ≈ 431.6%

---

#### 5.2: Check Segment Analysis

```bash
# Check contract type segments
echo "Contract Type Segments:"
cat outputs/reports/contract_type_analysis.csv | column -t -s,

# Check tenure group segments
echo -e "\nTenure Group Segments:"
cat outputs/reports/tenure_group_analysis.csv | column -t -s,

# Check monthly charges segments
echo -e "\nMonthly Charges Segments:"
cat outputs/reports/monthly_charges_analysis.csv | column -t -s,

# Check gender segments
echo -e "\nGender Segments:"
cat outputs/reports/gender_analysis.csv | column -t -s,
```

**Expected output (Contract Type example):**
```
Contract Type Segments:
Segment          N    Churn Rate  Precision  Recall  F1     FPR
Month-to-month   XXX  0.XXX       0.XXX      0.XXX   0.XXX  0.XXX
One year         XXX  0.XXX       0.XXX      0.XXX   0.XXX  0.XXX
Two year         XXX  0.XXX       0.XXX      0.XXX   0.XXX  0.XXX
```

✅ **Pass Criteria:**
- Contract Type: **3 segments** (Month-to-month, One year, Two year)
- Tenure Group: **5 segments** (<12mo, 12-24mo, 24-36mo, 36-48mo, >48mo)
- Monthly Charges: **4 segments** (Low, Medium, High, Very High)
- Gender: **2 segments** (Female, Male) - NOT "0.0"!
- Senior Citizen: **2 segments** (No, Yes) - NOT empty!
- All segment N values sum to **1,409** (test set size)

---

#### 5.3: Check Summary Report

```bash
# View the advanced evaluation summary
cat outputs/reports/advanced_evaluation_summary.txt
```

**Look for these key sections:**
```
## 5. SEGMENT-LEVEL PERFORMANCE

### Contract Type

            Segment    N  Churn Rate  Precision  Recall     F1     FPR
    Month-to-month  XXX      0.XXX      0.XXX   0.XXX  0.XXX   0.XXX
         One year   XXX      0.XXX      0.XXX   0.XXX  0.XXX   0.XXX
         Two year   XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX

**Best Performance**: Two year (F1=0.XXX)
**Worst Performance**: Month-to-month (F1=0.XXX)
```

✅ **Pass Criteria:**
- "Best" and "Worst" segments are **DIFFERENT** (not both "<12 months"!)
- Each segment section shows **multiple rows**
- No "Only one segment found" warnings

---

#### 5.4: Verify Model Metrics File

```bash
# Check that model_metrics.joblib has correct values
python -c "
import joblib
metrics = joblib.load('models/model_metrics.joblib')
print('Model Metrics File:')
print(f'  Customers Saved: {metrics.get(\"customers_saved\", \"N/A\")}')
print(f'  Total Campaigns: {metrics.get(\"total_campaigns\", \"N/A\")}')
print(f'  ROI: {metrics.get(\"roi_percentage\", \"N/A\"):.1f}%')
print(f'  Campaign Success Rate: {metrics.get(\"campaign_success_rate\", \"N/A\")}')
"
```

**Expected output:**
```
Model Metrics File:
  Customers Saved: 226
  Total Campaigns: 851
  ROI: 431.6%
  Campaign Success Rate: 0.65
```

✅ **Pass Criteria:**
- Customers Saved = 226 (matches enhanced ROI)
- Has `total_campaigns` field
- Has `campaign_success_rate` field

---

### Step 6: Test the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run app.py
```

**Navigate to:** `http://localhost:8501`

**Validate these dashboard metrics:**

1. **Executive Summary Page:**
   - Net Savings tile shows: **~$367,300**
   - ROI gauge shows: **431.6%**
   - Customers Saved shows: **226** (NOT 348!)

2. **Model Performance Page:**
   - Check "ROI Highlights" section exists
   - Metrics match enhanced ROI analysis
   - No error messages about missing enhanced reports

3. **Customer Risk Scoring Page:**
   - Segment analysis tables show multiple segments
   - Gender shows "Male"/"Female" (not "0.0")
   - No "only one segment found" messages

**If you see an error:**
```
⚠️ Enhanced ROI Analysis Not Found

The dashboard requires enhanced ROI metrics to ensure accuracy.
Please run: python src/run_advanced_evaluation.py
```

This means the enhanced reports weren't generated. Go back to Step 3.

---

### Step 7: Final Validation Checklist

Use this checklist to confirm everything is correct:

#### Metrics Consistency ✅
- [ ] `model_metrics.joblib` exists with customers_saved = 226
- [ ] `enhanced_roi_analysis.csv` shows customers_saved = 226.2
- [ ] README claims match regenerated reports
- [ ] Dashboard displays match backend calculations
- [ ] All "customers saved" references = 226 (not 348)

#### Segment Analysis ✅
- [ ] Contract Type analysis shows **3 segments** (not 1)
- [ ] Tenure analysis shows **5 segments** (not 1)
- [ ] Monthly Charges shows **4 segments** (not 1)
- [ ] Gender analysis shows **"Male"/"Female"** (not "0.0")
- [ ] Senior Citizen analysis returns data (not empty)
- [ ] All segment N values sum to **1,409**
- [ ] "Best" and "Worst" segments are **different**
- [ ] F1 scores **vary** across segments

#### Code Quality ✅
- [ ] No errors when running model_training.py
- [ ] No errors when running run_advanced_evaluation.py
- [ ] Dashboard loads without errors
- [ ] All segment CSVs generated

#### Documentation ✅
- [ ] SENIOR_DS_AUDIT_REPORT.md exists
- [ ] SEGMENT_ANALYSIS_FIX.md exists
- [ ] GLOSSARY.md exists
- [ ] This VALIDATION_GUIDE.md is accurate

---

### Step 8: Commit Regenerated Artifacts

Once all validations pass:

```bash
# Stage all changed files
git add models/model_metrics.joblib
git add outputs/reports/*.csv
git add outputs/reports/*.txt
git add outputs/reports/*.png

# Check what will be committed
git status

# Commit with clear message
git commit -m "Regenerate all artifacts with audit fixes

- model_metrics.joblib now uses enhanced ROI (customers_saved=226)
- All segment analyses regenerated with raw data (multiple segments)
- Validated all outputs match expected values
- See VALIDATION_GUIDE.md for validation results

Fixes:
- CRITICAL: Dual ROI calculation (SENIOR_DS_AUDIT_REPORT.md Issue #1)
- CRITICAL: Segment analysis data bug (Issue #2)
- All metrics now consistent across codebase"

# Push to remote
git push origin claude/analyze-pro-features-011CV53tkyWNsNoAdgeaiCYa
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: data/raw/Telco-Customer-Churn.csv"

**Solution:**
```bash
# Make sure you're in the project root directory
pwd  # Should show .../customer-churn-prediction

# Check if data file exists
ls -l data/raw/Telco-Customer-Churn.csv
```

### Issue: Segment analysis still shows only 1 segment

**Check:**
1. Did you pull the latest code? `git pull`
2. Did you run the regeneration script?
3. Check the logs in outputs/model_training.log

**Debug:**
```bash
# Check if raw data loading works
python -c "
from src.data_processing import load_raw_data, clean_data
df = load_raw_data()
df = clean_data(df)
print(f'Contract unique values: {df[\"Contract\"].unique()}')
print(f'Tenure range: {df[\"tenure\"].min()} to {df[\"tenure\"].max()}')
"
```

### Issue: Dashboard shows old metrics

**Solution:**
1. Stop the dashboard (Ctrl+C)
2. Delete cached data:
   ```bash
   rm -rf ~/.streamlit/cache
   ```
3. Restart dashboard:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Expected vs Actual Comparison

| Metric | Before Fix (WRONG) | After Fix (CORRECT) | Status |
|--------|-------------------|---------------------|--------|
| Customers Saved | 348 | 226 | ✅ Fixed |
| Total Campaigns | 348 | 851 | ✅ Fixed |
| Campaign Cost | $34,800 | $85,100 | ✅ Fixed |
| ROI | ~1,200%+ | 431.6% | ✅ Fixed |
| Contract segments | 1 | 3 | ✅ Fixed |
| Tenure segments | 1 | 5 | ✅ Fixed |
| Gender display | "0.0" | "Male"/"Female" | ✅ Fixed |

---

## 🎓 Understanding the Fixes

### Why did customers_saved change from 348 to 226?

**Old (Wrong):**
```python
customers_saved = TP  # Assumes 100% success
```
This assumed every retention campaign succeeded (unrealistic).

**New (Correct):**
```python
customers_saved = TP × success_rate = 348 × 0.65 = 226
```
This accounts for the fact that only 65% of campaigns successfully retain customers.

---

### Why did campaign cost increase from $34,800 to $85,100?

**Old (Wrong):**
```python
cost = TP × $100 = 348 × $100 = $34,800
```
This only counted campaigns for true positives, ignoring false positives.

**New (Correct):**
```python
cost = (TP + FP) × $100 = (348 + 503) × $100 = $85,100
```
In reality, we run campaigns on ALL flagged customers, including false alarms.

---

### Why did ROI drop from ~1,200% to 431.6%?

The old calculation was inflated due to:
1. Overcounting customers saved (348 vs 226)
2. Undercounting campaign costs ($34,800 vs $85,100)

The new calculation is **realistic** and still shows **excellent ROI**.

Even at 431.6%, this means every $1 spent returns $5.32!

---

## 📚 Reference Documents

- **Full Audit Report:** `SENIOR_DS_AUDIT_REPORT.md`
- **Segment Fix Details:** `SEGMENT_ANALYSIS_FIX.md`
- **Terminology Reference:** `GLOSSARY.md`
- **Generated Reports:** `outputs/reports/`

---

## ✅ Success Criteria

You're done when:
- ✅ All validation checks pass
- ✅ Dashboard displays correct metrics (226 customers saved, 431.6% ROI)
- ✅ Segment analyses show multiple segments per category
- ✅ All artifacts committed and pushed
- ✅ README claims match actual metrics

---

## 🤝 Need Help?

If you encounter issues:
1. Check the Troubleshooting section above
2. Review the audit report (SENIOR_DS_AUDIT_REPORT.md)
3. Check the logs: `outputs/model_training.log`
4. Verify your Python environment: `python --version` (should be 3.8+)

---

**Last Updated:** 2025-11-13
**Audit Reference:** SENIOR_DS_AUDIT_REPORT.md
**Branch:** claude/analyze-pro-features-011CV53tkyWNsNoAdgeaiCYa
