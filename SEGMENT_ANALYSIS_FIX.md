# Segment Analysis Bug Fix

## Problem Identified

The segment analysis was showing **identical F1 scores** for "best" and "worst" performance because all rows were falling into a single default segment. This was caused by using **scaled and encoded data** instead of raw original data.

### Specific Issues:
1. **Tenure values** were z-scores (e.g., -0.992) instead of months (e.g., 34 months)
2. **MonthlyCharges** were z-scores (e.g., 1.176) instead of dollars (e.g., $56.95)
3. **Gender** showed as "0.0" instead of "Male"/"Female"
4. **Contract** was one-hot encoded, making reconstruction unreliable
5. All segments defaulted to a single category, showing identical metrics

### Example of the Bug:
```
Before Fix (WRONG):
   Segment   N  Churn Rate  Precision   Recall       F1      FPR
<12 months 644    0.113354   0.240175 0.753425 0.364238 0.304729

✨ Key Insight: Model performs best on <12 months (F1=0.364) and worst on <12 months (F1=0.364).
```
**Same segment!** Only showing 1 of 5 expected tenure groups!

## Root Cause

In `src/run_advanced_evaluation.py` line 330:
```python
# OLD - WRONG:
test_data = pd.read_csv(config.TEST_DATA_FILE)  # This is PROCESSED data!

# The file has:
# - tenure = -0.992... (scaled, not months!)
# - MonthlyCharges = 1.176... (scaled, not dollars!)
# - gender = 0.0 (encoded, not "Male"/"Female"!)
```

When `analyze_all_segments()` tried to use:
```python
pd.cut(tenure, bins=[0, 12, 24, 36, 48, np.inf], ...)
```
It was cutting **scaled z-scores**, not months → all rows fell into one bin!

## The Fix

### 1. Updated `src/run_advanced_evaluation.py` (Lines 329-354)

**Changed from:**
```python
# WRONG - loads processed data
test_data = pd.read_csv(config.TEST_DATA_FILE)
```

**Changed to:**
```python
# ✅ CORRECT - Load RAW data (before encoding/scaling)
from data_processing import load_raw_data, clean_data, create_features

logger.info("Loading raw data for segment analysis...")
df_raw = load_raw_data()
df_raw = clean_data(df_raw)
df_raw = create_features(df_raw)

# Recreate the same train/test split to get test indices
from sklearn.model_selection import train_test_split

X_raw = df_raw.drop(config.TARGET_COLUMN, axis=1)
y_raw = df_raw[config.TARGET_COLUMN].map({'Yes': 1, 'No': 0})

# Same random_state ensures same indices as original split
_, X_test_raw, _, y_test_check = train_test_split(
    X_raw, y_raw,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=y_raw
)
```

### 2. Updated `src/model_evaluation.py` (Lines 516-584)

**Changed from:**
```python
def analyze_all_segments(X_test, y_test, y_pred, df_test):
    # Tried to work with scaled/encoded X_test
    tenure = X_test['tenure']  # ❌ This is a z-score!
    pd.cut(tenure, bins=[0, 12, 24, ...])  # ❌ Doesn't work!
```

**Changed to:**
```python
def analyze_all_segments(X_test, y_test, y_pred, df_test_raw):
    # ✅ Use raw data with original values

    # Contract Type - use original categorical column
    if 'Contract' in df_test_raw.columns:
        analyze_performance_by_segment(
            y_test, y_pred, df_test_raw['Contract'], 'Contract Type'
        )

    # Tenure - use ORIGINAL unscaled values (months, not z-scores)
    if 'tenure' in df_test_raw.columns:
        tenure_groups = pd.cut(
            df_test_raw['tenure'],  # ✅ Original months!
            bins=[0, 12, 24, 36, 48, np.inf],
            labels=['<12 months', '12-24 months', ...]
        )

    # MonthlyCharges - use ORIGINAL unscaled values (dollars, not z-scores)
    charge_tiers = pd.cut(
        df_test_raw['MonthlyCharges'],  # ✅ Original dollars!
        bins=[0, 35, 70, 90, np.inf],
        labels=['Low (<$35)', 'Medium ($35-70)', ...]
    )

    # Gender - use ORIGINAL categorical values
    df_test_raw['gender']  # ✅ "Male"/"Female", not 0/1!
```

### 3. Updated Report Generation (Lines 268-296)

Added better handling for edge cases:
- Check if segments are empty
- Only show "best" vs "worst" if there are multiple different segments
- Handle single-segment case with clear message

## How to Regenerate Reports

Run the following command to regenerate all segment analysis reports with the fix:

```bash
python src/run_advanced_evaluation.py
```

This will:
1. Load RAW data (before scaling/encoding)
2. Properly segment customers by:
   - **Contract Type**: Month-to-month, One year, Two year (3 segments)
   - **Tenure Group**: <12mo, 12-24mo, 24-36mo, 36-48mo, >48mo (5 segments)
   - **Monthly Charges**: Low, Medium, High, Very High (4 segments)
   - **Gender**: Male, Female (2 segments)
   - **Senior Citizen**: Yes, No (2 segments)

3. Generate corrected reports:
   - `outputs/reports/contract_type_analysis.csv`
   - `outputs/reports/tenure_group_analysis.csv`
   - `outputs/reports/monthly_charges_analysis.csv`
   - `outputs/reports/gender_analysis.csv`
   - `outputs/reports/senior_citizen_analysis.csv`
   - `outputs/reports/advanced_evaluation_summary.txt`

## Expected Output After Fix

### Contract Type (Should show 3 segments):
```
            Segment    N  Churn Rate  Precision  Recall      F1     FPR
    Month-to-month  XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX
         One year   XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX
         Two year   XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX

**Best Performance**: Two year (F1=0.XXX)
**Worst Performance**: Month-to-month (F1=0.XXX)
```

### Tenure Group (Should show 5 segments):
```
       Segment   N  Churn Rate  Precision  Recall     F1     FPR
   <12 months XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX
12-24 months  XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX
24-36 months  XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX
36-48 months  XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX
  >48 months  XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX

**Best Performance**: <12 months (F1=0.XXX)
**Worst Performance**: >48 months (F1=0.XXX)
```

### Gender (Should show 2 segments):
```
Segment    N  Churn Rate  Precision  Recall     F1     FPR
 Female  XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX
   Male  XXX      0.XXX      0.XXX   0.XXX   0.XXX   0.XXX
```

## Validation Checklist

After regenerating, verify:
- [ ] Contract Type shows **3 segments** (not 1)
- [ ] Tenure Group shows **5 segments** (not 1)
- [ ] Monthly Charges shows **4 segments** (not 1)
- [ ] Gender shows **"Male"/"Female"** (not "0.0")
- [ ] Senior Citizen shows **"Yes"/"No"** (not empty)
- [ ] All N values sum to **1,409** (test set size)
- [ ] "Best" and "Worst" segments are **different**
- [ ] F1 scores vary across segments

## Files Changed

1. **src/run_advanced_evaluation.py**
   - Lines 329-354: Load raw data instead of processed
   - Lines 268-296: Improved report generation
   - Line 366: Pass raw data to segment analysis

2. **src/model_evaluation.py**
   - Lines 516-584: Updated `analyze_all_segments()` to use raw data
   - Changed all segment definitions to use original unscaled/unencoded values
   - Added clear documentation about data requirements

## Impact

This fix enables:
- ✅ Accurate segment-level performance analysis
- ✅ Proper identification of high/low performing customer segments
- ✅ Validation of README claims about segment performance
- ✅ Data-driven business recommendations per segment

## Next Steps

1. Run `python src/run_advanced_evaluation.py` to regenerate reports
2. Check that all validations pass
3. Update any dashboard or documentation that references segment analysis
4. Review business insights from corrected segment data
