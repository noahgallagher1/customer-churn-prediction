# A/B Test Plan: ML-Driven Retention Campaigns

## Executive Summary

**Objective:** Validate that ML-predicted churn interventions deliver superior ROI compared to traditional retention approaches.

**Hypothesis:** Customers identified by our XGBoost churn model and targeted with personalized retention offers will show ≥20% higher retention rates than customers targeted through traditional high-value segmentation.

**Expected Impact:** $250K+ additional annual savings vs. status quo retention strategy.

**Timeline:** 12-week test (4 weeks ramp-up + 8 weeks measurement)

---

## 1. Test Design

### Test Structure: Randomized Controlled Trial (RCT)

**Treatment Group (A):** ML-Driven Targeted Retention
- Customers scored by XGBoost churn model
- Top 30% churn probability receive personalized interventions
- Retention offers tailored based on SHAP feature importance

**Control Group (B):** Traditional Business Rules
- Month-to-month customers with tenure <24 months
- MonthlyCharges >$70
- No phone/internet bundle
- Standard retention offer (generic discount)

**Holdout Group (C):** No Intervention
- 10% of eligible customers receive no outreach
- Establishes baseline churn rate
- Validates organic retention trends

### Sample Size Calculation

**Assumptions:**
- Baseline churn rate: 26.5% (from historical data)
- Expected lift from ML targeting: 20% relative improvement
- Statistical power: 80%
- Significance level: α = 0.05 (two-tailed)
- Minimum detectable effect: 5 percentage points

**Required Sample Size per Group:**
- Treatment (A): 1,200 customers
- Control (B): 1,200 customers
- Holdout (C): 300 customers
- **Total:** 2,700 customers

### Randomization Strategy

```
All eligible customers (churn probability >50% OR month-to-month contract)
    ↓
Stratified Random Assignment (by tenure group and monthly charges quartile)
    ↓
├─ 45% → Treatment (A): ML-Driven
├─ 45% → Control (B): Traditional
└─ 10% → Holdout (C): No intervention
```

**Stratification Variables:**
1. Tenure group: 0-6mo, 6-12mo, 12-24mo, 24+mo
2. MonthlyCharges quartile: Q1 (<$35), Q2 ($35-$55), Q3 ($55-$75), Q4 (>$75)

**Why Stratify?** Ensures groups are balanced on key confounders (customer value, lifecycle stage).

---

## 2. Intervention Details

### Treatment Group (A): ML-Driven Personalized Retention

**Step 1: Score All Customers**
```python
# Weekly batch scoring
churn_probs = model.predict_proba(customer_features)[:, 1]
eligible_customers = customers[churn_probs > 0.5]
```

**Step 2: Explain Predictions**
```python
# Generate SHAP explanations for each customer
shap_values = explainer.shap_values(customer_features)
top_risk_factors = get_top_features(shap_values, n=3)
```

**Step 3: Personalized Intervention Mapping**

| Top Risk Factor | Intervention | Offer | Cost |
|----------------|--------------|-------|------|
| **Contract_Month-to-month** | Contract upgrade incentive | "Lock in savings: Get 15% off for 12-month commitment" | $0 (discount offset by retention) |
| **tenure < 6 months** | Early engagement call | Dedicated onboarding specialist + first month 20% off | $100 (staff time + discount) |
| **TechSupport_No** | Service bundle upsell | "Add Premium Support for just $5/month (40% off)" | $0 (revenue-positive) |
| **PaymentMethod_Electronic check** | Auto-pay migration | "Switch to auto-pay, get $10 credit + waived convenience fees" | $10 |
| **MonthlyCharges high** | Loyalty discount | "Valued customer: We're reducing your rate by $10/month" | $120/year |
| **InternetService_Fiber optic + no bundles** | Service optimization | "Bundle and save: Get streaming package for $8/month (50% off)" | $0 (revenue-positive) |

**Delivery Channels:**
- Email (primary)
- SMS (high-risk customers >75% churn probability)
- Outbound call (VIP customers with CLV >$3,000)

### Control Group (B): Traditional Rule-Based Retention

**Eligibility Criteria:**
- Contract = Month-to-month
- tenure < 24 months
- MonthlyCharges > $70

**Standard Intervention:**
- Generic email: "We value your business! Here's 10% off for 6 months"
- No personalization
- No explanation of why they're receiving the offer
- Single channel (email only)

**Cost per customer:** $60 (6 months × $10 discount)

### Holdout Group (C): No Intervention

- No outreach
- Natural churn rate observed
- Establishes counterfactual

---

## 3. Measurement Plan

### Primary Outcome Metric

**30-Day Retention Rate:**
```
Retention Rate = (Customers still active at Day 30) / (Total customers in group)
```

**Why 30 days?** Aligns with our churn definition (customers who left in previous month).

**Success Criteria:**
- Treatment (A) retention ≥ 78% (20% relative improvement over baseline 65%)
- Treatment (A) retention significantly higher than Control (B) at p < 0.05

### Secondary Outcome Metrics

1. **60-Day Retention Rate** - Validates sustained impact
2. **90-Day Retention Rate** - Long-term behavior change
3. **Revenue per Customer** - Ensures discounts don't erode value
   ```
   Revenue/Customer = (MonthlyCharges × retention period) - (discount given)
   ```
4. **Customer Lifetime Value (CLV)** - Estimated future value
5. **Offer Acceptance Rate** - Engagement with intervention
6. **Net Promoter Score (NPS)** - Customer satisfaction impact

### Business Impact Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Cost per Retained Customer** | Total intervention cost / Customers retained | <$200 (vs. $1,500 acquisition cost) |
| **Incremental Revenue** | (Treatment retention - Control retention) × Avg CLV × Sample size | >$150K (for 2,700 customers) |
| **ROI** | (Incremental revenue - Incremental cost) / Incremental cost | >300% |
| **Net Savings** | Incremental revenue - Intervention cost | >$100K |

---

## 4. Data Collection & Tracking

### Pre-Test Data (T0 - Baseline)

Capture at randomization:
- Customer demographics (gender, SeniorCitizen, Partner, Dependents)
- Account info (tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges)
- Service subscriptions (all 10 service features)
- Churn probability score (Treatment group only)
- Top 3 SHAP features (Treatment group only)
- Group assignment (A, B, C)
- Randomization stratum

### Intervention Tracking (T1 - During Test)

**For Treatment & Control Groups:**
- Intervention sent: Date/time, channel (email/SMS/call)
- Offer details: Discount amount, offer type, duration
- Delivery status: Sent, opened, clicked, replied
- Acceptance: Did customer accept offer? (Yes/No/Partial)
- Response time: Hours from send to acceptance

### Outcome Tracking (T2 - Post-Test)

**Weekly Snapshots:**
- Active status: Is customer still subscribed? (Yes/No)
- Churn date: If churned, exact date
- MonthlyCharges: Current charges (did they upgrade/downgrade?)
- Contract changes: Did they change contract type?
- Service changes: Add/remove services?

### Attribution Window

- **Primary:** 30 days post-intervention
- **Secondary:** 60 and 90 days post-intervention

---

## 5. Statistical Analysis Plan

### Hypothesis Tests

**Null Hypothesis (H0):** Retention rate in Treatment (A) = Retention rate in Control (B)
**Alternative Hypothesis (H1):** Retention rate in Treatment (A) > Retention rate in Control (B)

**Test:** Two-proportion z-test (one-tailed)
**Significance level:** α = 0.05
**Power:** 1 - β = 0.80

### Analysis Steps

1. **Descriptive Statistics**
   - Compare baseline characteristics across groups (validate randomization)
   - Report retention rates for A, B, C with 95% confidence intervals

2. **Primary Analysis (Intention-to-Treat)**
   ```python
   from scipy.stats import proportions_ztest

   # Compare Treatment (A) vs. Control (B)
   count = [retained_A, retained_B]
   nobs = [n_A, n_B]
   stat, pval = proportions_ztest(count, nobs, alternative='larger')
   ```

3. **Secondary Analysis (Per-Protocol)**
   - Analyze only customers who engaged with intervention (opened email, clicked link)
   - Provides upper bound on effect size

4. **Subgroup Analysis**
   - By tenure group (early vs. late stage customers)
   - By monthly charges (high-value vs. low-value)
   - By top risk factor (contract, tenure, tech support, etc.)

   **Purpose:** Identify which customer segments benefit most from ML targeting

5. **Regression Analysis**
   ```python
   from statsmodels.api import Logit

   # Control for pre-test differences
   model = Logit(retained, treatment + tenure + monthly_charges + contract)
   results = model.fit()
   ```

6. **ROI Calculation**
   ```python
   incremental_retention = (retention_A - retention_B) * n_A
   incremental_revenue = incremental_retention * avg_CLV
   incremental_cost = (cost_per_intervention_A - cost_per_intervention_B) * n_A
   roi = (incremental_revenue - incremental_cost) / incremental_cost
   ```

### Multiple Testing Correction

With 2 primary comparisons (A vs. B, A vs. C) and 4 subgroups, we have 6 tests total.

**Bonferroni Correction:** α_adjusted = 0.05 / 6 = 0.0083

**Alternative:** Use Holm-Bonferroni for less conservative correction.

---

## 6. Timeline

### Week 1-2: Setup & Quality Assurance
- [ ] Finalize intervention copy and creative
- [ ] Set up tracking infrastructure (UTM tags, database tables)
- [ ] Test email/SMS delivery system
- [ ] QA churn model scoring pipeline
- [ ] Validate randomization code

### Week 3: Randomization & Launch
- [ ] Pull eligible customer list (week of November 18)
- [ ] Run stratified random assignment
- [ ] Validate group balance (check covariates)
- [ ] Send Treatment (A) interventions
- [ ] Send Control (B) interventions
- [ ] Monitor delivery rates (target >95%)

### Week 4-7: Intervention Period
- [ ] Weekly pulse checks (retention rates, engagement)
- [ ] Monitor for implementation issues (e.g., email bounces)
- [ ] No peeking at p-values (avoid bias)

### Week 8-11: Primary Measurement Window
- [ ] 30-day retention captured for all customers
- [ ] Continue monitoring 60-day and 90-day retention

### Week 12: Analysis & Reporting
- [ ] Run statistical tests
- [ ] Calculate business impact metrics
- [ ] Generate visualizations (retention curves, subgroup effects)
- [ ] Write executive summary report
- [ ] Present findings to stakeholders

---

## 7. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Insufficient sample size** (too few eligible customers) | Underpowered test, inconclusive results | Monitor weekly active customers; extend test if needed |
| **Seasonal effects** (holiday shopping, tax season) | Confounds treatment effect | Run test during "normal" period (avoid Nov-Dec, Apr-May) |
| **Data pipeline failure** (churn scores not updating) | Treatment group receives wrong offers | Daily monitoring dashboard; alerts on missing scores |
| **Implementation drift** (some customers get wrong intervention) | Dilutes treatment effect | Audit 10% of interventions weekly; ITT analysis robust to this |
| **Contamination** (Control customers hear about offers from Treatment) | Underestimates true effect | Discourage social sharing in offer terms; analyze network effects |
| **Early stopping pressure** (business wants to roll out immediately) | Type I error (false positive) | Pre-commit to 30-day window; explain multiple testing issues |
| **Negative customer reaction** (perceived unfairness of offers) | NPS decline, PR risk | Include "exclusive offer" framing; ensure all groups get *some* value |

---

## 8. Decision Framework

### If Treatment (A) Wins (p < 0.05, retention lift ≥20%)

**Action:** Roll out ML-driven retention to 100% of customer base
**Timeline:** 2-week ramp (10% → 50% → 100%)
**Monitoring:** Weekly retention tracking; revert if metrics degrade

### If Treatment (A) Shows Modest Win (p < 0.05, retention lift 10-20%)

**Action:** Refine intervention mapping, test again with enhanced personalization
**Timeline:** 4-week iteration cycle
**Focus:** Improve SHAP → Intervention translation logic

### If No Significant Difference (p ≥ 0.05)

**Action:** Analyze subgroups for heterogeneous treatment effects
**Question:** Does ML targeting work for specific customer segments?
**Follow-up:** Test on high-value customers only (CLV >$2,000)

### If Control (B) Wins (p < 0.05, Control retention > Treatment)

**Action:** Investigate root cause
**Hypotheses:**
- Are ML predictions accurate? (Check calibration)
- Are interventions poorly matched to risk factors?
- Is personalization confusing/off-putting to customers?
**Follow-up:** Qualitative research (customer interviews), model diagnostics

---

## 9. Success Metrics Dashboard

### Real-Time Monitoring (Updated Daily)

**Delivery Metrics:**
- Emails sent: Target 2,400 (Treatment + Control)
- Delivery rate: Target >95%
- Open rate: Target >25%
- Click rate: Target >10%

**Engagement Metrics:**
- Offer acceptance rate: Target >15%
- Response time: Median <48 hours

**Retention Metrics (Updated Weekly):**
- 7-day retention: Early signal
- 14-day retention: Mid-point check
- 30-day retention: Primary outcome

### Example Dashboard View

```
┌─────────────────────────────────────────────┐
│  A/B Test: ML-Driven Retention (Week 6/12) │
├─────────────────────────────────────────────┤
│  Treatment (A)  │  Control (B)  │ Holdout (C) │
├─────────────────┼───────────────┼─────────────┤
│  n = 1,200      │  n = 1,200    │  n = 300    │
│  Sent: 1,185    │  Sent: 1,192  │  n/a        │
│  Delivered: 98% │  Delivered: 98%│            │
│  Opened: 31%    │  Opened: 28%  │             │
│  Clicked: 14%   │  Clicked: 11% │             │
│  Accepted: 18%  │  Accepted: 12%│             │
├─────────────────┴───────────────┴─────────────┤
│  Retention (30-day, preliminary)              │
│  Treatment: 76% (±3%)                         │
│  Control:   68% (±3%)                         │
│  Holdout:   64% (±5%)                         │
│  Lift: +8 pp (p=0.03) ✓ Significant          │
│  ROI: 412% (preliminary)                      │
└───────────────────────────────────────────────┘
```

---

## 10. Reporting Template

### Executive Summary Report (Week 12)

**1. Test Overview**
- Dates: [Start] - [End]
- Sample size: 2,700 customers (1,200 Treatment, 1,200 Control, 300 Holdout)
- Intervention: ML-driven personalized retention vs. traditional rule-based

**2. Key Results**
- **30-Day Retention:**
  - Treatment (A): XX% (95% CI: [XX%, XX%])
  - Control (B): XX% (95% CI: [XX%, XX%])
  - Lift: +XX percentage points (p = X.XXX)

- **Statistical Significance:** [Yes/No] at α = 0.05
- **Business Impact:**
  - Incremental customers retained: XXX
  - Incremental revenue: $XXX,XXX
  - ROI: XXX%

**3. Subgroup Insights**
- [Best performing segment]
- [Worst performing segment]
- [Recommendation for targeting refinement]

**4. Recommendation**
- [Roll out / Iterate / Abandon]
- [Next steps]

---

## 11. Ethical Considerations

### Fairness & Equity

**Concern:** Does the ML model discriminate against protected groups (SeniorCitizen, gender)?

**Mitigation:**
- Analyze retention lift by demographic subgroups
- Ensure no group is systematically disadvantaged
- If disparate impact detected, consider fairness constraints in model

### Transparency

**Concern:** Are customers aware they're in an experiment?

**Approach:** Implicit consent (standard business practice testing). All groups receive value (Treatment gets personalized, Control gets standard offer, Holdout avoids spam).

### Holdout Group Ethics

**Concern:** Is it ethical to withhold intervention from Holdout group?

**Justification:**
- Holdout is only 10% of sample
- Establishes necessary counterfactual
- After test, Holdout can receive winning intervention

---

## 12. Post-Test Deployment Plan

### Assuming Treatment (A) Wins

**Week 13-14: Rollout Preparation**
- Finalize intervention mapping logic
- Scale ML scoring pipeline (from 2,700 to 7,000+ customers)
- Train customer service team on new retention offers
- Update CRM system with churn scores

**Week 15-16: Gradual Rollout**
- Week 15: 50% of eligible customers get ML-driven targeting
- Week 16: 100% of eligible customers

**Week 17+: Continuous Monitoring**
- Weekly retention dashboards
- Monthly A/B tests of intervention refinements
- Quarterly model retraining with new data

### Success Metrics for Deployment

- 30-day retention: >75% (sustained from test)
- Customer complaints: <1% increase
- NPS: No decline
- Annual incremental revenue: >$250K

---

## 13. Appendix: Sample Size Calculation Details

```python
from statsmodels.stats.proportion import proportion_effectsize, samplesize_proportions_2indep

# Parameters
p1 = 0.65  # Control retention rate (baseline)
p2 = 0.78  # Treatment retention rate (20% relative lift)
alpha = 0.05
power = 0.80
ratio = 1  # Equal sample sizes

# Calculate effect size
effect_size = proportion_effectsize(p1, p2)
# Output: 0.283 (medium effect)

# Calculate required sample size per group
n_per_group = samplesize_proportions_2indep(effect_size, alpha=alpha, power=power, ratio=ratio)
# Output: 391 per group

# Adjust for 30% attrition (some customers may churn before receiving intervention)
n_adjusted = n_per_group / 0.70
# Output: 559 per group

# Add buffer for stratification (6 strata, want ≥100 per stratum per group)
n_final = max(n_adjusted, 600)
# Output: 600 per group minimum

# Recommendation: 1,200 per group (provides extra power, allows deeper subgroup analysis)
```

**Conservative Assumptions:**
- 30% of customers may churn before intervention delivery → increase sample
- Want ≥100 customers per stratum per group → increase sample
- Desire 90% power instead of 80% → increase sample

**Final: 1,200 Treatment + 1,200 Control + 300 Holdout = 2,700 total**

---

## Contact

**Test Owner:** Noah Gallagher, Data Scientist
**Email:** noahgallagher1@gmail.com
**Stakeholders:** Retention Marketing Team, Customer Success, Finance

**Related Documentation:**
- [README.md](README.md) - Project overview
- [RESULTS_REPRODUCIBILITY.md](RESULTS_REPRODUCIBILITY.md) - Model training details
- [notebooks/threshold_roi_analysis.ipynb](notebooks/threshold_roi_analysis.ipynb) - Threshold optimization
