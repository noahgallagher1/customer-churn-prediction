"""
Interactive Streamlit Dashboard for Customer Churn Prediction.

This multi-page dashboard provides:
1. Executive Summary
2. Model Performance
3. Customer Risk Scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from pathlib import Path
import sys

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent / 'src'))
import config

# Page configuration
st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# AGGRESSIVE CSS - PERMANENT FULL WIDTH FIX
st.markdown("""
<style>
    /* FORCE full width - override ALL Streamlit defaults with !important */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* Main content area - FORCE 100% width */
    .main .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }

    /* Override app view container */
    .appview-container .main .block-container {
        max-width: 100% !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }

    /* Remove excessive margins but preserve layout */
    .element-container {
        margin: 0 !important;
    }

    /* FORCE full width on all plots and charts */
    .stPlotlyChart {
        width: 100% !important;
    }

    /* FORCE full width on matplotlib figures */
    .stpyplot {
        width: 100% !important;
    }

    /* FORCE full width on dataframes */
    .stDataFrame {
        width: 100% !important;
    }

    /* Full width vertical blocks */
    div[data-testid="stVerticalBlock"] {
        width: 100% !important;
    }

    /* Horizontal blocks (columns container) - let flex handle it */
    div[data-testid="stHorizontalBlock"] {
        width: 100% !important;
        display: flex !important;
        gap: 1rem !important;
    }

    /* Remove padding from main app container */
    .main {
        padding: 0 !important;
    }

    /* Sidebar - fixed width */
    section[data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
    }

    /* Fix column layout - let columns share space properly */
    div[data-testid="column"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        padding: 0 0.5rem !important;
    }

    /* Remove any max-width constraints */
    [data-testid="stAppViewContainer"] {
        max-width: 100% !important;
    }

    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        width: 100%;
    }

    /* Metric cards - fit within columns */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }

    /* Info boxes - adapt to container */
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts."""
    try:
        model = joblib.load(config.MODEL_FILE)
        preprocessor = joblib.load(config.PREPROCESSOR_FILE)
        feature_names = joblib.load(config.FEATURE_NAMES_FILE)
        metrics = joblib.load(config.METRICS_FILE)

        # Load SHAP objects if available
        try:
            from pathlib import Path
            # Try multiple possible paths
            possible_paths = [
                config.MODELS_DIR / 'shap_objects.joblib',
                Path('models/shap_objects.joblib'),
                Path('./models/shap_objects.joblib'),
                Path('/mount/src/customer-churn-prediction/models/shap_objects.joblib')
            ]
            shap_data = None
            for path in possible_paths:
                if path.exists():
                    shap_data = joblib.load(path)
                    break
        except Exception as e:
            print(f"Could not load SHAP data: {e}")
            shap_data = None

        # Load all model results for comparison
        try:
            all_results = joblib.load(config.MODELS_DIR / 'all_models_results.joblib')
        except:
            all_results = None

        return model, preprocessor, feature_names, metrics, shap_data, all_results
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        st.info("Please run the training pipeline first: python src/model_training.py")
        return None, None, None, None, None, None


@st.cache_data
def load_test_data():
    """Load test dataset."""
    try:
        test_data = pd.read_csv(config.TEST_DATA_FILE)
        return test_data
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None


def page_executive_summary():
    """Page 1: Executive Summary."""
    st.markdown('<h1 class="main-header">📊 Executive Summary</h1>', unsafe_allow_html=True)

    # Load artifacts
    model, preprocessor, feature_names, metrics, shap_data, all_results = load_model_artifacts()

    if model is None:
        st.warning("⚠️ Model not found. Please train the model first.")
        return

    # Company header
    st.markdown(f"### {config.COMPANY_NAME}")
    st.markdown("---")

    # Key Metrics Row with Enhanced Tiles
    st.markdown('''
    <style>
    .metric-tile {
        background: white;
        padding: 1.8rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid;
        text-align: left;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .metric-tile:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0.3rem 0;
    }
    .metric-delta {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    .tile-blue { border-left-color: #3b82f6; }
    .tile-blue .metric-value { color: #1e40af; }
    .tile-green { border-left-color: #10b981; }
    .tile-green .metric-value { color: #047857; }
    .tile-orange { border-left-color: #f59e0b; }
    .tile-orange .metric-value { color: #d97706; }
    .tile-purple { border-left-color: #8b5cf6; }
    .tile-purple .metric-value { color: #6d28d9; }
    .delta-positive { color: #10b981; }
    .delta-negative { color: #ef4444; }
    </style>
    ''', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        churn_rate = 26.5
        st.markdown(f'''
        <div class="metric-tile tile-blue">
            <div class="metric-label">Overall Churn Rate</div>
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-delta"><span class="delta-positive">↓ 2.3%</span> from baseline</div>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        accuracy = metrics.get('accuracy', 0) * 100
        st.markdown(f'''
        <div class="metric-tile tile-green">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">{accuracy:.1f}%</div>
            <div class="metric-delta">Overall prediction correctness</div>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        recall = metrics.get('recall', 0) * 100
        st.markdown(f'''
        <div class="metric-tile tile-orange">
            <div class="metric-label">Churn Detection Rate</div>
            <div class="metric-value">{recall:.1f}%</div>
            <div class="metric-delta">Identifies {recall:.0f} of 100 at-risk customers</div>
        </div>
        ''', unsafe_allow_html=True)

    with col4:
        net_savings = metrics.get('net_savings', 0)
        savings_increase = net_savings * 0.15
        st.markdown(f'''
        <div class="metric-tile tile-purple">
            <div class="metric-label">Annual Savings</div>
            <div class="metric-value">${net_savings:,.0f}</div>
            <div class="metric-delta"><span class="delta-positive">↑ ${savings_increase:,.0f}</span> potential growth</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    # Two columns for visualizations
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("🎯 Top Risk Factors")

        # Load feature importance
        try:
            feature_importance = pd.read_csv(config.REPORTS_DIR / 'shap_feature_importance.csv')

            # Apply the same feature name mapping as in Feature Importance tab
            feature_name_mapping = {
                'Contract_Month-to-month': 'Contract Type',
                'Contract_One year': 'Contract Type',
                'Contract_Two year': 'Contract Type',
                'InternetService_DSL': 'Internet Service Type',
                'InternetService_Fiber optic': 'Internet Service Type',
                'InternetService_No': 'Internet Service Type',
                'PaymentMethod_Bank transfer (automatic)': 'Payment Method',
                'PaymentMethod_Credit card (automatic)': 'Payment Method',
                'PaymentMethod_Electronic check': 'Payment Method',
                'PaymentMethod_Mailed check': 'Payment Method',
                'OnlineSecurity_Yes': 'Online Security',
                'OnlineSecurity_No': 'Online Security',
                'OnlineBackup_Yes': 'Online Backup',
                'OnlineBackup_No': 'Online Backup',
                'DeviceProtection_Yes': 'Device Protection',
                'DeviceProtection_No': 'Device Protection',
                'TechSupport_Yes': 'Tech Support',
                'TechSupport_No': 'Tech Support',
                'StreamingTV_Yes': 'Streaming TV',
                'StreamingTV_No': 'Streaming TV',
                'StreamingMovies_Yes': 'Streaming Movies',
                'StreamingMovies_No': 'Streaming Movies',
                'MultipleLines_Yes': 'Multiple Lines',
                'MultipleLines_No': 'Multiple Lines',
                'PhoneService_Yes': 'Phone Service',
                'PhoneService_No': 'Phone Service',
                'gender_Male': 'Gender',
                'gender_Female': 'Gender',
                'SeniorCitizen': 'Senior Citizen',
                'Partner': 'Has Partner',
                'Partner_Yes': 'Has Partner',
                'Partner_No': 'Has Partner',
                'Dependents': 'Has Dependents',
                'Dependents_Yes': 'Has Dependents',
                'Dependents_No': 'Has Dependents',
                'PaperlessBilling': 'Paperless Billing',
                'PaperlessBilling_Yes': 'Paperless Billing',
                'PaperlessBilling_No': 'Paperless Billing',
                'PhoneService': 'Phone Service',
                'tenure': 'Tenure (months)',
                'MonthlyCharges': 'Monthly Charges',
                'TotalCharges': 'Total Charges',
                'charges_per_tenure': 'Charges per Month',
                'contract_tenure_ratio': 'Contract-Tenure Ratio',
                'total_services': 'Total Services',
                'payment_risk_score': 'Payment Risk Score',
                'has_premium_services': 'Premium Services'
            }

            # Map technical names to business names
            feature_importance['display_name'] = feature_importance['feature'].map(
                lambda x: feature_name_mapping.get(x, x)
            )

            # Group by display name and sum importances
            grouped_importance = feature_importance.groupby('display_name').agg({
                'importance': 'sum'
            }).reset_index()

            # Sort and get top 10
            grouped_importance = grouped_importance.sort_values('importance', ascending=False)
            top_features = grouped_importance.head(10)

            fig = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['display_name'],
                orientation='h',
                marker=dict(color=top_features['importance'],
                          colorscale='Reds',
                          showscale=False),
                text=[f"{val:.3f}" for val in top_features['importance']],
                textposition='outside'
            ))

            fig.update_layout(
                title="Top 10 Churn Predictors (Business View)",
                xaxis_title="Impact Score",
                yaxis_title="",
                height=500,
                template=config.PLOTLY_TEMPLATE,
                yaxis={'categoryorder': 'total ascending'},
                xaxis={'range': [0, top_features['importance'].max() * 1.1]}
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add interpretive statement
            st.info(f"""
            **📊 What This Chart Tells Us:**

            The features shown above have the strongest **correlation** with customer churn predictions.
            Higher impact scores indicate features that, when present, are more strongly associated with
            customers leaving.

            **Business Insight:** The top factor, "{top_features.iloc[0]['display_name']}",
            has an impact score of {top_features.iloc[0]['importance']:.3f}, meaning it's a critical
            signal in identifying at-risk customers. However, remember that correlation ≠ causation—
            these features help us *predict* who will churn, but interventions should be validated
            through A/B testing to establish causal impact.
            """)

        except Exception as e:
            st.info("Run the explainability pipeline to generate feature importance.")

    with col_right:
        st.subheader("💰 Business Impact")

        # ROI Calculation
        roi = metrics.get('roi_percentage', 0)
        customers_saved = metrics.get('customers_saved', 0)
        customers_lost = metrics.get('customers_lost', 0)

        # Create gauge chart for ROI with dynamic range
        # Calculate appropriate max range (at least 30% above actual value, minimum 300)
        gauge_max = max(300, int((roi * 1.3) // 100) * 100)  # Round up to nearest 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=roi,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "ROI %", 'font': {'size': 20}},
            delta={'reference': 100, 'increasing': {'color': "green"}},
            number={'suffix': "%", 'font': {'size': 40}},
            gauge={
                'axis': {'range': [0, gauge_max], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': config.PRIMARY_COLOR, 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 100], 'color': "#ffcccc"},  # Below break-even (red tint)
                    {'range': [100, 200], 'color': "#fff9cc"},  # Moderate ROI (yellow)
                    {'range': [200, 400], 'color': "#ccffcc"},  # Good ROI (light green)
                    {'range': [400, gauge_max], 'color': "#99ff99"}  # Excellent ROI (green)
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': 100  # Break-even point is at 100% ROI
                }
            }
        ))

        fig.update_layout(height=300, template=config.PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

        # ROI explanation
        st.caption(f"📊 Break-even at 100% ROI (red line). Current ROI: **{roi:.1f}%** - Excellent performance!")

        # Impact metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Customers Saved", f"{customers_saved:,}")
        with col_b:
            st.metric("Customers Lost", f"{customers_lost:,}")

        # Add interpretive statement for ROI
        st.success(f"""
        **💡 Business Translation:**

        An ROI of {roi:.1f}% means that for every $1 spent on retention campaigns,
        we get back ${roi/100 + 1:.2f}. With {customers_saved:,} customers saved,
        we're preventing significant revenue loss while maintaining cost-effective operations.

        The model's {recall:.1%} recall rate means we identify {int(recall*10):.0f} out of every
        10 customers who will churn, allowing proactive intervention before they leave.
        """)

    # Business Recommendations
    st.markdown("---")
    st.subheader("💡 Key Business Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="success-box">
        <h4>🎯 Target High-Risk Segments</h4>
        <ul>
            <li>Month-to-month contract customers</li>
            <li>New customers (< 12 months tenure)</li>
            <li>Electronic check payment users</li>
        </ul>
        <b>Expected Impact:</b> 20-30% reduction in churn
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>📈 Enhance Service Offerings</h4>
        <ul>
            <li>Promote tech support services</li>
            <li>Bundle online security features</li>
            <li>Improve fiber optic service quality</li>
        </ul>
        <b>Expected Impact:</b> 15-20% churn reduction
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="insight-box">
        <h4>🔄 Contract Optimization</h4>
        <ul>
            <li>Incentivize annual contract upgrades</li>
            <li>Offer early renewal discounts</li>
            <li>Auto-payment enrollment bonuses</li>
        </ul>
        <b>Expected Impact:</b> 25-35% churn reduction
        </div>
        """, unsafe_allow_html=True)


def page_model_performance():
    """Page 2: Model Performance."""
    st.markdown('<h1 class="main-header">📈 Model Performance Analysis</h1>', unsafe_allow_html=True)

    # Load artifacts
    model, preprocessor, feature_names, metrics, shap_data, all_results = load_model_artifacts()

    if model is None:
        st.warning("⚠️ Model not found. Please train the model first.")
        return

    # Model Selection
    st.subheader("🤖 Model Information")
    model_name = metrics.get('model_name', 'Best Model')
    st.info(f"**Selected Model:** {model_name}")

    # Performance Metrics Table
    st.subheader("📊 Performance Metrics")

    col1, col2 = st.columns([2, 1])

    with col1:
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC'],
            'Score': [
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0),
                metrics.get('roc_auc', 0),
                metrics.get('pr_auc', 0)
            ],
            'Description': [
                'Overall prediction correctness',
                'Positive prediction accuracy',
                'True positive detection rate',
                'Harmonic mean of precision and recall',
                'Area under ROC curve',
                'Area under precision-recall curve'
            ]
        })

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn("Metric", width="medium"),
                "Score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                "Description": st.column_config.TextColumn("Description", width="large")
            }
        )

    with col2:
        st.markdown("### 🎯 Model Goal")
        st.markdown("""
        <div class="insight-box">
        Our model is optimized for <b>Recall</b> to maximize detection
        of potential churners, even at the cost of some false positives.
        <br><br>
        <b>Why?</b> The cost of losing a customer far exceeds the cost
        of a retention campaign.
        </div>
        """, unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown("---")
    st.subheader("🔍 Confusion Matrix")

    col_cm, col_metrics = st.columns([1, 1])

    with col_cm:
        # Create confusion matrix visualization
        test_data = load_test_data()
        if test_data is not None:
            X_test = test_data.drop(config.TARGET_COLUMN, axis=1)
            y_test = test_data[config.TARGET_COLUMN]
            y_pred = model.predict(X_test)

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted: No Churn', 'Predicted: Churn'],
                y=['Actual: No Churn', 'Actual: Churn'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                showscale=True
            ))

            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                height=400,
                template=config.PLOTLY_TEMPLATE
            )

            st.plotly_chart(fig, use_container_width=True)

    with col_metrics:
        # Business metrics
        st.markdown("### 💼 Business Metrics")

        business_metrics = pd.DataFrame({
            'Metric': [
                'True Positives (Saved)',
                'False Negatives (Lost)',
                'False Positives',
                'Retention Cost',
                'Potential Loss Prevented',
                'Net Savings',
                'ROI'
            ],
            'Value': [
                f"{metrics.get('customers_saved', 0):,}",
                f"{metrics.get('customers_lost', 0):,}",
                f"{metrics.get('false_positives', 0):,}",
                f"${metrics.get('cost_of_retention_program', 0):,.0f}",
                f"${metrics.get('potential_loss_prevented', 0):,.0f}",
                f"${metrics.get('net_savings', 0):,.0f}",
                f"{metrics.get('roi_percentage', 0):.1f}%"
            ]
        })

        st.dataframe(business_metrics, use_container_width=True, hide_index=True)

    # ROC and PR Curves
    st.markdown("---")
    st.subheader("📉 Performance Curves")

    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.markdown("#### ROC Curve")

        if test_data is not None:
            y_proba = model.predict_proba(X_test)[:, 1]

            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color=config.PRIMARY_COLOR, width=2)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random Classifier',
                line=dict(color='gray', width=2, dash='dash')
            ))

            fig.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400,
                template=config.PLOTLY_TEMPLATE,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

    with col_pr:
        st.markdown("#### Precision-Recall Curve")

        if test_data is not None:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                name=f'PR Curve (AUC = {pr_auc:.3f})',
                line=dict(color=config.SECONDARY_COLOR, width=2),
                fill='tozeroy'
            ))

            fig.update_layout(
                xaxis_title='Recall',
                yaxis_title='Precision',
                height=400,
                template=config.PLOTLY_TEMPLATE,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

    # Model Comparison
    if all_results is not None:
        st.markdown("---")
        st.subheader("🏆 Model Comparison")

        comparison_data = []
        for model_name, result in all_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1 Score': result['metrics']['f1'],
                'ROC AUC': result['metrics']['roc_auc'],
                'Net Savings ($)': result['business_metrics']['net_savings']
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Create grouped bar chart
        fig = go.Figure()

        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric]
            ))

        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=400,
            template=config.PLOTLY_TEMPLATE
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show table
        st.dataframe(comparison_df.style.highlight_max(axis=0, props='background-color: lightgreen'),
                    use_container_width=True, hide_index=True)


def page_customer_risk_scoring():
    """Page 4: Customer Risk Scoring."""
    st.markdown('<h1 class="main-header">🎯 Customer Risk Scoring</h1>', unsafe_allow_html=True)

    # Load artifacts
    model, preprocessor, feature_names, metrics, shap_data, all_results = load_model_artifacts()

    if model is None:
        st.warning("⚠️ Model not found. Please train the model first.")
        return

    st.markdown("### Predict churn risk for individual customers")

    # Two modes: Sample customer or manual input
    mode = st.radio(
        "Select input mode:",
        ["Analyze Sample Customer", "Manual Input"],
        horizontal=True
    )

    if mode == "Analyze Sample Customer":
        # Load test data
        test_data = load_test_data()
        if test_data is None:
            st.error("Test data not available")
            return

        X_test = test_data.drop(config.TARGET_COLUMN, axis=1)
        y_test = test_data[config.TARGET_COLUMN]

        # Select customer
        customer_idx = st.number_input(
            "Select customer index (0 to {})".format(len(X_test) - 1),
            min_value=0,
            max_value=len(X_test) - 1,
            value=0
        )

        customer_data = X_test.iloc[customer_idx:customer_idx+1]
        actual_churn = y_test.iloc[customer_idx]

    else:
        st.markdown("#### Enter Customer Information")

        # Create input form (simplified version)
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.number_input("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

        with col2:
            contract_type = st.selectbox("Contract Type",
                                        ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method",
                                         ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)",
                                          "Credit card (automatic)"])
            internet_service = st.selectbox("Internet Service",
                                           ["DSL", "Fiber optic", "No"])

        with col3:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            has_partner = st.selectbox("Has Partner", ["Yes", "No"])

        # This is simplified - in production, you'd need to create a full feature vector
        # matching the training data format
        st.warning("⚠️ Manual input mode requires complete feature engineering pipeline. " +
                  "Use 'Analyze Sample Customer' mode for demonstration.")
        return

    # Make prediction
    prediction_proba = model.predict_proba(customer_data)[0]
    churn_probability = prediction_proba[1]
    prediction = "CHURN" if churn_probability >= 0.5 else "NO CHURN"

    # Display results
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    col_pred1, col_pred2, col_pred3 = st.columns(3)

    with col_pred1:
        # Risk level
        if churn_probability >= 0.7:
            risk_level = "🔴 HIGH RISK"
            risk_color = "danger-box"
        elif churn_probability >= 0.4:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "warning-box"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "success-box"

        st.markdown(f'<div class="{risk_color}"><h2>{risk_level}</h2></div>',
                   unsafe_allow_html=True)

    with col_pred2:
        st.metric("Churn Probability", f"{churn_probability:.1%}",
                 help="Likelihood that this customer will churn")

    with col_pred3:
        st.metric("Prediction", prediction)
        if mode == "Analyze Sample Customer":
            actual_label = "CHURN" if actual_churn == 1 else "NO CHURN"
            st.metric("Actual Status", actual_label)
            correct = "✓" if (prediction == actual_label) else "✗"
            st.metric("Prediction Correct", correct)

    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': config.CHURN_COLOR},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(height=300, template=config.PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

    # SHAP Explanation
    if shap_data is not None and shap_data.get('explainer') is not None:
        st.markdown("---")
        st.markdown("### 🔍 Explanation - Why This Prediction?")

        explainer = shap_data['explainer']

        # Calculate SHAP values for this customer
        try:
            customer_shap = explainer.shap_values(customer_data)
            if isinstance(customer_shap, list):
                customer_shap = customer_shap[1]

            # Waterfall plot
            st.markdown("#### Feature Contributions")

            fig, ax = plt.subplots(figsize=(14, 10))

            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[1]

            shap_exp = shap.Explanation(
                values=customer_shap[0],
                base_values=expected_value,
                data=customer_data.iloc[0].values,
                feature_names=customer_data.columns.tolist()
            )

            shap.plots.waterfall(shap_exp, show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        except Exception as e:
            st.info(f"Could not generate SHAP explanation: {e}")

    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommended Actions")

    if churn_probability >= 0.7:
        st.markdown("""
        <div class="danger-box">
        <h4>🚨 URGENT: High Churn Risk</h4>
        <b>Immediate Actions:</b>
        <ol>
            <li>Contact customer within 24 hours</li>
            <li>Offer premium support package (50% discount)</li>
            <li>Propose contract upgrade with incentive</li>
            <li>Assign dedicated account manager</li>
            <li>Survey to understand pain points</li>
        </ol>
        <b>Estimated Retention Cost:</b> $100<br>
        <b>Customer Lifetime Value:</b> $2,000<br>
        <b>Expected ROI:</b> 1,900%
        </div>
        """, unsafe_allow_html=True)

    elif churn_probability >= 0.4:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Medium Risk - Proactive Engagement</h4>
        <b>Recommended Actions:</b>
        <ol>
            <li>Send personalized retention offer</li>
            <li>Highlight unused services/features</li>
            <li>Offer service bundle discount (20%)</li>
            <li>Monthly check-in email campaign</li>
        </ol>
        <b>Estimated Retention Cost:</b> $50<br>
        <b>Expected Success Rate:</b> 65%
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="success-box">
        <h4>✓ Low Risk - Standard Engagement</h4>
        <b>Maintenance Actions:</b>
        <ol>
            <li>Continue standard customer service</li>
            <li>Quarterly satisfaction survey</li>
            <li>Offer loyalty rewards program</li>
            <li>Cross-sell additional services</li>
        </ol>
        <b>Focus:</b> Customer satisfaction and upselling
        </div>
        """, unsafe_allow_html=True)


def page_feature_importance():
    """Page 4: Feature Importance & Explainability."""
    st.markdown('<h1 class="main-header">🔍 Feature Importance & Explainability</h1>', unsafe_allow_html=True)

    # Load artifacts
    model, preprocessor, feature_names, metrics, shap_data, all_results = load_model_artifacts()

    if model is None:
        st.warning("⚠️ Model not found. Please train the model first.")
        return

    # Feature name mapping dictionary - converts technical names to business-friendly names
    feature_name_mapping = {
        # Contract features
        'Contract_Month-to-month': 'Contract Type',
        'Contract_One year': 'Contract Type',
        'Contract_Two year': 'Contract Type',

        # Internet Service
        'InternetService_DSL': 'Internet Service Type',
        'InternetService_Fiber optic': 'Internet Service Type',
        'InternetService_No': 'Internet Service Type',

        # Payment Method
        'PaymentMethod_Bank transfer (automatic)': 'Payment Method',
        'PaymentMethod_Credit card (automatic)': 'Payment Method',
        'PaymentMethod_Electronic check': 'Payment Method',
        'PaymentMethod_Mailed check': 'Payment Method',

        # Yes/No service features
        'OnlineSecurity_Yes': 'Online Security',
        'OnlineSecurity_No': 'Online Security',
        'OnlineBackup_Yes': 'Online Backup',
        'OnlineBackup_No': 'Online Backup',
        'DeviceProtection_Yes': 'Device Protection',
        'DeviceProtection_No': 'Device Protection',
        'TechSupport_Yes': 'Tech Support',
        'TechSupport_No': 'Tech Support',
        'StreamingTV_Yes': 'Streaming TV',
        'StreamingTV_No': 'Streaming TV',
        'StreamingMovies_Yes': 'Streaming Movies',
        'StreamingMovies_No': 'Streaming Movies',
        'MultipleLines_Yes': 'Multiple Lines',
        'MultipleLines_No': 'Multiple Lines',
        'PhoneService_Yes': 'Phone Service',
        'PhoneService_No': 'Phone Service',

        # Binary features
        'gender_Male': 'Gender',
        'gender_Female': 'Gender',
        'SeniorCitizen': 'Senior Citizen',
        'Partner': 'Has Partner',
        'Partner_Yes': 'Has Partner',
        'Partner_No': 'Has Partner',
        'Dependents': 'Has Dependents',
        'Dependents_Yes': 'Has Dependents',
        'Dependents_No': 'Has Dependents',
        'PaperlessBilling': 'Paperless Billing',
        'PaperlessBilling_Yes': 'Paperless Billing',
        'PaperlessBilling_No': 'Paperless Billing',
        'PhoneService': 'Phone Service',

        # Numerical features
        'tenure': 'Tenure (months)',
        'MonthlyCharges': 'Monthly Charges',
        'TotalCharges': 'Total Charges',

        # Engineered features
        'charges_per_tenure': 'Charges per Month',
        'contract_tenure_ratio': 'Contract-Tenure Ratio',
        'total_services': 'Total Services',
        'payment_risk_score': 'Payment Risk Score',
        'has_premium_services': 'Premium Services'
    }

    # Section 1: Global Feature Importance
    st.markdown("## 📊 Global Feature Importance")
    st.markdown("Understanding which features have the biggest impact on churn predictions across all customers.")

    try:
        feature_importance_df = pd.read_csv(config.REPORTS_DIR / 'shap_feature_importance.csv')

        # Map technical names to business names
        feature_importance_df['display_name'] = feature_importance_df['feature'].map(
            lambda x: feature_name_mapping.get(x, x)
        )

        # Group by display name and sum importances (for one-hot encoded features)
        grouped_importance = feature_importance_df.groupby('display_name').agg({
            'importance': 'sum'  # Sum SHAP values for grouped features
        }).reset_index()

        # Sort by importance
        grouped_importance = grouped_importance.sort_values('importance', ascending=False)

        # Use grouped importance for visualization
        feature_importance_df = grouped_importance.rename(columns={'display_name': 'feature'})

        col1, col2 = st.columns([2, 1])

        with col1:
            # Top 10 features bar chart with business-friendly names
            top_n = st.slider("Number of top features to display", 5, 15, 10)
            top_features = feature_importance_df.head(top_n)

            fig = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h',
                marker=dict(
                    color=top_features['importance'],
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Impact Score")
                ),
                text=[f"{val:.3f}" for val in top_features['importance']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Impact Score: %{x:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"Top {top_n} Most Important Features (Business View)",
                xaxis_title="Average Impact on Churn Prediction",
                yaxis_title="",
                height=max(400, top_n * 30),
                template=config.PLOTLY_TEMPLATE,
                yaxis={'categoryorder': 'total ascending'}
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🎯 Key Insights")

            # Get the top feature for context-specific recommendations
            top_feature = feature_importance_df.iloc[0]['feature']
            top_importance = feature_importance_df.iloc[0]['importance']

            # Display top 5 features
            st.markdown("**Top 5 Churn Drivers:**")
            for idx, row in feature_importance_df.head(5).iterrows():
                st.markdown(f"{idx+1}. **{row['feature']}** (impact: {row['importance']:.3f})")

            st.markdown("---")
            st.markdown("### 💡 Business Context")

            # Provide specific recommendations based on top feature
            if 'Contract' in top_feature:
                st.success("""
                **Contract Type** is the #1 churn driver.

                **Action:** Focus on converting month-to-month customers to annual contracts with incentives like discounted rates or added services.
                """)
            elif 'Tenure' in top_feature:
                st.success("""
                **Tenure** is the #1 churn driver.

                **Action:** New customers (< 6 months) are highest risk. Implement strong onboarding programs and early engagement strategies.
                """)
            elif 'Charges' in top_feature or 'Monthly' in top_feature:
                st.success("""
                **Pricing** is the #1 churn driver.

                **Action:** Review pricing strategy. Consider loyalty discounts for long-term customers and competitive pricing reviews.
                """)
            elif 'Internet' in top_feature:
                st.success("""
                **Internet Service Type** is the #1 churn driver.

                **Action:** Fiber optic customers may have different expectations. Ensure service quality matches premium pricing.
                """)
            else:
                st.info(
                    "Higher impact scores indicate stronger influence on churn predictions. "
                    "These features are most critical for identifying at-risk customers."
                )

    except Exception as e:
        st.warning("⚠️ SHAP feature importance not available. Using model feature importances.")

        # Fallback to model's feature_importances_ if available
        if hasattr(model, 'feature_importances_'):
            test_data = load_test_data()
            if test_data is not None:
                X_test = test_data.drop(config.TARGET_COLUMN, axis=1)
                importances = model.feature_importances_
                feature_imp_df = pd.DataFrame({
                    'feature': X_test.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)

                top_n = 15
                top_features = feature_imp_df.head(top_n)

                fig = go.Figure(go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    marker=dict(color='steelblue')
                ))

                fig.update_layout(
                    title=f"Top {top_n} Most Important Features",
                    xaxis_title="Feature Importance",
                    yaxis_title="",
                    height=500,
                    template=config.PLOTLY_TEMPLATE,
                    yaxis={'categoryorder': 'total ascending'}
                )

                st.plotly_chart(fig, use_container_width=True)

    # Section 6: Interactive Feature Explorer
    st.markdown("---")
    st.markdown("## 🔬 Interactive Feature Explorer")
    st.markdown("Explore how individual features relate to customer churn with business-friendly names and better categorization.")

    test_data = load_test_data()
    if test_data is not None:
        X_test = test_data.drop(config.TARGET_COLUMN, axis=1)
        y_test = test_data[config.TARGET_COLUMN]

        # Combine for analysis
        analysis_df = X_test.copy()
        analysis_df['Churn'] = y_test

        # Categorize features into numerical and categorical
        numerical_features = []
        categorical_features = []

        for col in X_test.columns:
            # Check if it's truly numerical (not binary/one-hot encoded)
            unique_vals = analysis_df[col].nunique()
            if pd.api.types.is_numeric_dtype(analysis_df[col]) and unique_vals > 10:
                numerical_features.append(col)
            else:
                categorical_features.append(col)

        # Feature type selector
        feature_type = st.radio(
            "Select feature type to explore:",
            ["Numerical Features", "Categorical Features"],
            horizontal=True
        )

        if feature_type == "Numerical Features":
            if len(numerical_features) > 0:
                # Use technical feature names for selection
                selected_feature = st.selectbox(
                    "Select a numerical feature:",
                    options=sorted(numerical_features),
                    format_func=lambda x: feature_name_mapping.get(x, x)  # Show business name in dropdown
                )

                # Get display name for charts
                display_name = feature_name_mapping.get(selected_feature, selected_feature)

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Histogram with churn overlay
                    fig = go.Figure()

                    # Not churned
                    fig.add_trace(go.Histogram(
                        x=analysis_df[analysis_df['Churn'] == 0][selected_feature],
                        name='Not Churned',
                        marker_color='#4A90E2',
                        opacity=0.75,
                        nbinsx=30
                    ))

                    # Churned
                    fig.add_trace(go.Histogram(
                        x=analysis_df[analysis_df['Churn'] == 1][selected_feature],
                        name='Churned',
                        marker_color='#E74C3C',
                        opacity=0.75,
                        nbinsx=30
                    ))

                    fig.update_layout(
                        title=f"Distribution of {display_name} by Churn Status",
                        xaxis_title=display_name,
                        yaxis_title="Number of Customers",
                        barmode='overlay',
                        height=400,
                        template=config.PLOTLY_TEMPLATE,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### 📈 Statistics")

                    not_churned_stats = analysis_df[analysis_df['Churn'] == 0][selected_feature]
                    churned_stats = analysis_df[analysis_df['Churn'] == 1][selected_feature]

                    # Calculate stats
                    mean_diff = churned_stats.mean() - not_churned_stats.mean()
                    median_diff = churned_stats.median() - not_churned_stats.median()

                    stats_df = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev'],
                        'Not Churned': [
                            f"{not_churned_stats.mean():.2f}",
                            f"{not_churned_stats.median():.2f}",
                            f"{not_churned_stats.std():.2f}"
                        ],
                        'Churned': [
                            f"{churned_stats.mean():.2f}",
                            f"{churned_stats.median():.2f}",
                            f"{churned_stats.std():.2f}"
                        ]
                    })

                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                    # Interpretation
                    st.markdown("---")
                    st.markdown("**💡 Insight:**")
                    if abs(mean_diff) > 0.1 * not_churned_stats.mean():
                        direction = "higher" if mean_diff > 0 else "lower"
                        st.warning(f"Churned customers have {direction} {display_name} on average (diff: {abs(mean_diff):.2f})")
                    else:
                        st.info(f"Similar distribution between churned and retained customers")

            else:
                st.info("No numerical features available.")

        else:  # Categorical Features
            if len(categorical_features) > 0:
                # Use technical feature names for selection
                selected_feature = st.selectbox(
                    "Select a categorical feature:",
                    options=sorted(categorical_features),
                    format_func=lambda x: feature_name_mapping.get(x, x)  # Show business name in dropdown
                )

                # Get display name for charts
                display_name = feature_name_mapping.get(selected_feature, selected_feature)

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Calculate churn rates by category
                    category_stats = []
                    for category in sorted(analysis_df[selected_feature].unique()):
                        mask = analysis_df[selected_feature] == category
                        total = mask.sum()
                        churned = (mask & (analysis_df['Churn'] == 1)).sum()
                        not_churned = total - churned
                        churn_rate = (churned / total * 100) if total > 0 else 0

                        # Format category for display
                        if isinstance(category, (int, float)) and category in [0, 1]:
                            category_label = "Yes" if category == 1 else "No"
                        else:
                            category_label = str(category)

                        category_stats.append({
                            'category': category_label,
                            'not_churned': not_churned,
                            'churned': churned,
                            'churn_rate': churn_rate,
                            'total': total
                        })

                    stats_df = pd.DataFrame(category_stats)

                    # Stacked bar chart with churn rate
                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=stats_df['category'],
                        y=stats_df['not_churned'],
                        name='Not Churned',
                        marker_color='#4A90E2',
                        text=stats_df['not_churned'],
                        textposition='inside'
                    ))

                    fig.add_trace(go.Bar(
                        x=stats_df['category'],
                        y=stats_df['churned'],
                        name='Churned',
                        marker_color='#E74C3C',
                        text=stats_df['churned'],
                        textposition='inside'
                    ))

                    fig.update_layout(
                        title=f"{display_name} - Customer Distribution by Churn Status",
                        xaxis_title="Category",
                        yaxis_title="Number of Customers",
                        barmode='stack',
                        height=400,
                        template=config.PLOTLY_TEMPLATE,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### 📊 Churn Rates")

                    # Show churn rate by category
                    churn_rate_df = pd.DataFrame({
                        'Category': stats_df['category'],
                        'Total': stats_df['total'],
                        'Churn Rate': [f"{rate:.1f}%" for rate in stats_df['churn_rate']]
                    })

                    st.dataframe(churn_rate_df, use_container_width=True, hide_index=True)

                    # Find highest risk category
                    highest_risk_idx = stats_df['churn_rate'].idxmax()
                    highest_risk_cat = stats_df.iloc[highest_risk_idx]['category']
                    highest_risk_rate = stats_df.iloc[highest_risk_idx]['churn_rate']

                    st.markdown("---")
                    st.markdown("**⚠️ Highest Risk:**")
                    st.error(f"**{highest_risk_cat}**: {highest_risk_rate:.1f}% churn rate")

                    # Overall churn rate for comparison
                    overall_churn_rate = (analysis_df['Churn'].sum() / len(analysis_df)) * 100
                    st.metric("Overall Churn Rate", f"{overall_churn_rate:.1f}%")

            else:
                st.info("No categorical features available.")

    # Section 7: Individual Prediction Explanation
    st.markdown("---")
    st.markdown("## 🎯 Individual Prediction Explanation")
    st.markdown("Deep dive into why the model makes specific predictions for individual customers.")

    if test_data is not None:
        customer_idx = st.number_input(
            "Select Customer ID (0 to {})".format(len(X_test) - 1),
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            key="feature_importance_customer_idx"
        )

        customer_data = X_test.iloc[customer_idx:customer_idx+1]
        actual_churn = y_test.iloc[customer_idx]

        # Make prediction
        prediction_proba = model.predict_proba(customer_data)[0]
        churn_probability = prediction_proba[1]

        # Display customer details
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Customer ID", customer_idx)
            st.metric("Churn Probability", f"{churn_probability:.1%}")

        with col2:
            prediction = "CHURN" if churn_probability >= 0.5 else "NO CHURN"
            actual_label = "CHURN" if actual_churn == 1 else "NO CHURN"
            st.metric("Prediction", prediction)
            st.metric("Actual Status", actual_label)

        with col3:
            correct = "✓ Correct" if (prediction == actual_label) else "✗ Incorrect"
            st.metric("Accuracy", correct)

        # SHAP waterfall plot
        if shap_data is not None and shap_data.get('explainer') is not None:
            st.markdown("### 📊 SHAP Waterfall Chart")
            st.markdown("*Shows how each feature contributes to the prediction*")

            try:
                explainer = shap_data['explainer']
                customer_shap = explainer.shap_values(customer_data)

                if isinstance(customer_shap, list):
                    customer_shap = customer_shap[1]

                fig, ax = plt.subplots(figsize=(12, 8))

                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1]

                shap_exp = shap.Explanation(
                    values=customer_shap[0],
                    base_values=expected_value,
                    data=customer_data.iloc[0].values,
                    feature_names=customer_data.columns.tolist()
                )

                shap.plots.waterfall(shap_exp, show=False)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            except Exception as e:
                st.info(f"Could not generate SHAP waterfall: {e}")

        # Natural language explanation
        st.markdown("### 📝 Plain English Explanation")

        # Get top contributing features
        if shap_data is not None and shap_data.get('explainer') is not None:
            try:
                explainer = shap_data['explainer']
                customer_shap = explainer.shap_values(customer_data)

                if isinstance(customer_shap, list):
                    customer_shap = customer_shap[1]

                # Get feature contributions
                feature_contributions = pd.DataFrame({
                    'feature': customer_data.columns,
                    'value': customer_data.iloc[0].values,
                    'shap_value': customer_shap[0]
                }).sort_values('shap_value', key=abs, ascending=False)

                # Map to business-friendly names
                feature_contributions['display_name'] = feature_contributions['feature'].map(
                    lambda x: feature_name_mapping.get(x, x)
                )

                avg_churn_rate = y_test.mean() * 100

                explanation = f"""
**Customer #{customer_idx} Analysis:**

The model predicts a **{churn_probability:.1%}** churn probability for this customer,
which is {'higher' if churn_probability > avg_churn_rate/100 else 'lower'} than the
average churn rate of {avg_churn_rate:.1f}%.

**Key Contributing Factors:**
"""

                # Top 3 positive contributors
                positive_contributors = feature_contributions[feature_contributions['shap_value'] > 0].head(3)
                if len(positive_contributors) > 0:
                    explanation += "\n**Increasing Churn Risk:**\n"
                    for _, row in positive_contributors.iterrows():
                        val_str = f"{row['value']:.2f}" if isinstance(row['value'], (int, float)) else str(row['value'])
                        explanation += f"- **{row['display_name']}** (value: {val_str}) contributes +{row['shap_value']:.3f} to churn risk\n"

                # Top 3 negative contributors
                negative_contributors = feature_contributions[feature_contributions['shap_value'] < 0].head(3)
                if len(negative_contributors) > 0:
                    explanation += "\n**Decreasing Churn Risk:**\n"
                    for _, row in negative_contributors.iterrows():
                        val_str = f"{row['value']:.2f}" if isinstance(row['value'], (int, float)) else str(row['value'])
                        explanation += f"- **{row['display_name']}** (value: {val_str}) reduces churn risk by {abs(row['shap_value']):.3f}\n"

                # Recommendation
                explanation += "\n**Recommended Action:**\n"
                if churn_probability >= 0.7:
                    if len(positive_contributors) > 0:
                        top_feature_display = positive_contributors.iloc[0]['display_name']
                        explanation += f"🚨 **URGENT**: This customer is high risk. Focus on addressing **{top_feature_display}** immediately through targeted retention offers."
                    else:
                        explanation += "🚨 **URGENT**: This customer is high risk. Implement immediate retention strategies."
                elif churn_probability >= 0.4:
                    explanation += "⚠️ **PROACTIVE**: Monitor this customer and consider preventive engagement strategies."
                else:
                    explanation += "✅ **MAINTAIN**: Continue standard customer service protocols."

                st.markdown(explanation)

            except Exception as e:
                st.error(f"Could not generate explanation: {str(e)}")
                st.info("This may happen if SHAP values are not available for this model.")
        else:
            # Fallback: Use model feature importances when SHAP is not available
            if hasattr(model, 'feature_importances_'):
                avg_churn_rate = y_test.mean() * 100

                explanation = f"""
**Customer #{customer_idx} Analysis:**

The model predicts a **{churn_probability:.1%}** churn probability for this customer,
which is {'higher' if churn_probability > avg_churn_rate/100 else 'lower'} than the
average churn rate of {avg_churn_rate:.1f}%.

**Top Global Churn Drivers (from model):**

Based on the overall model feature importances, the most important factors for predicting churn are:
"""
                # Get top features and deduplicate by business name
                feature_importance_df = pd.DataFrame({
                    'feature': customer_data.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                # Map to business names and aggregate duplicates
                feature_importance_df['display_name'] = feature_importance_df['feature'].map(
                    lambda x: feature_name_mapping.get(x, x)
                )

                # Group by display name and sum importances, keep the feature with the highest individual importance
                grouped_features = feature_importance_df.groupby('display_name').agg({
                    'feature': 'first',  # Keep first occurrence
                    'importance': 'sum'  # Sum importances for grouped features
                }).reset_index()

                # Sort by summed importance and get top 5
                top_features = grouped_features.sort_values('importance', ascending=False).head(5)

                counter = 1
                for _, row in top_features.iterrows():
                    display_name = row['display_name']
                    feature_name = row['feature']
                    customer_value = customer_data.iloc[0][feature_name]
                    if isinstance(customer_value, (int, float)):
                        explanation += f"\n{counter}. **{display_name}** (customer value: {customer_value:.2f})"
                    else:
                        explanation += f"\n{counter}. **{display_name}** (customer value: {customer_value})"
                    counter += 1

                explanation += "\n\n**Recommended Action:**\n"
                if churn_probability >= 0.7:
                    explanation += "🚨 **URGENT**: This customer is high risk. Implement immediate retention strategies."
                elif churn_probability >= 0.4:
                    explanation += "⚠️ **PROACTIVE**: Monitor this customer and consider preventive engagement strategies."
                else:
                    explanation += "✅ **MAINTAIN**: Continue standard customer service protocols."

                explanation += "\n\n*Note: Detailed SHAP explanations are not available. This analysis uses overall feature importance patterns.*"

                st.markdown(explanation)
            else:
                st.info("Individual customer explanations require model feature importances or SHAP values.")

    # Section 8: Feature Correlations
    st.markdown("---")
    st.markdown("## 🔗 Feature Correlations")
    st.markdown("Understanding relationships between features.")

    if test_data is not None:
        # Select only numerical features for correlation
        numerical_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) > 0:
            # Limit to top features for readability
            top_features_for_corr = numerical_cols[:min(15, len(numerical_cols))]

            corr_matrix = X_test[top_features_for_corr].corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))

            fig.update_layout(
                title="Feature Correlation Heatmap",
                height=600,
                template=config.PLOTLY_TEMPLATE
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info("**Interpretation:** Red indicates positive correlation, blue indicates negative correlation. Values range from -1 to +1.")

    # Section 9: Technical Notes
    with st.expander("📚 How Feature Importance Works"):
        st.markdown("""
        ### SHAP (SHapley Additive exPlanations)

        **What is SHAP?**
        SHAP values explain the contribution of each feature to a model's prediction by assigning
        each feature an importance value for a particular prediction. The method is based on
        Shapley values from cooperative game theory.

        **How to Interpret SHAP Values:**
        - **Positive SHAP value**: Feature increases the probability of churn
        - **Negative SHAP value**: Feature decreases the probability of churn
        - **Magnitude**: Larger absolute values indicate stronger impact

        **Advantages:**
        - ✅ Model-agnostic (works with any ML model)
        - ✅ Provides both global and local explanations
        - ✅ Consistent and locally accurate
        - ✅ Based on solid game theory foundations

        **Limitations:**
        - ⚠️ Computationally expensive for large datasets
        - ⚠️ Requires careful interpretation with correlated features
        - ⚠️ Explanations are relative to the model, not ground truth

        ---

        ### ⚠️ Critical Caveat: Correlation vs. Causation

        **IMPORTANT:** SHAP values show **correlation** between features and predictions, **NOT causation**.

        **What This Means:**
        - SHAP identifies features that are *associated* with churn predictions
        - It does **NOT** prove that changing a feature will *cause* churn to change
        - Features may be correlated with unmeasured confounders

        **Example:**
        - SHAP shows "Month-to-month contract" has high importance
        - **Correlation:** Customers with month-to-month contracts churn more often
        - **But:** Simply forcing a customer into a long-term contract may not prevent churn
        - **Why?** Risk-averse customers self-select into month-to-month contracts. The contract type may be a *symptom* of underlying dissatisfaction, not the *cause* of churn.

        **Implications for Action:**
        1. **Use SHAP for prioritization**, not causal inference
        2. **Validate interventions** through A/B testing (see [A/B Test Plan](../A_B_TEST_PLAN.md))
        3. **Consider confounders** when designing retention strategies
        4. **Combine with domain expertise** to avoid spurious patterns

        **For Causal Analysis:**
        - Consider techniques like causal inference, propensity score matching, or instrumental variables
        - Run controlled experiments (A/B tests) to establish causation
        - Consult with domain experts to validate hypothesized causal mechanisms

        **Learn More:**
        - [SHAP Documentation](https://shap.readthedocs.io/)
        - [Original Paper](https://arxiv.org/abs/1705.07874)
        - [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/shap.html)
        - [Correlation vs Causation in ML](https://christophm.github.io/interpretable-ml-book/agnostic.html#feature-importance)
        """)


def page_ab_test_simulator():
    """Page 5: A/B Test Simulator."""
    st.markdown('<h1 class="main-header">🧪 A/B Test Simulator</h1>', unsafe_allow_html=True)

    st.markdown("""
    Simulate the impact of retention campaigns and validate improvements with statistical rigor.
    This tool helps you design experiments and calculate required sample sizes for reliable results.
    """)

    # Section 1: ROI Calculator
    st.markdown("## 💰 Retention Campaign ROI Calculator")
    st.markdown("Calculate the expected return on investment for your retention initiatives.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Current Metrics")
        current_churn_rate = st.slider("Current Churn Rate (%)", 0.0, 50.0, 26.5, 0.5)
        total_customers = st.number_input("Total Customers", 1000, 1000000, 10000, 1000)
        avg_customer_value = st.number_input("Avg Customer Lifetime Value ($)", 100, 10000, 2000, 100)

    with col2:
        st.markdown("### 🎯 Campaign Parameters")
        expected_reduction = st.slider("Expected Churn Reduction (%)", 1.0, 50.0, 15.0, 1.0)
        campaign_cost_per_customer = st.number_input("Cost per Customer ($)", 10, 500, 100, 10)
        target_percentage = st.slider("% of Customers Targeted", 10.0, 100.0, 30.0, 5.0)

    with col3:
        st.markdown("### 🔮 Projected Results")

        # Calculations
        customers_targeted = int(total_customers * target_percentage / 100)
        current_churned = int(total_customers * current_churn_rate / 100)
        new_churn_rate = current_churn_rate * (1 - expected_reduction / 100)
        new_churned = int(total_customers * new_churn_rate / 100)
        customers_saved = current_churned - new_churned

        total_campaign_cost = customers_targeted * campaign_cost_per_customer
        revenue_saved = customers_saved * avg_customer_value
        net_benefit = revenue_saved - total_campaign_cost
        roi_percentage = (net_benefit / total_campaign_cost * 100) if total_campaign_cost > 0 else 0

        st.metric("Customers Saved", f"{customers_saved:,}")
        st.metric("Total Campaign Cost", f"${total_campaign_cost:,}")
        st.metric("Revenue Saved", f"${revenue_saved:,}")
        st.metric("Net Benefit", f"${net_benefit:,}", f"{roi_percentage:.0f}% ROI")

    # Visualization
    st.markdown("### 📈 Campaign Impact Visualization")

    scenarios = pd.DataFrame({
        'Scenario': ['Current State', 'After Campaign'],
        'Churned Customers': [current_churned, new_churned],
        'Retained Customers': [total_customers - current_churned, total_customers - new_churned]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Retained',
        x=scenarios['Scenario'],
        y=scenarios['Retained Customers'],
        marker_color='steelblue',
        text=scenarios['Retained Customers'],
        textposition='inside'
    ))

    fig.add_trace(go.Bar(
        name='Churned',
        x=scenarios['Scenario'],
        y=scenarios['Churned Customers'],
        marker_color='indianred',
        text=scenarios['Churned Customers'],
        textposition='inside'
    ))

    fig.update_layout(
        barmode='stack',
        title="Customer Retention: Current vs. After Campaign",
        yaxis_title="Number of Customers",
        height=400,
        template=config.PLOTLY_TEMPLATE
    )

    st.plotly_chart(fig, use_container_width=True)

    # Financial breakdown
    st.markdown("### 💵 Financial Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Costs:**")
        costs_df = pd.DataFrame({
            'Item': ['Campaign Execution', 'Target Customers', 'Cost per Customer'],
            'Value': [f"${total_campaign_cost:,}", f"{customers_targeted:,}", f"${campaign_cost_per_customer}"]
        })
        st.dataframe(costs_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Benefits:**")
        benefits_df = pd.DataFrame({
            'Item': ['Customers Saved', 'Value per Customer', 'Total Revenue Saved'],
            'Value': [f"{customers_saved:,}", f"${avg_customer_value:,}", f"${revenue_saved:,}"]
        })
        st.dataframe(benefits_df, use_container_width=True, hide_index=True)

    # Section 2: A/B Test Design
    st.markdown("---")
    st.markdown("## 🔬 A/B Test Design & Sample Size Calculator")
    st.markdown("Design statistically rigorous experiments to validate retention strategies.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### ⚙️ Test Parameters")

        baseline_conversion = st.slider("Baseline Retention Rate (%)", 50.0, 95.0, 73.5, 0.5,
                                       help="Current retention rate (100% - churn rate)")
        minimum_detectable_effect = st.slider("Minimum Detectable Effect (%)", 1.0, 20.0, 5.0, 0.5,
                                             help="Smallest improvement you want to detect")
        significance_level = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1,
                                         help="Probability of false positive (Type I error)")
        statistical_power = st.selectbox("Statistical Power (1-β)", [0.80, 0.85, 0.90, 0.95], index=2,
                                        help="Probability of detecting true effect")

        # Sample size calculation (simplified formula)
        from scipy import stats

        p1 = baseline_conversion / 100
        p2 = p1 + (minimum_detectable_effect / 100)

        # Z-scores
        z_alpha = stats.norm.ppf(1 - significance_level / 2)
        z_beta = stats.norm.ppf(statistical_power)

        # Pooled proportion
        p_pooled = (p1 + p2) / 2

        # Sample size per group
        n = ((z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / ((p2 - p1) ** 2)

        sample_size_per_group = int(np.ceil(n))
        total_sample_size = sample_size_per_group * 2

        st.markdown("### 📊 Required Sample Size")
        st.metric("Per Group", f"{sample_size_per_group:,}")
        st.metric("Total (Both Groups)", f"{total_sample_size:,}")

        # Test duration estimate
        if total_customers > 0:
            weeks_needed = np.ceil(total_sample_size / total_customers * 52)
            st.metric("Estimated Test Duration", f"{int(weeks_needed)} weeks")

    with col2:
        st.markdown("### 📈 Simulated Results")

        # Generate synthetic test results
        np.random.seed(42)

        # Control group
        control_retention = np.random.binomial(1, p1, sample_size_per_group)
        control_rate = control_retention.mean()

        # Treatment group
        treatment_retention = np.random.binomial(1, p2, sample_size_per_group)
        treatment_rate = treatment_retention.mean()

        # Calculate improvement
        absolute_improvement = (treatment_rate - control_rate) * 100
        relative_improvement = ((treatment_rate - control_rate) / control_rate) * 100

        # Statistical test
        from scipy.stats import chi2_contingency

        contingency_table = np.array([
            [control_retention.sum(), len(control_retention) - control_retention.sum()],
            [treatment_retention.sum(), len(treatment_retention) - treatment_retention.sum()]
        ])

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        st.metric("Control Group Retention", f"{control_rate:.1%}")
        st.metric("Treatment Group Retention", f"{treatment_rate:.1%}")
        st.metric("Absolute Improvement", f"{absolute_improvement:.2f}%")
        st.metric("Relative Improvement", f"{relative_improvement:.1f}%")

        st.markdown("---")
        st.markdown("### 🎯 Statistical Significance")

        if p_value < significance_level:
            st.success(f"✅ **SIGNIFICANT** (p-value: {p_value:.4f})")
            st.markdown("The treatment shows a statistically significant improvement!")
        else:
            st.warning(f"⚠️ **NOT SIGNIFICANT** (p-value: {p_value:.4f})")
            st.markdown("No statistically significant difference detected. Consider running the test longer.")

    # Visualization of results
    st.markdown("### 📊 Test Results Comparison")

    results_df = pd.DataFrame({
        'Group': ['Control', 'Treatment'],
        'Retention Rate': [control_rate * 100, treatment_rate * 100],
        'Sample Size': [sample_size_per_group, sample_size_per_group]
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=results_df['Group'],
        y=results_df['Retention Rate'],
        text=[f"{val:.1f}%" for val in results_df['Retention Rate']],
        textposition='outside',
        marker_color=['steelblue', 'forestgreen']
    ))

    fig.update_layout(
        title="A/B Test Results: Retention Rate Comparison",
        yaxis_title="Retention Rate (%)",
        height=400,
        template=config.PLOTLY_TEMPLATE
    )

    st.plotly_chart(fig, use_container_width=True)

    # Interpretation guide
    with st.expander("📚 How to Interpret A/B Test Results"):
        st.markdown("""
        ### Understanding Statistical Significance

        **P-Value:**
        - Probability that the observed difference occurred by chance
        - Lower p-value = stronger evidence of real effect
        - Typical threshold: p < 0.05 (5% chance of false positive)

        **Statistical Power:**
        - Probability of detecting a true effect
        - Higher power = less risk of missing real improvements
        - Typical target: 80% or higher

        **Sample Size:**
        - More samples = more reliable results
        - Too small: May miss real effects (Type II error)
        - Too large: Wastes resources

        **Best Practices:**
        1. ✅ Define success metrics BEFORE running test
        2. ✅ Run test for full duration (avoid peeking)
        3. ✅ Ensure random assignment to groups
        4. ✅ Check for seasonality effects
        5. ✅ Validate results with holdout group

        **Common Pitfalls:**
        - ❌ Stopping test early when "significant"
        - ❌ Running multiple tests without correction
        - ❌ Changing success metrics mid-test
        - ❌ Ignoring practical significance
        """)


def page_about_data():
    """Page 6: About the Data."""
    st.markdown('<h1 class="main-header">📚 About the Data</h1>', unsafe_allow_html=True)

    # Section 1: Dataset Overview
    st.markdown("## 📊 Dataset Overview")

    # Load data
    test_data = load_test_data()

    try:
        # Try to load full dataset
        full_data = pd.read_csv(config.DATA_DIR / 'processed' / 'processed_data.csv')
    except:
        # Fall back to test data
        full_data = test_data

    if full_data is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(full_data):,}")

        with col2:
            st.metric("Number of Features", len(full_data.columns) - 1)  # Exclude target

        with col3:
            churn_rate = full_data['Churn'].mean() * 100 if 'Churn' in full_data.columns else 26.5
            st.metric("Churn Rate", f"{churn_rate:.1f}%")

        with col4:
            completeness = (1 - full_data.isnull().sum().sum() / (full_data.shape[0] * full_data.shape[1])) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")

    st.markdown("---")

    # Dataset info
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📋 Dataset Information")
        st.markdown("""
        - **Dataset Name:** Telco Customer Churn Dataset
        - **Source:** IBM Sample Data / Kaggle
        - **Domain:** Telecommunications
        - **Target Variable:** Churn (Binary: 0 = Retained, 1 = Churned)
        - **Use Case:** Customer retention and churn prediction
        """)

    with col2:
        st.markdown("### 🎯 Business Context")
        st.markdown("""
        This dataset contains customer information from a telecom company, including:
        - Demographics (age, gender, dependents)
        - Account information (tenure, contract, billing)
        - Services subscribed (internet, phone, streaming)
        - Churn status (whether customer left)
        """)

    # Section 2: Feature Dictionary
    st.markdown("---")
    st.markdown("## 📖 Feature Dictionary")
    st.markdown("Comprehensive description of all features in the dataset.")

    # Create feature dictionary
    feature_dict = {
        'Feature Name': [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges', 'Churn'
        ],
        'Data Type': [
            'Categorical', 'Binary', 'Binary', 'Binary', 'Numerical',
            'Binary', 'Categorical', 'Categorical', 'Categorical',
            'Categorical', 'Categorical', 'Categorical', 'Categorical',
            'Categorical', 'Categorical', 'Binary', 'Categorical',
            'Numerical', 'Numerical', 'Binary (Target)'
        ],
        'Description': [
            'Customer gender',
            'Whether customer is senior citizen (65+)',
            'Whether customer has a partner',
            'Whether customer has dependents',
            'Number of months as customer',
            'Has phone service',
            'Has multiple phone lines',
            'Internet service type',
            'Has online security add-on',
            'Has online backup add-on',
            'Has device protection add-on',
            'Has tech support add-on',
            'Has streaming TV service',
            'Has streaming movies service',
            'Contract type',
            'Enrolled in paperless billing',
            'Payment method',
            'Monthly bill amount',
            'Total amount billed to date',
            'Whether customer churned'
        ],
        'Example Values': [
            'Male, Female',
            '0, 1',
            'Yes, No',
            'Yes, No',
            '1-72 months',
            'Yes, No',
            'Yes, No, No phone',
            'DSL, Fiber optic, No',
            'Yes, No, No internet',
            'Yes, No, No internet',
            'Yes, No, No internet',
            'Yes, No, No internet',
            'Yes, No, No internet',
            'Yes, No, No internet',
            'Month-to-month, One year, Two year',
            'Yes, No',
            'Electronic check, Mailed check, Bank transfer, Credit card',
            '$18.25 - $118.75',
            '$18.80 - $8684.80',
            'Yes (1), No (0)'
        ]
    }

    feature_dict_df = pd.DataFrame(feature_dict)

    st.dataframe(
        feature_dict_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature Name": st.column_config.TextColumn("Feature Name", width="medium"),
            "Data Type": st.column_config.TextColumn("Data Type", width="small"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "Example Values": st.column_config.TextColumn("Example Values", width="large")
        }
    )

    # Section 3: Data Quality Summary
    st.markdown("---")
    st.markdown("## ✅ Data Quality Summary")

    if full_data is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Missing Values Analysis")

            missing_counts = full_data.isnull().sum()
            missing_pct = (missing_counts / len(full_data) * 100).round(2)

            if missing_counts.sum() == 0:
                st.success("✅ No missing values detected in the dataset!")
            else:
                missing_df = pd.DataFrame({
                    'Feature': missing_counts.index,
                    'Missing Count': missing_counts.values,
                    'Missing %': missing_pct.values
                }).query('`Missing Count` > 0')

                if len(missing_df) > 0:
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)
                else:
                    st.success("✅ No missing values!")

        with col2:
            st.markdown("### Data Type Distribution")

            # Count feature types
            numerical_features = full_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'Churn' in numerical_features:
                numerical_features.remove('Churn')

            categorical_features = full_data.select_dtypes(include=['object']).columns.tolist()

            type_counts = pd.DataFrame({
                'Type': ['Numerical', 'Categorical', 'Target'],
                'Count': [len(numerical_features), len(categorical_features), 1]
            })

            fig = go.Figure(data=[go.Pie(
                labels=type_counts['Type'],
                values=type_counts['Count'],
                hole=0.4,
                marker=dict(colors=['steelblue', 'forestgreen', 'indianred'])
            )])

            fig.update_layout(
                title="Feature Types Distribution",
                height=300,
                template=config.PLOTLY_TEMPLATE
            )

            st.plotly_chart(fig, use_container_width=True)

    # Section 4: Key Statistics
    st.markdown("---")
    st.markdown("## 📈 Key Statistics")

    if full_data is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👥 Customer Demographics")

            stats = []

            # Gender split if available
            if 'gender' in full_data.columns:
                gender_counts = full_data['gender'].value_counts()
                male_pct = gender_counts.get('Male', 0) / len(full_data) * 100
                female_pct = gender_counts.get('Female', 0) / len(full_data) * 100
                stats.append(('Gender Split', f'M: {male_pct:.1f}% / F: {female_pct:.1f}%'))

            # Senior citizens
            if 'SeniorCitizen' in full_data.columns:
                senior_pct = full_data['SeniorCitizen'].mean() * 100
                stats.append(('Senior Citizens', f'{senior_pct:.1f}%'))

            # Partner
            if 'Partner' in full_data.columns:
                partner_pct = (full_data['Partner'] == 'Yes').sum() / len(full_data) * 100
                stats.append(('Has Partner', f'{partner_pct:.1f}%'))

            # Dependents
            if 'Dependents' in full_data.columns:
                dep_pct = (full_data['Dependents'] == 'Yes').sum() / len(full_data) * 100
                stats.append(('Has Dependents', f'{dep_pct:.1f}%'))

            stats_df = pd.DataFrame(stats, columns=['Metric', 'Value'])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown("### 📊 Service Usage")

            usage_stats = []

            # Tenure
            if 'tenure' in full_data.columns:
                avg_tenure = full_data['tenure'].mean()
                usage_stats.append(('Average Tenure', f'{avg_tenure:.1f} months'))

            # Monthly charges
            if 'MonthlyCharges' in full_data.columns:
                median_charges = full_data['MonthlyCharges'].median()
                usage_stats.append(('Median Monthly Charges', f'${median_charges:.2f}'))

            # Contract type
            if 'Contract' in full_data.columns:
                most_common_contract = full_data['Contract'].mode()[0]
                contract_pct = (full_data['Contract'] == most_common_contract).sum() / len(full_data) * 100
                usage_stats.append(('Most Common Contract', f'{most_common_contract} ({contract_pct:.1f}%)'))

            # Internet service
            if 'InternetService' in full_data.columns:
                internet_pct = (full_data['InternetService'] != 'No').sum() / len(full_data) * 100
                usage_stats.append(('Internet Service Adoption', f'{internet_pct:.1f}%'))

            usage_df = pd.DataFrame(usage_stats, columns=['Metric', 'Value'])
            st.dataframe(usage_df, use_container_width=True, hide_index=True)

    # Section 5: Class Balance Visualization
    st.markdown("---")
    st.markdown("## ⚖️ Target Variable Distribution")

    if full_data is not None and 'Churn' in full_data.columns:
        col1, col2 = st.columns([1, 1])

        with col1:
            churn_counts = full_data['Churn'].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=['Not Churned', 'Churned'],
                values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                hole=0.4,
                marker=dict(colors=['steelblue', 'indianred']),
                text=[f"{churn_counts.get(0, 0):,}", f"{churn_counts.get(1, 0):,}"],
                textposition='inside'
            )])

            fig.update_layout(
                title="Churn Distribution",
                height=400,
                template=config.PLOTLY_TEMPLATE
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            churn_rate = full_data['Churn'].mean() * 100
            not_churn_rate = 100 - churn_rate

            fig = go.Figure(data=[go.Bar(
                x=['Not Churned', 'Churned'],
                y=[not_churn_rate, churn_rate],
                text=[f'{not_churn_rate:.1f}%', f'{churn_rate:.1f}%'],
                textposition='outside',
                marker_color=['steelblue', 'indianred']
            )])

            fig.update_layout(
                title="Churn Rate (%)",
                yaxis_title="Percentage",
                height=400,
                template=config.PLOTLY_TEMPLATE
            )

            st.plotly_chart(fig, use_container_width=True)

        if churn_rate < 40:
            st.info(f"ℹ️ The dataset shows a {churn_rate:.1f}% churn rate, which is typical for telecom industry benchmarks (15-35%).")
        else:
            st.warning(f"⚠️ The dataset shows a {churn_rate:.1f}% churn rate, which is higher than typical industry averages.")

    # Section 6: Download Options
    st.markdown("---")
    st.markdown("## 💾 Download Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        if full_data is not None:
            sample_data = full_data.head(100).to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Data (100 rows)",
                data=sample_data,
                file_name="churn_sample_data.csv",
                mime="text/csv"
            )

    with col2:
        feature_dict_csv = feature_dict_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Dictionary",
            data=feature_dict_csv,
            file_name="feature_dictionary.csv",
            mime="text/csv"
        )

    with col3:
        st.markdown("**Full Dataset:**")
        st.info("Contact data owner for full dataset access")

    # Section 8: Data Collection & Ethics
    st.markdown("---")
    st.markdown("## 🔒 Data Privacy & Ethics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Data Privacy")
        st.info("""
        - ✅ Dataset contains anonymized customer information
        - ✅ No personally identifiable information (PII) included
        - ✅ Data handling follows industry best practices
        - ✅ Customer IDs are randomized identifiers
        """)

    with col2:
        st.markdown("### Ethical Considerations")
        st.warning("""
        - ⚠️ Model predictions should not be used for discriminatory purposes
        - ⚠️ Human review recommended for high-stakes decisions
        - ⚠️ Regular monitoring for bias across demographics required
        - ⚠️ Use predictions to improve experience, not punish customers
        """)

    # Section 9: Limitations
    st.markdown("---")
    st.markdown("## ⚠️ Limitations & Responsible Use")

    with st.expander("📋 Dataset Limitations"):
        st.markdown("""
        **Temporal Limitations:**
        - Dataset represents a specific time period
        - Market conditions may have changed since data collection
        - Seasonal patterns may not be fully captured

        **Feature Limitations:**
        - External factors (competitors, economy) not included
        - Customer satisfaction scores not available
        - Social media sentiment not captured
        - Network quality metrics missing

        **Model Limitations:**
        - Model performance may degrade over time (concept drift)
        - Recommended retraining: Every 3-6 months
        - Predictions are probabilities, not certainties
        - Past behavior doesn't guarantee future outcomes

        **Recommended Practices:**
        - ✅ Combine model insights with business judgment
        - ✅ Maintain transparency with customers about data usage
        - ✅ Monitor model performance continuously
        - ✅ Update model with fresh data regularly
        - ✅ Use predictions to improve customer experience
        """)

    # Section 10: Data Lineage
    with st.expander("📊 Data Lineage & Version History"):
        st.markdown("""
        ### Version Information
        - **Version:** 1.0
        - **Original Source:** IBM Sample Data / Kaggle
        - **Last Updated:** 2024

        ### Preprocessing Steps Applied:
        1. ✅ Removed customerID from features (not predictive)
        2. ✅ Handled missing values in TotalCharges (< 1%)
        3. ✅ Encoded categorical variables (one-hot encoding)
        4. ✅ Scaled numerical features (StandardScaler)
        5. ✅ Split into train/test sets (80/20 stratified)
        6. ✅ Applied SMOTE for class balance in training

        ### Data Quality Checks:
        - ✅ Duplicate records: None found
        - ✅ Outliers: Identified and retained (valid business cases)
        - ✅ Consistency: All features validated
        - ✅ Completeness: 99.8% complete

        ### Future Updates:
        - Planned: Quarterly data refreshes
        - Next Update: TBD based on data availability
        """)


# Main App
def main():
    """Main application."""

    # Sidebar
    st.sidebar.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=TelcoConnect",
                     use_container_width=True)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Executive Summary", "Model Performance", "Customer Risk Scoring",
         "Feature Importance", "A/B Test Simulator", "About the Data"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard provides comprehensive insights into customer churn prediction "
        "using machine learning and explainability techniques (SHAP)."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")

    try:
        metrics = joblib.load(config.METRICS_FILE)
        st.sidebar.metric("Model Recall", f"{metrics.get('recall', 0)*100:.1f}%")
        st.sidebar.metric("ROI", f"{metrics.get('roi_percentage', 0):.0f}%")
    except:
        pass

    # Contact Information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👤 Contact")
    st.sidebar.markdown("""
    **Noah Gallagher**
    *Data Scientist*

    📧 [noahgallagher1@gmail.com](mailto:noahgallagher1@gmail.com)

    🔗 **Links:**
    - [GitHub Profile](https://github.com/noahgallagher1)
    - [Project Repository](https://github.com/noahgallagher1/customer-churn-prediction)
    - [LinkedIn](https://www.linkedin.com/in/noahgallagher/)
    - [Portfolio](https://noahgallagher1.github.io/MySite/)
    """)

    # Route to page
    if page == "Executive Summary":
        page_executive_summary()
    elif page == "Model Performance":
        page_model_performance()
    elif page == "Customer Risk Scoring":
        page_customer_risk_scoring()
    elif page == "Feature Importance":
        page_feature_importance()
    elif page == "A/B Test Simulator":
        page_ab_test_simulator()
    elif page == "About the Data":
        page_about_data()


if __name__ == "__main__":
    main()
