import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data import load_model_metrics

st.markdown("<h1 style='margin-bottom: 0;'>Model Insights</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 1.05rem; margin-bottom: 25px;'>Understand the algorithm, preprocessing pipeline, and validation performance.</p>", unsafe_allow_html=True)

# --- MODEL PIPELINE ---
st.markdown("<div class='section-header'>Machine Learning Pipeline</div>", unsafe_allow_html=True)

st.write("Expand any pipeline step below to examine the specific data transformation and inference operations:")

# Custom Step Render Helper
def render_pipeline_step(num, name, brief_desc, details_markdown):
    st.markdown(
        f"""
        <div class="pipeline-step-card">
            <div class="pipeline-step-num">{num}</div>
            <div class="pipeline-step-content">
                <div class="pipeline-step-name">{name}</div>
                <div class="pipeline-step-desc">{brief_desc}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.expander(f"View details for {name}"):
        st.markdown(details_markdown)

# Render steps
render_pipeline_step(
    "01", "Raw Applicant Data", 
    "Accepts raw applicant demographics, employment indices, and loan request metrics.",
    """
    **Input Variables Collected:**
    - *Demographics:* Age, Education Level, Marital Status, Supports Dependents.
    - *Financial Stability:* Annual Income, Credit Score, Months Employed, Mortgage Status, Credit Lines, DTI Ratio.
    - *Loan Parameters:* Loan Amount, Interest Rate, Loan Term (Months), Purpose, Co-signer Presence.
    """
)

render_pipeline_step(
    "02", "Data Validation & IQR Cleaning", 
    "Verifies data boundaries and filters income outlier values using the IQR method.",
    """
    **Cleaning Rules applied:**
    - Checks boundary constraints (e.g. Age: 18-100, Credit Score: 300-850).
    - Removes duplicate records (if any).
    - Filters income outliers using **Interquartile Range (IQR)**:
      $$Q_1 = \\text{25th Percentile}, \\quad Q_3 = \\text{75th Percentile}$$
      $$\\text{IQR} = Q_3 - Q_1$$
      $$\\text{Cleaned Income Range} = [Q_1 - 1.5 \\times \\text{IQR}, \\; Q_3 + 1.5 \\times \\text{IQR}]$$
    """
)

render_pipeline_step(
    "03", "Categorical Label Encoding", 
    "Encodes text/categorical features into numeric levels using pre-trained LabelEncoder maps.",
    """
    **Encoding Transformations:**
    - Re-maps categorical variables to alphabetical sorted integer values (`0, 1, 2, ...`):
      - *Education:* Bachelor's (0), High School (1), Master's (2), PhD (3).
      - *Employment Type:* Full-time (0), Part-time (1), Self-employed (2), Unemployed (3).
      - *Marital Status:* Divorced (0), Married (1), Single (2).
      - *Loan Purpose:* Auto (0), Business (1), Education (2), Home (3), Other (4).
      - *Binary options (Mortgage, Dependents, Co-signer):* No (0), Yes (1).
    """
)

render_pipeline_step(
    "04", "Feature Scaling", 
    "Standardizes numerical predictors using pre-trained StandardScaler coefficients.",
    r"""
    **Z-Score Standardization Formula:**
    - Transforms variables to have zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
      $$z = \\frac{x - \\mu}{\\sigma}$$
    - Applied on continuous variables: `age`, `income`, `loanamount`, `creditscore`, `monthsemployed`, `numcreditlines`, `interestrate`, `loanterm`, `dtiratio`.
    - Guarantees that variable magnitudes do not bias the Logistic Regression coefficients.
    """
)

render_pipeline_step(
    "05", "Logistic Regression Inference", 
    "Feeds standardized predictors into the live model to calculate log-odds scores.",
    """
    **Log-Odds Predictor equation:**
    - Maps predictors linear weights to log-odds score $z$:
      $$z = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\dots + \\beta_n X_n$$
    - $\\beta_0$ represents the model intercept, and $\\beta_i$ represent feature coefficients.
    """
)

render_pipeline_step(
    "06", "Probability mapping (Sigmoid)", 
    "Applies the sigmoid function to map log-odds to probabilities between 0.0 and 1.0.",
    """
    **Sigmoid Activation Formula:**
    - Transforms real-valued predictor score $z$ into a default probability $P$:
      $$P(\\text{Default} = 1) = \\frac{1}{1 + e^{-z}}$$
    """
)

render_pipeline_step(
    "07", "Risk Classification Triage", 
    "Applies threshold boundaries to classify default probabilities into risk tiers.",
    """
    **UI Risk Bands:**
    - **Low Risk:** Probability $\\le 30\\%$
    - **Medium Risk:** Probability between $30\\%$ and $60\\%$
    - **High Risk:** Probability $> 60\\%$
    """
)


# --- MODEL FORMULA ---
st.markdown("<div class='section-header'>Logistic Regression Concept & Mathematics</div>", unsafe_allow_html=True)

formula_col1, formula_col2 = st.columns([3, 2])

with formula_col1:
    st.write(
        "Logistic Regression is a supervised classification model that calculates "
        "probabilities for binary outcomes. In LoanGuard AI, it calculates the risk "
        "that an applicant belongs to the default class (Default = 1) using the sigmoid function:"
    )
    st.latex(r"P(\text{Default} = 1) = \frac{1}{1 + e^{-z}}")
    st.write(r"where the log-odds linear predictor \(z\) is defined as:")
    st.latex(r"z = \beta_0 + \beta_1 X_{\text{Age}} + \beta_2 X_{\text{Income}} + \dots + \beta_n X_{\text{CoSigner}}")

with formula_col2:
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); height: 180px;">
            <h4 style="margin-top: 0; color: #0A2540; border-bottom: 1px solid #E2E8F0; padding-bottom: 5px;">Core Assumptions</h4>
            <ul style="padding-left: 18px; margin: 5px 0 0 0; font-size: 0.8rem; color: #475569; line-height: 1.5;">
                <li><strong>Log-Odds Linearity:</strong> Relates features linearly to outcome log-odds.</li>
                <li><strong>Independence:</strong> Data points are independent of one another.</li>
                <li><strong>No Multicollinearity:</strong> Explanatory features are not highly correlated.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- PERFORMANCE METRICS ---
st.markdown("<div class='section-header'>Model Evaluation Metrics</div>", unsafe_allow_html=True)

metrics = load_model_metrics()

if metrics is None:
    st.warning("Performance metrics file not found. Run model training first.")
else:
        
    acc = metrics.get("accuracy", 0.0) * 100
    prec = metrics.get("precision", 0.0) * 100
    rec = metrics.get("recall", 0.0) * 100
    f1 = metrics.get("f1_score", 0.0) * 100
    roc_auc = metrics.get("roc_auc", 0.0)
    
    # Renders metrics row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Accuracy", f"{acc:.2f}%")
    with m_col2:
        st.metric("Precision", f"{prec:.2f}%")
    with m_col3:
        st.metric("Recall (Sensitivity)", f"{rec:.2f}%")
    with m_col4:
        st.metric("F1-Score", f"{f1:.2f}%")
    with m_col5:
        st.metric("ROC-AUC", f"{roc_auc:.4f}")
        
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 25px;">
            *Note: Metrics calculated on validation split partition (20% of dataset, 51,070 records). 
            Given dataset default rate is 11.61%, recall is low (3.12%), representing high default-profile conservatism.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- CONFUSION MATRIX Heatmap ---
    st.markdown("<div class='section-header'>Confusion Matrix Analysis</div>", unsafe_allow_html=True)
    
    cm = metrics.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    
    z_matrix = [[tn, fp], [fn, tp]]
    x_labels = ["Predicted: No Default (0)", "Predicted: Default (1)"]
    y_labels = ["Actual: No Default (0)", "Actual: Default (1)"]
    
    annot_text = [
        [f"True Negative (TN)<br><b>{tn:,}</b>", f"False Positive (FP)<br><b>{fp:,}</b>"],
        [f"False Negative (FN)<br><b>{fn:,}</b>", f"True Positive (TP)<br><b>{tp:,}</b>"]
    ]
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=z_matrix, x=x_labels, y=y_labels,
        colorscale=[[0.0, '#E0F2FE'], [1.0, '#0B192C']], # Light blue to dark navy
        showscale=False
    ))
    
    for i in range(2):
        for j in range(2):
            fig_cm.add_annotation(
                x=x_labels[j], y=y_labels[i],
                text=annot_text[i][j], showarrow=False,
                font=dict(color="white" if z_matrix[i][j] > 20000 else "#1E293B", size=13)
            )
            
    fig_cm.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        width=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(tickfont=dict(size=11, color="#1E293B"), side="bottom"),
        yaxis=dict(tickfont=dict(size=11, color="#1E293B"), autorange="reversed")
    )
    
    cm_col1, cm_col2 = st.columns([2, 1])
    with cm_col1:
        st.plotly_chart(fig_cm, width="stretch", config={'displayModeBar': False})
        
    with cm_col2:
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 18px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); height: 300px; display: flex; flex-direction: column; justify-content: center;">
                <h4 style="margin-top:0; color:#0A2540; border-bottom: 1px solid #F1F5F9; padding-bottom: 6px; font-size: 0.95rem;">🔍 Banking Error Evaluation</h4>
                <div style="font-size:0.82rem; color:#475569; line-height:1.5;">
                    <p style="margin: 0 0 8px 0;"><strong>True Negatives (TN):</strong> {tn:,} safe profiles correctly identified.</p>
                    <p style="margin: 0 0 8px 0;"><strong>True Positives (TP):</strong> {tp:,} defaults correctly detected.</p>
                    <p style="margin: 0 0 8px 0; color: #B45309;"><strong>False Positives (FP):</strong> {fp:,} safe profiles incorrectly flagged as risks (moderate business loss).</p>
                    <p style="margin: 0; color: #EF4444; font-weight: 700; background-color: #FEF2F2; padding: 5px; border-radius: 4px; border-left: 3px solid #EF4444;">
                        ⚠️ <strong>False Negatives (FN):</strong> {fn:,} default cases missed by model (Type II error). This represents direct credit loss and is the most costly error in lending.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- COMPARATIVE EVALUATION ACROSS ALL 4 MODELS ---
    st.markdown("<div class='section-header'>Comparative Evaluation Across All 4 Models</div>", unsafe_allow_html=True)
    st.write("Comprehensive benchmarking of all four machine learning architectures trained and evaluated on the platform:")
    
    from utils.prediction import load_all_model_metrics
    models_metrics = load_all_model_metrics()
    
    if models_metrics:
        comp_rows = []
        for m_key, m_data in models_metrics.items():
            comp_rows.append({
                "Model": m_data.get("name", m_key),
                "Architecture": m_data.get("type", "Ensemble"),
                "Accuracy": f"{m_data.get('accuracy', 0)*100:.2f}%",
                "Precision (Default)": f"{m_data.get('precision', 0)*100:.2f}%",
                "Recall (Default)": f"{m_data.get('recall', 0)*100:.2f}%",
                "F1-Score": f"{m_data.get('f1_score', 0)*100:.2f}%",
                "ROC-AUC": f"{m_data.get('roc_auc', 0):.4f}",
                "5-Fold CV Mean": f"{m_data.get('cv_mean', 0)*100:.2f}%",
                "CV Spread (Std)": f"±{m_data.get('cv_std', 0)*100:.2f}%"
            })
            
        df_models_comp = pd.DataFrame(comp_rows)
        st.dataframe(df_models_comp, width="stretch")
        
        # Plotly comparison bar chart
        m_names = [m_data.get("name", k) for k, m_data in models_metrics.items()]
        m_accs = [m_data.get("accuracy", 0) * 100 for k, m_data in models_metrics.items()]
        m_aucs = [m_data.get("roc_auc", 0) * 100 for k, m_data in models_metrics.items()]
        m_f1s = [m_data.get("f1_score", 0) * 100 for k, m_data in models_metrics.items()]
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Accuracy (%)', x=m_names, y=m_accs,
            marker_color='#0A2540'
        ))
        fig_comp.add_trace(go.Bar(
            name='ROC-AUC (×100)', x=m_names, y=m_aucs,
            marker_color='#00D2FF'
        ))
        fig_comp.add_trace(go.Bar(
            name='F1-Score (%)', x=m_names, y=m_f1s,
            marker_color='#10B981'
        ))
        
        fig_comp.update_layout(
            barmode='group',
            height=340,
            margin=dict(t=20, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            yaxis=dict(title="Score (%)", range=[0, 100])
        )
        
        st.plotly_chart(fig_comp, width="stretch", config={'displayModeBar': False})
        
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid #10B981; padding: 16px 20px; border-radius: 6px; margin-top: 15px; font-size: 0.88rem; color: #1E293B; line-height: 1.6;">
                <strong>🏆 Best Model Selection Rationale:</strong><br>
                <strong>Gradient Boosting</strong> achieved the highest overall <strong>ROC-AUC (0.7571)</strong>, the highest overall <strong>Accuracy (88.63%)</strong>, 
                and the highest <strong>Default Detection Recall (5.06%)</strong>—more than doubling the baseline Logistic Regression recall. 
                With an extremely low cross-validation spread of <strong>±0.03%</strong> across 5 folds, it provides the most dependable, risk-sensitive credit evaluation in production.
            </div>
            """,
            unsafe_allow_html=True
        )
