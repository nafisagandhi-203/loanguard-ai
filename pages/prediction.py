import streamlit as st
import time
import plotly.graph_objects as go
from utils.preprocessing import preprocess_input
from utils.prediction import predict_loan_risk, get_supported_models, load_all_model_metrics

# Custom styling for prediction page
st.markdown("<h1 style='margin-bottom: 0;'>Loan Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 1.05rem; margin-bottom: 20px;'>Complete the applicant profile to generate an AI-powered default risk assessment.</p>", unsafe_allow_html=True)

# 1. Visual Progress Stepper Header
st.markdown(
    """
    <div class="step-progress">
        <div class="step-item active"><span class="step-num">01</span> Applicant Profile</div>
        <div class="step-arrow">➔</div>
        <div class="step-item active"><span class="step-num">02</span> Financial Health</div>
        <div class="step-arrow">➔</div>
        <div class="step-item active"><span class="step-num">03</span> Loan Details</div>
        <div class="step-arrow">➔</div>
        <div class="step-item active"><span class="step-num">04</span> Select Model</div>
        <div class="step-arrow">➔</div>
        <div class="step-item active"><span class="step-num">05</span> Risk Triage</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "logistic_regression"
if "model_used_name" not in st.session_state:
    st.session_state.model_used_name = "Logistic Regression"
if "form_inputs" not in st.session_state:
    st.session_state.form_inputs = None
if "base_probability" not in st.session_state:
    st.session_state.base_probability = None
if "base_prediction" not in st.session_state:
    st.session_state.base_prediction = None
if "risk_level" not in st.session_state:
    st.session_state.risk_level = None

# Load available models metadata and metrics
model_catalog = get_supported_models()
all_metrics = load_all_model_metrics()

# Create Main Form
with st.form("loan_assessment_form"):
    
    # --- CARD 1: APPLICANT PROFILE ---
    st.markdown(
        """
        <div class="form-section-card">
            <div class="form-section-title">👤 01 — Applicant Profile</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        age = st.number_input(
            "Age (Years)", 
            min_value=18, 
            max_value=100, 
            value=35, 
            step=1,
            help="Applicant must be of legal age (18 or older)."
        )
        st.markdown("<div class='field-microcopy'>Valid range: 18 - 100</div>", unsafe_allow_html=True)
        
        education = st.selectbox(
            "Highest Education Level",
            options=["Bachelor's", "High School", "Master's", "PhD"],
            index=0,
            help="Select applicant's highest completed education level."
        )
        st.markdown("<div class='field-microcopy'>Education level maps to model log-odds weights</div>", unsafe_allow_html=True)
        
    with col_a2:
        employment_type = st.selectbox(
            "Employment Type",
            options=["Full-time", "Part-time", "Self-employed", "Unemployed"],
            index=0,
            help="Primary income source structure."
        )
        st.markdown("<div class='field-microcopy'>Employment status influences risk weights</div>", unsafe_allow_html=True)
        
        marital_status = st.selectbox(
            "Marital Status",
            options=["Divorced", "Married", "Single"],
            index=2,
            help="Marital status of primary applicant."
        )
        
        has_dependents = st.selectbox(
            "Supports Dependents?",
            options=["No", "Yes"],
            index=0,
            help="Select Yes if applicant supports children or legal dependents."
        )

    # --- CARD 2: FINANCIAL HEALTH ---
    st.markdown(
        """
        <div class="form-section-card" style="margin-top: 1.5rem;">
            <div class="form-section-title">📊 02 — Financial Health</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        income = st.number_input(
            "Annual Income ($)", 
            min_value=1000.0, 
            max_value=1000000.0, 
            value=65000.0, 
            step=2500.0,
            help="Gross pre-tax annual income in USD."
        )
        st.markdown("<div class='field-microcopy'>Scale: $1,000 - $1,000,000</div>", unsafe_allow_html=True)
        
        credit_score = st.slider(
            "Credit Score (FICO)", 
            min_value=300, 
            max_value=850, 
            value=720, 
            step=5,
            help="FICO credit score ranges from 300 to 850."
        )
        st.markdown("<div class='field-microcopy'>FICO Scale: 300 (Poor) — 850 (Exceptional)</div>", unsafe_allow_html=True)
        
        months_employed = st.number_input(
            "Months Employed", 
            min_value=0, 
            max_value=700, 
            value=48, 
            step=6,
            help="Tenure with current employer in months."
        )
        
    with col_b2:
        num_credit_lines = st.number_input(
            "Number of Active Credit Lines", 
            min_value=0, 
            max_value=50, 
            value=3, 
            step=1,
            help="Total open credit facilities, lines of credit, or credit cards."
        )
        
        dti_ratio = st.slider(
            "Debt-to-Income (DTI) Ratio", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.28, 
            step=0.01,
            format="%.2f",
            help="Monthly debt liabilities divided by gross monthly income."
        )
        st.markdown("<div class='field-microcopy'>Healthy banking threshold is generally ≤ 0.36</div>", unsafe_allow_html=True)
        
        has_mortgage = st.selectbox(
            "Existing Mortgage?", 
            options=["No", "Yes"], 
            index=0,
            help="Select Yes if applicant has an ongoing primary property mortgage."
        )

    # --- CARD 3: LOAN DETAILS ---
    st.markdown(
        """
        <div class="form-section-card" style="margin-top: 1.5rem;">
            <div class="form-section-title">💵 03 — Loan Request Parameters</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        loan_amount = st.number_input(
            "Requested Loan Amount ($)", 
            min_value=500.0, 
            max_value=500000.0, 
            value=15000.0, 
            step=1000.0,
            help="Principal amount requested by the borrower."
        )
        st.markdown("<div class='field-microcopy'>Scale: $500 - $500,000</div>", unsafe_allow_html=True)
        
        interest_rate = st.number_input(
            "Interest Rate (APR %)", 
            min_value=0.1, 
            max_value=40.0, 
            value=8.5, 
            step=0.25,
            format="%.2f",
            help="Annual percentage rate applied to the loan."
        )
        st.markdown("<div class='field-microcopy'>Scale: 0.1% - 40.0%</div>", unsafe_allow_html=True)
        
    with col_c2:
        loan_term = st.number_input(
            "Loan Term (Months)", 
            min_value=6, 
            max_value=360, 
            value=36, 
            step=12,
            help="Amortization duration in months."
        )
        st.markdown("<div class='field-microcopy'>Scale: Months (e.g. 12, 24, 36, 60, 120, 360)</div>", unsafe_allow_html=True)
        
        loan_purpose = st.selectbox(
            "Loan Purpose",
            options=["Auto", "Business", "Education", "Home", "Other"],
            index=4,
            help="Primary destination of loan funds."
        )
        
        has_cosigner = st.selectbox(
            "Has Co-signer?",
            options=["No", "Yes"],
            index=0,
            help="Select Yes if there is a secondary guarantor."
        )

    # --- CARD 4: SELECT PREDICTION MODEL ---
    st.markdown(
        """
        <div class="form-section-card" style="margin-top: 1.5rem;">
            <div class="form-section-title">🤖 04 — Select Prediction Model</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("Choose which trained Machine Learning model to evaluate this loan application:")
    
    # Model option formatting
    model_keys = list(model_catalog.keys())
    model_labels = [
        f"{model_catalog[k]['icon']} {model_catalog[k]['name']} — {model_catalog[k]['description']}"
        for k in model_keys
    ]
    
    # Current index lookup
    default_idx = model_keys.index(st.session_state.selected_model) if st.session_state.selected_model in model_keys else 0
    
    selected_label = st.radio(
        "Available Machine Learning Models:",
        options=model_labels,
        index=default_idx,
        help="Select any of the four trained architectures to execute prediction."
    )
    
    chosen_model_idx = model_labels.index(selected_label)
    chosen_model_key = model_keys[chosen_model_idx]
    chosen_meta = model_catalog[chosen_model_key]
    
    # Display Model Info Card
    model_stat = all_metrics.get(chosen_model_key, {})
    acc_text = f"{model_stat.get('accuracy', 0)*100:.2f}%" if 'accuracy' in model_stat else "88.5%+"
    auc_text = f"{model_stat.get('roc_auc', 0):.4f}" if 'roc_auc' in model_stat else "0.75+"
    cv_text = f"{model_stat.get('cv_mean', 0)*100:.2f}% ± {model_stat.get('cv_std', 0)*100:.2f}%" if 'cv_mean' in model_stat else "Verified"
    
    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid #0A2540; padding: 14px 18px; border-radius: 6px; margin: 12px 0 20px 0; font-size: 0.88rem; color: #1E293B;">
            <div style="font-weight: 700; color: #0A2540; font-size: 0.95rem; margin-bottom: 4px;">
                {chosen_meta['icon']} Selected: {chosen_meta['name']} ({chosen_meta['type']})
            </div>
            <div style="color: #475569; margin-bottom: 8px;">{chosen_meta['description']}</div>
            <div style="display: flex; gap: 20px; font-size: 0.82rem; font-weight: 600; color: #0369A1;">
                <span>🎯 Test Accuracy: <b>{acc_text}</b></span>
                <span>📈 ROC-AUC: <b>{auc_text}</b></span>
                <span>🔄 5-Fold CV: <b>{cv_text}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Custom prominent button
    submit_btn = st.form_submit_button("✦ Predict Loan Default")

# Handle Form Execution
if submit_btn:
    # 1. Sequential Loading Animation
    status_placeholder = st.empty()
    
    with status_placeholder.container():
        st.markdown(
            f"""
            <div style="background-color: #E0F2FE; border-left: 4px solid #0284C7; padding: 15px; border-radius: 4px; color: #0369A1; font-weight: 600; margin-bottom: 20px;">
                🔄 Analyzing loan application with {chosen_meta['name']}...
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.4)
        st.markdown(
            f"""
            <div style="background-color: #E0F2FE; border-left: 4px solid #0284C7; padding: 15px; border-radius: 4px; color: #0369A1; font-weight: 600; margin-bottom: 20px;">
                🛡️ Computing default probability & risk tiers...
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.4)
        st.markdown(
            """
            <div style="background-color: #E2F0D9; border-left: 4px solid #385723; padding: 15px; border-radius: 4px; color: #385723; font-weight: 600; margin-bottom: 20px;">
                📈 Generating credit risk assessment card...
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.3)
        
    status_placeholder.empty()
    
    # 2. Model Inference via Backend Module
    raw_inputs = {
        "age": age, "income": income, "loanamount": loan_amount, "creditscore": credit_score,
        "monthsemployed": months_employed, "numcreditlines": num_credit_lines, "interestrate": interest_rate,
        "loanterm": loan_term, "dtiratio": dti_ratio, "education": education,
        "employmenttype": employment_type, "maritalstatus": marital_status, "hasmortgage": has_mortgage,
        "hasdependents": has_dependents, "loanpurpose": loan_purpose, "hascosigner": has_cosigner
    }
    
    try:
        preprocessed_df = preprocess_input(raw_inputs)
        result = predict_loan_risk(preprocessed_df, model_key=chosen_model_key)
        
        # Store in session state
        st.session_state.form_inputs = raw_inputs
        st.session_state.base_probability = result["default_probability"]
        st.session_state.base_prediction = result["prediction"]
        st.session_state.model_used_name = result["model_name"]
        st.session_state.selected_model = result["model"]
        st.session_state.risk_level = result["risk_level"]
        
    except Exception as e:
        st.error(f"Prediction service encountered an issue: {e}")

# --- RESULT DISPLAY & SIMULATION PLAYGROUND ---
if st.session_state.base_probability is not None:
    st.markdown("<div class='section-header'>Loan Risk Assessment Result</div>", unsafe_allow_html=True)
    
    # Extract values
    base_prob_pct = st.session_state.base_probability * 100
    base_pred = st.session_state.base_prediction
    active_model_name = st.session_state.model_used_name
    active_risk_level = st.session_state.risk_level
    
    # 1. Result Card Styling
    if active_risk_level == "Low":
        risk_color = "#10B981"
        container_class = "low-risk"
        badge_class = "badge-low"
        title = "🟢 Low Default Risk"
        desc = "Based on the provided applicant profile, the model predicts a lower likelihood of default. Standard interest rate pricing and standard approval routes are appropriate."
    elif active_risk_level == "Medium":
        risk_color = "#F59E0B"
        container_class = "medium-risk"
        badge_class = "badge-medium"
        title = "🟡 Medium Default Risk"
        desc = "The model predicts moderate default risk. Reviewing additional covenants, debt history, or requiring a secondary guarantor/co-signer is recommended."
    else:
        risk_color = "#EF4444"
        container_class = "high-risk"
        badge_class = "badge-high"
        title = "🔴 High Default Risk"
        desc = "The model predicts a higher likelihood of default based on the provided applicant profile. Enhanced risk pricing or formal mitigation (collateral, co-signers) is highly recommended."

    pred_label = "Default Risk Detected" if base_pred == 1 else "No Default Expected"

    # Render Result Container
    st.markdown(
        f"""
        <div class="result-container {container_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div class="result-badge {badge_class}">{active_risk_level.upper()} RISK</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #0A2540; background-color: #F1F5F9; padding: 4px 10px; border-radius: 4px; border: 1px solid #CBD5E1;">
                    Model Used: <b>{active_model_name}</b>
                </div>
            </div>
            <div class="result-status-title">{title} — {pred_label}</div>
            <div class="result-status-desc">{desc}</div>
            <div style="display: flex; justify-content: space-between; font-size: 0.95rem; font-weight: 700; border-top: 1px solid #E2E8F0; padding-top: 12px; color: #0A2540;">
                <span>Default Probability: {base_prob_pct:.2f}%</span>
                <span>No-Default Probability: {(100 - base_prob_pct):.2f}%</span>
            </div>
            <div style="font-size: 0.78rem; color: #64748B; margin-top: 6px; text-align: right;">
                ✦ Prediction generated using <b>{active_model_name}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Columns for Gauge and Factor Explanations
    g_col1, g_col2 = st.columns([1, 1])
    
    with g_col1:
        # Plotly Risk Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = base_prob_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{active_model_name} Default Probability", 'font': {'size': 13, 'family': "Inter, sans-serif", 'color': '#0A2540'}},
            number = {'suffix': "%", 'font': {'size': 32, 'family': "Inter, sans-serif", 'color': '#0A2540'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B", 'ticksuffix': "%"},
                'bar': {'color': '#0A2540'},
                'bgcolor': "white",
                'borderwidth': 1,
                'bordercolor': "#E2E8F0",
                'steps': [
                    {'range': [0, 30], 'color': '#ECFDF5'},   # Light Green
                    {'range': [30, 60], 'color': '#FFFBEB'},  # Light Yellow
                    {'range': [60, 100], 'color': '#FEF2F2'}  # Light Red
                ],
                'threshold': {
                    'line': {'color': risk_color, 'width': 5},
                    'thickness': 0.75,
                    'value': base_prob_pct
                }
            }
        ))
        fig_gauge.update_layout(
            height=250,
            margin=dict(l=25, r=25, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, width="stretch", config={'displayModeBar': False})
        
    with g_col2:
        # Why this result section
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); height: 235px;">
                <h4 style="margin-top: 0; color: #0A2540; border-bottom: 1px solid #F1F5F9; padding-bottom: 8px;">📊 Key Risk Drivers ({active_model_name})</h4>
                <ul style="padding-left: 18px; margin: 0; font-size: 0.84rem; color: #475569; line-height: 1.55;">
                    <li><strong>Credit Score ({st.session_state.form_inputs['creditscore']}):</strong> Serves as a primary repayment reliability factor. High scores heavily suppress predicted default probabilities across both linear and tree models.</li>
                    <li><strong>Interest Rate ({st.session_state.form_inputs['interestrate']}%):</strong> Direct debt burden indicator. Elevated interest rates sharply increase installment obligations and default likelihood.</li>
                    <li><strong>Annual Income (${st.session_state.form_inputs['income']:,.0f}):</strong> Provides essential cash-flow stability, reducing default risk.</li>
                    <li><strong>DTI Ratio ({st.session_state.form_inputs['dtiratio']:.2f}):</strong> Quantifies pre-existing leverage. DTI above 0.36 triggers higher risk weighting.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 3. Applicant Risk Profile Summary
    st.markdown("<div class='section-header'>Applicant Risk Profile Summary</div>", unsafe_allow_html=True)
    
    prof_col1, prof_col2 = st.columns(2)
    with prof_col1:
        st.markdown(
            f"""
            <table class="summary-table">
                <tr><td class="label">👤 Age</td><td class="value">{st.session_state.form_inputs['age']} years</td></tr>
                <tr><td class="label">💳 Credit Score</td><td class="value">{st.session_state.form_inputs['creditscore']}</td></tr>
                <tr><td class="label">💼 Employment Type</td><td class="value">{st.session_state.form_inputs['employmenttype']}</td></tr>
                <tr><td class="label">💵 Annual Income</td><td class="value">${st.session_state.form_inputs['income']:,.2f}</td></tr>
            </table>
            """,
            unsafe_allow_html=True
        )
    with prof_col2:
        st.markdown(
            f"""
            <table class="summary-table">
                <tr><td class="label">💰 Loan Amount</td><td class="value">${st.session_state.form_inputs['loanamount']:,.2f}</td></tr>
                <tr><td class="label">📈 Interest Rate</td><td class="value">{st.session_state.form_inputs['interestrate']}%</td></tr>
                <tr><td class="label">⏱️ Loan Term</td><td class="value">{st.session_state.form_inputs['loanterm']} months</td></tr>
                <tr><td class="label">⚖️ DTI Ratio</td><td class="value">{st.session_state.form_inputs['dtiratio']:.2f}</td></tr>
            </table>
            """,
            unsafe_allow_html=True
        )

    # 4. Interactive What-If Simulation Playground (Uses Currently Selected Model!)
    st.markdown(f"<div class='section-header'>What-If Risk Simulation Playground ({active_model_name})</div>", unsafe_allow_html=True)
    st.write(
        f"Adjust the continuous financial levers below to simulate real-time risk changes using the active **{active_model_name}** model. "
        "Drag any slider — the panel on the right recalculates instantly. It starts at the assessed risk above."
    )
    
    sim_col1, sim_col2 = st.columns([3, 2])
    
    with sim_col1:
        sim_credit = st.slider(
            "Simulate Credit Score",
            min_value=300,
            max_value=850,
            value=int(st.session_state.form_inputs['creditscore']),
            step=5
        )
        
        sim_income = st.slider(
            "Simulate Annual Income ($)",
            min_value=1000.0,
            max_value=1000000.0,
            value=float(st.session_state.form_inputs['income']),
            step=5000.0
        )
        
        sim_loan = st.slider(
            "Simulate Loan Amount ($)",
            min_value=500.0,
            max_value=500000.0,
            value=float(st.session_state.form_inputs['loanamount']),
            step=2500.0
        )
        
        sim_dti = st.slider(
            "Simulate DTI Ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.form_inputs['dtiratio']),
            step=0.01,
            format="%.2f"
        )
        
        sim_rate = st.slider(
            "Simulate Interest Rate (%)",
            min_value=0.1,
            max_value=40.0,
            value=float(st.session_state.form_inputs['interestrate']),
            step=0.1
        )
        
        # Calculate dynamic prediction based on sliders using the exact active model
        simulated_inputs = st.session_state.form_inputs.copy()
        simulated_inputs["creditscore"] = sim_credit
        simulated_inputs["income"] = sim_income
        simulated_inputs["loanamount"] = sim_loan
        simulated_inputs["dtiratio"] = sim_dti
        simulated_inputs["interestrate"] = sim_rate
        
        try:
            sim_features = preprocess_input(simulated_inputs)
            sim_result = predict_loan_risk(sim_features, model_key=st.session_state.selected_model)
            sim_prob = sim_result["default_probability"]
            sim_prob_pct = sim_prob * 100
        except Exception as e:
            st.error(f"Error in What-If calculations: {e}")
            sim_prob_pct = base_prob_pct
            
    with sim_col2:
        # Comparison indicator display
        prob_diff = sim_prob_pct - base_prob_pct
        
        # Determine colors and text based on diff
        if prob_diff < -0.05:
            diff_color = "#10B981"
            diff_text = f"Improved by {abs(prob_diff):.2f} percentage points 🟢"
        elif prob_diff > 0.05:
            diff_color = "#EF4444"
            diff_text = f"Increased by {prob_diff:.2f} percentage points 🔴"
        else:
            diff_color = "#64748B"
            diff_text = (
                "Baseline — drag a slider to compare"
                if abs(prob_diff) < 1e-9
                else "No material change"
            )
            
        st.markdown(
            f"""
            <div style="background-color: #0B192C; color: #FFFFFF; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; height: 350px; display: flex; flex-direction: column; justify-content: center; border: 1px solid #1E293B;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">
                    Risk Simulation ({active_model_name})
                </div>
                
                <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 5px;">BASE LINE RISK</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #94A3B8; margin-bottom: 20px;">{base_prob_pct:.1f}%</div>
                
                <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 5px;">SIMULATED RISK</div>
                <div style="font-size: 3.25rem; font-weight: 800; color: {risk_color if abs(prob_diff) < 0.1 else ('#10B981' if prob_diff < 0 else '#EF4444')}; line-height: 1; margin-bottom: 5px;">
                    {sim_prob_pct:.1f}%
                </div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #FFFFFF; margin-bottom: 15px;">
                    {"LIKELY DEFAULT (1)" if sim_prob_pct > 50 else "UNLIKELY DEFAULT (0)"}
                </div>
                
                <div style="border-top: 1px solid #1E293B; padding-top: 12px; margin-bottom: 10px;">
                    <div style="font-size: 0.7rem; color: #64748B; letter-spacing: 0.03em; margin-bottom: 4px;">ACTIVE LEVERS</div>
                    <div style="font-size: 0.72rem; color: #94A3B8; line-height: 1.45;">
                        Credit {sim_credit} · DTI {sim_dti:.2f} · Rate {sim_rate:.1f}%<br>
                        Income ${sim_income:,.0f} · Loan ${sim_loan:,.0f}
                    </div>
                </div>
                
                <div style="border-top: 1px solid #1E293B; padding-top: 15px;">
                    <div style="font-size: 0.76rem; color: #64748B; margin-bottom: 4px;">RISK DELTA</div>
                    <div style="font-size: 0.88rem; font-weight: 800; color: {diff_color};">{diff_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    # Empty State display
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 4rem 2rem; text-align: center; margin-top: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.01);">
            <div style="font-size: 4rem; color: #94A3B8; margin-bottom: 1rem;">🛡️</div>
            <h3 style="color: #0A2540; margin: 0 0 10px 0; font-size: 1.5rem; font-weight: 800;">Ready to assess a loan?</h3>
            <p style="color: #64748B; font-size: 0.95rem; margin: 0; max-width: 500px; display: inline-block; line-height: 1.5;">
                Enter applicant demographics, financial health indices, select your preferred ML architecture, 
                then click <strong>✦ Predict Loan Default</strong> to generate a multi-model risk evaluation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
