import streamlit as st
import pandas as pd
from utils.data import load_dataset

st.markdown("<h1 style='margin-bottom: 0;'>About the Project</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 1.05rem; margin-bottom: 25px;'>Technical portfolio, workflow architecture, and dataset exploration.</p>", unsafe_allow_html=True)

# --- HOW IT WORKS TIMELINE FLOWCHART ---
st.markdown("<div class='section-header'>How LoanGuard AI Works</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="flowchart-container">
        <div class="flowchart-node">
            <div style="font-size: 1.5rem; margin-bottom: 4px;">📝</div>
            <div class="flowchart-title">1. Input Profile</div>
            <div class="flowchart-desc">Enter demographic & financial data</div>
        </div>
        <div class="flowchart-arrow">➔</div>
        <div class="flowchart-node">
            <div style="font-size: 1.5rem; margin-bottom: 4px;">✅</div>
            <div class="flowchart-title">2. Validate Inputs</div>
            <div class="flowchart-desc">Client-side range checking</div>
        </div>
        <div class="flowchart-arrow">➔</div>
        <div class="flowchart-node">
            <div style="font-size: 1.5rem; margin-bottom: 4px;">⚙️</div>
            <div class="flowchart-title">3. Encode & Scale</div>
            <div class="flowchart-desc">Label map & Z-score scaling</div>
        </div>
        <div class="flowchart-arrow">➔</div>
        <div class="flowchart-node">
            <div style="font-size: 1.5rem; margin-bottom: 4px;">🧠</div>
            <div class="flowchart-title">4. ML Inference</div>
            <div class="flowchart-desc">Logistic Regression model</div>
        </div>
        <div class="flowchart-arrow">➔</div>
        <div class="flowchart-node">
            <div style="font-size: 1.5rem; margin-bottom: 4px;">📊</div>
            <div class="flowchart-title">5. Risk Triage</div>
            <div class="flowchart-desc">Bands assigned & gauge plots</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- PROJECT OVERVIEW CARDS ---
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 20px; min-height: 250px;">
            <h3 style="margin-top: 0; color: #0A2540; font-size: 1.15rem;">📌 Project Overview</h3>
            <p style="margin: 0; color: #475569; font-size: 0.88rem; line-height: 1.6;">
                <strong>LoanGuard AI</strong> is a machine learning-backed financial risk assessment platform. 
                It leverages historic lending records to identify high-risk profiles before credit disbursement. 
                By deploying automated, mathematical predictive algorithms, the system helps credit underwriting 
                teams streamline loan pipelines while securing capital reserves against non-performing assets.
            </p>
        </div>
        
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 20px; min-height: 230px;">
            <h3 style="margin-top: 0; color: #0A2540; font-size: 1.15rem;">💡 Problem Statement</h3>
            <p style="margin: 0; color: #475569; font-size: 0.88rem; line-height: 1.6;">
                Defaulting loans represent the primary source of credit loss in the commercial banking industry. 
                Evaluating loan applicants manually is slow, error-prone, and struggles to capture complex interaction 
                dynamics among multiple predictors (e.g. combined effect of high DTI ratio and low credit score). 
                An automated binary classification pipeline mathematically quantifies risk to support secure lending decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
with col2:
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 20px; min-height: 250px;">
            <h3 style="margin-top: 0; color: #0A2540; font-size: 1.15rem;">⚙️ Machine Learning Approach</h3>
            <p style="margin: 0; color: #475569; font-size: 0.88rem; line-height: 1.6;">
                The platform utilizes a standardized <strong>Logistic Regression</strong> model. 
                This algorithm provides a high degree of transparency and interpretability via standardized coefficients—a 
                critical compliance requirement for banking risk models. The features undergo outlier removal (IQR), 
                alphabetical label mapping, and Z-score standardization ($z = \\frac{x-\\mu}{\\sigma}$) to guarantee stable convergence.
            </p>
        </div>
        
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 20px; min-height: 230px;">
            <h3 style="margin-top: 0; color: #0A2540; font-size: 1.15rem;">🛠️ Technology Stack</h3>
            <p style="margin: 0 0 10px 0; color: #475569; font-size: 0.88rem; line-height: 1.6;">
                Built strictly using modern open-source Python data science and web development packages:
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 8px; font-size: 0.8rem; font-weight: 700; color: #0A2540;">
                <span style="background-color: #E0F2FE; padding: 4px 10px; border-radius: 4px; border: 1px solid #BAE6FD;">🐍 Python</span>
                <span style="background-color: #E0F2FE; padding: 4px 10px; border-radius: 4px; border: 1px solid #BAE6FD;">🎨 Streamlit</span>
                <span style="background-color: #E0F2FE; padding: 4px 10px; border-radius: 4px; border: 1px solid #BAE6FD;">🐼 Pandas</span>
                <span style="background-color: #E0F2FE; padding: 4px 10px; border-radius: 4px; border: 1px solid #BAE6FD;">🔢 NumPy</span>
                <span style="background-color: #E0F2FE; padding: 4px 10px; border-radius: 4px; border: 1px solid #BAE6FD;">🧠 Scikit-Learn</span>
                <span style="background-color: #E0F2FE; padding: 4px 10px; border-radius: 4px; border: 1px solid #BAE6FD;">📊 Plotly</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- FUTURE SCOPE & STUDENT DETAILS ---
col_spec1, col_spec2 = st.columns(2)

with col_spec1:
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 20px; min-height: 200px;">
            <h3 style="margin-top: 0; color: #0A2540; font-size: 1.15rem;">🚀 Future Scope</h3>
            <ul style="padding-left: 18px; margin: 0; color: #475569; font-size: 0.85rem; line-height: 1.6;">
                <li><strong>Class Balancing:</strong> Incorporate SMOTE or focal loss weights to improve model recall on default predictions.</li>
                <li><strong>Ensemble Models:</strong> Deploy XGBoost or Random Forests for non-linear feature boundary classification.</li>
                <li><strong>Explainable AI (XAI):</strong> Integrate SHAP/LIME tools to provide detailed explanations for individual predictions.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_spec2:
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 20px; min-height: 200px;">
            <h3 style="margin-top: 0; color: #0A2540; font-size: 1.15rem;">🎓 Academic Information</h3>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.65;">
                <strong>Degree Program:</strong> Bachelor of Technology (B.Tech)<br>
                <strong>Department:</strong> Computer Science and Engineering (CSE)<br>
                <strong>Domain:</strong> Machine Learning & Applied Financial Analytics<br>
                <strong>Advisor:</strong> Department ML Project Supervisor
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- DATASET EXPLORER ---
st.markdown("<div class='section-header'>Historical Dataset Explorer</div>", unsafe_allow_html=True)

df_exp = load_dataset()

if df_exp is not None:
    st.write("Browse and filter records from the historical loan application database below:")
    
    # Explorer Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 2, 1])
    
    with ctrl_col1:
        num_rows = st.slider("Records to display", min_value=5, max_value=200, value=15, step=5)
        
    with ctrl_col2:
        search_id = st.text_input("Search by Loan ID (Exact match, e.g. I38PQUQS96)", "").strip()
        
    with ctrl_col3:
        status_filter = st.selectbox(
            "Filter Default Status",
            options=["All Records", "No Default (0) Only", "Default (1) Only"],
            index=0
        )
        
    filtered_df = df_exp.copy()
    
    if search_id:
        filtered_df = filtered_df[filtered_df["LoanID"].astype(str) == search_id]
        
    if status_filter == "No Default (0) Only":
        filtered_df = filtered_df[filtered_df["Default"] == 0]
    elif status_filter == "Default (1) Only":
        filtered_df = filtered_df[filtered_df["Default"] == 1]
        
    all_cols = list(df_exp.columns)
    selected_cols = st.multiselect("Select columns to view", options=all_cols, default=all_cols)
    
    if len(filtered_df) == 0:
        st.info("No records match the active search/filters.")
    else:
        st.dataframe(filtered_df[selected_cols].head(num_rows), width="stretch")
        st.write(f"Showing top {min(num_rows, len(filtered_df))} of {len(filtered_df):,} filtered records.")
        
    # Basic statistics
    st.markdown("#### Feature Descriptive Statistics", unsafe_allow_html=True)
    stat_opt = st.selectbox("Select variable type to view summary statistics", ["Numerical Features", "Categorical Features"])
    
    if stat_opt == "Numerical Features":
        num_cols = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
        st.dataframe(df_exp[num_cols].describe().round(2), width="stretch")
    else:
        cat_cols = ["Education", "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"]
        cat_summaries = []
        for col in cat_cols:
            val_counts = df_exp[col].value_counts()
            top_val = val_counts.index[0]
            top_freq = val_counts.iloc[0]
            top_pct = (top_freq / len(df_exp)) * 100
            cat_summaries.append({
                "Variable": col,
                "Unique Classes": df_exp[col].nunique(),
                "Top Category": top_val,
                "Top Freq": f"{top_freq:,}",
                "Top Pct": f"{top_pct:.2f}%"
            })
        st.dataframe(pd.DataFrame(cat_summaries), width="stretch")

else:
    st.info("Dataset 'Loan_default.csv' not found. Explorer is disabled.")
