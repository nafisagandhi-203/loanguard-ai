import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page titles
st.markdown("<h1 style='margin-bottom: 0;'>Risk Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 1.05rem; margin-bottom: 25px;'>Explore how the trained Logistic Regression model associates input features with default risk.</p>", unsafe_allow_html=True)

# Path to serialized model metrics
metrics_path = "model/metrics.json"

if not os.path.exists(metrics_path):
    st.warning("Model metrics file not found. Please run model training first to populate the coefficients.")
else:
    # 1. Load trained coefficients
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    coefficients = metrics.get("coefficients", {})
    
    st.markdown("<div class='section-header'>Logistic Regression Model Coefficients</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 20px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 25px; font-size: 0.92rem; color: #334155; border-left: 4px solid #0A2540; line-height: 1.65;">
            Logistic regression models log-odds of the default class. 
            <strong>Positive coefficients</strong> increase the predicted probability of default, while 
            <strong>negative coefficients</strong> decrease the predicted probability of default, holding all other features constant. 
            The magnitude represents the relative impact of a standard-deviation change in that feature (since features are standardized).
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 2. Horizontal Plotly Bar Chart
    coef_data = []
    for feat, val in coefficients.items():
        feat_label = feat.replace("_", " ").title()
        # Clean specific labels
        if feat_label == "Dtiratio":
            feat_label = "DTI Ratio"
        elif feat_label == "Numcreditlines":
            feat_label = "Credit Lines"
        elif feat_label == "Monthsemployed":
            feat_label = "Months Employed"
        elif feat_label == "Hascosigner":
            feat_label = "Has Co-signer"
        elif feat_label == "Hasdependents":
            feat_label = "Has Dependents"
        elif feat_label == "Hasmortgage":
            feat_label = "Has Mortgage"
            
        coef_data.append({"Feature": feat_label, "Coefficient": val})
        
    coef_df = pd.DataFrame(coef_data)
    # Sort for plotting
    coef_df = coef_df.sort_values(by="Coefficient", ascending=True)
    
    # Impact mapping
    coef_df["Impact"] = coef_df["Coefficient"].apply(lambda x: "Increases Default Risk (Positive)" if x > 0 else "Reduces Default Risk (Negative)")
    
    fig_coef = px.bar(
        coef_df,
        x="Coefficient",
        y="Feature",
        orientation="h",
        color="Impact",
        color_discrete_map={
            "Increases Default Risk (Positive)": "#EF4444",
            "Reduces Default Risk (Negative)": "#10B981"
        },
        labels={"Feature": "Model Predictor", "Coefficient": "Standardized Coefficient Effect Size"},
        title="Impact of Standardized Features on Default Log-Odds"
    )
    
    fig_coef.update_layout(
        margin=dict(t=40, b=10, l=10, r=10),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=11, color="#1E293B"),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=True, zerolinecolor="#64748B", zerolinewidth=1.5),
        yaxis=dict(showgrid=False)
    )
    
    st.plotly_chart(fig_coef, width="stretch", config={'displayModeBar': False})
    
    # 3. Factor Analysis Breakdown
    st.markdown("<div class='section-header'>Predictive Risk Factor Breakdown</div>", unsafe_allow_html=True)
    
    factor_col1, factor_col2 = st.columns(2)
    
    with factor_col1:
        st.markdown(
            """
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 15px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); margin-bottom: 15px;">
                <h4 style="margin: 0 0 5px 0; color: #EF4444; font-size: 0.95rem;">📈 Interest Rate</h4>
                <p style="margin: 0; font-size: 0.85rem; color: #475569; line-height: 1.55;">
                    The model coefficients indicate that higher interest rates have a significant positive coefficient. 
                    A higher interest rate increases the borrower's monthly payments, raising the debt burden and increasing the probability of default, holding other factors constant.
                </p>
            </div>
            
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 15px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); margin-bottom: 15px;">
                <h4 style="margin: 0 0 5px 0; color: #EF4444; font-size: 0.95rem;">📊 Debt-to-Income (DTI) Ratio</h4>
                <p style="margin: 0; font-size: 0.85rem; color: #475569; line-height: 1.55;">
                    A higher DTI ratio indicates that a large portion of the applicant's income is already dedicated to debt payments. 
                    This reduces the financial cushion available to handle emergencies, and the positive model coefficient reflects this increased default risk.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with factor_col2:
        st.markdown(
            """
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 15px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); margin-bottom: 15px;">
                <h4 style="margin: 0 0 5px 0; color: #10B981; font-size: 0.95rem;">💳 Credit Score</h4>
                <p style="margin: 0; font-size: 0.85rem; color: #475569; line-height: 1.55;">
                    A higher credit score represents a solid repayment history and creditworthiness. 
                    This feature exhibits a large negative coefficient, indicating that higher credit scores significantly reduce the predicted log-odds of default.
                </p>
            </div>
            
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 15px; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); margin-bottom: 15px;">
                <h4 style="margin: 0 0 5px 0; color: #10B981; font-size: 0.95rem;">💵 Annual Income</h4>
                <p style="margin: 0; font-size: 0.85rem; color: #475569; line-height: 1.55;">
                    Applicant annual income is a crucial mitigating factor with a negative coefficient. 
                    Higher incomes provide a larger financial capacity to cover loan installments, lowering default probability in the trained model.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 15px; border-top: 1px solid #E2E8F0; padding-top: 10px;">
            *Note: Standardized coefficients display effect sizes relative to the standard deviation of each variable. This allows direct magnitude comparison between features measured in different units (e.g. credit score vs annual income). Holding other features constant is a standard assumption of regression coefficients.
        </div>
        """,
        unsafe_allow_html=True
    )
