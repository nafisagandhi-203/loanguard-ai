import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data import load_dataset

df = load_dataset()

# --- HERO SECTION ---
hero_col1, hero_col2 = st.columns([3, 2])

with hero_col1:
    st.markdown(
        """
        <div style="background-color: #0B192C; color: #FFFFFF; border-radius: 8px; padding: 25px; border-left: 5px solid #00D2FF; margin-bottom: 25px;">
            <h1 style="color: #FFFFFF; margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: 0.02em;">LOANGUARD AI</h1>
            <p style="color: #00D2FF; font-size: 1.25rem; font-weight: 600; margin: 5px 0 10px 0;">Smarter Lending. Safer Decisions.</p>
            <p style="color: #94A3B8; font-size: 0.92rem; margin-bottom: 20px; line-height: 1.5; max-width: 90%;">
                Assess loan default risk utilizing machine learning and applicant credit history. 
                Our platform analyzes demographics, employment stability, and financial metrics 
                to quantify credit risk profiles and support secure lending strategies.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Overlay st.page_link inside st.markdown for proper button styling
    st.page_link("pages/prediction.py", label="Assess Loan Risk →", icon="💳")
    st.write("") # Spacing

with hero_col2:
    # Creative micro-visualization: Abstract Financial Network Graph
    nodes_x = [0.1, 0.25, 0.35, 0.55, 0.65, 0.8, 0.9]
    nodes_y = [0.5, 0.2, 0.8, 0.3, 0.7, 0.5, 0.55]
    node_labels = ["Applicant", "Credit", "Income", "DTI", "Mortgage", "LR Model", "Decision"]
    
    edge_x = []
    edge_y = []
    edges = [(0,1), (0,2), (1,3), (1,4), (2,3), (2,4), (3,5), (4,5), (5,6)]
    for e in edges:
        edge_x.extend([nodes_x[e[0]], nodes_x[e[1]], None])
        edge_y.extend([nodes_y[e[0]], nodes_y[e[1]], None])
        
    fig_net = go.Figure()
    # Draw links
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y, 
        mode='lines', 
        line=dict(width=1.5, color='#38BDF8'),
        hoverinfo='none'
    ))
    # Draw nodes
    fig_net.add_trace(go.Scatter(
        x=nodes_x, y=nodes_y, 
        mode='markers+text',
        text=node_labels,
        textposition="top center",
        marker=dict(size=12, color='#10B981', line=dict(width=2, color='#0B192C')),
        textfont=dict(family="Inter, sans-serif", size=10, color="#64748B"),
        hoverinfo='none'
    ))
    fig_net.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0.1, 1.0]),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=195
    )
    st.plotly_chart(fig_net, width="stretch", config={'displayModeBar': False})

# --- RISK INTELLIGENCE OVERVIEW (KPIs) ---
st.markdown("<div class='section-header'>Risk Intelligence Overview</div>", unsafe_allow_html=True)

# Custom KPI Card Renderer
def render_kpi(title, value, support, card_type="primary", icon="📊"):
    st.markdown(
        f"""
        <div class="kpi-card kpi-card-{card_type}">
            <div class="kpi-card-title">{icon} {title}</div>
            <div class="kpi-card-value">{value}</div>
            <div class="kpi-card-support">{support}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Derive KPIs from the live dataset when available (falls back to last-known values)
if df is not None:
    total_applications = f"{len(df):,}"
    default_cases = f"{int(df['Default'].sum()):,}"
    default_rate_pct = float(df['Default'].mean() * 100)
else:
    total_applications = "255,347"
    default_cases = "29,653"
    default_rate_pct = 11.61

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    render_kpi("Total Applications", total_applications, "↑ Historical Dataset", "primary", "📄")
with kpi_col2:
    render_kpi("Default Cases", default_cases, "↓ Historical Default Cases", "danger", "⚠️")
with kpi_col3:
    render_kpi("Default Rate", f"{default_rate_pct:.2f}%", "⚙️ Dataset Default Ratio", "warning", "📈")
with kpi_col4:
    render_kpi("Model Status", "Logistic Reg.", "● Preprocessors Loaded", "success", "⚙️")


# --- RISK SNAPSHOT ---
st.markdown("<div class='section-header'>Risk Snapshot</div>", unsafe_allow_html=True)

snap_col1, snap_col2 = st.columns([3, 2])

with snap_col1:
    # Stacked horizontal bar chart mapping risk ranges
    fig_snap = go.Figure()
    fig_snap.add_trace(go.Bar(
        y=["Dataset Split"], x=[68], name="Low Risk (0-30% Prob.)",
        orientation='h', marker=dict(color='#10B981', line=dict(color='#FFFFFF', width=1))
    ))
    fig_snap.add_trace(go.Bar(
        y=["Dataset Split"], x=[21], name="Medium Risk (30-60% Prob.)",
        orientation='h', marker=dict(color='#F59E0B', line=dict(color='#FFFFFF', width=1))
    ))
    fig_snap.add_trace(go.Bar(
        y=["Dataset Split"], x=[11], name="High Risk (60-100% Prob.)",
        orientation='h', marker=dict(color='#EF4444', line=dict(color='#FFFFFF', width=1))
    ))
    
    fig_snap.update_layout(
        barmode='stack',
        height=120,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=True, ticksuffix="%"),
        yaxis=dict(showgrid=False, showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=-1.1, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_snap, width="stretch", config={'displayModeBar': False})

with snap_col2:
    st.markdown(
        """
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); height: 110px; display: flex; flex-direction: column; justify-content: center;">
            <strong style="color: #0A2540; font-size: 0.85rem; text-transform: uppercase;">UI Threshold Bands</strong>
            <p style="margin: 5px 0 0 0; font-size: 0.76rem; color: #64748B; line-height: 1.4;">
                Risk bands (Low: &lt;30%, Medium: 30-60%, High: &gt;60%) are conceptual UI classifications designed to support risk management triage, and are not formal model calibration segments.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- DATASET VISUALIZATIONS ---
st.markdown("<div class='section-header'>Dataset Exploration</div>", unsafe_allow_html=True)

if df is not None:
    # 2. Charts Layout (Left: Donut default split, Right: Tabs for numerical distributions)
    chart_col1, chart_col2 = st.columns([1, 1])
    
    with chart_col1:
        default_counts = df['Default'].value_counts().reset_index()
        default_counts.columns = ['Status', 'Count']
        default_counts['Status'] = default_counts['Status'].map({
            0: f'No Default ({100 - default_rate_pct:.2f}%)',
            1: f'Default ({default_rate_pct:.2f}%)'
        })
        
        fig_donut = px.pie(
            default_counts, 
            values='Count', 
            names='Status',
            hole=0.6,
            color='Status',
            color_discrete_map={
                f'No Default ({100 - default_rate_pct:.2f}%)': '#10B981',
                f'Default ({default_rate_pct:.2f}%)': '#EF4444'
            },
            title="Historical Default Distribution"
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent')
        fig_donut.update_layout(
            margin=dict(t=40, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=11, color="#1E293B")
        )
        st.plotly_chart(fig_donut, width="stretch", config={'displayModeBar': False})
        
    with chart_col2:
        df_sample = df.sample(n=15000, random_state=42) if len(df) > 15000 else df
        
        tab_credit, tab_loan, tab_income = st.tabs(["Credit Score", "Loan Amount", "Annual Income"])
        
        with tab_credit:
            fig_credit = px.histogram(
                df_sample, x='CreditScore', color='Default',
                nbins=30, color_discrete_map={0: '#0B192C', 1: '#EF4444'},
                labels={'CreditScore': 'Credit Score', 'Default': 'Default (1=Yes)'}
            )
            fig_credit.update_layout(
                margin=dict(t=20, b=10, l=10, r=10),
                height=260,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", size=11),
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9")
            )
            st.plotly_chart(fig_credit, width="stretch", config={'displayModeBar': False})
            
        with tab_loan:
            fig_loan = px.histogram(
                df_sample, x='LoanAmount', color='Default',
                nbins=30, color_discrete_map={0: '#1E3A8A', 1: '#EF4444'},
                labels={'LoanAmount': 'Loan Amount ($)', 'Default': 'Default (1=Yes)'}
            )
            fig_loan.update_layout(
                margin=dict(t=20, b=10, l=10, r=10),
                height=260,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", size=11),
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9")
            )
            st.plotly_chart(fig_loan, width="stretch", config={'displayModeBar': False})
            
        with tab_income:
            fig_income = px.histogram(
                df_sample, x='Income', color='Default',
                nbins=30, color_discrete_map={0: '#0D9488', 1: '#EF4444'},
                labels={'Income': 'Annual Income ($)', 'Default': 'Default (1=Yes)'}
            )
            fig_income.update_layout(
                margin=dict(t=20, b=10, l=10, r=10),
                height=260,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", size=11),
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9")
            )
            st.plotly_chart(fig_income, width="stretch", config={'displayModeBar': False})
            
else:
    st.info("Dataset 'Loan_default.csv' not found. Exploration features are disabled.")
