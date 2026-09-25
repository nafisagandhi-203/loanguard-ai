import streamlit as st
import os

# 1. Page Configuration
st.set_page_config(
    page_title="LoanGuard AI - Risk Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom CSS
css_path = os.path.join("assets", "custom.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("Styling assets not found. Interface layout may look default.")

# 3. Dynamic Top Navigation Header Bar (drawn on content section)
st.markdown(
    """
    <div class="top-nav">
        <div class="top-nav-brand">🏦🛡️ LoanGuard AI</div>
        <div class="top-nav-info">
            <span><strong>Risk Intelligence Platform</strong></span>
            <span style="color: #10B981; font-weight: bold; display: flex; align-items: center; gap: 5px;">
                <span style="font-size: 1.1rem; line-height: 1;">●</span> Model Online
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# 4. Custom Sidebar Header (branding layout)
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 15px 0; border-bottom: 1px solid #1E293B; margin-bottom: 15px;">
        <div style="font-size: 3rem; margin-bottom: 5px;">🛡️🏦</div>
        <h2 style="margin: 0; color: #FFFFFF; font-size: 1.35rem; font-weight: 800; letter-spacing: 0.08em; line-height: 1.2;">LOANGUARD AI</h2>
        <p style="margin: 5px 0 0 0; color: #94A3B8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
            Risk Intelligence Platform
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# 5. Define Navigation Pages
dashboard_page = st.Page("pages/dashboard.py", title="Dashboard", icon=":material/dashboard:", default=True)
prediction_page = st.Page("pages/prediction.py", title="Loan Assessment", icon=":material/shield_with_heart:")
risk_analysis_page = st.Page("pages/risk_analysis.py", title="Risk Intelligence", icon=":material/analytics:")
model_info_page = st.Page("pages/model_info.py", title="Model Insights", icon=":material/psychology:")
about_page = st.Page("pages/about.py", title="About", icon=":material/info:")

# Orchestrate navigation links
pg = st.navigation([
    dashboard_page,
    prediction_page,
    risk_analysis_page,
    model_info_page,
    about_page
])

# Run Selected Page
pg.run()

# 6. Sidebar Bottom Status & Metadata Panel
st.sidebar.markdown(
    """
    <div class="sidebar-status-box">
        <div class="sidebar-status-item">
            <span style="color: #64748B; font-weight: 700;">MODEL STATUS</span>
            <span style="color: #10B981; font-weight: 800;">● ONLINE</span>
        </div>
        <div class="sidebar-status-item">
            <span style="color: #64748B; font-weight: 700;">ALGORITHM</span>
            <span style="color: #F1F5F9; font-weight: 700;">Logistic Reg.</span>
        </div>
        <div class="sidebar-status-item">
            <span style="color: #64748B; font-weight: 700;">VERSION</span>
            <span style="color: #F1F5F9; font-weight: 700;">v1.0</span>
        </div>
    </div>
    <div class="sidebar-footer">
        <strong>Academic Project:</strong> B.Tech CSE<br>
        <strong>Subject:</strong> Machine Learning Lab
    </div>
    """,
    unsafe_allow_html=True
)
