"""
frontend.py
------------------------------------------------------------
Virtual Patient Recruitment System for Clinical Trials
Main entry point / router for the true multi-page Streamlit app.

This file only builds the app shell (page config, global CSS,
sidebar with dataset upload + Gemini status, and the page
navigation list). Each feature lives in its own file under
app_pages/ and is a genuinely separate page - clicking a sidebar
item navigates to that page, it does not just scroll a single
long page.

All business logic still lives in backend.py.
All Gemini/LLM logic still lives in llm_integration.py.
Shared session state / styling helpers live in state.py.

Run with:
    streamlit run frontend.py
------------------------------------------------------------
"""

import streamlit as st

import state as st_state

st.set_page_config(
    page_title="Virtual Patient Recruitment System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st_state.init_session_state()
st_state.inject_global_css()

pages = [
    st.Page("app_pages/dashboard.py", title="Dashboard", icon="🏠", default=True),
    st.Page("app_pages/trial_criteria.py", title="Trial Criteria", icon="📋"),
    st.Page("app_pages/patients.py", title="Patients", icon="👥"),
    st.Page("app_pages/add_patient.py", title="Add Patient", icon="➕"),
    st.Page("app_pages/eligibility.py", title="Eligibility", icon="🎯"),
    st.Page("app_pages/ranking.py", title="Patient Ranking", icon="🏆"),
    st.Page("app_pages/explainability.py", title="Explainability", icon="🔍"),
    st.Page("app_pages/whatif.py", title="What-If Analysis", icon="🔄"),
]

pg = st.navigation(pages)

st_state.render_sidebar()

pg.run()
