"""app_pages/dashboard.py - Dashboard page (real multi-page app: this is its own page)."""

import pandas as pd
import plotly.express as px
import streamlit as st

import backend as be
import state as st_state

st_state.init_session_state()

st.title("🏥 Virtual Patient Recruitment System")
st.caption("AI-Powered Clinical Trial Recruitment Decision Support")

if not st_state.require_dataset():
    st.stop()

df = st.session_state.dataset

# ------------------------------------------------------------
# Key metrics (all computed from actual session data, never hard-coded)
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Patients", len(df))

n_eligible = len(st.session_state.eligible_df) if st.session_state.eligible_df is not None else "—"
n_ineligible = len(st.session_state.ineligible_df) if st.session_state.ineligible_df is not None else "—"
c2.metric("Eligible Patients", n_eligible)
c3.metric("Not Eligible Patients", n_ineligible)

n_factors = len(st.session_state.selected_factors) if st.session_state.selected_factors else 0
c4.metric("Ranking Factors Selected", n_factors)

c5, c6, c7 = st.columns(3)
if st.session_state.ranked_df is not None:
    ranked = st.session_state.ranked_df
    c5.metric("High Priority Patients", int((ranked["priority"] == "High Priority").sum()))
else:
    c5.metric("High Priority Patients", "—")
c6.metric("Current Trial", st.session_state.trial_name or "Not set")
status = "Ranking complete" if st.session_state.ranked_df is not None else (
    "Eligibility complete" if st.session_state.eligible_df is not None else "Not started")
c7.metric("Trial Status", status)

st.divider()

# ------------------------------------------------------------
# Charts
# ------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.session_state.eligible_df is not None:
        elig_counts = pd.DataFrame({
            "Status": ["Eligible", "Not Eligible"],
            "Count": [len(st.session_state.eligible_df), len(st.session_state.ineligible_df)],
        })
        fig = px.pie(elig_counts, names="Status", values="Count", title="Eligibility Distribution",
                     color="Status", color_discrete_map={"Eligible": "#22c55e", "Not Eligible": "#ef4444"}, hole=0.4)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Run eligibility filtering (Trial Criteria → Eligibility) to see this chart.")

with col2:
    if st.session_state.ranked_df is not None:
        pr_counts = st.session_state.ranked_df["priority"].value_counts().reset_index()
        pr_counts.columns = ["Priority", "Count"]
        fig = px.pie(pr_counts, names="Priority", values="Count", title="Priority Distribution",
                     color="Priority",
                     color_discrete_map={"High Priority": "#22c55e", "Medium Priority": "#eab308", "Low Priority": "#ef4444"},
                     hole=0.4)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Complete Patient Ranking to see this chart.")

col3, col4 = st.columns(2)
with col3:
    source_df = st.session_state.eligible_df if st.session_state.eligible_df is not None else df
    int_counts = source_df["interest"].value_counts().reset_index()
    int_counts.columns = ["Interest", "Count"]
    fig = px.bar(int_counts, x="Interest", y="Count", title="Interest Distribution" + (" (Eligible)" if st.session_state.eligible_df is not None else " (All Patients)"))
    st.plotly_chart(fig, width='stretch')

with col4:
    source_df = st.session_state.eligible_df if st.session_state.eligible_df is not None else df
    avail_counts = source_df["availability"].value_counts().reset_index()
    avail_counts.columns = ["Availability", "Count"]
    fig = px.bar(avail_counts, x="Availability", y="Count", title="Availability Distribution" + (" (Eligible)" if st.session_state.eligible_df is not None else " (All Patients)"))
    st.plotly_chart(fig, width='stretch')

st.divider()

if st.session_state.ranked_df is not None:
    st.subheader("🏆 Top Ranked Patient")
    top = st.session_state.ranked_df.iloc[0]
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Patient ID", top["patient_id"])
    tc2.metric("Final Score", f"{top['final_score']:.1f} / 100")
    tc3.markdown(f"**Priority:** {st_state.priority_badge(top['priority'])}", unsafe_allow_html=True)
else:
    st.info("Once ranking is complete, the top-ranked patient will be highlighted here.")

st.divider()
st.caption(st_state.DISCLAIMER)
