"""app_pages/explainability.py - Explainable Patient Prioritization: why was this patient ranked here?"""

import pandas as pd
import plotly.express as px
import streamlit as st

import backend as be
import state as st_state

st_state.init_session_state()

st.title("🔍 Explainability")
st.caption("Explainable Patient Prioritization — every number below is traceable to an actual calculation.")

if not st_state.require_dataset() or not st_state.require_ranking():
    st.stop()

ranked = st.session_state.ranked_df
factors = st.session_state.selected_factors
weights = st.session_state.ahp_result["weights"]
df = st.session_state.dataset

pid_list = ranked["patient_id"].tolist()
default_idx = pid_list.index(st.session_state.selected_patient_id) if st.session_state.selected_patient_id in pid_list else 0
pid = st.selectbox("Select a ranked patient", pid_list, index=default_idx)
st.session_state.selected_patient_id = pid

# Patient info + eligibility status
row = df[df["patient_id"] == pid].iloc[0]
st.subheader("Patient Information")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Patient ID", pid)
p2.metric("Age", row["age"])
p3.metric("Primary Disease", row["primary_disease"])
p4.metric("Disease Severity", row["disease_severity"])

if st.session_state.explanations and pid in st.session_state.explanations:
    exp_elig = st.session_state.explanations[pid]
    st.markdown(f"**Eligibility Status:** {st_state.eligibility_badge(exp_elig['eligible'])}", unsafe_allow_html=True)

st.divider()

explanation = be.generate_ranking_explanation(pid, ranked, st.session_state.raw_df, factors, weights)

c1, c2, c3 = st.columns(3)
c1.metric("Final Score", f"{explanation['final_score']:.1f} / 100")
c2.metric("Rank", f"#{explanation['rank']}")
c3.markdown(f"**Priority:** {st_state.priority_badge(explanation['priority'])}", unsafe_allow_html=True)

st.subheader("Ranking Factors & AHP Weights")
breakdown_df = pd.DataFrame(explanation["breakdown"])[
    ["factor", "raw_value", "normalized_score", "ahp_weight", "contribution"]
].rename(columns={
    "factor": "Factor", "raw_value": "Raw Value", "normalized_score": "Patient Score (0-100)",
    "ahp_weight": "AHP Weight", "contribution": "Contribution",
})
breakdown_df["Raw Value"] = breakdown_df["Raw Value"].astype(str)
st.dataframe(breakdown_df, width='stretch', hide_index=True)

fig = px.bar(breakdown_df, x="Factor", y="Contribution", title="Contribution to Final Score by Factor")
st.plotly_chart(fig, width='stretch')

st.subheader("Why was this patient ranked here?")
st.write(explanation["narrative"])

if "interest" in factors:
    interest_raw = st.session_state.raw_df[st.session_state.raw_df["patient_id"] == pid]["interest"].iloc[0]
    interest_score = be.calculate_interest_score(interest_raw)
    st.info(f"**Interest Level:** {interest_raw}  |  **Interest Score:** {interest_score:.0f} / 100 "
            f"(the recorded interest level in the synthetic dataset — not a prediction of future behavior).")
