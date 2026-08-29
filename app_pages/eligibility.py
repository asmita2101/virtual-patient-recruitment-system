"""app_pages/eligibility.py - Run the deterministic hard eligibility filter and show results."""

import pandas as pd
import streamlit as st

import backend as be
import state as st_state

st_state.init_session_state()

st.title("🎯 Eligibility")

if not st_state.require_dataset():
    st.stop()

df = st.session_state.dataset

st.subheader("Current Trial")
tc1, tc2 = st.columns(2)
tc1.metric("Trial Name", st.session_state.trial_name or "Not set")
tc2.metric("Rules Confirmed", "Yes" if st.session_state.rules_confirmed else "No")

if not st.session_state.confirmed_rules:
    st.info("No confirmed eligibility rules yet. Go to **📋 Trial Criteria** to generate and confirm rules first.")
    st.stop()

rules = st.session_state.confirmed_rules
with st.expander("View confirmed structured eligibility criteria"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Inclusion**")
        for r in rules["inclusion"]:
            st.markdown(f"- {r['feature']} {be._requirement_text(r['operator'], r['value'])}")
    with col2:
        st.markdown("**Exclusion**")
        for r in rules["exclusion"]:
            st.markdown(f"- {r['feature']} {be._requirement_text(r['operator'], r['value'])}")

st.divider()

if st.button("▶️ Run Eligibility", type="primary"):
    with st.spinner(f"Evaluating {len(df)} patients against the hard eligibility filter..."):
        eligible_df, ineligible_df, explanations = be.apply_eligibility_rules(df, rules["inclusion"], rules["exclusion"])
    st.session_state.eligible_df = eligible_df
    st.session_state.ineligible_df = ineligible_df
    st.session_state.explanations = explanations
    st_state.reset_downstream_after_eligibility()
    if len(eligible_df) == 0:
        st.warning("No patients are eligible under these criteria. Please review your rules on the Trial Criteria page.")
    else:
        st.success(f"{len(eligible_df)} of {len(df)} patients are eligible.")

if st.session_state.eligible_df is not None:
    eligible_df = st.session_state.eligible_df
    ineligible_df = st.session_state.ineligible_df

    st.divider()
    st.subheader("Filtering Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients", len(df))
    c2.metric("✅ Eligible", len(eligible_df))
    c3.metric("❌ Not Eligible", len(ineligible_df))
    st.progress(len(eligible_df) / len(df) if len(df) else 0)

    tab1, tab2 = st.tabs(["✅ Eligible Patients", "❌ Not Eligible Patients"])

    display_cols = [
        "patient_id", "age", "gender", "primary_disease", "disease_severity",
        "hba1c", "kidney_function", "distance_from_trial_site_km", "availability",
        "interest", "consent_to_contact",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    with tab1:
        st.dataframe(eligible_df[display_cols], width='stretch', hide_index=True)
        st.download_button(
            "⬇️ Download Eligible Patients (CSV)", eligible_df[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="eligible_patients.csv", mime="text/csv",
        )

    with tab2:
        st.caption("Select a patient to see exactly why they did not qualify.")
        if len(ineligible_df) > 0:
            pid = st.selectbox("Patient", ineligible_df["patient_id"].tolist())
            exp = st.session_state.explanations[pid]
            exp_df = pd.DataFrame(exp["criteria"]).rename(columns={
                "type": "Type", "feature": "Criterion", "patient_value": "Patient Value",
                "requirement": "Requirement", "result": "Result",
            })
            exp_df["Patient Value"] = exp_df["Patient Value"].astype(str)
            failed_only = exp_df[exp_df["Result"] == "FAIL"]
            st.markdown(f"**Reason(s) patient {pid} was excluded:**")
            st.dataframe(failed_only, width='stretch', hide_index=True)
            with st.expander("View full criterion-by-criterion breakdown"):
                st.dataframe(exp_df, width='stretch', hide_index=True)
        st.dataframe(ineligible_df[display_cols], width='stretch', hide_index=True)

    st.success("Only eligible patients will proceed to the **🏆 Patient Ranking** page.")
