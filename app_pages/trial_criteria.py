"""app_pages/trial_criteria.py - Define a trial and generate/review/confirm eligibility rules."""

import json

import streamlit as st

import backend as be
import llm_integration as llm
import state as st_state

st_state.init_session_state()

st.title("📋 Trial Criteria")

if not st_state.require_dataset():
    st.stop()

df = st.session_state.dataset

tab1, tab2 = st.tabs(["1️⃣ Define Trial", "2️⃣ Review & Confirm Rules"])

# ------------------------------------------------------------
# TAB 1: Natural-language criteria -> Gemini -> structured rules
# ------------------------------------------------------------
with tab1:
    st.session_state.trial_name = st.text_input("Trial Name", st.session_state.trial_name)
    st.session_state.trial_description = st.text_area("Trial Description", st.session_state.trial_description, height=80)
    st.session_state.nl_criteria = st.text_area(
        "Enter Clinical Trial Eligibility Criteria", st.session_state.nl_criteria, height=160,
        placeholder=("Example: Patients aged 40 to 65 with Type 2 Diabetes, HbA1c above 7, "
                     "moderate disease severity, and no recent surgery."),
    )

    if st.button("🔮 Generate Eligibility Rules", type="primary"):
        if not st.session_state.trial_name.strip():
            st.warning("Please enter a trial name.")
        else:
            with st.spinner("Sending criteria to Gemini..."):
                dataset_summary = be.inspect_dataset(df)["columns"]
                dataset_summary = {k: v for k, v in dataset_summary.items() if k != "patient_id"}
                success, rules, message = llm.generate_eligibility_rules(st.session_state.nl_criteria, dataset_summary)
            if success:
                st.session_state.generated_rules = rules
                st.session_state.rules_confirmed = False
                st.session_state.confirmed_rules = None
                st_state.reset_downstream_after_eligibility()
                st.session_state.eligible_df = None
                st.session_state.ineligible_df = None
                st.session_state.explanations = None
                st.success(message + " Go to the **Review & Confirm Rules** tab.")
            else:
                st.error(message)

# ------------------------------------------------------------
# TAB 2: Review, edit, validate, confirm
# ------------------------------------------------------------
with tab2:
    if not st.session_state.generated_rules:
        st.info("No rules generated yet. Use the **Define Trial** tab first.")
    else:
        rules = st.session_state.generated_rules
        valid, errors, warnings = be.validate_rules(rules, df)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Inclusion Criteria")
            if rules["inclusion"]:
                for r in rules["inclusion"]:
                    st.markdown(f"- **{r.get('feature')}** {be._requirement_text(r.get('operator'), r.get('value'))}")
            else:
                st.write("_None specified_")
        with col2:
            st.subheader("🚫 Exclusion Criteria")
            if rules["exclusion"]:
                for r in rules["exclusion"]:
                    st.markdown(f"- **{r.get('feature')}** {be._requirement_text(r.get('operator'), r.get('value'))}")
            else:
                st.write("_None specified_")

        with st.expander("View raw structured JSON"):
            st.json(rules)

        if errors:
            st.error("The generated rules have validation errors and cannot be executed yet:")
            for e in errors:
                st.markdown(f"- {e}")
        if warnings:
            st.warning("Warnings (review before confirming):")
            for w in warnings:
                st.markdown(f"- {w}")

        st.divider()
        st.subheader("Edit Rules (optional)")
        edited_json = st.text_area("Edit the JSON directly if needed", value=json.dumps(rules, indent=2), height=240)

        c1, c2 = st.columns(2)
        if c1.button("💾 Apply Edited JSON"):
            try:
                new_rules = json.loads(edited_json)
                st.session_state.generated_rules = new_rules
                st.session_state.rules_confirmed = False
                st.success("Edited rules applied. Re-checking validation...")
                st.rerun()
            except Exception as exc:
                st.error(f"Invalid JSON: {exc}")

        if c2.button("✔️ Confirm Rules", type="primary", disabled=not valid):
            st.session_state.confirmed_rules = rules
            st.session_state.rules_confirmed = True
            # Eligibility itself now runs on the dedicated Eligibility page.
            st.session_state.eligible_df = None
            st.session_state.ineligible_df = None
            st.session_state.explanations = None
            st_state.reset_downstream_after_eligibility()
            st.success("Rules confirmed! Go to the **🎯 Eligibility** page to run the eligibility engine.")

if st.session_state.rules_confirmed:
    st.sidebar.success(f"Trial '{st.session_state.trial_name}' — rules confirmed.")
