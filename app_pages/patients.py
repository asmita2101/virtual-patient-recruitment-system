"""app_pages/patients.py - Patient management: search, filter, sort, view details."""

import pandas as pd
import streamlit as st

import backend as be
import state as st_state

st_state.init_session_state()

st.title("👥 Patients")

if not st_state.require_dataset():
    st.stop()

df = st.session_state.dataset

st.caption(f"{len(df)} patient records in the active dataset.")

# ------------------------------------------------------------
# Search / filter / sort
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
search_id = c1.text_input("Search Patient ID")
disease_filter = c2.multiselect("Primary Disease", sorted(df["primary_disease"].dropna().unique()))
gender_filter = c3.multiselect("Gender", sorted(df["gender"].dropna().unique()))
interest_filter = c4.multiselect("Interest", sorted(df["interest"].dropna().unique()))

sort_col, sort_dir = st.columns(2)
sortable_cols = ["patient_id", "age", "hba1c", "bmi", "distance_from_trial_site_km", "disease_duration_years"]
sort_by = sort_col.selectbox("Sort by", sortable_cols)
ascending = sort_dir.selectbox("Order", ["Ascending", "Descending"]) == "Ascending"

filtered = df.copy()
if search_id:
    filtered = filtered[filtered["patient_id"].astype(str).str.contains(search_id, case=False, na=False)]
if disease_filter:
    filtered = filtered[filtered["primary_disease"].isin(disease_filter)]
if gender_filter:
    filtered = filtered[filtered["gender"].isin(gender_filter)]
if interest_filter:
    filtered = filtered[filtered["interest"].isin(interest_filter)]
filtered = filtered.sort_values(sort_by, ascending=ascending)

# Eligibility badge if eligibility has already been run.
if st.session_state.explanations:
    filtered = filtered.copy()
    filtered["eligibility_status"] = filtered["patient_id"].map(
        lambda pid: "Eligible" if st.session_state.explanations.get(str(pid), {}).get("eligible") else "Not Eligible"
    )

st.divider()
st.subheader(f"Results ({len(filtered)} patients)")

display_cols = [
    "patient_id", "age", "gender", "primary_disease", "disease_severity",
    "hba1c", "distance_from_trial_site_km", "availability", "interest",
]
if "eligibility_status" in filtered.columns:
    display_cols.append("eligibility_status")
display_cols = [c for c in display_cols if c in filtered.columns]

# Simple pagination for readability on normal laptop screens.
PAGE_SIZE = 25
total_pages = max(1, (len(filtered) - 1) // PAGE_SIZE + 1)
page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
start, end = (page_num - 1) * PAGE_SIZE, page_num * PAGE_SIZE

st.dataframe(filtered[display_cols].iloc[start:end], width='stretch', hide_index=True)
st.caption(f"Page {page_num} of {total_pages}")

st.divider()
st.subheader("🔍 View Patient Details")
pid_options = filtered["patient_id"].tolist()
if pid_options:
    pid = st.selectbox("Select a patient", pid_options)
    row = df[df["patient_id"] == pid].iloc[0]

    fields = [
        ("Patient ID", "patient_id"), ("Age", "age"), ("Gender", "gender"), ("Location", "location"),
        ("Primary Disease", "primary_disease"), ("Disease Duration (years)", "disease_duration_years"),
        ("Disease Severity", "disease_severity"), ("Comorbidities", "comorbidities"),
        ("Previous Treatment", "previous_treatment"), ("BMI", "bmi"), ("Systolic BP", "systolic_bp"),
        ("Diastolic BP", "diastolic_bp"), ("HbA1c", "hba1c"), ("Fasting Glucose", "fasting_glucose"),
        ("Kidney Function", "kidney_function"), ("Liver Function", "liver_function"),
        ("Cholesterol", "cholesterol"), ("Smoking Status", "smoking_status"), ("Alcohol Use", "alcohol_use"),
        ("Pregnancy Status", "pregnancy_status"), ("Allergies", "allergies"), ("Recent Surgery", "recent_surgery"),
        ("Other Serious Condition", "other_serious_condition"), ("Distance (km)", "distance_from_trial_site_km"),
        ("Availability", "availability"), ("Contact Preference", "contact_preference"),
        ("Consent to Contact", "consent_to_contact"), ("Interest", "interest"),
    ]

    with st.expander("Full patient record", expanded=True):
        cols = st.columns(3)
        for i, (label, key) in enumerate(fields):
            value = row.get(key)
            value = "—" if pd.isna(value) else value
            cols[i % 3].metric(label, value)

    if st.session_state.explanations and pid in st.session_state.explanations:
        exp = st.session_state.explanations[pid]
        st.markdown(f"**Eligibility:** {st_state.eligibility_badge(exp['eligible'])}", unsafe_allow_html=True)
        with st.expander("Eligibility criterion-by-criterion breakdown"):
            exp_df = pd.DataFrame(exp["criteria"]).rename(columns={
                "type": "Type", "feature": "Criterion", "patient_value": "Patient Value",
                "requirement": "Requirement", "result": "Result",
            })
            exp_df["Patient Value"] = exp_df["Patient Value"].astype(str)
            st.dataframe(exp_df, width='stretch', hide_index=True)

    st.divider()
    st.subheader("✏️ Edit or Delete This Patient")
    with st.form(f"edit_form_{pid}"):
        new_interest = st.selectbox("Interest", ["Low", "Medium", "High"],
                                     index=["Low", "Medium", "High"].index(row["interest"]) if row["interest"] in ["Low", "Medium", "High"] else 0)
        new_availability = st.selectbox("Availability", ["Low", "Medium", "High"],
                                         index=["Low", "Medium", "High"].index(row["availability"]) if row["availability"] in ["Low", "Medium", "High"] else 0)
        new_consent = st.selectbox("Consent to Contact", ["Yes", "No"],
                                    index=["Yes", "No"].index(row["consent_to_contact"]) if row["consent_to_contact"] in ["Yes", "No"] else 0)
        submitted = st.form_submit_button("Save Changes")
        if submitted:
            updated_df, ok, msg = be.update_patient_record(
                df, pid, {"interest": new_interest, "availability": new_availability, "consent_to_contact": new_consent}
            )
            if ok:
                st.session_state.dataset = updated_df
                if st.session_state.dataset_path:
                    be.save_dataset(updated_df, st.session_state.dataset_path)
                st.success(msg + " (Re-run Eligibility/Ranking to reflect this change.)")
                st.rerun()
            else:
                st.error(msg)

    with st.expander("⚠️ Delete this patient"):
        st.warning(f"This will permanently remove patient **{pid}** from the active dataset.")
        confirm = st.checkbox(f"I understand this will delete patient {pid}", key=f"confirm_delete_{pid}")
        if st.button("🗑️ Delete Patient", disabled=not confirm, type="secondary"):
            updated_df, ok, msg = be.delete_patient_record(df, pid)
            if ok:
                st.session_state.dataset = updated_df
                if st.session_state.dataset_path:
                    be.save_dataset(updated_df, st.session_state.dataset_path)
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
else:
    st.info("No patients match the current search/filter.")
