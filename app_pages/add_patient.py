"""app_pages/add_patient.py - Add a new patient record to the dataset (persists to CSV)."""

import streamlit as st

import backend as be
import state as st_state

st_state.init_session_state()

st.title("➕ Add New Patient")

if not st_state.require_dataset():
    st.stop()

df = st.session_state.dataset
st.caption("New patients are added to the active dataset immediately and become available "
           "for eligibility checks and ranking right away.")

with st.form("add_patient_form", clear_on_submit=False):
    st.subheader("Basic Information")
    b1, b2, b3, b4 = st.columns(4)
    patient_id = b1.text_input("Patient ID *", placeholder="e.g. P05001")
    age = b2.number_input("Age *", min_value=0, max_value=120, value=45)
    gender = b3.selectbox("Gender *", sorted(df["gender"].dropna().unique().tolist()))
    location = b4.text_input("Location *", placeholder="City / region")

    st.subheader("Medical Information")
    m1, m2 = st.columns(2)
    primary_disease = m1.selectbox("Primary Disease *", sorted(df["primary_disease"].dropna().unique().tolist()))
    disease_duration_years = m2.number_input("Disease Duration (years) *", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
    m3, m4 = st.columns(2)
    disease_severity = m3.selectbox("Disease Severity *", sorted(df["disease_severity"].dropna().unique().tolist()))
    comorbidities = m4.selectbox("Comorbidities *", sorted(df["comorbidities"].dropna().unique().tolist()))
    previous_treatment = st.selectbox("Previous Treatment *", sorted(df["previous_treatment"].dropna().unique().tolist()))

    st.subheader("Clinical Measurements")
    c1, c2, c3 = st.columns(3)
    bmi = c1.number_input("BMI *", min_value=5.0, max_value=100.0, value=25.0, step=0.1)
    systolic_bp = c2.number_input("Systolic BP *", min_value=50, max_value=260, value=120)
    diastolic_bp = c3.number_input("Diastolic BP *", min_value=30, max_value=180, value=80)
    c4, c5, c6 = st.columns(3)
    hba1c = c4.number_input("HbA1c *", min_value=2.0, max_value=20.0, value=6.0, step=0.1)
    fasting_glucose = c5.number_input("Fasting Glucose *", min_value=30.0, max_value=600.0, value=100.0, step=1.0)
    cholesterol = c6.number_input("Cholesterol *", min_value=50.0, max_value=500.0, value=180.0, step=1.0)
    c7, c8 = st.columns(2)
    kidney_function = c7.selectbox("Kidney Function *", sorted(df["kidney_function"].dropna().unique().tolist()))
    liver_function = c8.selectbox("Liver Function *", sorted(df["liver_function"].dropna().unique().tolist()))
    c9, c10, c11 = st.columns(3)
    smoking_status = c9.selectbox("Smoking Status *", sorted(df["smoking_status"].dropna().unique().tolist()))
    alcohol_use = c10.selectbox("Alcohol Use *", sorted(df["alcohol_use"].dropna().unique().tolist()))
    pregnancy_status = c11.selectbox("Pregnancy Status *", sorted(df["pregnancy_status"].dropna().unique().tolist()))
    c12, c13 = st.columns(2)
    allergies = c12.selectbox("Allergies *", sorted(df["allergies"].dropna().unique().tolist()))
    recent_surgery = c13.selectbox("Recent Surgery *", sorted(df["recent_surgery"].dropna().unique().tolist()))
    other_serious_condition = st.selectbox("Other Serious Condition *", sorted(df["other_serious_condition"].dropna().unique().tolist()))

    st.subheader("Recruitment Information")
    r1, r2 = st.columns(2)
    distance_from_trial_site_km = r1.number_input("Distance from Trial Site (km) *", min_value=0.0, max_value=5000.0, value=10.0, step=1.0)
    availability = r2.selectbox("Availability *", sorted(df["availability"].dropna().unique().tolist()))
    r3, r4, r5 = st.columns(3)
    contact_preference = r3.selectbox("Contact Preference *", sorted(df["contact_preference"].dropna().unique().tolist()))
    consent_to_contact = r4.selectbox("Consent to Contact *", sorted(df["consent_to_contact"].dropna().unique().tolist()))
    interest = r5.selectbox("Interest *", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("💾 Add Patient", type="primary")

if submitted:
    new_patient = {
        "patient_id": patient_id.strip(), "age": age, "gender": gender, "location": location,
        "primary_disease": primary_disease, "disease_duration_years": disease_duration_years,
        "disease_severity": disease_severity, "comorbidities": comorbidities,
        "previous_treatment": previous_treatment, "bmi": bmi, "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp, "hba1c": hba1c, "fasting_glucose": fasting_glucose,
        "kidney_function": kidney_function, "liver_function": liver_function, "cholesterol": cholesterol,
        "smoking_status": smoking_status, "alcohol_use": alcohol_use, "pregnancy_status": pregnancy_status,
        "allergies": allergies, "recent_surgery": recent_surgery,
        "other_serious_condition": other_serious_condition,
        "distance_from_trial_site_km": distance_from_trial_site_km, "availability": availability,
        "contact_preference": contact_preference, "consent_to_contact": consent_to_contact, "interest": interest,
    }

    updated_df, ok, errors = be.add_patient_record(df, new_patient)
    if ok:
        st.session_state.dataset = updated_df
        # Reset downstream eligibility/ranking results since the population changed.
        st.session_state.eligible_df = None
        st.session_state.ineligible_df = None
        st.session_state.explanations = None
        st_state.reset_downstream_after_eligibility()

        if st.session_state.dataset_path:
            saved, save_msg = be.save_dataset(updated_df, st.session_state.dataset_path)
            if saved:
                st.success(f"Patient '{patient_id}' added and saved to '{st.session_state.dataset_path}'. "
                           f"Total patients: {len(updated_df)}.")
            else:
                st.warning(f"Patient '{patient_id}' added to the in-memory dataset, but could not be saved to disk: {save_msg}")
        else:
            st.success(f"Patient '{patient_id}' added to the active (uploaded) dataset. Total patients: {len(updated_df)}. "
                       f"Note: this dataset was uploaded, not loaded from disk, so download an updated copy if you need to keep it.")
        st.info("Re-run Eligibility/Ranking on the corresponding pages to include this new patient.")
    else:
        st.error("Could not add patient. Please fix the following:")
        for e in errors:
            st.markdown(f"- {e}")
