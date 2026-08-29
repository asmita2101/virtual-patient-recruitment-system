"""
state.py
------------------------------------------------------------
Shared session-state initialization, global styling, and small
UI helpers used by every page of the multi-page Streamlit app.

This module exists so that all pages under app_pages/ behave like
ONE connected application (shared data, shared trial, shared
ranking results) instead of disconnected mini-tools, and so that
common setup code isn't duplicated across every page file.
------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

import backend as be

DISCLAIMER = (
    "Prototype for academic/research/demo purposes using synthetic patient data. "
    "This system is not a substitute for clinical judgment, formal clinical-trial "
    "screening, or medical advice."
)

DEFAULTS: Dict[str, Any] = {
    "dataset": None,
    "dataset_path": None,          # path the active dataset was loaded from (for persistence)
    "trial_name": "",
    "trial_description": "",
    "nl_criteria": "",
    "generated_rules": None,
    "rules_confirmed": False,
    "confirmed_rules": None,
    "eligible_df": None,
    "ineligible_df": None,
    "explanations": None,
    "selected_factors": [],
    "ahp_comparisons": {},
    "ahp_result": None,
    "scores_df": None,
    "raw_df": None,
    "ranked_df": None,
    "selected_patient_id": None,
    "whatif_result": None,
}


def init_session_state() -> None:
    """Initialize every shared session_state key exactly once per session."""
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state.dataset is None:
        try:
            df = be.load_dataset(be.DATASET_PATH)
            valid, _errors = be.validate_dataset_schema(df)
            if valid:
                st.session_state.dataset = df
                st.session_state.dataset_path = be.DATASET_PATH
        except Exception:
            pass


def inject_global_css() -> None:
    st.markdown("""
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 14px 18px;
    }
    .badge-high {background-color:#dcfce7;color:#166534;padding:3px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;}
    .badge-medium {background-color:#fef9c3;color:#854d0e;padding:3px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;}
    .badge-low {background-color:#fee2e2;color:#991b1b;padding:3px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;}
    .badge-eligible {background-color:#dcfce7;color:#166534;padding:3px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;}
    .badge-ineligible {background-color:#fee2e2;color:#991b1b;padding:3px 10px;border-radius:12px;font-weight:600;font-size:0.85rem;}
    </style>
    """, unsafe_allow_html=True)


def priority_badge(priority: str) -> str:
    cls = {"High Priority": "badge-high", "Medium Priority": "badge-medium", "Low Priority": "badge-low"}.get(priority, "badge-low")
    return f'<span class="{cls}">{priority}</span>'


def eligibility_badge(eligible: bool) -> str:
    cls = "badge-eligible" if eligible else "badge-ineligible"
    text = "✅ Eligible" if eligible else "❌ Not Eligible"
    return f'<span class="{cls}">{text}</span>'


def reset_downstream_after_eligibility() -> None:
    st.session_state.selected_factors = []
    st.session_state.ahp_comparisons = {}
    st.session_state.ahp_result = None
    st.session_state.scores_df = None
    st.session_state.raw_df = None
    st.session_state.ranked_df = None
    st.session_state.whatif_result = None


def reset_downstream_after_ahp() -> None:
    st.session_state.scores_df = None
    st.session_state.raw_df = None
    st.session_state.ranked_df = None
    st.session_state.whatif_result = None


def require_dataset() -> bool:
    if st.session_state.dataset is None:
        st.error(f"No dataset loaded. Please upload a CSV from the sidebar, or place '{be.DATASET_PATH}' next to frontend.py.")
        return False
    return True


def require_eligibility() -> bool:
    if st.session_state.eligible_df is None or len(st.session_state.eligible_df) == 0:
        st.info("No eligible patients yet. Complete the **Trial Criteria** and **Eligibility** pages first.")
        return False
    return True


def require_ranking() -> bool:
    if st.session_state.ranked_df is None:
        st.info("No ranking computed yet. Complete the **Patient Ranking** page first.")
        return False
    return True


def render_sidebar() -> None:
    """Global sidebar elements shown above the page-navigation list on every page:
    dataset upload/status, Gemini key status, and the disclaimer footer."""
    import llm_integration as llm

    st.sidebar.title("🏥 Virtual Patient Recruitment")
    st.sidebar.caption("AI-Powered Clinical Trial Recruitment Decision Support")
    st.sidebar.divider()

    st.sidebar.subheader("Dataset")
    uploaded = st.sidebar.file_uploader("Upload patient CSV", type=["csv"], key="global_dataset_uploader")
    if uploaded is not None:
        try:
            df = be.load_dataset(uploaded)
            valid, errors = be.validate_dataset_schema(df)
            if valid:
                st.session_state.dataset = df
                st.session_state.dataset_path = None  # uploaded in-memory file has no reliable disk path
                st.sidebar.success(f"Loaded {len(df)} patients.")
            else:
                st.sidebar.error("Uploaded file is missing required columns:\n" + "\n".join(errors))
        except Exception as exc:
            st.sidebar.error(f"Could not read file: {exc}")

    if st.session_state.dataset is not None:
        st.sidebar.info(f"Active dataset: {len(st.session_state.dataset)} patients")
    else:
        st.sidebar.warning(f"No dataset loaded. Upload a CSV or place '{be.DATASET_PATH}' next to frontend.py.")

    st.sidebar.divider()
    if not llm.is_configured():
        st.sidebar.warning("Gemini API key is not configured (set GEMINI_API_KEY).")
    else:
        st.sidebar.success("Gemini API key configured.")

    st.sidebar.divider()
    st.sidebar.caption(DISCLAIMER)
