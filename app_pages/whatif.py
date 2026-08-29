"""app_pages/whatif.py - What-If Analysis: real recalculation of ranking under simulated changes."""

import numpy as np
import pandas as pd
import streamlit as st

import backend as be
import state as st_state

st_state.init_session_state()

st.title("🔄 What-If Analysis")
st.caption("Simulate a changed factor value or changed AHP priorities. The original dataset and "
           "confirmed ranking are never modified — this is a live recalculation, not an animation.")

if not st_state.require_dataset() or not st_state.require_ranking():
    st.stop()

ranked = st.session_state.ranked_df
scores_df = st.session_state.scores_df
raw_df = st.session_state.raw_df
factors = st.session_state.selected_factors
weights = st.session_state.ahp_result["weights"]

tab1, tab2 = st.tabs(["Change a Patient's Factor Value", "Change AHP Priorities"])

# ------------------------------------------------------------
with tab1:
    pid = st.selectbox("Select patient", ranked["patient_id"].tolist(), key="whatif_pid")
    factor_to_change = st.selectbox(
        "Select factor to change", factors, format_func=lambda f: be.RANKING_FACTOR_DEFINITIONS[f]["label"],
    )
    current_score = float(scores_df[scores_df["patient_id"] == pid][factor_to_change].iloc[0])
    st.write(f"Current normalized score for this factor: **{current_score:.1f} / 100**")

    if factor_to_change == "interest":
        new_level = st.selectbox("New interest level", ["Low", "Medium", "High"])
        new_score = be.calculate_interest_score(new_level)
    elif factor_to_change == "availability":
        new_level = st.selectbox("New availability level", ["Low", "Medium", "High"])
        new_score = be.calculate_availability_score(new_level)
    else:
        new_score = st.slider("New normalized score (0-100)", 0.0, 100.0, current_score)

    if st.button("Run What-If Simulation", type="primary", key="run_whatif_factor"):
        result = be.perform_what_if_analysis(
            scores_df, raw_df, factors, weights,
            patient_id=pid, factor_overrides={factor_to_change: new_score},
        )
        st.session_state.whatif_result = result

# ------------------------------------------------------------
with tab2:
    st.write("Adjust AHP weights directly for simulation (values are re-normalized to sum to 1).")
    new_weight_inputs = {}
    cols = st.columns(len(factors))
    for i, f in enumerate(factors):
        default_w = float(weights[factors.index(f)]) * 100
        new_weight_inputs[f] = cols[i].slider(
            be.RANKING_FACTOR_DEFINITIONS[f]["label"], 0, 100, int(round(default_w)), key=f"whatif_w_{f}",
        )
    total_w = sum(new_weight_inputs.values()) or 1
    normalized_weights = np.array([new_weight_inputs[f] / total_w for f in factors])
    st.caption("Normalized weights: " + ", ".join(f"{be.RANKING_FACTOR_DEFINITIONS[f]['label']}={w*100:.1f}%" for f, w in zip(factors, normalized_weights)))

    whatif_pid_2 = st.selectbox("Track a specific patient (optional)", ["(None)"] + ranked["patient_id"].tolist(), key="whatif_pid_weights")

    if st.button("Run What-If Simulation", type="primary", key="run_whatif_weights"):
        result = be.perform_what_if_analysis(
            scores_df, raw_df, factors, weights,
            patient_id=None if whatif_pid_2 == "(None)" else whatif_pid_2,
            new_weights=normalized_weights,
        )
        st.session_state.whatif_result = result

# ------------------------------------------------------------
if st.session_state.whatif_result:
    result = st.session_state.whatif_result
    st.divider()
    st.subheader("Results")

    if "patient_id" in result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Score", f"{result['original_score']:.1f}")
        c2.metric("New Score", f"{result['new_score']:.1f}", delta=f"{result['score_difference']:+.1f}")
        c3.metric("Rank Change", f"#{result['original_rank']} → #{result['new_rank']}",
                  delta=f"{result['rank_difference']:+d} positions")

    st.markdown("**BEFORE (original top 10)**")
    before_display = result["original_ranking"][["rank", "patient_id", "final_score", "priority"]].head(10).rename(
        columns={"rank": "Rank", "patient_id": "Patient ID", "final_score": "Final Score", "priority": "Priority"})
    st.dataframe(before_display, width='stretch', hide_index=True)

    st.markdown("**AFTER (new top 10)**")
    after_display = result["new_ranking"][["rank", "patient_id", "final_score", "priority"]].head(10).rename(
        columns={"rank": "Rank", "patient_id": "Patient ID", "final_score": "Final Score", "priority": "Priority"})
    st.dataframe(after_display, width='stretch', hide_index=True)

    # Highlight rank changes across the two full rankings.
    merged = result["original_ranking"][["patient_id", "rank"]].rename(columns={"rank": "original_rank"}).merge(
        result["new_ranking"][["patient_id", "rank"]].rename(columns={"rank": "new_rank"}), on="patient_id")
    merged["rank_change"] = merged["original_rank"] - merged["new_rank"]
    movers = merged[merged["rank_change"] != 0].sort_values("rank_change", ascending=False)
    if len(movers) > 0:
        st.markdown("**Biggest Rank Changes**")
        movers_display = movers.head(10).rename(columns={
            "patient_id": "Patient ID", "original_rank": "Original Rank", "new_rank": "New Rank", "rank_change": "Positions Moved",
        })
        st.dataframe(movers_display, width='stretch', hide_index=True)
    else:
        st.info("No rank changes under this simulation.")

    st.download_button(
        "⬇️ Download What-If Ranking (CSV)", result["new_ranking"].to_csv(index=False).encode("utf-8"),
        file_name="whatif_ranking.csv", mime="text/csv",
    )
