"""app_pages/ranking.py - Trial-specific ranking: factor selection -> Adaptive AHP -> Weighted Sum -> final ranking."""

import itertools

import pandas as pd
import plotly.express as px
import streamlit as st

import backend as be
import state as st_state

st_state.init_session_state()

st.title("🏆 Patient Ranking")

if not st_state.require_dataset() or not st_state.require_eligibility():
    st.stop()

eligible_df = st.session_state.eligible_df
inclusion_rules = st.session_state.confirmed_rules["inclusion"]

tab1, tab2, tab3 = st.tabs(["1️⃣ Ranking Factors", "2️⃣ Adaptive AHP Weights", "3️⃣ Final Ranking"])

# ------------------------------------------------------------
# TAB 1: Trial-specific ranking factor selection
# ------------------------------------------------------------
with tab1:
    st.write("Select the ranking factors that matter for **this specific trial**. "
             "Only eligible patients are ranked — ranking never overrides hard eligibility.")

    factors = be.get_available_ranking_factors(st.session_state.dataset)
    options = {f["key"]: f"{f['label']} — {f['description']}" for f in factors}
    default_selection = st.session_state.selected_factors or ["clinical_match", "interest", "availability", "distance_from_trial_site_km"]

    selected = st.multiselect(
        "Ranking factors for this trial", options=list(options.keys()), default=default_selection,
        format_func=lambda k: options[k],
    )

    if st.button("Save Ranking Factors", type="primary"):
        if len(selected) < 2:
            st.warning("Select at least 2 ranking factors so AHP pairwise comparison can be performed.")
        else:
            st.session_state.selected_factors = selected
            st.session_state.ahp_comparisons = {}
            st_state.reset_downstream_after_ahp()
            st.success(f"Saved {len(selected)} ranking factors. Go to **Adaptive AHP Weights** next.")

    if st.session_state.selected_factors:
        st.info("Currently selected: " + ", ".join(be.RANKING_FACTOR_DEFINITIONS[f]["label"] for f in st.session_state.selected_factors))

# ------------------------------------------------------------
# TAB 2: Adaptive AHP
# ------------------------------------------------------------
with tab2:
    factors = st.session_state.selected_factors
    if not factors or len(factors) < 2:
        st.info("Select at least 2 ranking factors in the **Ranking Factors** tab first.")
    else:
        labels = {f: be.RANKING_FACTOR_DEFINITIONS[f]["label"] for f in factors}
        st.caption("For each pair, indicate how much more important the first factor is than the second "
                   "(Saaty scale 1-9). Reciprocal values are generated automatically.")

        saaty_options = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        saaty_labels = {
            1: "1 — Equal importance", 2: "2", 3: "3 — Moderately more important", 4: "4",
            5: "5 — Strongly more important", 6: "6", 7: "7 — Very strongly more important",
            8: "8", 9: "9 — Extremely more important",
        }

        pairs = list(itertools.combinations(factors, 2))
        comparisons = {}
        for fi, fj in pairs:
            default_val = st.session_state.ahp_comparisons.get((fi, fj), 1)
            val = st.select_slider(
                f"{labels[fi]}  vs.  {labels[fj]}", options=saaty_options,
                value=default_val if default_val in saaty_options else 1,
                format_func=lambda v: saaty_labels[v], key=f"ahp_{fi}_{fj}",
            )
            comparisons[(fi, fj)] = val

        if st.button("Calculate AHP Weights", type="primary"):
            st.session_state.ahp_comparisons = comparisons
            matrix = be.build_ahp_matrix(factors, comparisons)
            result = be.calculate_ahp_weights(matrix)
            result["matrix"] = matrix
            st.session_state.ahp_result = result
            st_state.reset_downstream_after_ahp()
            st.success("AHP weights calculated.")

        if st.session_state.ahp_result:
            result = st.session_state.ahp_result
            st.divider()

            weights_df = pd.DataFrame({"Factor": [labels[f] for f in factors], "Weight": result["weights"]}).sort_values("Weight", ascending=False)
            weights_df["Weight (%)"] = (weights_df["Weight"] * 100).round(1)

            colw1, colw2 = st.columns([1, 1])
            with colw1:
                st.subheader("AHP Weights")
                st.dataframe(weights_df[["Factor", "Weight (%)"]], width='stretch', hide_index=True)
            with colw2:
                fig = px.bar(weights_df, x="Factor", y="Weight (%)", title="Trial-Specific AHP Weights", color="Factor")
                st.plotly_chart(fig, width='stretch')

            st.subheader("Consistency Check")
            c1, c2, c3 = st.columns(3)
            c1.metric("Principal Eigenvalue (λmax)", round(result["lambda_max"], 4))
            c2.metric("Consistency Index (CI)", round(result["CI"], 4))
            c3.metric("Consistency Ratio (CR)", round(result["CR"], 4))

            if result["consistent"]:
                st.success("✅ Consistent (CR ≤ 0.10).")
                if st.button("Proceed to Weighted Sum & Final Ranking", type="primary"):
                    scores_df, raw_df = be.compute_all_factor_scores(eligible_df, inclusion_rules, factors)
                    weighted = be.calculate_weighted_scores(scores_df, factors, result["weights"])
                    ranked = be.rank_patients(weighted)
                    st.session_state.scores_df = scores_df
                    st.session_state.raw_df = raw_df
                    st.session_state.ranked_df = ranked
                    st.success("Ranking computed! Go to the **Final Ranking** tab.")
            else:
                st.error("⚠️ Review Preferences — pairwise comparisons may be inconsistent (CR > 0.10). Adjust the sliders above.")

# ------------------------------------------------------------
# TAB 3: Final ranking table
# ------------------------------------------------------------
with tab3:
    if not st_state.require_ranking():
        pass
    else:
        ranked = st.session_state.ranked_df
        factors = st.session_state.selected_factors

        colf1, colf2, colf3 = st.columns(3)
        priority_filter = colf1.multiselect("Filter by priority", ["High Priority", "Medium Priority", "Low Priority"])
        search_id = colf2.text_input("Search Patient ID", key="ranking_search")
        sort_order = colf3.selectbox("Sort order", ["Rank (best first)", "Rank (worst first)"])

        display = ranked.copy()
        if priority_filter:
            display = display[display["priority"].isin(priority_filter)]
        if search_id:
            display = display[display["patient_id"].str.contains(search_id, case=False, na=False)]
        if sort_order == "Rank (worst first)":
            display = display.sort_values("rank", ascending=False)

        show_cols = ["rank", "patient_id", "final_score", "priority"] + factors
        show_cols = [c for c in show_cols if c in display.columns]
        table = display[show_cols].copy()
        for f in factors:
            table[f] = table[f].round(1)
        table = table.rename(columns={f: be.RANKING_FACTOR_DEFINITIONS[f]["label"] for f in factors})
        table = table.rename(columns={"rank": "Rank", "patient_id": "Patient ID", "final_score": "Final Score", "priority": "Priority"})

        st.dataframe(table, width='stretch', hide_index=True)
        st.download_button(
            "⬇️ Download Ranked Patients (CSV)", table.to_csv(index=False).encode("utf-8"),
            file_name="ranked_patients.csv", mime="text/csv",
        )

        st.divider()
        selected_pid = st.selectbox("Explain a patient's ranking:", display["patient_id"].tolist())
        if st.button("🔍 Explain Selected Patient"):
            st.session_state.selected_patient_id = selected_pid
            st.info("Go to the **🔍 Explainability** page to see the full factor-level breakdown.")
