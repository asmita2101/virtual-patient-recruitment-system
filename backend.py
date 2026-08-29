"""
backend.py
------------------------------------------------------------
Virtual Patient Recruitment System for Clinical Trials
Backend: dataset handling, dynamic eligibility engine, AHP,
Weighted Sum Model, explainability and What-If analysis.

NOTE: This module contains NO machine-learning models and NO
recruitment feedback loop. All logic is deterministic and
rule-based, exactly as specified for this academic prototype.
------------------------------------------------------------
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# CONSTANTS
# ============================================================

DATASET_PATH = "patients.csv"

REQUIRED_COLUMNS = [
    "patient_id", "age", "gender", "location", "primary_disease",
    "disease_duration_years", "disease_severity", "comorbidities",
    "previous_treatment", "bmi", "systolic_bp", "diastolic_bp", "hba1c",
    "fasting_glucose", "kidney_function", "liver_function", "cholesterol",
    "smoking_status", "alcohol_use", "pregnancy_status", "allergies",
    "recent_surgery", "other_serious_condition",
    "distance_from_trial_site_km", "availability", "contact_preference",
    "consent_to_contact", "interest",
]

CATEGORICAL_COLUMNS = [
    "gender", "location", "primary_disease", "disease_severity",
    "comorbidities", "previous_treatment", "kidney_function",
    "liver_function", "smoking_status", "alcohol_use", "pregnancy_status",
    "allergies", "recent_surgery", "other_serious_condition", "availability",
    "contact_preference", "consent_to_contact", "interest",
]

NUMERIC_COLUMNS = [
    "age", "disease_duration_years", "bmi", "systolic_bp", "diastolic_bp",
    "hba1c", "fasting_glucose", "cholesterol", "distance_from_trial_site_km",
]

SUPPORTED_OPERATORS = [
    "equals", "not_equals", ">", "<", ">=", "<=", "between",
    "contains", "not_contains", "in", "not_in",
]

# Transparent ordinal mapping used for the Low/Medium/High style fields.
LEVEL_SCORE_MAP = {"low": 0.0, "medium": 50.0, "high": 100.0}

# Random Index (RI) values used for AHP Consistency Ratio calculation.
RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}

# Core clinical fields used to build the composite "Clinical Match" factor.
CORE_CLINICAL_FEATURES = [
    "age", "hba1c", "bmi", "systolic_bp", "diastolic_bp",
    "cholesterol", "fasting_glucose", "disease_duration_years",
]

# Definition of every ranking factor the recruiter may choose from.
RANKING_FACTOR_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "clinical_match": {
        "label": "Clinical Match",
        "description": (
            "Composite score showing how closely the patient's clinical "
            "values (age, HbA1c, BMI, blood pressure, cholesterol, "
            "fasting glucose, disease duration) align with the trial's "
            "stated eligibility criteria."
        ),
    },
    "interest": {
        "label": "Patient Interest",
        "description": "Patient's recorded interest level (Low / Medium / High).",
    },
    "availability": {
        "label": "Availability",
        "description": "Patient's recorded availability level (Low / Medium / High).",
    },
    "distance_from_trial_site_km": {
        "label": "Distance to Trial Site",
        "description": "Distance from the patient's location to the trial site (closer = higher score).",
    },
    "age_match": {
        "label": "Age Match",
        "description": "How closely the patient's age matches the trial's age criterion (if any).",
    },
    "disease_severity_match": {
        "label": "Disease Severity Match",
        "description": "Match between patient's disease severity and the trial's severity criterion (if any).",
    },
    "disease_duration_match": {
        "label": "Disease Duration Match",
        "description": "Match between patient's disease duration and the trial's duration criterion (if any).",
    },
    "previous_treatment_match": {
        "label": "Previous Treatment Match",
        "description": "Match between patient's previous treatment and the trial's treatment criterion (if any).",
    },
    "bmi_match": {
        "label": "BMI Match",
        "description": "Match between patient's BMI and the trial's BMI criterion (if any).",
    },
    "kidney_function_match": {
        "label": "Kidney Function Match",
        "description": "Match between patient's kidney function and the trial's kidney-function criterion (if any).",
    },
    "liver_function_match": {
        "label": "Liver Function Match",
        "description": "Match between patient's liver function and the trial's liver-function criterion (if any).",
    },
    "hba1c_match": {
        "label": "HbA1c Match",
        "description": "Match between patient's HbA1c and the trial's HbA1c criterion (if any).",
    },
    "blood_pressure_match": {
        "label": "Blood Pressure Match",
        "description": "Match between patient's systolic/diastolic BP and the trial's BP criteria (if any).",
    },
    "cholesterol_match": {
        "label": "Cholesterol Match",
        "description": "Match between patient's cholesterol and the trial's cholesterol criterion (if any).",
    },
}

# Maps each "_match" ranking factor to the dataset feature(s) it is derived from.
FACTOR_TO_FEATURE = {
    "age_match": "age",
    "disease_severity_match": "disease_severity",
    "disease_duration_match": "disease_duration_years",
    "previous_treatment_match": "previous_treatment",
    "bmi_match": "bmi",
    "kidney_function_match": "kidney_function",
    "liver_function_match": "liver_function",
    "hba1c_match": "hba1c",
    "cholesterol_match": "cholesterol",
}


# ============================================================
# 1. DATASET LOADING & INSPECTION
# ============================================================

def load_dataset(path_or_buffer) -> pd.DataFrame:
    """Load the patient dataset from a path or an uploaded file-like object."""
    df = pd.read_csv(path_or_buffer)
    df.columns = [c.strip() for c in df.columns]
    if "patient_id" in df.columns:
        df["patient_id"] = df["patient_id"].astype(str)
    return df


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check that the dataset contains all columns the system expects."""
    errors = []
    if df is None or df.empty:
        errors.append("The dataset is empty or could not be loaded.")
        return False, errors
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    return len(errors) == 0, errors


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Produce a structured summary of the dataset for display / rule validation."""
    summary: Dict[str, Any] = {"n_rows": len(df), "n_columns": len(df.columns), "columns": {}}
    for col in df.columns:
        col_info: Dict[str, Any] = {
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
        }
        if col in NUMERIC_COLUMNS:
            col_info["min"] = float(df[col].min())
            col_info["max"] = float(df[col].max())
            col_info["mean"] = float(df[col].mean())
        elif col in CATEGORICAL_COLUMNS:
            col_info["unique_values"] = sorted(df[col].dropna().unique().tolist())
        summary["columns"][col] = col_info
    return summary


def get_dataset_features(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "patient_id"]


# ============================================================
# 2A. PATIENT RECORD MANAGEMENT (ADD / UPDATE / DELETE)
# ============================================================

def save_dataset(df: pd.DataFrame, path: str) -> Tuple[bool, str]:
    """Persist the current dataset back to disk (CSV). Returns (success, message)."""
    try:
        df.to_csv(path, index=False)
        return True, f"Dataset saved to '{path}'."
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not save dataset to '{path}': {exc}"


def validate_new_patient(patient: Dict[str, Any], df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate a new (or edited) patient record before it is added to the dataset."""
    errors: List[str] = []

    patient_id = str(patient.get("patient_id", "")).strip()
    if not patient_id:
        errors.append("Patient ID is required.")
    elif patient_id in set(df["patient_id"].astype(str)):
        errors.append(f"Patient ID '{patient_id}' already exists. Please choose a unique ID.")

    for col in NUMERIC_COLUMNS:
        if col not in patient:
            continue
        val = patient.get(col)
        if val is None or val == "":
            errors.append(f"'{col}' is required and must be numeric.")
            continue
        try:
            float(val)
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be a numeric value.")

    for col in CATEGORICAL_COLUMNS:
        if col not in patient:
            continue
        val = patient.get(col)
        if val is None or str(val).strip() == "":
            errors.append(f"'{col}' is required.")

    # Simple, transparent range sanity-checks (not clinical validation - just data-entry safety).
    range_checks = {
        "age": (0, 120), "bmi": (5, 100), "systolic_bp": (50, 260), "diastolic_bp": (30, 180),
        "hba1c": (2, 20), "fasting_glucose": (30, 600), "cholesterol": (50, 500),
        "distance_from_trial_site_km": (0, 5000), "disease_duration_years": (0, 100),
    }
    for col, (lo, hi) in range_checks.items():
        if col in patient and patient.get(col) not in (None, ""):
            try:
                val_f = float(patient[col])
                if not (lo <= val_f <= hi):
                    errors.append(f"'{col}' value {val_f} looks out of a plausible range ({lo}-{hi}). Please double-check.")
            except (TypeError, ValueError):
                pass

    return len(errors) == 0, errors


def add_patient_record(df: pd.DataFrame, patient: Dict[str, Any]) -> Tuple[pd.DataFrame, bool, List[str]]:
    """
    Validate and append a new patient record to the in-memory dataset.
    Returns (updated_df, success, errors). The original df is not mutated in place.
    """
    is_valid, errors = validate_new_patient(patient, df)
    if not is_valid:
        return df, False, errors

    ordered_patient = {col: patient.get(col) for col in df.columns}
    new_row_df = pd.DataFrame([ordered_patient])
    updated_df = pd.concat([df, new_row_df], ignore_index=True)
    updated_df["patient_id"] = updated_df["patient_id"].astype(str)
    return updated_df, True, []


def update_patient_record(df: pd.DataFrame, patient_id: str, updates: Dict[str, Any]) -> Tuple[pd.DataFrame, bool, str]:
    """Update fields of an existing patient. Returns (updated_df, success, message)."""
    if patient_id not in set(df["patient_id"].astype(str)):
        return df, False, f"Patient ID '{patient_id}' was not found."
    updated_df = df.copy()
    mask = updated_df["patient_id"] == patient_id
    for key, val in updates.items():
        if key in updated_df.columns and key != "patient_id":
            updated_df.loc[mask, key] = val
    return updated_df, True, f"Patient '{patient_id}' updated."


def delete_patient_record(df: pd.DataFrame, patient_id: str) -> Tuple[pd.DataFrame, bool, str]:
    """Remove a patient from the dataset. Returns (updated_df, success, message)."""
    if patient_id not in set(df["patient_id"].astype(str)):
        return df, False, f"Patient ID '{patient_id}' was not found."
    updated_df = df[df["patient_id"] != patient_id].reset_index(drop=True)
    return updated_df, True, f"Patient '{patient_id}' deleted."


# ============================================================
# 2. RULE VALIDATION
# ============================================================

def _closest_column_matches(feature: str, columns: List[str], n: int = 3) -> List[str]:
    """Very small helper for suggesting likely-intended column names."""
    feature_l = feature.lower().replace(" ", "_")
    scored = []
    for col in columns:
        col_l = col.lower()
        score = 0
        if feature_l == col_l:
            score = 100
        elif feature_l in col_l or col_l in feature_l:
            score = 50
        else:
            common = set(feature_l.split("_")) & set(col_l.split("_"))
            score = 10 * len(common)
        if score > 0:
            scored.append((score, col))
    scored.sort(reverse=True)
    return [c for _, c in scored[:n]]


def validate_rules(rules: Dict[str, Any], df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a structured eligibility-rule dictionary (as produced by Gemini)
    against the actual dataset schema.

    Returns: (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []
    columns = list(df.columns)

    if not isinstance(rules, dict):
        return False, ["The generated rules are not a valid JSON object."], []

    if "inclusion" not in rules or "exclusion" not in rules:
        errors.append("Rules JSON must contain both 'inclusion' and 'exclusion' lists.")
        return False, errors, warnings

    for section in ("inclusion", "exclusion"):
        section_rules = rules.get(section, [])
        if not isinstance(section_rules, list):
            errors.append(f"'{section}' must be a list of rule objects.")
            continue

        for i, rule in enumerate(section_rules):
            label = f"{section}[{i}]"
            if not isinstance(rule, dict):
                errors.append(f"{label}: rule must be an object.")
                continue

            feature = rule.get("feature")
            operator = rule.get("operator")
            value = rule.get("value", None)

            if not feature:
                errors.append(f"{label}: missing 'feature'.")
                continue
            if feature not in columns:
                suggestions = _closest_column_matches(feature, columns)
                sugg_txt = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                errors.append(f"{label}: unknown feature '{feature}'.{sugg_txt}")
                continue

            if not operator or operator not in SUPPORTED_OPERATORS:
                errors.append(
                    f"{label}: unsupported operator '{operator}' for feature '{feature}'. "
                    f"Supported operators: {', '.join(SUPPORTED_OPERATORS)}."
                )
                continue

            if value is None:
                errors.append(f"{label}: missing 'value' for feature '{feature}'.")
                continue

            # Datatype / range checks
            if feature in NUMERIC_COLUMNS:
                if operator == "between":
                    if not (isinstance(value, (list, tuple)) and len(value) == 2):
                        errors.append(f"{label}: 'between' requires a [min, max] list of numbers.")
                    else:
                        try:
                            float(value[0]); float(value[1])
                        except (TypeError, ValueError):
                            errors.append(f"{label}: 'between' values must be numeric.")
                elif operator in (">", "<", ">=", "<="):
                    try:
                        float(value)
                    except (TypeError, ValueError):
                        errors.append(f"{label}: value for '{feature}' with operator '{operator}' must be numeric.")
                elif operator in ("equals", "not_equals"):
                    try:
                        float(value)
                    except (TypeError, ValueError):
                        errors.append(f"{label}: value for numeric feature '{feature}' must be numeric.")
                elif operator in ("in", "not_in"):
                    if not isinstance(value, list):
                        errors.append(f"{label}: '{operator}' requires a list of numeric values.")
                else:
                    warnings.append(f"{label}: operator '{operator}' is unusual for numeric feature '{feature}'.")

            elif feature in CATEGORICAL_COLUMNS:
                known_values = set(str(v).lower() for v in df[feature].dropna().unique())
                if operator in ("equals", "not_equals", "contains", "not_contains"):
                    if isinstance(value, str) and known_values:
                        match_found = any(
                            value.lower() == kv or value.lower() in kv or kv in value.lower()
                            for kv in known_values
                        )
                        if not match_found:
                            warnings.append(
                                f"{label}: value '{value}' does not closely match any known value of "
                                f"'{feature}' ({sorted(df[feature].dropna().unique().tolist())})."
                            )
                elif operator in ("in", "not_in"):
                    if not isinstance(value, list):
                        errors.append(f"{label}: '{operator}' requires a list of values.")
                elif operator == "between":
                    errors.append(f"{label}: 'between' cannot be used on categorical feature '{feature}'.")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


# ============================================================
# 3. DYNAMIC ELIGIBILITY ENGINE
# ============================================================

def _values_equal(a: Any, b: Any) -> bool:
    """Case-insensitive, whitespace-tolerant equality with substring fallback
    for categorical text (helps generic terms like 'diabetes' match
    'Type 2 Diabetes')."""
    if a is None or (isinstance(a, float) and pd.isna(a)):
        return False
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        pass
    a_s, b_s = str(a).strip().lower(), str(b).strip().lower()
    if a_s == b_s:
        return True
    return a_s in b_s or b_s in a_s


def evaluate_rule(value: Any, rule: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Evaluate a single rule against a single patient value.
    Returns (passed, human_readable_requirement_string).
    """
    operator = rule["operator"]
    target = rule["value"]

    if value is None or (isinstance(value, float) and pd.isna(value)):
        # Missing data can never satisfy a criterion (see requirement #58).
        return False, _requirement_text(operator, target)

    req_text = _requirement_text(operator, target)

    if operator == "equals":
        return _values_equal(value, target), req_text
    if operator == "not_equals":
        return not _values_equal(value, target), req_text
    if operator == "contains":
        return str(target).strip().lower() in str(value).strip().lower(), req_text
    if operator == "not_contains":
        return str(target).strip().lower() not in str(value).strip().lower(), req_text
    if operator == "in":
        return any(_values_equal(value, t) for t in target), req_text
    if operator == "not_in":
        return not any(_values_equal(value, t) for t in target), req_text

    # Numeric operators
    try:
        val_f = float(value)
    except (TypeError, ValueError):
        return False, req_text

    if operator == "between":
        lo, hi = float(target[0]), float(target[1])
        return lo <= val_f <= hi, req_text
    if operator == ">":
        return val_f > float(target), req_text
    if operator == "<":
        return val_f < float(target), req_text
    if operator == ">=":
        return val_f >= float(target), req_text
    if operator == "<=":
        return val_f <= float(target), req_text

    return False, req_text


def _requirement_text(operator: str, target: Any) -> str:
    if operator == "between":
        return f"between {target[0]} and {target[1]}"
    mapping = {
        "equals": f"= {target}", "not_equals": f"!= {target}",
        ">": f"> {target}", "<": f"< {target}",
        ">=": f">= {target}", "<=": f"<= {target}",
        "contains": f"contains '{target}'", "not_contains": f"does not contain '{target}'",
        "in": f"one of {target}", "not_in": f"none of {target}",
    }
    return mapping.get(operator, str(target))


def get_eligibility_explanation(
    row: pd.Series, inclusion: List[Dict[str, Any]], exclusion: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build a criterion-by-criterion explanation for a single patient."""
    criteria_results = []
    eligible = True

    for rule in inclusion:
        feature = rule["feature"]
        value = row.get(feature, None)
        passed, requirement = evaluate_rule(value, rule)
        criteria_results.append({
            "type": "Inclusion", "feature": feature, "patient_value": value,
            "requirement": requirement, "result": "PASS" if passed else "FAIL",
        })
        if not passed:
            eligible = False

    for rule in exclusion:
        feature = rule["feature"]
        value = row.get(feature, None)
        violated, requirement = evaluate_rule(value, rule)
        # For exclusion rules, "violated" means the patient MATCHES the excluded condition.
        passed = not violated
        criteria_results.append({
            "type": "Exclusion", "feature": feature, "patient_value": value,
            "requirement": f"must NOT be {requirement}", "result": "PASS" if passed else "FAIL",
        })
        if not passed:
            eligible = False

    return {
        "patient_id": row.get("patient_id"),
        "criteria": criteria_results,
        "eligible": eligible,
        "status": "ELIGIBLE" if eligible else "NOT ELIGIBLE",
    }


def apply_eligibility_rules(
    df: pd.DataFrame, inclusion: List[Dict[str, Any]], exclusion: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluate every patient against the confirmed rules.
    Returns (eligible_df, ineligible_df, explanations_by_patient_id).
    """
    explanations: Dict[str, Dict[str, Any]] = {}
    eligible_flags = []

    for _, row in df.iterrows():
        explanation = get_eligibility_explanation(row, inclusion, exclusion)
        explanations[str(row["patient_id"])] = explanation
        eligible_flags.append(explanation["eligible"])

    df = df.copy()
    df["_eligible"] = eligible_flags
    eligible_df = df[df["_eligible"]].drop(columns=["_eligible"]).reset_index(drop=True)
    ineligible_df = df[~df["_eligible"]].drop(columns=["_eligible"]).reset_index(drop=True)
    return eligible_df, ineligible_df, explanations


# ============================================================
# 4. RANKING FACTORS
# ============================================================

def get_available_ranking_factors(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Return the list of ranking factors supported by the current dataset."""
    factors = []
    for key, meta in RANKING_FACTOR_DEFINITIONS.items():
        factors.append({"key": key, "label": meta["label"], "description": meta["description"]})
    return factors


def calculate_interest_score(value: Any) -> float:
    """Low = 0, Medium = 50, High = 100 (transparent ordinal mapping)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 50.0
    return LEVEL_SCORE_MAP.get(str(value).strip().lower(), 50.0)


def calculate_availability_score(value: Any) -> float:
    """Low = 0, Medium = 50, High = 100 (transparent ordinal mapping)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 50.0
    return LEVEL_SCORE_MAP.get(str(value).strip().lower(), 50.0)


def normalize_factor(value: float, min_val: float, max_val: float, inverse: bool = False) -> float:
    """Generic min-max normalization to a 0-100 scale."""
    if pd.isna(value) or min_val == max_val:
        return 50.0
    score = (value - min_val) / (max_val - min_val) * 100.0
    if inverse:
        score = 100.0 - score
    return float(max(0.0, min(100.0, score)))


def _find_rule_for_feature(feature: str, inclusion_rules: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for rule in inclusion_rules:
        if rule.get("feature") == feature:
            return rule
    return None


def generic_match_score(
    value: Any, feature: str, inclusion_rules: List[Dict[str, Any]], eligible_df: pd.DataFrame
) -> float:
    """
    Trial-aware "match" scoring used for every *_match ranking factor.

    - If the trial defines an inclusion criterion for this feature, the score
      reflects how well the patient's value aligns with that criterion
      (closeness to the midpoint of a 'between' range, or scaled distance
      above/below a threshold).
    - If no inclusion criterion touches this feature, the factor is neutral
      (50 for every eligible patient) since the trial has stated no
      preference for it. This keeps the model fully transparent and avoids
      inventing an arbitrary preference direction.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 50.0

    rule = _find_rule_for_feature(feature, inclusion_rules)
    if rule is None:
        return 50.0

    operator = rule["operator"]
    target = rule["value"]

    if feature in NUMERIC_COLUMNS or feature == "disease_duration_years":
        try:
            val_f = float(value)
        except (TypeError, ValueError):
            return 50.0

        if operator == "between":
            lo, hi = float(target[0]), float(target[1])
            if hi == lo:
                return 100.0
            mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            return float(max(0.0, 100.0 * (1 - abs(val_f - mid) / half)))

        if operator in (">", ">="):
            threshold = float(target)
            col_max = float(eligible_df[feature].max())
            if col_max <= threshold:
                return 100.0
            return normalize_factor(val_f, threshold, col_max, inverse=False)

        if operator in ("<", "<="):
            threshold = float(target)
            col_min = float(eligible_df[feature].min())
            if threshold <= col_min:
                return 100.0
            return normalize_factor(val_f, col_min, threshold, inverse=True)

        if operator in ("equals", "in"):
            return 100.0
        return 50.0

    # Categorical feature
    if operator in ("equals", "in", "contains"):
        return 100.0
    if operator in ("not_equals", "not_in", "not_contains"):
        return 100.0
    return 50.0


def calculate_clinical_match(row: pd.Series, inclusion_rules: List[Dict[str, Any]], eligible_df: pd.DataFrame) -> float:
    """Composite Clinical Match score averaged over the core clinical features."""
    scores = [
        generic_match_score(row.get(f), f, inclusion_rules, eligible_df)
        for f in CORE_CLINICAL_FEATURES
    ]
    return float(np.mean(scores)) if scores else 50.0


def calculate_factor_score(
    row: pd.Series, factor_key: str, inclusion_rules: List[Dict[str, Any]], eligible_df: pd.DataFrame
) -> Tuple[float, Any]:
    """
    Compute the normalized (0-100) score AND the raw value for one ranking
    factor, for one patient.
    """
    if factor_key == "interest":
        raw = row.get("interest")
        return calculate_interest_score(raw), raw

    if factor_key == "availability":
        raw = row.get("availability")
        return calculate_availability_score(raw), raw

    if factor_key == "distance_from_trial_site_km":
        raw = row.get("distance_from_trial_site_km")
        score = normalize_factor(
            float(raw), float(eligible_df["distance_from_trial_site_km"].min()),
            float(eligible_df["distance_from_trial_site_km"].max()), inverse=True,
        )
        return score, raw

    if factor_key == "clinical_match":
        return calculate_clinical_match(row, inclusion_rules, eligible_df), None

    if factor_key == "blood_pressure_match":
        raw_sys, raw_dia = row.get("systolic_bp"), row.get("diastolic_bp")
        s1 = generic_match_score(raw_sys, "systolic_bp", inclusion_rules, eligible_df)
        s2 = generic_match_score(raw_dia, "diastolic_bp", inclusion_rules, eligible_df)
        return float(np.mean([s1, s2])), f"{raw_sys}/{raw_dia} mmHg"

    if factor_key in FACTOR_TO_FEATURE:
        feature = FACTOR_TO_FEATURE[factor_key]
        raw = row.get(feature)
        score = generic_match_score(raw, feature, inclusion_rules, eligible_df)
        return score, raw

    return 50.0, None


def compute_all_factor_scores(
    eligible_df: pd.DataFrame, inclusion_rules: List[Dict[str, Any]], selected_factors: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two DataFrames indexed by patient_id:
      - scores_df: normalized 0-100 score per selected factor
      - raw_df: the raw value behind each factor (for display / explanation)
    """
    score_rows, raw_rows = [], []
    for _, row in eligible_df.iterrows():
        score_row = {"patient_id": row["patient_id"]}
        raw_row = {"patient_id": row["patient_id"]}
        for factor in selected_factors:
            score, raw_val = calculate_factor_score(row, factor, inclusion_rules, eligible_df)
            score_row[factor] = score
            raw_row[factor] = raw_val
        score_rows.append(score_row)
        raw_rows.append(raw_row)
    return pd.DataFrame(score_rows), pd.DataFrame(raw_rows)


# ============================================================
# 5. AHP (ANALYTIC HIERARCHY PROCESS)
# ============================================================

def build_ahp_matrix(factors: List[str], comparisons: Dict[Tuple[str, str], float]) -> np.ndarray:
    """
    Build the pairwise comparison matrix from recruiter inputs.
    `comparisons` maps (factor_i, factor_j) -> Saaty value meaning
    "factor_i is X times as important as factor_j". Reciprocals are
    generated automatically.
    """
    n = len(factors)
    matrix = np.ones((n, n))
    for i, fi in enumerate(factors):
        for j, fj in enumerate(factors):
            if i == j:
                matrix[i, j] = 1.0
            elif (fi, fj) in comparisons:
                matrix[i, j] = comparisons[(fi, fj)]
                matrix[j, i] = 1.0 / comparisons[(fi, fj)]
            elif (fj, fi) in comparisons:
                matrix[j, i] = comparisons[(fj, fi)]
                matrix[i, j] = 1.0 / comparisons[(fj, fi)]
    return matrix


def calculate_ahp_weights(matrix: np.ndarray) -> Dict[str, Any]:
    """
    Compute AHP priority weights, principal eigenvalue, Consistency Index (CI)
    and Consistency Ratio (CR) using the standard normalize-and-average method.
    """
    n = matrix.shape[0]
    col_sums = matrix.sum(axis=0)
    normalized = matrix / col_sums
    weights = normalized.mean(axis=1)

    weighted_sum_vector = matrix @ weights
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(weighted_sum_vector, weights, out=np.zeros_like(weighted_sum_vector), where=weights != 0)
    lambda_max = float(np.mean(ratios))

    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0

    return {
        "weights": weights, "lambda_max": lambda_max,
        "CI": ci, "CR": cr,
        "consistent": cr <= 0.10,
    }


# ============================================================
# 6. WEIGHTED SUM MODEL & RANKING
# ============================================================

def calculate_weighted_scores(
    scores_df: pd.DataFrame, factors: List[str], weights: np.ndarray
) -> pd.DataFrame:
    """Apply the Weighted Sum Model to compute each patient's final score."""
    result = scores_df.copy()
    result["final_score"] = 0.0
    for factor, weight in zip(factors, weights):
        result["final_score"] += result[factor] * weight
    result["final_score"] = result["final_score"].round(2)
    return result


def _priority_category(score: float) -> str:
    if score >= 80:
        return "High Priority"
    if score >= 60:
        return "Medium Priority"
    return "Low Priority"


def rank_patients(weighted_scores_df: pd.DataFrame) -> pd.DataFrame:
    """Sort by final_score descending and assign rank + priority category."""
    ranked = weighted_scores_df.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked["priority"] = ranked["final_score"].apply(_priority_category)
    return ranked


# ============================================================
# 7. EXPLAINABILITY
# ============================================================

def generate_ranking_explanation(
    patient_id: str, ranked_df: pd.DataFrame, raw_df: pd.DataFrame,
    factors: List[str], weights: np.ndarray,
) -> Dict[str, Any]:
    """Produce the full factor-level breakdown + narrative for one patient."""
    row = ranked_df[ranked_df["patient_id"] == patient_id].iloc[0]
    raw_row = raw_df[raw_df["patient_id"] == patient_id].iloc[0]

    breakdown = []
    strong_factors, weak_factors = [], []
    for factor, weight in zip(factors, weights):
        norm_score = float(row[factor])
        contribution = round(norm_score * weight, 2)
        raw_value = raw_row[factor]
        label = RANKING_FACTOR_DEFINITIONS[factor]["label"]
        breakdown.append({
            "factor": label, "factor_key": factor, "raw_value": raw_value,
            "normalized_score": round(norm_score, 1), "ahp_weight": round(float(weight), 3),
            "contribution": contribution,
        })
        if norm_score >= 70:
            strong_factors.append(label)
        elif norm_score <= 30:
            weak_factors.append(label)

    final_score = float(row["final_score"])
    priority = row["priority"]
    rank = int(row["rank"])

    narrative_parts = [f"Patient {patient_id} received a final priority score of {final_score:.1f} "
                        f"({priority}, rank #{rank})."]
    if strong_factors:
        narrative_parts.append(
            f"This is driven mainly by strong scores in: {', '.join(strong_factors)}."
        )
    if weak_factors:
        narrative_parts.append(
            f"The score is reduced by weaker performance in: {', '.join(weak_factors)}."
        )
    if "interest" in factors:
        interest_raw = raw_row["interest"]
        if str(interest_raw).strip().lower() == "high":
            narrative_parts.append("Interest contributed strongly to this patient's score because the recorded interest level is High.")
        elif str(interest_raw).strip().lower() == "low":
            narrative_parts.append("Interest reduced the patient's priority score because the recorded interest level is Low.")
        else:
            narrative_parts.append("Interest had a moderate effect on this patient's score because the recorded interest level is Medium.")

    return {
        "patient_id": patient_id, "final_score": final_score, "priority": priority,
        "rank": rank, "breakdown": breakdown, "narrative": " ".join(narrative_parts),
    }


# ============================================================
# 8. WHAT-IF ANALYSIS
# ============================================================

def perform_what_if_analysis(
    scores_df: pd.DataFrame, raw_df: pd.DataFrame, factors: List[str], weights: np.ndarray,
    patient_id: Optional[str] = None, factor_overrides: Optional[Dict[str, float]] = None,
    new_weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Simulate the effect of changing either:
      (a) one patient's factor score(s) (factor_overrides: {factor_key: new_normalized_score}), and/or
      (b) the AHP weights (new_weights),
    WITHOUT mutating the original dataset or original scores.

    Returns original & new ranking info for comparison.
    """
    original_ranked = rank_patients(calculate_weighted_scores(scores_df, factors, weights))

    sim_scores_df = scores_df.copy()
    if patient_id is not None and factor_overrides:
        mask = sim_scores_df["patient_id"] == patient_id
        for factor, new_val in factor_overrides.items():
            sim_scores_df.loc[mask, factor] = float(new_val)

    sim_weights = new_weights if new_weights is not None else weights
    new_ranked = rank_patients(calculate_weighted_scores(sim_scores_df, factors, sim_weights))

    result = {
        "original_ranking": original_ranked, "new_ranking": new_ranked,
        "weights_changed": new_weights is not None,
        "factor_changed": bool(factor_overrides),
    }

    if patient_id is not None:
        orig_row = original_ranked[original_ranked["patient_id"] == patient_id].iloc[0]
        new_row = new_ranked[new_ranked["patient_id"] == patient_id].iloc[0]
        result["patient_id"] = patient_id
        result["original_score"] = float(orig_row["final_score"])
        result["new_score"] = float(new_row["final_score"])
        result["score_difference"] = round(result["new_score"] - result["original_score"], 2)
        result["original_rank"] = int(orig_row["rank"])
        result["new_rank"] = int(new_row["rank"])
        result["rank_difference"] = result["original_rank"] - result["new_rank"]

    return result
