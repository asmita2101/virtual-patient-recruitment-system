"""
llm_integration.py
------------------------------------------------------------
All Google Gemini related functionality for the Virtual Patient
Recruitment System.

Responsibility of Gemini in this project (and ONLY this):
    Natural Language Eligibility Criteria  -->  Structured JSON rules

Gemini NEVER decides eligibility, ranks patients, computes AHP
weights, computes weighted scores, or predicts patient behaviour.
All of that logic lives in backend.py and is fully deterministic.
------------------------------------------------------------
SETUP:
    1. Get a Gemini API key from https://aistudio.google.com/app/apikey
    2. Set it as the environment variable GEMINI_API_KEY. Locally, the
       easiest way is to create a ".env" file (never committed to git)
       in the project root containing:
           GEMINI_API_KEY=your_key_here
       This file is loaded automatically below via python-dotenv.
    3. In Docker / cloud deployments, pass GEMINI_API_KEY as a runtime
       environment variable / platform secret (see README.md).
    4. Run: streamlit run frontend.py

SECURITY NOTE: This file must NEVER contain a real, hard-coded API key.
The key is always read from the environment at runtime.
------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# Load variables from a local .env file if python-dotenv is installed and
# a .env file is present. This has no effect in Docker/cloud deployments
# where GEMINI_API_KEY is already provided as a real environment variable
# - it simply does nothing if no .env file exists.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Model name used for rule generation. Change if your account has access
# to a different Gemini model. (Google deprecated "gemini-2.0-flash" and
# now recommends "gemini-3.6-flash" as of the error message returned by
# their API — update this string again in the future if Google deprecates
# this model too.)
GEMINI_MODEL_NAME = "gemini-3.6-flash"

# Tried in order if the primary model returns a transient "overloaded" error
# (HTTP 503 / UNAVAILABLE). Having a fallback avoids a demo/evaluation being
# blocked by a temporary capacity spike on Google's most popular model.
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash"]

# Retry behaviour for transient overload errors (503 / UNAVAILABLE).
MAX_RETRIES_PER_MODEL = 3
RETRY_BACKOFF_SECONDS = [2, 5, 10]  # wait time before each retry attempt


def get_api_key() -> Optional[str]:
    """Resolve the Gemini API key from the environment (populated either by
    a real environment variable, a platform secret, or a local .env file
    loaded above). Returns None if it is not configured anywhere.

    SECURITY: Do NOT hard-code a real API key in this function or anywhere
    else in this file. A key pasted into source code, a chat, or a document
    must be treated as leaked and revoked immediately at
    https://aistudio.google.com/app/apikey. Only ever put a real key in your
    local, git-ignored ".env" file or your deployment platform's secret
    manager (see README.md).
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key if key else None


def is_configured() -> bool:
    return get_api_key() is not None


def _build_prompt(natural_language_criteria: str, dataset_features: Dict[str, Any]) -> str:
    """Build the strict instruction prompt sent to Gemini."""

    schema_lines = []
    for col, info in dataset_features.items():
        if "unique_values" in info:
            schema_lines.append(f"- {col} (categorical): {info['unique_values']}")
        elif "min" in info:
            schema_lines.append(f"- {col} (numeric): range {info['min']} to {info['max']}")
        else:
            schema_lines.append(f"- {col}")
    schema_text = "\n".join(schema_lines)

    prompt = f"""
You are a strict natural-language-to-JSON converter for a clinical-trial
eligibility system. You must convert the recruiter's plain-English trial
criteria into a structured JSON object describing INCLUSION and EXCLUSION
rules.

STRICT RULES YOU MUST FOLLOW:
1. Use ONLY the exact dataset feature names listed below. Never invent a
   new feature name.
2. Use ONLY these supported operators: equals, not_equals, >, <, >=, <=,
   between, contains, not_contains, in, not_in.
3. For categorical features, use one of the exact listed category values
   whenever possible.
4. Return ONLY valid JSON. No explanations, no markdown fences, no prose
   before or after the JSON.
5. Do NOT decide whether any patient is eligible.
6. Do NOT rank, score, or prioritize patients.
7. Do NOT calculate AHP weights or weighted scores.
8. Do NOT predict patient interest or participation.
9. Treat "interest" as a plain dataset field (Low/Medium/High) — do not
   infer or predict it.
10. Treat "consent_to_contact" as a separate field from "interest". Only
    add a rule for consent_to_contact if the recruiter explicitly
    mentions consent.
11. Your ONLY job is translating natural language into structured rules.

DATASET FEATURES AVAILABLE:
{schema_text}

REQUIRED JSON OUTPUT FORMAT (example structure only):
{{
    "inclusion": [
        {{"feature": "age", "operator": "between", "value": [40, 65]}},
        {{"feature": "primary_disease", "operator": "equals", "value": "Type 2 Diabetes"}},
        {{"feature": "hba1c", "operator": ">", "value": 7}}
    ],
    "exclusion": [
        {{"feature": "recent_surgery", "operator": "equals", "value": "Yes"}}
    ]
}}

RECRUITER'S NATURAL LANGUAGE TRIAL CRITERIA:
\"\"\"{natural_language_criteria}\"\"\"

Return ONLY the JSON object now.
"""
    return prompt.strip()


def _extract_json(text: str) -> str:
    """Strip markdown code fences / stray text and isolate the JSON object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    # If there's still leading/trailing prose, isolate the outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text


def _is_overloaded_error(exc: Exception) -> bool:
    """Detect Google's transient 'model overloaded' error so we know it is
    worth retrying/falling back, as opposed to a permanent error (bad key,
    invalid request, etc.) that would just fail the same way again."""
    text = str(exc).lower()
    return "503" in text or "unavailable" in text or "overloaded" in text or "high demand" in text


def _call_gemini_with_retries(client, prompt: str) -> Tuple[bool, Optional[str], str]:
    """
    Try GEMINI_MODEL_NAME first, retrying on transient overload errors with
    a short backoff. If it keeps failing, fall back to each model in
    GEMINI_FALLBACK_MODELS in turn. Returns (success, raw_text_or_None, message).
    """
    models_to_try = [GEMINI_MODEL_NAME] + GEMINI_FALLBACK_MODELS
    last_error_message = "Unknown error."

    for model_index, model_name in enumerate(models_to_try):
        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                raw_text = getattr(response, "text", None)
                if raw_text:
                    note = f" (using fallback model '{model_name}')" if model_index > 0 else ""
                    return True, raw_text, f"Success{note}."
                last_error_message = f"Model '{model_name}' returned an empty response."
                break  # empty response isn't an overload issue - move to next model, don't retry
            except Exception as exc:  # noqa: BLE001
                last_error_message = str(exc)
                if _is_overloaded_error(exc) and attempt < MAX_RETRIES_PER_MODEL - 1:
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                    continue
                break  # not an overload error, or out of retries for this model - try next model

    return False, None, last_error_message


def generate_eligibility_rules(
    natural_language_criteria: str, dataset_features: Dict[str, Any]
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Send the recruiter's natural-language criteria to Gemini and parse the
    structured JSON response.

    Returns: (success, rules_dict_or_None, message)
    """
    api_key = get_api_key()
    if not api_key:
        return False, None, (
            "Gemini API key is not configured. Please add your Gemini API key "
            "(see the top of llm_integration.py or set the GEMINI_API_KEY "
            "environment variable)."
        )

    if not natural_language_criteria or not natural_language_criteria.strip():
        return False, None, "Please enter natural-language eligibility criteria before generating rules."

    try:
        from google import genai
    except ImportError:
        return False, None, (
            "The 'google-genai' package is not installed. "
            "Run: pip install google-genai"
        )

    try:
        client = genai.Client(api_key=api_key)
        prompt = _build_prompt(natural_language_criteria, dataset_features)
    except Exception as exc:  # noqa: BLE001
        return False, None, f"Could not initialize the Gemini client: {exc}"

    success, raw_text, call_message = _call_gemini_with_retries(client, prompt)
    if not success:
        if _is_overloaded_error(Exception(call_message)):
            return False, None, (
                "Gemini's servers are currently overloaded (high demand on this model). "
                "We automatically retried and tried a fallback model, but it is still "
                "unavailable. Please wait a minute and click 'Generate Eligibility Rules' "
                "again, or edit the rules JSON manually on this page in the meantime."
            )
        return False, None, f"Gemini API request failed: {call_message}"

    json_text = _extract_json(raw_text)
    try:
        rules = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return False, None, f"Gemini did not return valid JSON ({exc}). Raw response:\n{raw_text}"

    if not isinstance(rules, dict) or "inclusion" not in rules or "exclusion" not in rules:
        return False, None, (
            "Gemini's response was valid JSON but did not contain the expected "
            "'inclusion' / 'exclusion' structure."
        )

    rules.setdefault("inclusion", [])
    rules.setdefault("exclusion", [])
    return True, rules, "Eligibility rules generated successfully. Please review before confirming."