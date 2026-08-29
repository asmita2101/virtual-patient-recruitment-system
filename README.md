# Virtual Patient Recruitment System

An explainable, trial-specific decision-support prototype that helps a clinical-trial
recruiter identify eligible patients and prioritize them transparently — with **no
machine-learning prediction model and no recruitment feedback loop**.

> **Disclaimer:** This system is an academic/research prototype using synthetic patient
> data. It is not a substitute for clinical judgment, formal clinical-trial screening,
> or medical advice.

---

## 1. Project Overview

Clinical-trial recruiters need to (1) identify which patients in a dataset are
eligible for a specific trial, and (2) decide which eligible patients to contact
first. This project solves both problems with fully deterministic, explainable logic:

- **Eligibility** is a hard filter: a patient must pass every inclusion rule and
  violate no exclusion rule.
- **Prioritization** ranks only the eligible patients using trial-specific weights
  derived from the recruiter's own priorities (via AHP), combined with a Weighted
  Sum Model.

## 2. System Architecture

```
Recruiter
   -> Natural Language Trial Eligibility Criteria
   -> Gemini LLM (text -> structured JSON rules ONLY)
   -> Rule Validation
   -> Recruiter Review / Confirmation
   -> Dynamic Eligibility Engine (hard filter)
   -> Eligible Patients
   -> Trial-Specific Ranking Factor Selection
   -> Adaptive AHP (pairwise comparisons -> dynamic weights)
   -> Factor Normalization (0-100)
   -> Weighted Sum Model
   -> Final Patient Priority Score
   -> Patient Ranking
   -> Explainable Ranking
   -> What-If Analysis
   -> Recruiter Dashboard
```

Gemini's role is intentionally narrow: it only converts natural language into
structured JSON rules. It never decides eligibility, ranks patients, computes AHP
weights, computes final scores, or predicts patient interest/participation — all of
that is deterministic Python in `backend.py`.

"Adaptive" in this project means **trial-specific**, not self-learning: the AHP
weights adapt to whatever priorities the recruiter sets for the *current* trial. The
system does not learn from past recruitment outcomes.

## 3. Main Features

- Natural-language eligibility input
- Gemini-powered eligibility rule extraction
- Rule validation against the actual dataset schema (with "did you mean...?" suggestions)
- Human-in-the-loop review/edit/confirm of generated rules before execution
- Generic, dynamic eligibility engine (no hard-coded trial logic)
- Hard eligibility filtering (inclusion + exclusion)
- Trial-specific ranking factor selection
- Interest-based prioritization (Low / Medium / High, kept separate from consent)
- Adaptive AHP (Analytic Hierarchy Process) with automatic reciprocal values and
  Consistency Ratio checking
- Weighted Sum Model for the final 0-100 priority score
- Factor-level explainable ranking ("why was this patient ranked here?")
- What-If analysis (simulate a changed factor value or changed AHP weights without
  touching the original data)
- CSV export of eligible/ranked patients
- Professional Streamlit dashboard
- Docker support for portable, reproducible deployment

## 4. Dataset

The app ships with a synthetic patient dataset, `patients.csv` (5,000 rows), containing:

`patient_id, age, gender, location, primary_disease, disease_duration_years,
disease_severity, comorbidities, previous_treatment, bmi, systolic_bp, diastolic_bp,
hba1c, fasting_glucose, kidney_function, liver_function, cholesterol,
smoking_status, alcohol_use, pregnancy_status, allergies, recent_surgery,
other_serious_condition, distance_from_trial_site_km, availability,
contact_preference, consent_to_contact, interest`

Two fields deserve special mention:

- **`interest`** (`Low` / `Medium` / `High`) is used as a **ranking** factor
  (mapped transparently to 0 / 50 / 100). It is not a hard eligibility criterion
  unless the recruiter explicitly asks for that in their natural-language input.
- **`consent_to_contact`** is a *separate* field from `interest`. It only becomes a
  hard eligibility rule if the recruiter's text explicitly mentions consent (e.g.
  "only include patients who have consented to be contacted").

You can also upload a different (but schema-compatible) CSV from the sidebar at
runtime.

## 5. Installation Without Docker

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Gemini API key
cp .env.example .env
# then open .env and paste your real key after GEMINI_API_KEY=

# 4. Run the app
streamlit run frontend.py
```

Open the URL Streamlit prints in the terminal (typically `http://localhost:8501`).

> Get a Gemini API key at https://aistudio.google.com/app/apikey

## 6. Docker Installation

Build the image:

```bash
docker build -t virtual-patient-recruitment .
```

Run the container, passing your Gemini key as a runtime environment variable
(never baked into the image):

```bash
docker run -p 8501:8501 -e GEMINI_API_KEY="YOUR_KEY" virtual-patient-recruitment
```

Then open your browser at:

```
localhost:8501
```

## 7. Docker Environment Variable

The container reads the key at runtime only — it is never stored inside the image,
`Dockerfile`, or source code:

```
GEMINI_API_KEY
```

If you omit `-e GEMINI_API_KEY=...`, the app still starts; every page works except
rule generation, which shows a friendly "Gemini API key is not configured" message
instead of crashing.

## 8. GitHub Setup

```bash
git init
git add .
git commit -m "Initial project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

## 9. GitHub Security

**Never commit:**

- `.env` (your real API key lives here — it is already listed in `.gitignore`)
- API keys, passwords, tokens, or secrets of any kind
- `.streamlit/secrets.toml` if you choose to use Streamlit's secrets mechanism

`.gitignore` is already configured to exclude `.env` and `.streamlit/secrets.toml`
so a normal `git add .` will not accidentally stage them. Always double-check with
`git status` before your first commit.

## 10. Deployment

General Docker-based deployment workflow on most cloud platforms (e.g. a container
service that builds from a Dockerfile):

```
GitHub repository
   -> Cloud platform pulls the repository
   -> Platform runs `docker build` using the included Dockerfile
   -> Container starts, exposing port 8501
   -> Platform injects GEMINI_API_KEY as a secret/environment variable
   -> Live Streamlit application, reachable at the platform's assigned URL
```

Configure `GEMINI_API_KEY` in your platform's secret/environment-variable settings
(the exact screen name varies by provider) — never inside the repository itself.

**Note:** Docker guarantees that any system with Docker installed can run this
container without manually installing Python or the project's dependencies. It does
not mean the app runs on a system with no Docker installed at all — Docker (or a
Docker-compatible container runtime) is still required on the host.

## 11. Project Structure

```
virtual-patient-recruitment/
├── frontend.py             # App shell/router: page config, global CSS, sidebar, navigation
├── state.py                # Shared session-state init, styling, sidebar (used by every page)
├── backend.py              # Eligibility engine, AHP, Weighted Sum, explainability, patient CRUD
├── llm_integration.py      # All Gemini API calls (rule extraction only)
├── app_pages/              # One file per sidebar page (true multi-page navigation)
│   ├── dashboard.py
│   ├── trial_criteria.py   # Define trial + generate/review/confirm eligibility rules
│   ├── patients.py         # Search, filter, sort, view, edit, delete patients
│   ├── add_patient.py      # Add a new patient (persists to the dataset)
│   ├── eligibility.py      # Run the hard eligibility filter
│   ├── ranking.py          # Ranking factors + Adaptive AHP + Weighted Sum + final ranking
│   ├── explainability.py
│   └── whatif.py
├── patients.csv            # Synthetic patient dataset
├── requirements.txt        # Exact runtime dependencies
├── Dockerfile              # Production container build
├── .dockerignore           # Files excluded from the Docker build context
├── .gitignore              # Files excluded from git (incl. secrets)
├── .env.example            # Template for local environment variables
├── .streamlit/config.toml  # Streamlit server/theme configuration
└── README.md               # This file
```

No `app.py` exists — the app always runs via `streamlit run frontend.py`, both
locally and inside Docker. `frontend.py` is a thin shell that builds real,
separate pages from `app_pages/` using Streamlit's native `st.Page` /
`st.navigation` multi-page API — clicking a sidebar item navigates to a
genuinely separate page rather than scrolling one long page. All pages share
state (dataset, confirmed trial rules, eligibility results, AHP weights,
ranking) via `st.session_state`, initialized consistently by `state.py`.

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Gemini API key is not configured" | `GEMINI_API_KEY` not set | Add it to `.env` (local) or pass `-e GEMINI_API_KEY=...` (Docker) |
| `ModuleNotFoundError: google` or similar | Dependencies not installed in the environment actually running Streamlit | Re-run `pip install -r requirements.txt` in the correct venv/container |
| Gemini API request fails with a 404 mentioning a model name | Google deprecated the configured model | Update `GEMINI_MODEL_NAME` in `llm_integration.py` to the model name Google's error message recommends |
| "No dataset loaded" | `patients.csv` missing from the working directory | Make sure `patients.csv` sits next to `frontend.py`, or upload a CSV from the sidebar |
| Docker container exits immediately | Check `docker logs <container_id>` for a Python traceback | Usually a missing dependency or a dataset file not copied into the image |
| Can't reach `localhost:8501` | Port not published, or container crashed | Confirm `-p 8501:8501` was passed to `docker run` and check `docker ps` / `docker logs` |
| Edited JSON rules won't confirm | Validation errors listed on the Eligibility Criteria page | Fix the reported feature/operator/value issues, or click Regenerate |

## 13. Why This Now Works Reliably on Another System

- **No absolute paths anywhere** — the dataset is loaded via the relative path
  `patients.csv`, and an in-app CSV uploader is available as a fallback, so the
  project does not depend on any particular folder location (no `C:\Users\...`,
  no `Desktop`/`OneDrive` paths).
- **No hard-coded API key** — `llm_integration.py` reads `GEMINI_API_KEY`
  exclusively from the environment (populated by a real env var, a container
  secret, or an optional local `.env` file). The key never appears in source code,
  the Dockerfile, or the dataset.
- **Pinned, verified dependencies** — `requirements.txt` was built by inspecting the
  actual imports in all three Python files and was tested with a clean install in
  an isolated virtual environment.
- **Containerized runtime** — any machine with Docker installed can build and run
  the exact same environment this project was tested in, without needing to
  install Python, pip packages, or worry about OS-level differences. (Docker does
  not remove the need to have Docker itself installed on the host.)
- **Headless, externally-reachable Streamlit config** — `.streamlit/config.toml`
  and the Dockerfile's environment variables ensure the server binds to
  `0.0.0.0:8501` instead of `localhost` only, so it's reachable from outside the
  container.
