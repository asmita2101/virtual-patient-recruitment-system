# ============================================================
# Virtual Patient Recruitment System - Dockerfile
# ============================================================
# Builds a container that runs the existing Streamlit application
# (frontend.py + backend.py + llm_integration.py) exactly as-is.
#
# The Gemini API key is NEVER baked into this image. It must be
# supplied at container run time via the GEMINI_API_KEY environment
# variable (see README.md).
# ============================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr,
# which makes container logs show up immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer whenever
# only application code changes (faster rebuilds).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code, shared state module, page files, and dataset.
COPY frontend.py backend.py llm_integration.py state.py patients.csv ./
COPY app_pages/ ./app_pages/
COPY .streamlit/ ./.streamlit/

# Streamlit-specific environment variables so the app is reachable
# from outside the container and does not prompt for anything
# interactively (email prompt, telemetry prompt, etc.).
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Basic container health check hitting Streamlit's built-in health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# GEMINI_API_KEY is intentionally NOT set here. Provide it at runtime:
#   docker run -p 8501:8501 -e GEMINI_API_KEY="YOUR_KEY" virtual-patient-recruitment
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
