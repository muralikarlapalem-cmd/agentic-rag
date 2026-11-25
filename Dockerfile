FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    DATA_DIR=/data

# Fixed dark theme so the UI looks the same whatever the browser's light/dark setting,
# and hide Streamlit's developer toolbar from visitors.
ENV STREAMLIT_THEME_BASE=dark \
    STREAMLIT_THEME_PRIMARY_COLOR="#3b82f6" \
    STREAMLIT_THEME_BACKGROUND_COLOR="#0f1117" \
    STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR="#1e293b" \
    STREAMLIT_THEME_TEXT_COLOR="#e2e8f0" \
    STREAMLIT_CLIENT_TOOLBAR_MODE=viewer

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

# Run as non-root; /data holds the persisted FAISS indexes (mount a volume here).
RUN useradd -m appuser && mkdir -p /data && chown appuser /data
USER appuser
VOLUME /data

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

# Secrets arrive via environment variables at runtime, never baked into the image.
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--server.maxUploadSize=50", "--server.enableXsrfProtection=true"]
