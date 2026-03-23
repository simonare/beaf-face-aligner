# ── Build stage ───────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile mediapipe / OpenCV wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.12-slim

# Runtime shared libraries (libgles2 + libegl1 required by mediapipe tasks API)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    libgles2 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY aligner.py main.py ./

# Pre-download the MediaPipe face landmarker model at build time so the
# container starts immediately without a network round-trip at runtime.
# Set SKIP_MODEL_DOWNLOAD=1 to skip this step (e.g. for dev builds).
ARG SKIP_MODEL_DOWNLOAD=0
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "0" ]; then \
    python -c "from aligner import ensure_model; ensure_model(); print('Model ready.')"; \
    fi

EXPOSE 8000

# Adjust workers / timeout to taste; 1 worker is fine for n8n usage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
    "--workers", "1", "--timeout-keep-alive", "120"]