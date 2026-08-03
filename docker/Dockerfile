# ==============================================================================
# Medical Image Diagnosis & Anomaly Detection Platform
# Production Multi-Stage Dockerfile — Security & OCI Hardened
# Module 22 — Production Deployment Hardening Pass (v0.22.1.0)
# ==============================================================================

# Target Platform Build Arguments (AMD64 / ARM64 multi-arch support)
ARG BUILDPLATFORM=linux/amd64
ARG TARGETPLATFORM=linux/amd64

# GPU Acceleration Note:
# For NVIDIA GPU hardware acceleration, change base image from python:3.11-slim
# to nvidia/cuda:11.8.0-runtime-ubuntu22.04 and install Python 3.11 via apt.

# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1: Builder — Dependency Compilation & Virtualenv Setup
# ──────────────────────────────────────────────────────────────────────────────
FROM --platform=${TARGETPLATFORM} python:3.11-slim AS builder

LABEL stage="builder"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch>=2.0.0 torchvision>=0.15.0 && \
    pip install -r /tmp/requirements.txt && \
    pip install \
        fastapi>=0.100.0 \
        uvicorn[standard]>=0.23.0 \
        python-multipart>=0.0.6 \
        gradio>=4.0.0 \
        httpx>=0.24.0 \
        pydantic>=2.0.0 \
        python-dotenv>=1.0.0


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: Runtime — Minimal, Non-Root, OCI-Compliant Production Image
# ──────────────────────────────────────────────────────────────────────────────
FROM --platform=${TARGETPLATFORM} python:3.11-slim AS runtime

# Priority 1: OCI-Compliant Metadata Labels
LABEL org.opencontainers.image.title="Medical Image Diagnosis & Anomaly Detection Platform" \
      org.opencontainers.image.description="Enterprise Multi-Modality Medical AI Engine, REST API, Anomaly Detection & Gradio Dashboard" \
      org.opencontainers.image.version="0.22.1.0" \
      org.opencontainers.image.authors="Fida Hussain Mohammed" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/your-username/medical-image-diagnosis" \
      org.opencontainers.image.documentation="https://github.com/your-username/medical-image-diagnosis/tree/main/docs" \
      org.opencontainers.image.vendor="Medical AI Systems" \
      org.opencontainers.image.created="2026-08-03T18:20:00Z" \
      org.opencontainers.image.revision="v0.22.1.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    ENVIRONMENT=production \
    DEBUG=false \
    AUTH_REQUIRED=false \
    ENABLE_DOCS=true \
    DATABASE_PATH=/app/data/medical_ai.db \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    GRADIO_PORT=7860 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    LOG_LEVEL=info

# Install runtime-only C-libraries for OpenCV & Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user medai (UID 1001)
RUN groupadd --gid 1001 medai && \
    useradd --uid 1001 --gid medai --shell /bin/bash --create-home medai

WORKDIR /app

# Copy application source with strict ownership
COPY --chown=medai:medai api/              ./api/
COPY --chown=medai:medai medical_ai_engine/ ./medical_ai_engine/
COPY --chown=medai:medai app.py            ./app.py
COPY --chown=medai:medai requirements.txt  ./requirements.txt
COPY --chown=medai:medai deployment/       ./deployment/
COPY --chown=medai:medai configs/          ./configs/

# Create runtime directories & enforce non-root ownership
RUN mkdir -p /app/data /app/logs /app/outputs /app/saved_models /tmp && \
    chown -R medai:medai /app /tmp

USER medai

EXPOSE 8000 7860

# Enhanced Container Health Check Probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD /bin/bash /app/deployment/healthcheck.sh || exit 1

ENTRYPOINT ["/bin/bash", "/app/deployment/startup.sh"]
