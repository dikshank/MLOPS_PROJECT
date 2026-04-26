# ============================================================
# airflow/Dockerfile
# Custom Airflow image with ML dependencies pre-installed.
#
# Based on apache/airflow:2.8.1-python3.10
# Adds: torch, torchvision, mlflow, scikit-learn, pillow, pandas
#
# Build once → fast restarts, no pip install on every start
# ============================================================

FROM apache/airflow:2.8.1-python3.10

USER root

# Install system dependencies needed for torch
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install ML dependencies
# torch CPU-only to keep image size manageable
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    mlflow==2.12.1 \
    scikit-learn \
    pillow \
    pandas \
    pyyaml