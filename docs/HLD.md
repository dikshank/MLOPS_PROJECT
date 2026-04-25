# High Level Design — Melanoma Detection MLOps System

## 1. Problem Statement

Early melanoma detection in resource-constrained environments where only
low-resolution images are available (rural healthcare, telemedicine, basic
mobile devices). The system assists non-medical users in deciding whether
to seek professional consultation.

**Primary metric:** Recall — minimising false negatives (missed melanomas)
is more important than minimising false positives (unnecessary consultations).

---

## 2. Design Paradigm

The system follows the **functional programming paradigm**.

Each module is a collection of pure functions with clearly defined inputs
and outputs. Side effects (file I/O, model loading, logging) are isolated
to dedicated modules (`logger.py`, `model_loader.py`).

Exceptions: `Dataset` and `Model` classes in the training pipeline are
object-oriented because PyTorch requires `nn.Module` and `Dataset`
subclassing. These are standard library conventions, not design choices.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                         │
│              nginx frontend (port 80)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API (HTTP)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               FastAPI Backend (port 8000)                   │
│  /predict  /health  /ready  /metrics  /feedback             │
│                                                             │
│  model_loader.py ──► MLflow Registry (mlruns/)              │
│  predictor.py    ──► PyTorch model inference                │
│  monitoring.py   ──► Prometheus metrics                     │
└──────────┬──────────────────────────────────────────────────┘
           │ scrape /metrics
           ▼
┌─────────────────────┐        ┌──────────────────────────────┐
│  Prometheus (9090)  │───────►│  Grafana Dashboard (3000)    │
└─────────────────────┘        └──────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              MLflow UI (port 5000)                          │
│  Experiment tracking, model registry, artifact store        │
│  Reads from: mlruns/ (local filesystem)                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Apache Airflow (port 8080)                     │
│  Data engineering pipeline orchestration                    │
│  DAG: validate → split → preprocess → baseline_stats        │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Design Principles

### 4.1 Loose Coupling
Frontend and backend are completely independent services connected
only via REST API calls. The frontend has no knowledge of the model
architecture, inference logic, or MLflow registry. This allows either
component to be updated independently.

### 4.2 Reproducibility
Every experiment is reproducible via:
```
git checkout <commit_hash>
dvc pull
mlflow run . -P config=<config_path>
```
The combination of Git commit hash + DVC data version + MLflow run ID
uniquely identifies every experiment.

### 4.3 Champion/Challenger Model Promotion
Models are never manually promoted to Production. After every training run,
the new model's val_recall is compared with the current Production model.
Only if the new model outperforms the existing champion is it promoted.
This ensures FastAPI always serves the best model seen so far.

### 4.4 Environment Parity
All services run in Docker containers defined by explicit Dockerfiles.
`docker-compose.yml` ensures identical environments across development
and demonstration.

---

## 5. Data Flow

```
Raw images (Google Drive)
        ↓
Airflow Pipeline (local Docker):
    validate → split → preprocess → baseline_stats
        ↓
Processed images + manifests (local filesystem)
        ↓
Training (local CPU):
    dataset.py → model.py → train.py → evaluate.py
        ↓
MLflow Registry (mlruns/ local):
    Auto-promote best recall model to Production
        ↓
FastAPI Backend:
    Load Production model at startup
    Serve predictions via /predict
    Collect feedback via /feedback
        ↓
Prometheus + Grafana:
    Monitor real-world recall decay
    Alert if recall drops or drift detected
        ↓
Airflow Retraining DAG:
    Triggered manually or by drift alert
    Retrains model, re-evaluates, auto-promotes if better
```

---

## 6. Technology Choices and Rationale

| Component | Choice | Rationale |
|---|---|---|
| Model architecture | MobileNetV3-Small / EfficientNet-B0 | Lightweight, pretrained, CPU-deployable |
| Data pipeline | Apache Airflow | Satisfies rubric, provides visual DAG and error tracking |
| Experiment tracking | MLflow (local) | No cloud dependency, full registry support |
| Data versioning | DVC + Google Drive | Reproducible data lineage, no cloud compute |
| Serving | FastAPI | Fast, async, automatic OpenAPI docs, Prometheus-compatible |
| Monitoring | Prometheus + Grafana | Industry standard, lightweight, NRT dashboards |
| Containerisation | Docker + docker-compose | Environment parity, loose coupling between services |
| Metric focus | Recall | False negatives (missed cancers) are clinically more dangerous |

---

## 7. Acceptance Criteria

| Criterion | Target |
|---|---|
| val_recall (v2 model) | ≥ 0.80 |
| /predict latency | < 500ms on CPU |
| Error rate | < 5% |
| All unit tests | Pass |
| All Docker services start | Without errors |
