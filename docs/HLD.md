# High Level Design — Melanoma Detection MLOps System

## 1. Problem Statement

Early melanoma detection in resource-constrained environments where only
low-resolution images are available (rural healthcare, telemedicine, basic
mobile devices). The system assists non-medical users in deciding whether
to seek professional consultation.

**Primary metric:** Recall — minimising false negatives (missed melanomas)
is more important than minimising false positives (unnecessary consultations).

**Secondary metric:** F1 score — used as tiebreaker when recall is equal
between models, to avoid promoting models that flag everything as malignant.

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
│         nginx frontend (port 80) — two tabs:                │
│         Screening Tab | Pipeline Dashboard Tab              │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST API (HTTP)
                           │ API_BASE = window.API_BASE || "http://localhost:8000"
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               FastAPI Backend (port 8000)                   │
│  /predict  /health  /ready  /metrics  /feedback             │
│                                                             │
│  model_loader.py ──► MLflow Registry (mlruns/)              │
│  predictor.py    ──► PyTorch model inference                │
│  monitoring.py   ──► Prometheus metrics                     │
│  main.py         ──► drift detection, retraining trigger    │
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
│  Data engineering pipeline + retraining pipeline            │
│  DAG 1: validate → split → preprocess → baseline_stats      │
│  DAG 2: check_trigger → prepare_feedback → train →          │
│          evaluate → cleanup (auto every 30 mins)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Design Principles

### 4.1 Loose Coupling
Frontend and backend are completely independent services connected
only via REST API calls. The frontend has no knowledge of the model
architecture, inference logic, or MLflow registry. The API base URL
is configurable via `window.API_BASE` (falls back to localhost:8000).
This allows either component to be updated independently.

### 4.2 Reproducibility
Every experiment is reproducible via:
```
git checkout <commit_hash>
dvc checkout
mlflow run . -P config=<config_path>
```
The combination of Git commit hash + DVC data version + MLflow run ID
uniquely identifies every experiment.

### 4.3 Champion/Challenger Model Promotion (Two-Stage)
Models are never manually promoted to Production. After every training run,
a two-stage comparison is performed:

**Stage 1 — Recall (primary metric, patient safety):**
- If new_recall > current_recall + 0.02 → promote
- If new_recall < current_recall - 0.02 → keep current

**Stage 2 — F1 tiebreaker (when recall is tied within ±0.02):**
- If new_f1 > current_f1 → promote (better balance)
- Otherwise → keep current (stability wins)

This ensures FastAPI always serves the safest AND most balanced model.

### 4.4 Environment Parity
All services run in Docker containers defined by explicit Dockerfiles.
`docker-compose.yml` ensures identical environments across development
and demonstration. All 5 services run on a shared `mlops_network`.

### 4.5 Automated Retraining
The retraining DAG runs automatically every 30 minutes and checks for:
- Misclassification rate > 10% (after >= 10 feedback submissions)
- Drift score > 20% (pixel distribution vs baseline histogram)

When triggered, it merges feedback images into training data and retrains
all 3 models, auto-promoting the best to Production.

---

## 5. Model Architectures

Three models are trained and compared:

| Model | Type | Parameters | Notes |
|---|---|---|---|
| MobileNetV3-Small | Pretrained (ImageNet) | 1.5M (592K trainable) | Lightweight, fast on CPU |
| EfficientNet-B0 | Pretrained (ImageNet) | 4M (2K trainable) | Stronger baseline |
| SimpleCNN | From scratch | 102K (all trainable) | Proves resolution is bottleneck |

All models use:
- Input: 64x64 RGB images (32x32 raw upscaled during preprocessing)
- Output: 2 classes (malignant, benign)
- Loss: CrossEntropyLoss with class weights [1.0, 2.0]
- Threshold: Auto-tuned on validation set to maximise recall

---

## 6. Data Flow

```
Raw images (32x32, local filesystem, DVC tracked)
        |
Airflow Data Pipeline DAG:
    validate -> split -> preprocess (->64x64) -> baseline_stats
        |
Processed images + manifests + baseline stats (local filesystem)
        |
Training (local CPU, MLflow tracked):
    dataset.py -> model.py -> train.py -> evaluate.py -> mlflow_utils.py
    3 models trained: MobileNetV3, EfficientNet-B0, SimpleCNN
        |
MLflow Registry (mlruns/ local):
    Two-stage champion/challenger: recall primary, F1 tiebreaker
    Best model auto-promoted to Production
        |
FastAPI Backend:
    Load Production model at startup
    Serve predictions via /predict (saves image for retraining)
    Collect feedback via /feedback (updates confusion matrix)
    Detect drift via pixel histogram comparison
        |
Prometheus + Grafana:
    Monitor real-world recall, precision, drift score
    Misclassification rate, retraining triggers
        |
Airflow Retraining DAG (auto every 30 mins):
    check_trigger -> prepare_feedback_data -> trigger_training
    -> evaluate_new_model -> cleanup
    Retrains all 3 models, auto-promotes best
```

---

## 7. Technology Choices and Rationale

| Component | Choice | Rationale |
|---|---|---|
| Model architectures | MobileNetV3-Small, EfficientNet-B0, SimpleCNN | Pretrained vs from-scratch comparison; CPU-deployable |
| Data pipeline | Apache Airflow | Visual DAG, error tracking, task-level logging |
| Experiment tracking | MLflow (local) | No cloud dependency, full registry + artifact support |
| Data versioning | DVC + local remote | Reproducible data lineage, on-premise constraint |
| Serving | FastAPI | Async, automatic OpenAPI docs, Prometheus-compatible |
| Monitoring | Prometheus + Grafana | Industry standard, NRT dashboards, alert support |
| Containerisation | Docker + docker-compose | Environment parity, 5 services on shared network |
| Primary metric | Recall | False negatives (missed cancers) are clinically dangerous |
| Secondary metric | F1 | Tiebreaker to avoid all-malignant degenerate models |

---

## 8. Acceptance Criteria

| Criterion | Target |
|---|---|
| /predict latency | < 200ms on CPU |
| Error rate | < 5% of requests |
| All unit tests | 100% pass rate |
| All Docker services start | Without errors |
| Model loaded at startup | Production model auto-loaded |
| Drift detection | Operational (threshold 20%) |
| Retraining trigger | Operational (threshold 10% misclassification) |
