# Melanoma Detection MLOps Pipeline

End-to-end MLOps pipeline for low-resolution melanoma classification.
Built for resource-constrained, on-premise environments (CPU only, no cloud).

## Architecture

```
Raw Data (32x32 images)
        ↓
[Airflow] Data Pipeline DAG
  validate → split → preprocess (upscale to 64x64) → baseline_stats
        ↓
[MLflow] Training (3 models)
  MobileNetV3-Small | EfficientNet-B0 | SimpleCNN
  Champion/Challenger auto-promotion (recall → F1 tiebreaker)
        ↓
[FastAPI] Serving
  POST /predict | POST /feedback | GET /metrics
        ↓
[Prometheus + Grafana] Monitoring
  Drift score | Misclassification rate | Real-world recall
        ↓
[Airflow] Retraining DAG (auto every 30 mins)
  Triggered by misclassification > 10% or drift > 20%
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data pipeline | Apache Airflow 2.8.1 (SequentialExecutor + SQLite) |
| ML framework | PyTorch 2.x (CPU only) |
| Models | MobileNetV3-Small, EfficientNet-B0, SimpleCNN (from scratch) |
| Experiment tracking | MLflow 3.x |
| Data versioning | DVC + Google Drive |
| Serving | FastAPI + nginx |
| Monitoring | Prometheus + Grafana |
| Containerisation | Docker + Docker Compose |
| CI | GitHub Actions + dvc.yaml |
| Testing | pytest (46 tests) |

## Prerequisites

- Windows 11 / Linux
- Python 3.10+
- Docker Desktop
- Git

## Reproducing from Scratch

### Step 0 — Clone and setup

```bash
git clone <repo_url>
cd mlops_project
pip install -r requirements.txt
```

### Step 1 — Get the data (DVC)

```bash
# Pull data from Google Drive remote
dvc pull

# Data will appear at:
# data/raw/v1/malignant/  ← 32x32 malignant images
# data/raw/v1/benign/     ← 32x32 benign images
```

### Step 2 — Create Docker network

```bash
docker network create mlops_network
```

### Step 3 — Start Airflow and run data pipeline

```bash
# Build Airflow image (first time only — installs torch, ~5 mins)
cd airflow
docker compose -f docker-compose-airflow.yml build

# Initialise Airflow database
docker compose -f docker-compose-airflow.yml up airflow-init
# Wait for "Airflow initialized" then Ctrl+C

# Start Airflow
docker compose -f docker-compose-airflow.yml up airflow-webserver airflow-scheduler
```

Open `http://localhost:8080` (admin / admin)

Trigger **melanoma_data_pipeline** DAG manually. Wait for all 4 tasks to turn green:
```
validate_images → split_data → preprocess_images → compute_baseline_stats
```

This produces:
```
data/processed/v1/train/    ← 164 training images (64x64)
data/processed/v1/val/      ← 36 validation images
data/processed/v1/test/     ← 600 test images
data/baseline_stats/v1_stats.json
```

### Step 4 — Train models

```bash
cd ..  # back to project root

# Option A: Using MLflow Projects (recommended for reproducibility)
mlflow run . -P config=training/configs/config_v1_mobilenet.yaml
mlflow run . -e train_efficientnet
mlflow run . -e train_simplecnn

# Option B: Direct (faster)
python training/src/train.py --config training/configs/config_v1_mobilenet.yaml
python training/src/train.py --config training/configs/config_v1_efficientnet.yaml
python training/src/train.py --config training/configs/config_v1_simplecnn.yaml
```

Champion/Challenger auto-promotion:
- **Stage 1**: New recall > current recall → promote
- **Stage 2**: Recalls tied → compare F1, promote if better
- Best model goes to **Production** in MLflow registry

View experiments at `http://localhost:5000` (run MLflow locally):
```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

### Step 5 — Start all services

```bash
docker compose up --build
```

Services:
| Service | URL |
|---------|-----|
| Frontend | http://localhost:80 |
| Backend API | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/admin) |

### Step 6 — Run tests

```bash
pytest tests/ -v
# Expected: 46/46 passed
```

### Step 7 — Batch prediction (optional demo)

```bash
python scripts/batch_predict.py \
    --malignant path/to/malignant/images \
    --benign path/to/benign/images \
    --count 100
```

This sends 100 images per class to `/predict` + `/feedback`.
Check Grafana for updated metrics.

### Step 8 — Retraining DAG

The retraining DAG runs automatically every 30 minutes.
It triggers when:
- Misclassification rate > 10% (after ≥10 feedback submissions)
- Drift score > 20%

To trigger manually: Airflow UI → **melanoma_retraining_pipeline** → Trigger DAG

Pipeline:
```
check_trigger → prepare_feedback_data → trigger_training → evaluate_new_model → cleanup
```

Feedback images are merged into training data, all 3 models retrained,
best model auto-promoted to Production.

---

## Project Structure

```
mlops_project/
├── airflow/
│   ├── Dockerfile                      ← Custom Airflow image with torch
│   ├── docker-compose-airflow.yml
│   ├── dags/
│   │   ├── data_pipeline_dag.py        ← 4-task data pipeline
│   │   └── retraining_dag.py           ← 5-task retraining pipeline
│   ├── scripts/                        ← validate, split, preprocess, baseline_stats
│   └── configs/pipeline_config_v1.yaml
├── training/
│   ├── src/                            ← dataset, model, train, evaluate, mlflow_utils
│   └── configs/                        ← config_v1_mobilenet/efficientnet/simplecnn
├── backend/
│   ├── main.py                         ← FastAPI app
│   ├── model_loader.py                 ← MLflow registry integration
│   ├── predictor.py                    ← inference logic
│   ├── monitoring.py                   ← Prometheus metrics
│   └── schemas.py                      ← Pydantic schemas
├── frontend/                           ← nginx + HTML/CSS/JS
├── prometheus/prometheus.yml
├── grafana/provisioning/               ← dashboards + datasources
├── tests/                              ← 46 pytest tests
├── scripts/batch_predict.py            ← batch prediction + feedback
├── docs/                               ← HLD, LLD, test plan, test report
├── docker-compose.yml                  ← 5 services on mlops_network
├── MLproject                           ← MLflow reproducibility
├── python_env.yaml                     ← training environment spec
├── dvc.yaml                            ← data pipeline DAG
└── requirements.txt                    ← project dependencies
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SequentialExecutor + SQLite | No extra containers needed for Airflow |
| CPU only | On-premise constraint, Intel UHD 620 |
| 32x32 → 64x64 upscaling | v1 baseline proves resolution is bottleneck; models ready for v2 (224x224 → 64x64) without config changes |
| 3 model architectures | Pretrained (MobileNet, EfficientNet) vs from-scratch (SimpleCNN) comparison |
| Recall as primary metric | Missing cancer (FN) is worse than false alarm (FP) |
| F1 as tiebreaker | When recall is tied, pick more balanced model |
| No conda/venv | Docker isolation is sufficient |
| Functional programming | Throughout pipeline scripts |

## Reproducing a Specific MLflow Run

```bash
# Get the run ID from MLflow UI
git checkout <commit_hash_from_mlflow_tags>
dvc checkout
mlflow run . -P config=training/configs/config_v1_simplecnn.yaml
```

Every MLflow run is tagged with:
- `git_commit`: exact code state
- `data_version`: v1
- `model_name`: architecture used
