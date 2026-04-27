# Melanoma Detection MLOps Pipeline

End-to-end MLOps pipeline for low-resolution melanoma classification.
Built for resource-constrained, on-premise environments (CPU only, no cloud).

## Architecture

```
Raw Data (32x32 images, DVC tracked)
        ↓
[Airflow] Data Pipeline DAG
  validate → split → preprocess (upscale to 64x64) → baseline_stats
        ↓
[MLflow] Training (3 models)
  MobileNetV3-Small | EfficientNet-B0 | SimpleCNN (from scratch)
  Champion/Challenger auto-promotion (recall primary → F1 tiebreaker)
        ↓
[FastAPI] Serving
  POST /predict | POST /feedback | GET /metrics | GET /health | GET /ready
        ↓
[Prometheus + Grafana] Monitoring (13 panels + 3 alert rules)
  Drift score | Misclassification rate | Real-world recall | Alerting
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
| Experiment tracking | MLflow (local mlruns/) |
| Data versioning | DVC + local remote |
| Serving | FastAPI + nginx |
| Monitoring | Prometheus + Grafana (13 panels, 3 alert rules) |
| Containerisation | Docker + Docker Compose (5 services) |
| CI | GitHub Actions + dvc.yaml DAG |
| Testing | pytest (68 tests, 100% pass rate) |

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

### Step 1 — Get the data

Raw images are tracked via Git LFS — pulled automatically on clone:

```bash
git lfs pull
```

Alternatively via DVC local remote:
```bash
dvc remote modify local_remote url /path/to/mlops_dvc_remote     # This won't work for as it is a local registry
dvc pull
```

Data will appear at:
```
data/raw/malignant/   ← 32x32 malignant images
data/raw/benign/      ← 32x32 benign images
```

### Step 2 — Create Docker network

```bash
docker network create mlops_network
```

### Step 3 — Start Airflow and run data pipeline

```bash
cd airflow

# Build Airflow image (first time only — installs torch, ~5-10 mins)
docker compose -f docker-compose-airflow.yml build

# Initialise Airflow database
docker compose -f docker-compose-airflow.yml up airflow-init
# Wait for "Airflow initialized" then Ctrl+C

# Start Airflow
docker compose -f docker-compose-airflow.yml up airflow-webserver airflow-scheduler
cd ..
```

Open `http://localhost:8080` (admin / admin)

Trigger **melanoma_data_pipeline** DAG manually. Wait for all 4 tasks green:
```
validate_images → split_data → preprocess_images → compute_baseline_stats
```

This produces:
```
data/processed/v1/train/          ← 164 training images (64x64)
data/processed/v1/val/            ← 36 validation images
data/processed/v1/test/           ← 600 test images
data/baseline_stats/v1_stats.json ← pixel histogram for drift detection
```

### Step 4 — Train models

#### Prerequisites for Windows (one-time setup) (FIRST TRY THE MLFLOW RUN COMMANDS IF THEY DON"T WORK THEN DO THIS SETUP OR OTHERWISE JUST TRAIN DIRECTLY)

```powershell
# 1. Allow scripts (run PowerShell as Administrator)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Install pyenv-win (restart PowerShell after this)
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"

# 3. After restarting PowerShell, set PYENV_ROOT permanently
[System.Environment]::SetEnvironmentVariable("PYENV_ROOT", "$env:USERPROFILE\.pyenv\pyenv-win", "User")
[System.Environment]::SetEnvironmentVariable("PATH", "$env:USERPROFILE\.pyenv\pyenv-win\bin;$env:USERPROFILE\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable("PATH", "User"), "User")

# 4. Restart PowerShell again, then verify
pyenv --version
```

```bash
# Option A: Using MLflow Projects (recommended for reproducibility)
mlflow run . --experiment-name baseline-v1 -P config=training/configs/config_v1_mobilenet.yaml
mlflow run . --experiment-name baseline-v1 -e train_efficientnet
mlflow run . --experiment-name baseline-v1 -e train_simplecnn

# Option B: Direct (faster)
python training/src/train.py --config training/configs/config_v1_mobilenet.yaml
python training/src/train.py --config training/configs/config_v1_efficientnet.yaml
python training/src/train.py --config training/configs/config_v1_simplecnn.yaml
```

Champion/Challenger auto-promotion:
- Stage 1: New recall > current recall + 0.02 → promote
- Stage 2: Recalls tied → compare F1, promote if better
- Best model goes to Production in MLflow registry

View experiments:
```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

### Step 5 — Start all services

```bash
docker network create mlops_network
docker compose up --build
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Frontend | http://localhost:80 | — |
| Backend API | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin/admin |
| Airflow | http://localhost:8080 | admin/admin |

### Step 6 — Run tests

```bash
pytest tests/ -v
# Expected: 68/68 passed
```

### Step 7 — Batch prediction (optional demo)

```bash
python scripts/batch_predict.py --malignant path/to/malignant --benign path/to/benign --count 100
```

This sends 100 images per class to `/predict` + `/feedback`.
Check Grafana for updated metrics.

### Step 8 — Retraining DAG

The retraining DAG runs automatically every 30 minutes.
It triggers when:
- Misclassification rate > 10% (after >= 10 feedback submissions)
- Drift score > 20%

To trigger manually: Airflow UI → **melanoma_retraining_pipeline** → Trigger DAG

```
check_trigger → prepare_feedback_data → trigger_training → evaluate_new_model → cleanup
```

---

## Project Structure

```
mlops_project/
├── airflow/
│   ├── Dockerfile                      ← Custom Airflow image with torch pre-installed
│   ├── docker-compose-airflow.yml
│   ├── dags/
│   │   ├── data_pipeline_dag.py        ← 4-task data pipeline
│   │   └── retraining_dag.py           ← 5-task auto-retraining pipeline
│   ├── scripts/                        ← validate, split, preprocess, baseline_stats
│   └── configs/pipeline_config_v1.yaml
├── training/
│   ├── src/                            ← dataset, model, train, evaluate, mlflow_utils
│   └── configs/                        ← config_v1_mobilenet/efficientnet/simplecnn
├── backend/
│   ├── main.py                         ← FastAPI app + drift detection
│   ├── model_loader.py                 ← MLflow registry integration
│   ├── model.py                        ← model architecture definitions
│   ├── predictor.py                    ← inference logic
│   ├── monitoring.py                   ← Prometheus metrics (15 metrics)
│   ├── schemas.py                      ← Pydantic request/response schemas
│   ├── logger.py                       ← structured logging
│   ├── requirements.txt                ← backend Docker dependencies
│   └── Dockerfile
├── frontend/
│   ├── index.html                      ← Screening + Pipeline Dashboard tabs
│   ├── style.css                       ← colorblind-friendly styles
│   ├── app.js                          ← configurable API_BASE
│   └── Dockerfile
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── provisioning/
│       ├── dashboards/melanoma.json    ← 13-panel NRT dashboard
│       ├── datasources/               ← Prometheus datasource
│       └── alerting/                  ← 3 provisioned alert rules
├── tests/
│   ├── conftest.py
│   ├── test_predict.py                 ← 15 tests
│   ├── test_health.py                  ← 17 tests
│   ├── test_feedback.py                ← 14 tests
│   └── test_pipeline.py               ← 22 tests
├── scripts/
│   └── batch_predict.py               ← batch prediction + feedback script
├── docs/
│   ├── HLD.md                         ← high-level design
│   ├── LLD.md                         ← API endpoint specifications
│   ├── architecture.html              ← diagram + block explanations
│   ├── test_plan.md                   ← test plan + acceptance criteria
│   ├── test_report.md                 ← 68/68 passing test report
│   └── user_manual.md                 ← non-technical user guide
├── .github/workflows/ci.yml           ← GitHub Actions CI
├── docker-compose.yml                 ← 5 services on mlops_network
├── MLproject                          ← MLflow reproducibility
├── python_env.yaml                    ← training environment spec
├── dvc.yaml                           ← 5-stage CI pipeline DAG
├── .dvcignore
└── requirements.txt                   ← project dependencies
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SequentialExecutor + SQLite | No extra containers needed for Airflow |
| CPU only | On-premise constraint, Intel UHD 620 |
| 32x32 → 64x64 upscaling | Proves resolution is bottleneck; models ready for v2 without config changes |
| 3 model architectures | Pretrained (MobileNet, EfficientNet) vs from-scratch (SimpleCNN) comparison |
| Recall as primary metric | Missing cancer (FN) is worse than false alarm (FP) |
| F1 as tiebreaker | When recall is tied, pick more balanced model |
| Local DVC remote | On-premise constraint — no cloud allowed |
| Functional programming | Throughout all pipeline scripts |
| Colorblind-friendly UI | Amber/blue instead of red/green for results |
| Configurable API_BASE | window.API_BASE ensures loose coupling between frontend and backend |

## Reproducing a Specific MLflow Run

```bash
git checkout <commit_hash_from_mlflow_tags>
dvc checkout
mlflow run . -P config=training/configs/config_v1_simplecnn.yaml
```

Every MLflow run is tagged with:
- `git_commit_hash`: exact code state
- `data_version`: v1
- `model_name`: architecture used
- `promotion_result`: champion/challenger outcome

## Data Reproduction Note

Data is versioned with DVC. The `data/raw.dvc` pointer file is committed
to Git. The actual images are stored in a local DVC remote at:
```
C:\Users\HP\Desktop\mlops_dvc_remote
```

To reproduce on a different machine, copy this folder and run:
```bash
dvc remote modify local_remote url /path/to/mlops_dvc_remote
dvc pull
```
