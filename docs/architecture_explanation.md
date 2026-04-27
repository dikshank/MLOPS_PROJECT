# Architecture Diagram — Block Explanations

See `architecture_diagram.svg` for the visual diagram.

---

## Block Descriptions

### Row 1 — Data Layer

**Raw Data**
- 32×32 pixel skin lesion images stored locally
- Tracked by DVC (801 files, local remote)
- Two classes: malignant and benign
- Git tracks the `.dvc` pointer file for reproducibility

**Airflow Pipeline**
- Apache Airflow 2.8.1 (SequentialExecutor + SQLite)
- 4-task DAG: `validate → split → preprocess → compute_baseline_stats`
- validate: checks corrupt/zero-byte files, minimum image counts
- split: stratified 70/15/15 train/val/test split, produces CSV manifests
- preprocess: resizes 32×32 → 64×64 RGB, normalises images
- compute_baseline_stats: computes pixel histogram for drift detection baseline
- Runs in Docker container on `mlops_network`

**Processed Data**
- 64×64 RGB images in train/val/test folders
- CSV manifests with filepath and label columns
- baseline_stats/v1_stats.json — pixel histogram baseline for drift detection
- All stored on local filesystem, mounted into Docker containers

---

### Row 2 — Training Layer

**MobileNetV3-Small**
- Pretrained on ImageNet, base layers frozen
- Only classifier head trained (592K trainable / 1.5M total params)
- Lightweight — suitable for CPU-only inference
- Tracked in MLflow experiment `baseline-v1`

**EfficientNet-B0**
- Pretrained on ImageNet, base layers frozen
- Only classifier head trained (2K trainable / 4M total params)
- Stronger feature extractor than MobileNet
- Tracked in MLflow experiment `baseline-v1`

**SimpleCNN**
- Custom CNN built from scratch — no pretrained weights
- 3 conv blocks + global average pooling + FC layers
- 102K parameters, all trainable
- Proves that resolution is the bottleneck, not model architecture
- Tracked in MLflow experiment `baseline-v1`

All models:
- Input: 64×64 RGB tensors
- Loss: CrossEntropyLoss with class weights [1.0, 2.0] (upweight malignant)
- Threshold: auto-tuned on validation set to maximise recall
- Early stopping: patience=3 on val_recall

---

### Row 3 — Model Registry

**MLflow Registry**
- Local MLflow registry stored in `mlruns/` folder
- Every training run logged: params, metrics (per epoch), artifacts
- Champion/Challenger auto-promotion (two-stage):
  - Stage 1: compare val_recall (primary metric)
  - Stage 2: compare val_F1 (tiebreaker when recall tied)
- Production model auto-loaded by FastAPI at startup
- MLflow UI accessible at `http://localhost:5000`

---

### Row 4 — Serving Layer

**FastAPI Backend**
- Python 3.10, FastAPI 0.111.0, Uvicorn
- 5 endpoints: `/predict`, `/health`, `/ready`, `/metrics`, `/feedback`
- Loads Production model from MLflow registry at startup
- Saves images to `logs/pending_feedback/` for retraining
- Detects drift via pixel histogram comparison
- Runs in Docker container on port 8000

**nginx Frontend**
- nginx:alpine serving static HTML/CSS/JS
- Two tabs: Screening and Pipeline Dashboard
- Communicates with backend via configurable REST API only
- No direct access to model or MLflow
- Runs in Docker container on port 80

**Docker Compose**
- 5 services on shared `mlops_network`:
  - melanoma_frontend (nginx, port 80)
  - melanoma_backend (FastAPI, port 8000)
  - melanoma_mlflow (MLflow UI, port 5000)
  - melanoma_prometheus (Prometheus, port 9090)
  - melanoma_grafana (Grafana, port 3000)
- All services use volume mounts for mlruns/, data/, logs/

---

### Row 5 — Monitoring Layer

**Prometheus**
- Scrapes `/metrics` endpoint from FastAPI every 15 seconds
- Stores time-series metrics locally
- 15 custom metrics tracked (requests, predictions, drift, recall, etc.)
- Accessible at `http://localhost:9090`

**Grafana**
- NRT dashboard reading from Prometheus
- 13 panels: request rate, latency, predictions, drift score,
  misclassification rate, real-world recall/precision, model loaded, etc.
- Auto-provisioned dashboard from `grafana/provisioning/`
- Accessible at `http://localhost:3000` (admin/admin)

**Drift Detection**
- Runs on every `/predict` call
- Computes pixel intensity histogram (10 bins) of incoming image
- Compares with baseline histogram (from Airflow pipeline)
- Drift score = mean absolute error between histograms
- If drift > 20%: writes `logs/retrain_needed.flag`

---

### Row 6 — Retraining Layer

**Retraining DAG**
- Apache Airflow DAG: `melanoma_retraining_pipeline`
- Runs automatically every 30 minutes
- 5 tasks: `check_trigger → prepare_feedback_data → trigger_training → evaluate_new_model → cleanup`
- check_trigger: reads `retrain_needed.flag`, stops if not present
- prepare_feedback_data: merges feedback images into training manifests
- trigger_training: retrains all 3 models via `train.py`
- evaluate_new_model: compares with current Production via MLflow
- cleanup: archives feedback images, removes flag

**Feedback Loop**
- Users (medical professionals) submit ground truth via `/feedback`
- Images moved from `pending_feedback/` to `feedback_data/<label>/`
- Confusion matrix updated: TP, FP, FN, TN
- Real-world recall = TP / (TP + FN)
- Misclassification rate = (FP + FN) / total
- When rate > 10% (after >= 10 submissions): retraining triggered

**Git + DVC**
- Git: tracks all code, configs, .dvc pointer files, MLproject
- DVC: tracks raw data (801 images), local remote storage
- Every experiment tied to git commit hash + MLflow run ID
- `dvc.yaml`: 5-stage CI pipeline DAG
- GitHub Actions: runs pytest + linting on every push

---

## Data Flow Summary

```
Raw Data (DVC) → Airflow Pipeline → Processed Data
                                          ↓
                              Training (3 models, MLflow)
                                          ↓
                              MLflow Registry (Production model)
                                          ↓
                              FastAPI Backend (serving)
                                    ↓         ↓
                              nginx UI    Prometheus
                                          ↓
                                       Grafana
                                          ↓
                              Drift/Misclass detected
                                          ↓
                              Airflow Retraining DAG
                                          ↓
                              (back to Training)
```
