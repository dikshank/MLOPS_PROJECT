"""
conftest.py
-----------
Phase 5 | Executed: Local

Shared pytest fixtures used across all test files.
"""

import io
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch


# ── Add backend to path ───────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_image_bytes(width: int = 64, height: int = 64, fmt: str = "JPEG") -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ── FastAPI client fixtures ───────────────────────────────────────────────────

@pytest.fixture
def client_with_model():
    """FastAPI test client with a mocked loaded model."""
    import torch
    from fastapi.testclient import TestClient

    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.3, 0.7]])

    with patch("model_loader._model", mock_model), \
         patch("model_loader._model_meta", {
             "version": "1",
             "name": "mobilenet_v3_small",
             "run_id": "test_run_id",
             "threshold": 0.35
         }), \
         patch("model_loader.load_model", return_value=True), \
         patch("main._load_baseline_histogram", return_value=None):
        from main import app
        with TestClient(app) as client:
            yield client


@pytest.fixture
def client_no_model():
    """FastAPI test client with NO model loaded."""
    from fastapi.testclient import TestClient

    with patch("model_loader._model", None), \
         patch("model_loader._model_meta", {
             "version": None,
             "name": None,
             "run_id": None,
             "threshold": 0.35
         }), \
         patch("model_loader.load_model", return_value=False), \
         patch("main._load_baseline_histogram", return_value=None):
        from main import app
        with TestClient(app) as client:
            yield client


# ── Image fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def valid_jpeg():
    return make_image_bytes(64, 64, "JPEG")

@pytest.fixture
def valid_png():
    return make_image_bytes(64, 64, "PNG")

@pytest.fixture
def tiny_image():
    return make_image_bytes(1, 1, "JPEG")

@pytest.fixture
def large_image():
    return make_image_bytes(1000, 1000, "JPEG")

@pytest.fixture
def empty_bytes():
    return b""

@pytest.fixture
def pdf_bytes():
    return b"%PDF-1.4 fake pdf content"


# ── Pipeline fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def flat_dataset(temp_dir):
    """Flat dataset: temp_dir/raw/malignant/ and temp_dir/raw/benign/"""
    for class_name in ["malignant", "benign"]:
        class_dir = temp_dir / "raw" / class_name
        class_dir.mkdir(parents=True)
        for i in range(10):
            img = Image.new("RGB", (32, 32), color=(i * 20, 80, 60))
            img.save(class_dir / f"img_{i:03d}.jpg")
    return temp_dir


@pytest.fixture
def presplit_dataset(temp_dir):
    """Presplit dataset: train/test with malignant/benign subfolders."""
    for split in ["train", "test"]:
        count = 10 if split == "train" else 5
        for class_name in ["malignant", "benign"]:
            class_dir = temp_dir / "raw" / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(count):
                img = Image.new("RGB", (32, 32), color=(i * 20, 80, 60))
                img.save(class_dir / f"img_{i:03d}.jpg")
    return temp_dir


@pytest.fixture
def pipeline_config_flat(temp_dir, flat_dataset):
    """Config dict for flat dataset pipeline tests."""
    # Create required directories
    (temp_dir / "manifests").mkdir(exist_ok=True)
    (temp_dir / "processed").mkdir(exist_ok=True)
    (temp_dir / "stats").mkdir(exist_ok=True)
    return {
        "version": "test_v1",
        "paths": {
            "raw_data_dir":        str(temp_dir / "raw"),
            "processed_data_dir":  str(temp_dir / "processed"),
            "baseline_stats_path": str(temp_dir / "stats" / "stats.json"),
            "split_manifest_dir":  str(temp_dir / "manifests")
        },
        "classes": ["malignant", "benign"],
        "valid_extensions": [".jpg", ".jpeg", ".png"],
        "preprocessing": {
            "target_size": [32, 32],
            "convert_to_rgb": True
        },
        "split": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
            "stratified": True,
            "random_seed": 42
        },
        "validation": {
            "min_images_per_class": 5,
            "max_corrupt_pct": 0.05
        }
    }


@pytest.fixture
def pipeline_config_presplit(temp_dir, presplit_dataset):
    """Config dict for presplit dataset pipeline tests."""
    (temp_dir / "manifests").mkdir(exist_ok=True)
    (temp_dir / "processed").mkdir(exist_ok=True)
    (temp_dir / "stats").mkdir(exist_ok=True)
    return {
        "version": "test_v1",
        "paths": {
            "raw_data_dir":        str(temp_dir / "raw"),
            "processed_data_dir":  str(temp_dir / "processed"),
            "baseline_stats_path": str(temp_dir / "stats" / "stats.json"),
            "split_manifest_dir":  str(temp_dir / "manifests")
        },
        "classes": ["malignant", "benign"],
        "valid_extensions": [".jpg", ".jpeg", ".png"],
        "preprocessing": {
            "target_size": [32, 32],
            "convert_to_rgb": True
        },
        "split": {
            "train": 0.70,
            "val": 0.15,
            "test": 0.15,
            "stratified": True,
            "random_seed": 42
        },
        "validation": {
            "min_images_per_class": 3,
            "max_corrupt_pct": 0.05
        }
    }