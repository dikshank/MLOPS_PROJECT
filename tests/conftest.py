"""
conftest.py
-----------
Phase 5 | Executed: Local

Shared pytest fixtures used across all test files.

Provides:
    - FastAPI test client
    - Synthetic test images (valid JPEG, PNG, tiny, large)
    - Temporary directory for pipeline tests
    - Mock config dictionaries
"""

import io
import pytest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_image_bytes(width: int = 64, height: int = 64, fmt: str = "JPEG") -> bytes:
    """
    Generate synthetic image bytes for testing.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        fmt (str): Image format — 'JPEG' or 'PNG'.

    Returns:
        bytes: Raw image bytes.
    """
    img = Image.new("RGB", (width, height), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ── FastAPI client fixtures ───────────────────────────────────────────────────

@pytest.fixture
def client_with_model():
    """
    FastAPI test client with a mocked loaded model.
    Use for tests that require the model to be present.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

    # Mock the model so we don't need actual weights
    mock_model = MagicMock()

    import torch
    # Return fake logits: slightly higher malignant score
    mock_model.return_value = torch.tensor([[0.3, 0.7]])

    with patch("model_loader._model", mock_model), \
         patch("model_loader._model_meta", {
             "version": "1",
             "name": "mobilenet_v3_small",
             "run_id": "test_run_id",
             "threshold": 0.35
         }):
        from main import app
        with TestClient(app) as client:
            yield client


@pytest.fixture
def client_no_model():
    """
    FastAPI test client with NO model loaded.
    Use for tests that check 503 behaviour.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

    with patch("model_loader._model", None), \
         patch("model_loader._model_meta", {
             "version": None,
             "name": None,
             "run_id": None,
             "threshold": 0.35
         }):
        from main import app
        with TestClient(app) as client:
            yield client


# ── Image fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def valid_jpeg():
    """Standard 64x64 JPEG image bytes."""
    return make_image_bytes(64, 64, "JPEG")


@pytest.fixture
def valid_png():
    """Standard 64x64 PNG image bytes."""
    return make_image_bytes(64, 64, "PNG")


@pytest.fixture
def tiny_image():
    """1x1 pixel JPEG — edge case."""
    return make_image_bytes(1, 1, "JPEG")


@pytest.fixture
def large_image():
    """1000x1000 pixel JPEG — should be resized correctly."""
    return make_image_bytes(1000, 1000, "JPEG")


@pytest.fixture
def empty_bytes():
    """Zero-byte content."""
    return b""


@pytest.fixture
def pdf_bytes():
    """Fake PDF bytes to test wrong file type rejection."""
    return b"%PDF-1.4 fake pdf content"


# ── Pipeline fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    """
    Temporary directory for pipeline tests.
    Automatically cleaned up after each test.
    """
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def flat_dataset(temp_dir):
    """
    Create a minimal flat dataset structure for pipeline tests.

    Structure:
        temp_dir/raw/malignant/ (10 images)
        temp_dir/raw/benign/    (10 images)
    """
    for class_name in ["malignant", "benign"]:
        class_dir = temp_dir / "raw" / class_name
        class_dir.mkdir(parents=True)
        for i in range(10):
            img = Image.new("RGB", (32, 32), color=(i * 20, 80, 60))
            img.save(class_dir / f"img_{i:03d}.jpg")

    return temp_dir


@pytest.fixture
def presplit_dataset(temp_dir):
    """
    Create a minimal presplit dataset structure for pipeline tests.

    Structure:
        temp_dir/raw/train/malignant/ (10 images)
        temp_dir/raw/train/benign/    (10 images)
        temp_dir/raw/test/malignant/  (5 images)
        temp_dir/raw/test/benign/     (5 images)
    """
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
