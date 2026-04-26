"""
test_pipeline.py
----------------
Phase 5 | Executed: Local

Unit tests for Airflow pipeline scripts.
"""

import json
import sys
import pytest
from pathlib import Path
from PIL import Image

# ── Add airflow scripts to path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "airflow" / "scripts"))

import validate
import split
import preprocess
import baseline_stats


# ─────────────────────────────────────────────────────────────────────────────
# validate.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidate:

    def test_valid_dataset_passes(self, pipeline_config_flat):
        """Valid dataset should pass without exception."""
        try:
            validate.run(pipeline_config_flat)
        except Exception as e:
            pytest.fail(f"validate.run() raised unexpectedly: {e}")

    def test_missing_class_folder_raises(self, pipeline_config_flat, temp_dir):
        """Missing class folder should raise an error."""
        import shutil
        shutil.rmtree(temp_dir / "raw" / "benign")
        with pytest.raises((FileNotFoundError, ValueError)):
            validate.run(pipeline_config_flat)

    def test_below_min_images_raises(self, pipeline_config_flat):
        """Below min_images_per_class threshold should raise ValueError."""
        pipeline_config_flat["validation"]["min_images_per_class"] = 9999
        with pytest.raises(ValueError, match="Minimum required"):
            validate.run(pipeline_config_flat)

    def test_zero_byte_file_detected(self, pipeline_config_flat, temp_dir):
        """Zero-byte file should be counted as corrupt and trigger threshold."""
        zero_file = temp_dir / "raw" / "malignant" / "corrupt.jpg"
        zero_file.write_bytes(b"")
        pipeline_config_flat["validation"]["max_corrupt_pct"] = 0.0
        with pytest.raises(ValueError, match="Corrupt"):
            validate.run(pipeline_config_flat)

    def test_returns_summary_dict(self, pipeline_config_flat):
        """validate_images() should return a summary dict with class entries."""
        summary = validate.validate_images(pipeline_config_flat)
        assert isinstance(summary, dict)
        # Keys are in format "root/malignant" or "train/malignant" etc.
        keys = list(summary.keys())
        assert any("malignant" in k for k in keys)
        assert any("benign" in k for k in keys)

    def test_summary_has_expected_keys(self, pipeline_config_flat):
        """Each entry in summary should have total, valid, corrupt keys."""
        summary = validate.validate_images(pipeline_config_flat)
        for key, value in summary.items():
            assert "total"   in value, f"{key}: missing 'total'"
            assert "valid"   in value, f"{key}: missing 'valid'"
            assert "corrupt" in value, f"{key}: missing 'corrupt'"


# ─────────────────────────────────────────────────────────────────────────────
# split.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSplit:

    def test_flat_structure_detected(self, pipeline_config_flat, temp_dir):
        """Flat structure should be detected correctly."""
        structure = split.detect_structure(
            Path(pipeline_config_flat["paths"]["raw_data_dir"]),
            pipeline_config_flat["classes"]
        )
        assert structure == "flat"

    def test_presplit_structure_detected(self, pipeline_config_presplit, temp_dir):
        """Presplit structure should be detected correctly."""
        structure = split.detect_structure(
            Path(pipeline_config_presplit["paths"]["raw_data_dir"]),
            pipeline_config_presplit["classes"]
        )
        assert structure == "presplit"

    def test_unknown_structure_raises(self, temp_dir):
        """Unknown folder structure should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot detect dataset structure"):
            split.detect_structure(temp_dir / "nonexistent", ["malignant", "benign"])

    def test_flat_split_creates_manifests(self, pipeline_config_flat):
        """Running split on flat dataset should create 3 CSV manifests."""
        split.run(pipeline_config_flat)
        manifest_dir = Path(pipeline_config_flat["paths"]["split_manifest_dir"])
        assert (manifest_dir / "train_manifest.csv").exists()
        assert (manifest_dir / "val_manifest.csv").exists()
        assert (manifest_dir / "test_manifest.csv").exists()

    def test_manifests_have_correct_columns(self, pipeline_config_flat):
        """Each manifest CSV must have filepath and label columns."""
        import pandas as pd
        split.run(pipeline_config_flat)
        manifest_dir = Path(pipeline_config_flat["paths"]["split_manifest_dir"])
        for split_name in ["train", "val", "test"]:
            df = pd.read_csv(manifest_dir / f"{split_name}_manifest.csv")
            assert "filepath" in df.columns
            assert "label"    in df.columns

    def test_flat_split_covers_all_images(self, pipeline_config_flat):
        """Train + val + test should cover all 20 images."""
        import pandas as pd
        split.run(pipeline_config_flat)
        manifest_dir = Path(pipeline_config_flat["paths"]["split_manifest_dir"])
        total = sum(
            len(pd.read_csv(manifest_dir / f"{s}_manifest.csv"))
            for s in ["train", "val", "test"]
        )
        assert total == 20

    def test_presplit_test_size_unchanged(self, pipeline_config_presplit):
        """In presplit mode, test set should match original test folder (10 images)."""
        import pandas as pd
        split.run(pipeline_config_presplit)
        manifest_dir = Path(pipeline_config_presplit["paths"]["split_manifest_dir"])
        test_df = pd.read_csv(manifest_dir / "test_manifest.csv")
        assert len(test_df) == 10

    def test_labels_are_valid(self, pipeline_config_flat):
        """All labels in manifests must be malignant or benign."""
        import pandas as pd
        split.run(pipeline_config_flat)
        manifest_dir = Path(pipeline_config_flat["paths"]["split_manifest_dir"])
        valid_labels = {"malignant", "benign"}
        for split_name in ["train", "val", "test"]:
            df = pd.read_csv(manifest_dir / f"{split_name}_manifest.csv")
            assert set(df["label"].unique()).issubset(valid_labels)


# ─────────────────────────────────────────────────────────────────────────────
# preprocess.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocess:

    def _run_full_pipeline(self, config):
        validate.run(config)
        split.run(config)
        preprocess.run(config)

    def test_output_images_are_correct_size(self, pipeline_config_flat):
        """All processed images should be resized to target_size."""
        self._run_full_pipeline(pipeline_config_flat)
        processed_dir = Path(pipeline_config_flat["paths"]["processed_data_dir"])
        target = tuple(pipeline_config_flat["preprocessing"]["target_size"])
        for img_path in processed_dir.rglob("*.jpg"):
            with Image.open(img_path) as img:
                assert img.size == target

    def test_output_images_are_rgb(self, pipeline_config_flat):
        """All processed images should be in RGB mode."""
        self._run_full_pipeline(pipeline_config_flat)
        processed_dir = Path(pipeline_config_flat["paths"]["processed_data_dir"])
        for img_path in processed_dir.rglob("*.jpg"):
            with Image.open(img_path) as img:
                assert img.mode == "RGB"

    def test_output_folder_structure_created(self, pipeline_config_flat):
        """Processed folder should have train/val/test subfolders."""
        self._run_full_pipeline(pipeline_config_flat)
        processed_dir = Path(pipeline_config_flat["paths"]["processed_data_dir"])
        for split_name in ["train", "val", "test"]:
            assert (processed_dir / split_name).exists()


# ─────────────────────────────────────────────────────────────────────────────
# baseline_stats.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselineStats:

    def _run_full_pipeline(self, config):
        validate.run(config)
        split.run(config)
        preprocess.run(config)
        baseline_stats.run(config)

    def test_stats_json_created(self, pipeline_config_flat):
        """Baseline stats JSON file should be created."""
        self._run_full_pipeline(pipeline_config_flat)
        stats_path = Path(pipeline_config_flat["paths"]["baseline_stats_path"])
        assert stats_path.exists()

    def test_stats_has_required_top_level_keys(self, pipeline_config_flat):
        """Stats JSON must have required top-level keys."""
        self._run_full_pipeline(pipeline_config_flat)
        with open(pipeline_config_flat["paths"]["baseline_stats_path"]) as f:
            stats = json.load(f)
        for key in ["version", "total_train_images", "class_distribution",
                    "channel_stats", "pixel_histogram"]:
            assert key in stats

    def test_channel_stats_have_required_fields(self, pipeline_config_flat):
        """Each channel R/G/B must have mean, std, min, max, variance."""
        self._run_full_pipeline(pipeline_config_flat)
        with open(pipeline_config_flat["paths"]["baseline_stats_path"]) as f:
            stats = json.load(f)
        for channel in ["R", "G", "B"]:
            assert channel in stats["channel_stats"]
            for field in ["mean", "std", "min", "max", "variance"]:
                assert field in stats["channel_stats"][channel]

    def test_class_distribution_sums_correctly(self, pipeline_config_flat):
        """Sum of class counts must equal total_train_images."""
        self._run_full_pipeline(pipeline_config_flat)
        with open(pipeline_config_flat["paths"]["baseline_stats_path"]) as f:
            stats = json.load(f)
        total = stats["total_train_images"]
        dist_total = sum(v["count"] for v in stats["class_distribution"].values())
        assert total == dist_total

    def test_histogram_has_bin_edges_and_counts(self, pipeline_config_flat):
        """Pixel histogram must have bin_edges and counts lists."""
        self._run_full_pipeline(pipeline_config_flat)
        with open(pipeline_config_flat["paths"]["baseline_stats_path"]) as f:
            stats = json.load(f)
        hist = stats["pixel_histogram"]
        assert "bin_edges" in hist
        assert "counts"    in hist
        assert isinstance(hist["bin_edges"], list)
        assert isinstance(hist["counts"],    list)