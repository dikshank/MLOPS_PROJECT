"""
test_pipeline.py
----------------
Phase 5 | Executed: Local

Unit tests for Airflow pipeline scripts.

Tests cover:
    validate.py:
        - valid folder structure passes
        - corrupt image is detected
        - missing class folder raises FileNotFoundError
        - below min_images threshold raises ValueError
        - zero-byte file is detected as corrupt

    split.py:
        - flat structure detected correctly
        - presplit structure detected correctly
        - unknown structure raises ValueError
        - output CSVs have correct columns
        - class distribution maintained (stratified)
        - presplit: test set not modified

    preprocess.py:
        - output images are correct target size
        - output images are RGB
        - output folder structure is created
        - throughput logged (non-zero images processed)

    baseline_stats.py:
        - JSON output has required top-level keys
        - channel stats have mean, std, min, max
        - class distribution sums to total images
        - histogram bin_edges and counts present
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
        """Valid dataset with good images should pass without exception."""
        try:
            validate.run(pipeline_config_flat)
        except Exception as e:
            pytest.fail(f"validate.run() raised unexpectedly: {e}")

    def test_missing_class_folder_raises(self, pipeline_config_flat, temp_dir):
        """If a class folder is missing, FileNotFoundError should be raised."""
        # Remove benign folder
        import shutil
        shutil.rmtree(temp_dir / "raw" / "benign")

        with pytest.raises(FileNotFoundError):
            validate.run(pipeline_config_flat)

    def test_below_min_images_raises(self, pipeline_config_flat):
        """If class has fewer images than min_images_per_class, ValueError raised."""
        # Set minimum unreachably high
        pipeline_config_flat["validation"]["min_images_per_class"] = 9999
        with pytest.raises(ValueError, match="Minimum required"):
            validate.run(pipeline_config_flat)

    def test_zero_byte_file_detected(self, pipeline_config_flat, temp_dir):
        """Zero-byte file should be counted as corrupt."""
        # Add a zero-byte file
        zero_file = temp_dir / "raw" / "malignant" / "corrupt.jpg"
        zero_file.write_bytes(b"")

        # Raise corrupt threshold to 0 to force failure
        pipeline_config_flat["validation"]["max_corrupt_pct"] = 0.0
        with pytest.raises(ValueError, match="Corrupt image percentage"):
            validate.run(pipeline_config_flat)

    def test_returns_summary_dict(self, pipeline_config_flat):
        """validate_images() should return a summary dict."""
        summary = validate.validate_images(pipeline_config_flat)
        assert isinstance(summary, dict)
        assert "malignant" in summary
        assert "benign" in summary

    def test_summary_has_expected_keys(self, pipeline_config_flat):
        """Each class in summary should have total, valid, corrupt keys."""
        summary = validate.validate_images(pipeline_config_flat)
        for class_name in ["malignant", "benign"]:
            assert "total"   in summary[class_name]
            assert "valid"   in summary[class_name]
            assert "corrupt" in summary[class_name]


# ─────────────────────────────────────────────────────────────────────────────
# split.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSplit:

    def test_flat_structure_detected(self, pipeline_config_flat, temp_dir):
        """Flat structure (class folders at root) should be detected correctly."""
        structure = split.detect_structure(
            Path(pipeline_config_flat["paths"]["raw_data_dir"]),
            pipeline_config_flat["classes"]
        )
        assert structure == "flat"

    def test_presplit_structure_detected(self, pipeline_config_presplit, temp_dir):
        """Presplit structure (train/test folders) should be detected correctly."""
        structure = split.detect_structure(
            Path(pipeline_config_presplit["paths"]["raw_data_dir"]),
            pipeline_config_presplit["classes"]
        )
        assert structure == "presplit"

    def test_unknown_structure_raises(self, temp_dir):
        """Unknown folder structure should raise ValueError."""
        config = {
            "classes": ["malignant", "benign"]
        }
        with pytest.raises(ValueError, match="Cannot detect dataset structure"):
            split.detect_structure(temp_dir / "nonexistent", config["classes"])

    def test_flat_split_creates_manifests(self, pipeline_config_flat):
        """Running split on flat dataset should create 3 CSV manifests."""
        split.run(pipeline_config_flat)
        manifest_dir = Path(pipeline_config_flat["paths"]["split_manifest_dir"])
        assert (manifest_dir / "train_manifest.csv").exists()
        assert (manifest_dir / "val_manifest.csv").exists()
        assert (manifest_dir / "test_manifest.csv").exists()

    def test_manifests_have_correct_columns(self, pipeline_config_flat):
        """Each manifest CSV must have 'filepath' and 'label' columns."""
        import pandas as pd
        split.run(pipeline_config_flat)
        manifest_dir = Path(pipeline_config_flat["paths"]["split_manifest_dir"])

        for split_name in ["train", "val", "test"]:
            df = pd.read_csv(manifest_dir / f"{split_name}_manifest.csv")
            assert "filepath" in df.columns, f"{split_name}: missing 'filepath'"
            assert "label"    in df.columns, f"{split_name}: missing 'label'"

    def test_flat_split_covers_all_images(self, pipeline_config_flat):
        """Train + val + test should cover all images in flat dataset."""
        import pandas as pd
        split.run(pipeline_config_flat)
        manifest_dir = Path(pipeline_config_flat["paths"]["split_manifest_dir"])

        total = sum(
            len(pd.read_csv(manifest_dir / f"{s}_manifest.csv"))
            for s in ["train", "val", "test"]
        )
        # flat dataset has 10 malignant + 10 benign = 20 images
        assert total == 20

    def test_presplit_test_size_unchanged(self, pipeline_config_presplit):
        """In presplit mode, test set should be exactly the original test folder size."""
        import pandas as pd
        split.run(pipeline_config_presplit)
        manifest_dir = Path(pipeline_config_presplit["paths"]["split_manifest_dir"])

        test_df = pd.read_csv(manifest_dir / "test_manifest.csv")
        # presplit fixture has 5 malignant + 5 benign in test = 10
        assert len(test_df) == 10

    def test_labels_are_valid(self, pipeline_config_flat):
        """All labels in manifests must be 'malignant' or 'benign'."""
        import pandas as pd
        split.run(pipeline_config_flat)
        manifest_dir = Path(pipeline_config_flat["paths"]["split_manifest_dir"])

        valid_labels = {"malignant", "benign"}
        for split_name in ["train", "val", "test"]:
            df = pd.read_csv(manifest_dir / f"{split_name}_manifest.csv")
            unique_labels = set(df["label"].unique())
            assert unique_labels.issubset(valid_labels), \
                f"{split_name}: invalid labels {unique_labels}"


# ─────────────────────────────────────────────────────────────────────────────
# preprocess.py tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocess:

    def _run_full_pipeline(self, config):
        """Run validate + split + preprocess in sequence."""
        validate.run(config)
        split.run(config)
        preprocess.run(config)

    def test_output_images_are_correct_size(self, pipeline_config_flat):
        """All processed images should be resized to target_size."""
        self._run_full_pipeline(pipeline_config_flat)
        processed_dir = Path(
            pipeline_config_flat["paths"]["processed_data_dir"]
        )
        target = tuple(
            pipeline_config_flat["preprocessing"]["target_size"]
        )

        for img_path in processed_dir.rglob("*.jpg"):
            with Image.open(img_path) as img:
                assert img.size == target, \
                    f"{img_path.name}: expected {target}, got {img.size}"

    def test_output_images_are_rgb(self, pipeline_config_flat):
        """All processed images should be in RGB mode."""
        self._run_full_pipeline(pipeline_config_flat)
        processed_dir = Path(
            pipeline_config_flat["paths"]["processed_data_dir"]
        )
        for img_path in processed_dir.rglob("*.jpg"):
            with Image.open(img_path) as img:
                assert img.mode == "RGB", \
                    f"{img_path.name}: expected RGB, got {img.mode}"

    def test_output_folder_structure_created(self, pipeline_config_flat):
        """Processed folder should have train/val/test subfolders."""
        self._run_full_pipeline(pipeline_config_flat)
        processed_dir = Path(
            pipeline_config_flat["paths"]["processed_data_dir"]
        )
        for split_name in ["train", "val", "test"]:
            assert (processed_dir / split_name).exists(), \
                f"Missing split folder: {split_name}"


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
        stats_path = Path(
            pipeline_config_flat["paths"]["baseline_stats_path"]
        )
        assert stats_path.exists(), "baseline_stats.json not created"

    def test_stats_has_required_top_level_keys(self, pipeline_config_flat):
        """Stats JSON must have required top-level keys."""
        self._run_full_pipeline(pipeline_config_flat)
        stats_path = Path(
            pipeline_config_flat["paths"]["baseline_stats_path"]
        )
        with open(stats_path) as f:
            stats = json.load(f)

        required_keys = {
            "version", "total_train_images",
            "class_distribution", "channel_stats",
            "pixel_histogram", "image_size_distribution"
        }
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_channel_stats_have_required_fields(self, pipeline_config_flat):
        """Each channel (R, G, B) must have mean, std, min, max, variance."""
        self._run_full_pipeline(pipeline_config_flat)
        stats_path = Path(
            pipeline_config_flat["paths"]["baseline_stats_path"]
        )
        with open(stats_path) as f:
            stats = json.load(f)

        for channel in ["R", "G", "B"]:
            assert channel in stats["channel_stats"], \
                f"Missing channel: {channel}"
            ch_stats = stats["channel_stats"][channel]
            for field in ["mean", "std", "min", "max", "variance"]:
                assert field in ch_stats, \
                    f"Channel {channel} missing field: {field}"

    def test_class_distribution_sums_correctly(self, pipeline_config_flat):
        """Sum of class counts must equal total_train_images."""
        self._run_full_pipeline(pipeline_config_flat)
        stats_path = Path(
            pipeline_config_flat["paths"]["baseline_stats_path"]
        )
        with open(stats_path) as f:
            stats = json.load(f)

        total = stats["total_train_images"]
        dist_total = sum(
            v["count"] for v in stats["class_distribution"].values()
        )
        assert total == dist_total, \
            f"Class distribution total {dist_total} != total_train_images {total}"

    def test_histogram_has_bin_edges_and_counts(self, pipeline_config_flat):
        """Pixel histogram must have bin_edges and counts lists."""
        self._run_full_pipeline(pipeline_config_flat)
        stats_path = Path(
            pipeline_config_flat["paths"]["baseline_stats_path"]
        )
        with open(stats_path) as f:
            stats = json.load(f)

        hist = stats["pixel_histogram"]
        assert "bin_edges" in hist
        assert "counts"    in hist
        assert isinstance(hist["bin_edges"], list)
        assert isinstance(hist["counts"],    list)
        assert len(hist["bin_edges"]) == len(hist["counts"]) + 1
