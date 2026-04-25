"""
baseline_stats.py
-----------------
Phase 1 & 2 | Executed: Local (inside Airflow Docker container)

Airflow Task 4: Compute and save baseline statistics for drift detection.

Responsibilities:
- Compute per-channel mean, variance, min, max across the full processed dataset
- Compute class distribution (count and percentage per class)
- Compute image size distribution (sanity check all images are correct size)
- Compute pixel intensity distribution (histogram buckets)
- Save all statistics as a JSON file to baseline_stats_path

This JSON is later used by Prometheus/monitoring to detect data drift
when new inference requests come in.

Output:
    baseline_stats/v1_stats.json (or v2_stats.json)
"""

import json
import logging
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("baseline_stats")

# Number of histogram bins for pixel intensity distribution
HISTOGRAM_BINS = 10


def collect_image_arrays(processed_data_dir: Path, classes: list) -> tuple:
    """
    Load all processed images from train split into numpy arrays.
    Uses train split only — val/test should not influence baselines.

    Args:
        processed_data_dir (Path): Root of processed data output.
        classes (list): Class names to iterate over.

    Returns:
        tuple: (pixel_arrays, class_counts)
            pixel_arrays: list of flattened float32 numpy arrays (one per image)
            class_counts: dict mapping class_name -> count
    """
    pixel_arrays = []
    class_counts = defaultdict(int)

    train_dir = processed_data_dir / "train"

    for class_name in classes:
        class_dir = train_dir / class_name

        if not class_dir.exists():
            logger.warning("Class directory not found: %s — skipping", class_dir)
            continue

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            try:
                with Image.open(img_path) as img:
                    arr = np.array(img, dtype=np.float32) / 255.0
                    pixel_arrays.append(arr)
                    class_counts[class_name] += 1

            except Exception as e:
                logger.warning("Could not read [%s]: %s", img_path.name, str(e))

    logger.info(
        "Loaded %d images for baseline computation. Class counts: %s",
        len(pixel_arrays), dict(class_counts)
    )
    return pixel_arrays, dict(class_counts)


def compute_channel_stats(pixel_arrays: list) -> dict:
    """
    Compute per-channel (R, G, B) statistics across all images.

    Args:
        pixel_arrays (list): List of numpy arrays with shape (H, W, C),
                             values normalized to [0, 1].

    Returns:
        dict: Per-channel mean, variance, std, min, max.
    """
    # Stack all images: shape (N, H, W, C)
    stacked = np.stack(pixel_arrays, axis=0)

    channel_stats = {}
    channel_names = ["R", "G", "B"]

    for i, ch in enumerate(channel_names):
        channel_data = stacked[:, :, :, i].flatten()
        channel_stats[ch] = {
            "mean": float(np.mean(channel_data)),
            "variance": float(np.var(channel_data)),
            "std": float(np.std(channel_data)),
            "min": float(np.min(channel_data)),
            "max": float(np.max(channel_data))
        }
        logger.info(
            "Channel [%s] → mean=%.4f std=%.4f min=%.4f max=%.4f",
            ch,
            channel_stats[ch]["mean"],
            channel_stats[ch]["std"],
            channel_stats[ch]["min"],
            channel_stats[ch]["max"]
        )

    return channel_stats


def compute_pixel_histogram(pixel_arrays: list) -> dict:
    """
    Compute a histogram of pixel intensity values across all images and channels.
    Used later to detect distributional drift on incoming inference images.

    Args:
        pixel_arrays (list): List of numpy arrays, values in [0, 1].

    Returns:
        dict: Histogram bin edges and counts.
    """
    all_pixels = np.concatenate(
        [arr.flatten() for arr in pixel_arrays]
    )

    counts, bin_edges = np.histogram(all_pixels, bins=HISTOGRAM_BINS, range=(0.0, 1.0))

    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "total_pixels": int(len(all_pixels))
    }


def compute_image_size_distribution(processed_data_dir: Path, classes: list) -> dict:
    """
    Verify all images are the expected size and report any anomalies.

    Args:
        processed_data_dir (Path): Root of processed data.
        classes (list): Class names.

    Returns:
        dict: Size distribution mapping "WxH" -> count.
    """
    size_counts = defaultdict(int)
    train_dir = processed_data_dir / "train"

    for class_name in classes:
        class_dir = train_dir / class_name
        if not class_dir.exists():
            continue

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                with Image.open(img_path) as img:
                    size_key = f"{img.width}x{img.height}"
                    size_counts[size_key] += 1
            except Exception:
                pass

    return dict(size_counts)


def run(config: dict) -> None:
    """
    Entry point called by Airflow PythonOperator.

    Args:
        config (dict): Parsed pipeline config YAML as a dictionary.
    """
    try:
        processed_data_dir = Path(config["paths"]["processed_data_dir"])
        baseline_stats_path = Path(config["paths"]["baseline_stats_path"])
        classes = config["classes"]
        version = config["version"]

        # ── Load images ───────────────────────────────────────────────────
        pixel_arrays, class_counts = collect_image_arrays(processed_data_dir, classes)

        if len(pixel_arrays) == 0:
            raise ValueError(
                "No images found in processed_data_dir train split. "
                "Run preprocess.py first."
            )

        # ── Compute stats ─────────────────────────────────────────────────
        channel_stats = compute_channel_stats(pixel_arrays)
        pixel_histogram = compute_pixel_histogram(pixel_arrays)
        size_distribution = compute_image_size_distribution(processed_data_dir, classes)

        # ── Class distribution percentages ────────────────────────────────
        total_images = sum(class_counts.values())
        class_distribution = {
            cls: {
                "count": count,
                "percentage": round(count / total_images * 100, 2)
            }
            for cls, count in class_counts.items()
        }

        # ── Assemble final stats object ───────────────────────────────────
        stats = {
            "version": version,
            "total_train_images": total_images,
            "class_distribution": class_distribution,
            "channel_stats": channel_stats,
            "pixel_histogram": pixel_histogram,
            "image_size_distribution": size_distribution
        }

        # ── Save to JSON ──────────────────────────────────────────────────
        baseline_stats_path.parent.mkdir(parents=True, exist_ok=True)

        with open(baseline_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("✅ Baseline stats saved to: %s", baseline_stats_path)
        logger.info("Class distribution: %s", class_distribution)

    except Exception as e:
        logger.error("❌ Baseline stats computation failed: %s", str(e))
        raise
