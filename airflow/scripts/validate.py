"""
validate.py
-----------
Phase 1 & 2 | Executed: Local (inside Airflow Docker container)

Airflow Task 1: Validate raw images before any processing.

Responsibilities:
- Check every file in raw_data_dir can be opened as a valid image
- Check file extensions are in the allowed list
- Check for zero-byte / corrupt files
- Check minimum image count per class is met
- Log counts of valid vs rejected images
- Fail the task if corrupt percentage exceeds threshold

Raises:
    ValueError: If corrupt image percentage exceeds max_corrupt_pct
    ValueError: If any class has fewer images than min_images_per_class
"""

import os
import logging
from pathlib import Path
from PIL import Image

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("validate")


def validate_images(config: dict) -> dict:
    """
    Validate all raw images in the configured data directory.

    Args:
        config (dict): Parsed pipeline config YAML as a dictionary.

    Returns:
        dict: Validation summary with counts per class.

    Raises:
        ValueError: If validation thresholds are breached.
    """
    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    classes = config["classes"]
    valid_extensions = set(config["valid_extensions"])
    min_images = config["validation"]["min_images_per_class"]
    max_corrupt_pct = config["validation"]["max_corrupt_pct"]

    logger.info("Starting image validation at: %s", raw_data_dir)

    summary = {}

    for class_name in classes:
        class_dir = raw_data_dir / class_name

        # ── Check class folder exists ─────────────────────────────────────
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class folder not found: {class_dir}. "
                f"Expected folders: {classes}"
            )

        all_files = list(class_dir.iterdir())
        total = len(all_files)
        valid_count = 0
        corrupt_files = []

        for file_path in all_files:
            # ── Check extension ───────────────────────────────────────────
            if file_path.suffix.lower() not in valid_extensions:
                logger.warning("Skipping unsupported extension: %s", file_path.name)
                corrupt_files.append(str(file_path))
                continue

            # ── Check zero-byte files ─────────────────────────────────────
            if file_path.stat().st_size == 0:
                logger.warning("Zero-byte file detected: %s", file_path.name)
                corrupt_files.append(str(file_path))
                continue

            # ── Try opening as image ──────────────────────────────────────
            try:
                with Image.open(file_path) as img:
                    img.verify()  # verify does not decode, just checks header
                valid_count += 1

            except Exception as e:
                logger.warning("Corrupt image [%s]: %s", file_path.name, str(e))
                corrupt_files.append(str(file_path))

        # ── Compute corrupt percentage ────────────────────────────────────
        corrupt_count = len(corrupt_files)
        corrupt_pct = corrupt_count / total if total > 0 else 1.0

        logger.info(
            "Class [%s] → Total: %d | Valid: %d | Corrupt: %d (%.1f%%)",
            class_name, total, valid_count, corrupt_count, corrupt_pct * 100
        )

        # ── Enforce thresholds ────────────────────────────────────────────
        if corrupt_pct > max_corrupt_pct:
            raise ValueError(
                f"Class [{class_name}]: Corrupt image percentage "
                f"({corrupt_pct:.1%}) exceeds threshold ({max_corrupt_pct:.1%}). "
                f"Corrupt files: {corrupt_files[:5]} ..."
            )

        if valid_count < min_images:
            raise ValueError(
                f"Class [{class_name}]: Only {valid_count} valid images found. "
                f"Minimum required: {min_images}."
            )

        summary[class_name] = {
            "total": total,
            "valid": valid_count,
            "corrupt": corrupt_count,
            "corrupt_pct": round(corrupt_pct * 100, 2),
            "corrupt_files": corrupt_files
        }

    logger.info("Validation complete. Summary: %s", summary)
    return summary


def run(config: dict) -> None:
    """
    Entry point called by Airflow PythonOperator.

    Args:
        config (dict): Parsed pipeline config YAML as a dictionary.
    """
    try:
        summary = validate_images(config)
        logger.info("✅ Validation passed. Summary: %s", summary)

    except (ValueError, FileNotFoundError) as e:
        logger.error("❌ Validation failed: %s", str(e))
        raise  # re-raise so Airflow marks task as failed
