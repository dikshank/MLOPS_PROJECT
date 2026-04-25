"""
preprocess.py
-------------
Phase 1 & 2 | Executed: Local (inside Airflow Docker container)

Airflow Task 3: Preprocess images based on split manifests.

Responsibilities:
- Read train/val/test manifests produced by split.py
- Resize all images to target_size from config
- Convert all images to RGB (handles grayscale edge cases)
- Save processed images to processed_data_dir maintaining train/val/test structure
- Log throughput (images per second) and total processing time

Output structure:
    processed/v1/
        train/
            malignant/
            benign/
        val/
            malignant/
            benign/
        test/
            malignant/
            benign/
"""

import time
import logging
from pathlib import Path
from PIL import Image
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("preprocess")


def process_split(
    manifest_path: Path,
    split_name: str,
    output_dir: Path,
    target_size: tuple,
    convert_to_rgb: bool
) -> dict:
    """
    Process all images for a single split (train, val, or test).

    Args:
        manifest_path (Path): Path to the split CSV manifest.
        split_name (str): One of 'train', 'val', 'test'.
        output_dir (Path): Root processed output directory.
        target_size (tuple): Target (width, height) for resizing.
        convert_to_rgb (bool): Whether to convert images to RGB mode.

    Returns:
        dict: Processing summary with counts and throughput.
    """
    df = pd.read_csv(manifest_path)
    total = len(df)
    processed_count = 0
    failed_files = []

    start_time = time.time()

    for _, row in df.iterrows():
        src_path = Path(row["filepath"])
        label = row["label"]

        # ── Build output path maintaining class subfolder structure ───────
        out_dir = output_dir / split_name / label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / src_path.name

        try:
            with Image.open(src_path) as img:
                # ── Convert to RGB if needed ──────────────────────────────
                if convert_to_rgb and img.mode != "RGB":
                    img = img.convert("RGB")

                # ── Resize to target size ─────────────────────────────────
                img_resized = img.resize(target_size, Image.LANCZOS)

                # ── Save processed image ──────────────────────────────────
                img_resized.save(out_path)
                processed_count += 1

        except Exception as e:
            logger.warning(
                "Failed to process [%s]: %s", src_path.name, str(e)
            )
            failed_files.append(str(src_path))

    elapsed = time.time() - start_time
    throughput = processed_count / elapsed if elapsed > 0 else 0

    summary = {
        "split": split_name,
        "total": total,
        "processed": processed_count,
        "failed": len(failed_files),
        "elapsed_seconds": round(elapsed, 2),
        "throughput_imgs_per_sec": round(throughput, 2),
        "failed_files": failed_files
    }

    logger.info(
        "Split [%s] → Processed: %d/%d | Failed: %d | "
        "Time: %.2fs | Throughput: %.1f imgs/sec",
        split_name, processed_count, total,
        len(failed_files), elapsed, throughput
    )

    return summary


def run(config: dict) -> None:
    """
    Entry point called by Airflow PythonOperator.

    Args:
        config (dict): Parsed pipeline config YAML as a dictionary.
    """
    try:
        manifest_dir = Path(config["paths"]["split_manifest_dir"])
        processed_data_dir = Path(config["paths"]["processed_data_dir"])
        target_size = tuple(config["preprocessing"]["target_size"])
        convert_to_rgb = config["preprocessing"]["convert_to_rgb"]

        splits = ["train", "val", "test"]
        all_summaries = {}

        for split_name in splits:
            manifest_path = manifest_dir / f"{split_name}_manifest.csv"

            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Manifest not found: {manifest_path}. "
                    f"Run split.py first."
                )

            summary = process_split(
                manifest_path=manifest_path,
                split_name=split_name,
                output_dir=processed_data_dir,
                target_size=target_size,
                convert_to_rgb=convert_to_rgb
            )
            all_summaries[split_name] = summary

        # ── Overall throughput summary ────────────────────────────────────
        total_images = sum(s["processed"] for s in all_summaries.values())
        total_time = sum(s["elapsed_seconds"] for s in all_summaries.values())
        overall_throughput = total_images / total_time if total_time > 0 else 0

        logger.info(
            "✅ Preprocessing complete. "
            "Total: %d images | Total time: %.2fs | "
            "Overall throughput: %.1f imgs/sec",
            total_images, total_time, overall_throughput
        )

    except Exception as e:
        logger.error("❌ Preprocessing failed: %s", str(e))
        raise
