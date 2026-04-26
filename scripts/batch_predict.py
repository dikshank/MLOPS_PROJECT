"""
batch_predict.py
----------------
Sends images from two folders to /predict and /feedback endpoints.
Simulates real-world usage with known ground truth labels.

Usage:
    python scripts/batch_predict.py \
        --malignant path/to/malignant/folder \
        --benign path/to/benign/folder \
        --count 100 \
        --api http://localhost:8000
"""

import os
import time
import argparse
import requests
from pathlib import Path


def process_images(folder: Path, true_label: str, count: int, api_base: str):
    images = list(folder.glob("*.jpg")) + \
             list(folder.glob("*.jpeg")) + \
             list(folder.glob("*.png"))

    if not images:
        print(f"No images found in {folder}")
        return 0, 0

    images = images[:count]
    print(f"\n{'='*50}")
    print(f"Processing {len(images)} {true_label} images from {folder}")
    print(f"{'='*50}")

    success = 0
    errors  = 0

    for i, img_path in enumerate(images, 1):
        try:
            with open(img_path, "rb") as f:
                response = requests.post(
                    f"{api_base}/predict",
                    files={"file": (img_path.name, f, "image/jpeg")},
                    timeout=30
                )

            if response.status_code != 200:
                print(f"[{i}/{len(images)}] PREDICT FAILED {img_path.name}: {response.status_code}")
                errors += 1
                continue

            data            = response.json()
            image_id        = data["image_id"]
            predicted_label = data["label"]
            confidence      = data["confidence"]

            fb_response = requests.post(
                f"{api_base}/feedback",
                json={
                    "image_id":        image_id,
                    "predicted_label": predicted_label,
                    "true_label":      true_label
                },
                timeout=10
            )

            correct = "✅" if predicted_label == true_label else "❌"
            print(
                f"[{i}/{len(images)}] {correct} {img_path.name} | "
                f"predicted={predicted_label} | true={true_label} | "
                f"confidence={confidence:.2f}"
            )
            success += 1

        except Exception as e:
            print(f"[{i}/{len(images)}] ERROR {img_path.name}: {e}")
            errors += 1

        time.sleep(0.1)

    print(f"\nDone: {success} success, {errors} errors")
    return success, errors


def main():
    parser = argparse.ArgumentParser(description="Batch predict and feedback")
    parser.add_argument("--malignant", type=str, required=True)
    parser.add_argument("--benign",    type=str, required=True)
    parser.add_argument("--count",     type=int, default=100)
    parser.add_argument("--api",       type=str, default="http://localhost:8000")
    args = parser.parse_args()

    malignant_dir = Path(args.malignant)
    benign_dir    = Path(args.benign)

    if not malignant_dir.exists():
        print(f"Malignant folder not found: {malignant_dir}")
        return
    if not benign_dir.exists():
        print(f"Benign folder not found: {benign_dir}")
        return

    try:
        r = requests.get(f"{args.api}/health", timeout=5)
        if r.status_code != 200:
            print(f"API not healthy: {r.status_code}")
            return
        print(f"✅ API is healthy at {args.api}")
    except Exception as e:
        print(f"❌ Cannot reach API: {e}")
        return

    process_images(malignant_dir, "malignant", args.count, args.api)
    process_images(benign_dir,    "benign",    args.count, args.api)

    print("\n✅ Batch complete. Check Grafana for updated metrics.")


if __name__ == "__main__":
    main()