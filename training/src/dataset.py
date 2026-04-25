"""
dataset.py
----------
Phase 3 & 4 | Executed: Local (on-prem, CPU)

PyTorch Dataset class for melanoma classification.

Reads images from the processed data directory using split manifests
produced by the Airflow pipeline. Applies augmentation transforms
during training only.

Supports debug mode: loads only a small subset of images for quick
pipeline verification without full training time.
"""

import logging
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dataset")

# ── Label mapping ─────────────────────────────────────────────────────────────
# Must be consistent across all scripts
LABEL_MAP = {
    "benign": 0,
    "malignant": 1
}


def get_transforms(img_size: int, split: str) -> transforms.Compose:
    """
    Return the appropriate transforms for a given split.

    Training split gets augmentation transforms.
    Val and test splits get only resize + normalize (no augmentation).

    Augmentation is applied here (not in the pipeline) to avoid
    storing augmented images on disk.

    Args:
        img_size (int): Target image size (square). E.g. 32 or 64.
        split (str): One of 'train', 'val', 'test'.

    Returns:
        transforms.Compose: Composed transform pipeline.
    """
    # ImageNet normalization stats (used since we have pretrained weights)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        # ── Heavy augmentation for training ──────────────────────────────
        # Critical for small datasets (600 images) to prevent overfitting
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    else:
        # ── No augmentation for val and test ─────────────────────────────
        # Only resize and normalize — deterministic, reproducible
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


class MelanomaDataset(Dataset):
    """
    PyTorch Dataset for melanoma classification.

    Reads image paths and labels from a CSV manifest file
    produced by the Airflow split pipeline.

    Args:
        manifest_path (str or Path): Path to split manifest CSV.
                                     Columns: filepath, label.
        img_size (int): Target image size (square).
        split (str): One of 'train', 'val', 'test'.
        debug (bool): If True, loads only debug_size images.
        debug_size (int): Number of images to load in debug mode.
    """

    def __init__(
        self,
        manifest_path: str,
        img_size: int,
        split: str,
        debug: bool = False,
        debug_size: int = 20
    ):
        self.split = split
        self.img_size = img_size
        self.debug = debug
        self.transform = get_transforms(img_size, split)

        # ── Load manifest ─────────────────────────────────────────────────
        manifest_path = Path(manifest_path)

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run the Airflow pipeline first."
            )

        df = pd.read_csv(manifest_path)

        # ── Validate manifest columns ─────────────────────────────────────
        if "filepath" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"Manifest must have 'filepath' and 'label' columns. "
                f"Found: {df.columns.tolist()}"
            )

        # ── Debug mode: subsample ─────────────────────────────────────────
        if debug:
            # Take equal samples from each class if possible
            df = (
                df.groupby("label", group_keys=False)
                .apply(lambda x: x.sample(
                    min(len(x), debug_size // 2),
                    random_state=42
                ))
                .reset_index(drop=True)
            )
            logger.info(
                "DEBUG MODE: Loaded %d images for split [%s]", len(df), split
            )

        self.filepaths = df["filepath"].tolist()
        self.labels = [LABEL_MAP[label] for label in df["label"].tolist()]

        logger.info(
            "Dataset [%s] → %d images | "
            "Malignant: %d | Benign: %d",
            split,
            len(self.filepaths),
            self.labels.count(1),
            self.labels.count(0)
        )

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        """
        Load and return a single image and its label.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image_tensor, label) where label is 0=benign, 1=malignant.
        """
        img_path = self.filepaths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, label

        except Exception as e:
            logger.error(
                "Failed to load image [%s]: %s", img_path, str(e)
            )
            raise
