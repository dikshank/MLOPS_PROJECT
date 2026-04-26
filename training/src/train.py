"""
train.py
--------
Phase 3 & 4 | Executed: Local (on-prem, CPU)

Main training script for melanoma classification.

Usage:
    python train.py --config configs/config_v1_mobilenet.yaml
    python train.py --config configs/config_v1_efficientnet.yaml

Training strategy:
    Phase 1 (epochs 1 to unfreeze_epoch):
        - Base CNN frozen
        - Only classifier head trained
        - Higher learning rate (lr_head)

    Phase 2 (epochs unfreeze_epoch+1 to end):
        - Last N layers of base unfrozen
        - Full fine-tuning at lower learning rate (lr_finetune)

Debug mode:
    Set debug.enabled: true in config
    Loads only 20 images, runs 2 epochs
    Verifies entire pipeline end-to-end in ~2 minutes
"""

import argparse
import logging
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

# ── Local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from dataset import MelanomaDataset
from model import get_model, unfreeze_last_layers, count_parameters
from evaluate import evaluate
from mlflow_utils import (
    setup_mlflow,
    log_config_params,
    log_tags,
    log_epoch_metrics,
    log_threshold,
    log_artifacts,
    log_model
)

import mlflow

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train")


def load_config(config_path: str) -> dict:
    """
    Load and parse a YAML training config file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed config dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config: %s", config_path)
    return config


def build_dataloaders(config: dict) -> tuple:
    """
    Build train, val, and test DataLoaders from config.

    Args:
        config (dict): Parsed training config.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    debug       = config["debug"]["enabled"]
    debug_size  = config["debug"]["num_images"]
    img_size    = config["model"]["img_size"]
    batch_size  = config["debug"]["batch_size"] if debug else config["training"]["batch_size"]
    manifest_dir = Path(config["data"]["manifest_dir"])

    train_ds = MelanomaDataset(
        manifest_path=manifest_dir / "train_manifest.csv",
        img_size=img_size,
        split="train",
        debug=debug,
        debug_size=debug_size
    )
    val_ds = MelanomaDataset(
        manifest_path=manifest_dir / "val_manifest.csv",
        img_size=img_size,
        split="val",
        debug=debug,
        debug_size=debug_size
    )
    test_ds = MelanomaDataset(
        manifest_path=manifest_dir / "test_manifest.csv",
        img_size=img_size,
        split="test",
        debug=debug,
        debug_size=debug_size
    )

    # num_workers=0 for Windows compatibility (avoids multiprocessing issues)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> tuple:
    """
    Run one full training epoch.

    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): Training dataloader.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        device (str): Device to train on.

    Returns:
        tuple: (avg_loss, recall, f1)
    """
    from sklearn.metrics import recall_score, f1_score

    model.train()
    running_loss = 0.0
    all_preds  = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # ── Forward pass ──────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss   = criterion(logits, labels)

        # ── Backward pass ─────────────────────────────────────────────────
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / len(dataloader)
    recall   = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1       = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

    return avg_loss, float(recall), float(f1)


def run_training(config: dict) -> None:
    """
    Full training pipeline: build model, train, evaluate, log to MLflow.

    Args:
        config (dict): Parsed training config.
    """
    # ── Device ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # ── Debug mode override ───────────────────────────────────────────────
    if config["debug"]["enabled"]:
        logger.info("=" * 50)
        logger.info("DEBUG MODE ACTIVE — 20 images, 2 epochs")
        logger.info("=" * 50)
        config["training"]["epochs"] = config["debug"]["epochs"]

    # ── Directories ───────────────────────────────────────────────────────
    model_dir     = Path(config["output"]["model_dir"])
    artifacts_dir = Path(config["output"]["artifacts_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / f"{config['model']['name']}_best.pth"

    # ── MLflow setup ──────────────────────────────────────────────────────
    setup_mlflow(config)

    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):

        # ── Model ─────────────────────────────────────────────────────────
        model = get_model(
            model_name=config["model"]["name"],
            num_classes=2,
            freeze_base=config["model"]["freeze_base"]
        )
        model = model.to(device)

        param_counts = count_parameters(model)
        logger.info("Parameter counts: %s", param_counts)

        # ── Log params and tags ───────────────────────────────────────────
        log_config_params(config)
        log_tags(config, param_counts)

        # ── Dataloaders ───────────────────────────────────────────────────
        train_loader, val_loader, test_loader = build_dataloaders(config)

        # ── Loss: weighted to penalize missing malignant (recall focus) ───
        # Weight malignant class higher since false negatives are more costly
        class_weights = torch.tensor(
            config["training"]["class_weights"], dtype=torch.float
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ── Optimizer: head-only phase ────────────────────────────────────
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["lr_head"],
            weight_decay=config["training"]["weight_decay"]
        )

        # ── Training loop ─────────────────────────────────────────────────
        best_val_recall   = 0.0
        best_val_f1       = 0.0
        best_threshold    = 0.5
        patience_counter  = 0
        patience          = config["training"]["early_stopping_patience"]
        unfreeze_epoch    = config["model"]["unfreeze_epoch"]
        total_epochs      = config["training"]["epochs"]
        phase_switched    = False

        for epoch in range(1, total_epochs + 1):

            # ── Switch to fine-tuning phase ───────────────────────────────
            if epoch == unfreeze_epoch and not phase_switched:
                logger.info(
                    "Epoch %d: Switching to fine-tuning phase "
                    "(unfreezing last %d layers)",
                    epoch, config["model"]["num_layers_to_unfreeze"]
                )
                model = unfreeze_last_layers(
                    model,
                    config["model"]["name"],
                    config["model"]["num_layers_to_unfreeze"]
                )
                # Lower learning rate for fine-tuning
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=config["training"]["lr_finetune"],
                    weight_decay=config["training"]["weight_decay"]
                )
                phase_switched = True

            # ── Train one epoch ───────────────────────────────────────────
            train_loss, train_recall, train_f1 = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )

            # ── Validate ──────────────────────────────────────────────────
            val_metrics, val_threshold = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                split="val",
                artifacts_dir=artifacts_dir,
                threshold=None  # auto-tune on val
            )

            logger.info(
                "Epoch %02d/%02d | "
                "train_loss=%.4f | train_recall=%.4f | train_f1=%.4f | "
                "val_recall=%.4f | val_f1=%.4f | val_auc=%.4f | threshold=%.2f",
                epoch, total_epochs,
                train_loss, train_recall, train_f1,
                val_metrics["recall"], val_metrics["f1"],
                val_metrics["auc"], val_threshold
            )

            # ── Log to MLflow ─────────────────────────────────────────────
            log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_recall=train_recall,
                train_f1=train_f1,
                val_recall=val_metrics["recall"],
                val_f1=val_metrics["f1"],
                val_accuracy=val_metrics["accuracy"],
                val_auc=val_metrics["auc"]
            )

            # ── Save best model ───────────────────────────────────────────
            if val_metrics["recall"] > best_val_recall:
                # Stage 1: recall improved — always save
                best_val_recall = val_metrics["recall"]
                best_val_f1     = val_metrics["f1"]
                best_threshold  = val_threshold
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0
                logger.info(
                    "✔ New best model saved | val_recall=%.4f", best_val_recall
                )
            elif val_metrics["recall"] == best_val_recall and val_metrics["f1"] > best_val_f1:
                # Stage 2: recall tied but F1 improved — save better balanced model
                best_val_f1    = val_metrics["f1"]
                best_threshold = val_threshold
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0
                logger.info(
                    "✔ New best model saved | val_recall=%.4f | val_f1=%.4f (F1 improved)",
                    best_val_recall, best_val_f1
                )
            else:
                patience_counter += 1
                logger.info(
                    "No improvement. Patience: %d/%d",
                    patience_counter, patience
                )

            # ── Early stopping ────────────────────────────────────────────
            if patience_counter >= patience:
                logger.info(
                    "Early stopping at epoch %d. "
                    "No recall improvement for %d epochs.",
                    epoch, patience
                )
                break

        # ── Final evaluation on test set ──────────────────────────────────
        logger.info("Loading best checkpoint for final test evaluation...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        test_metrics, _ = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            split="test",
            artifacts_dir=artifacts_dir,
            threshold=best_threshold   # use val-tuned threshold on test
        )

        logger.info("=" * 50)
        logger.info("FINAL TEST RESULTS")
        logger.info("Recall:   %.4f", test_metrics["recall"])
        logger.info("F1:       %.4f", test_metrics["f1"])
        logger.info("Accuracy: %.4f", test_metrics["accuracy"])
        logger.info("AUC:      %.4f", test_metrics["auc"])
        logger.info("Threshold:%.4f", best_threshold)
        logger.info("=" * 50)

        # ── Log final test metrics ────────────────────────────────────────
        mlflow.log_metrics({
            "test_recall":   test_metrics["recall"],
            "test_f1":       test_metrics["f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_auc":      test_metrics["auc"]
        })

        # ── Log threshold ─────────────────────────────────────────────────
        log_threshold(best_threshold)

        # ── Log artifacts ─────────────────────────────────────────────────
        log_artifacts(
            artifacts_dir=artifacts_dir,
            baseline_stats_path=config["data"].get("baseline_stats_path")
        )

        # ── Log model + auto-promote if best recall so far ───────────────
        log_model(
            model=model,
            model_name=config["model"]["name"],
            best_checkpoint_path=checkpoint_path,
            new_val_recall=best_val_recall,
            new_val_f1=best_val_f1
        )

        logger.info("✅ Training complete. MLflow run finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Train melanoma classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()

    