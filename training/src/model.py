"""
model.py
--------
Phase 3 & 4 | Executed: Local (on-prem, CPU)

Model definitions for melanoma classification.

Supports two architectures:
    - mobilenet_v3_small   : Lightweight, fast on CPU, good for small datasets
    - efficientnet_b0      : Stronger baseline, slower on CPU

Both use pretrained ImageNet weights with a custom classification head.

Training strategy:
    Phase 1 (frozen base): Only the classifier head is trained.
                           Base CNN weights are frozen.
                           Prevents destroying pretrained features early on.
    Phase 2 (fine-tuning): Last few layers of base are unfrozen.
                           Full model fine-tuned at lower learning rate.
"""

import logging
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    EfficientNet_B0_Weights
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("model")

# ── Supported architectures ───────────────────────────────────────────────────
SUPPORTED_MODELS = ["mobilenet_v3_small", "efficientnet_b0"]


def get_mobilenet_v3_small(num_classes: int, freeze_base: bool) -> nn.Module:
    """
    Load MobileNetV3-Small with pretrained ImageNet weights.
    Replace the classifier head for binary melanoma classification.

    Args:
        num_classes (int): Number of output classes (2 for binary).
        freeze_base (bool): If True, freeze all base layers except classifier.

    Returns:
        nn.Module: MobileNetV3-Small model ready for training.
    """
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    # ── Freeze base if required ───────────────────────────────────────────
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
        logger.info("MobileNetV3-Small: base frozen, training head only")

    # ── Replace classifier head ───────────────────────────────────────────
    # Original head: Linear(576, 1000) for ImageNet
    # New head: Linear(576, num_classes) for melanoma
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    logger.info(
        "MobileNetV3-Small loaded | freeze_base=%s | out_classes=%d",
        freeze_base, num_classes
    )
    return model


def get_efficientnet_b0(num_classes: int, freeze_base: bool) -> nn.Module:
    """
    Load EfficientNet-B0 with pretrained ImageNet weights.
    Replace the classifier head for binary melanoma classification.

    Args:
        num_classes (int): Number of output classes (2 for binary).
        freeze_base (bool): If True, freeze all base layers except classifier.

    Returns:
        nn.Module: EfficientNet-B0 model ready for training.
    """
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # ── Freeze base if required ───────────────────────────────────────────
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
        logger.info("EfficientNet-B0: base frozen, training head only")

    # ── Replace classifier head ───────────────────────────────────────────
    # Original head: Linear(1280, 1000) for ImageNet
    # New head: Linear(1280, num_classes) for melanoma
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    logger.info(
        "EfficientNet-B0 loaded | freeze_base=%s | out_classes=%d",
        freeze_base, num_classes
    )
    return model


def get_model(model_name: str, num_classes: int = 2, freeze_base: bool = True) -> nn.Module:
    """
    Factory function to get a model by name.

    Args:
        model_name (str): One of 'mobilenet_v3_small', 'efficientnet_b0'.
        num_classes (int): Number of output classes. Default: 2.
        freeze_base (bool): Whether to freeze base layers initially.

    Returns:
        nn.Module: Requested model with pretrained weights.

    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Choose from: {SUPPORTED_MODELS}"
        )

    if model_name == "mobilenet_v3_small":
        return get_mobilenet_v3_small(num_classes, freeze_base)

    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes, freeze_base)


def unfreeze_last_layers(model: nn.Module, model_name: str, num_layers: int = 2) -> nn.Module:
    """
    Unfreeze the last N layers of the base for fine-tuning phase.

    Called after initial head-only training to allow fine-tuning
    of the top layers of the pretrained base.

    Args:
        model (nn.Module): Model with frozen base.
        model_name (str): Architecture name.
        num_layers (int): Number of feature layers to unfreeze from the end.

    Returns:
        nn.Module: Model with last N base layers unfrozen.
    """
    if model_name == "mobilenet_v3_small":
        # MobileNetV3 features is a Sequential of blocks
        layers = list(model.features.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    elif model_name == "efficientnet_b0":
        layers = list(model.features.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Unfroze last %d layers of %s | Trainable params: %d",
        num_layers, model_name, trainable
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count total and trainable parameters in a model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        dict: total and trainable parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }
