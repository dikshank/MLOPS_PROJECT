"""
model.py
--------
Phase 4 | Executed: Local (backend Docker container)

Copy of training/src/model.py for use in the FastAPI backend.
Required because MLflow needs the model architecture definition
at load time to reconstruct the model from the registry.

Keep this file in sync with training/src/model.py.
"""

import logging
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    EfficientNet_B0_Weights
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("model")

SUPPORTED_MODELS = ["mobilenet_v3_small", "efficientnet_b0"]


def get_mobilenet_v3_small(num_classes: int, freeze_base: bool = False) -> nn.Module:
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def get_efficientnet_b0(num_classes: int, freeze_base: bool = False) -> nn.Module:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    if freeze_base:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_model(model_name: str, num_classes: int = 2, freeze_base: bool = False) -> nn.Module:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Choose from: {SUPPORTED_MODELS}"
        )
    if model_name == "mobilenet_v3_small":
        return get_mobilenet_v3_small(num_classes, freeze_base)
    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes, freeze_base)
