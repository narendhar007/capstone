"""model.py — Stage 2: transfer-learning model + embedding extractor.

Configure a pretrained ResNet18 backbone for transfer learning (freeze the backbone,
replace the final layer with a fresh 2-class head). The same backbone is reused as a
512-dim feature extractor for embedding drift. See notebook "Model Development".
"""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

from torchvision.models import ResNet18_Weights, resnet18

def build_model(freeze: bool | None = None) -> nn.Module:
    # TODO 2: load torchvision resnet18 with ImageNet weights; if freeze, set
    #         requires_grad=False on backbone params; replace net.fc with a
    #         nn.Linear(in_features, config.NUM_CLASSES) trainable head.
    """
    Build an ImageNet-pretrained ResNet18 for two-class classification.

    When freeze is True, only the new classification head is trainable.
    """
    if freeze is None:
        freeze = config.FREEZE_BACKBONE

    # Use a fixed ImageNet weight version for reproducibility.
    net = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )

    # If freeze is True, freeze the backbone parameters (requires_grad=False).
    if freeze:
        for parameter in net.parameters():
            parameter.requires_grad = False

    # Replace the original 1000-class ImageNet head.
    input_features = net.fc.in_features
    net.fc = nn.Linear(
        input_features,
        config.NUM_CLASSES,
    )

    return net
    raise NotImplementedError("Build the ResNet18 transfer-learning model")


def trainable_parameters(net: nn.Module):
    return [p for p in net.parameters() if p.requires_grad]


class EmbeddingExtractor(nn.Module):
    """Expose the 512-dim penultimate features (drop the fc layer)."""
    def __init__(self, net: nn.Module):
        super().__init__()
        # TODO 4 (embedding drift): keep all layers except the final fc.
        raise NotImplementedError("Wrap the backbone to output pre-fc embeddings")

    @torch.no_grad()
    def forward(self, x):
        raise NotImplementedError


def save_model(net: nn.Module, path: Path | None = None) -> None:
    torch.save(net.state_dict(), path or config.MODEL_PATH)


def load_model(path: Path | None = None, freeze: bool = True) -> nn.Module:
    net = build_model(freeze=freeze)
    net.load_state_dict(torch.load(path or config.MODEL_PATH, map_location=config.DEVICE))
    net.eval()
    return net
