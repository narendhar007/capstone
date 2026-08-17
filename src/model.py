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

def build_model(freeze: bool | None = None, pretrained: bool = True,) -> nn.Module:
    # TODO 2: load torchvision resnet18 with ImageNet weights; if freeze, set
    #         requires_grad=False on backbone params; replace net.fc with a
    #         nn.Linear(in_features, config.NUM_CLASSES) trainable head.
    """
    Build an ImageNet-pretrained ResNet18 for two-class classification.

    When freeze is True, only the new classification head is trainable.
    """
    # If freeze is None, use the default value from config.FREEZE_BACKBONE.
    if freeze is None:
        freeze = config.FREEZE_BACKBONE

    # Load the ResNet18 model with pretrained ImageNet weights if specified.
    weights = (
        ResNet18_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )

    # Use a fixed ImageNet weight version for reproducibility.
    net = resnet18(
        weights=weights
    )

    # If freeze is True, freeze the backbone parameters (requires_grad=False).
    if freeze:
        for parameter in net.parameters():
            parameter.requires_grad = False

    # Replace the original 1000-class ImageNet head.
    input_features = net.fc.in_features

    # Replace the final fully connected layer with a new linear layer for 2-class classification.
    net.fc = nn.Linear(
        input_features,
        config.NUM_CLASSES,
    )

    return net


def trainable_parameters(net: nn.Module):
    return [p for p in net.parameters() if p.requires_grad]


class EmbeddingExtractor(nn.Module):
    """Expose the 512-dim penultimate features (drop the fc layer)."""
    def __init__(self, net: nn.Module):
        super().__init__()
        # TODO 4 (embedding drift): keep all layers except the final fc.
        # ResNet18 children end with: avgpool -> fc
        # Keep everything through avgpool and remove fc.
        self.features = nn.Sequential(
            *list(net.children())[:-1]
        )

        self.features.eval() #To stop the network from training when extracting embeddings

    @torch.no_grad() # To prevent gradient computation during embedding extraction

    def forward(self, x):
        # forward pass through the feature extractor to obtain embeddings
        embeddings = self.features(x) 

        # ResNet output before fc is:
        # [batch, 512, 1, 1]
        # Flatten to [batch, 512].
        embeddings = torch.flatten(
            embeddings,
            1,
        )

        return embeddings


def save_model(net: nn.Module, path: Path | None = None) -> None:
    torch.save(net.state_dict(), path or config.MODEL_PATH)


def load_model(path: Path | None = None, freeze: bool = True) -> nn.Module:
    """
    Load trained model weights without downloading ImageNet weights.

    """
    # Use the supplied path or the default model path from config.
    model_path = (
        Path(path)
        if path is not None
        else config.MODEL_PATH
    )

    # check if the model path exists, raise an error if it doesn't
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    # build the model architecture with the specified freeze option and without pretrained weights
    net = build_model(freeze=freeze,pretrained=False,)

    # load the state dictionary from the specified model path, mapping it to the appropriate device
    state_dict = torch.load(
        model_path,
        map_location=config.DEVICE,
        weights_only=True,
    )

    # load the state dictionary into the model
    net.load_state_dict(state_dict, strict=True)
    net.to(config.DEVICE)
    net.eval()

    return net
