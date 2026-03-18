import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, dropout: float) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


def unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True