import json
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model
from src.utils import softmax_np, get_device


def predict_image(image_path: str, model_path: str, serving_config_path: str) -> Dict:
    with open(serving_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    class_names = cfg["class_names"]
    image_size = cfg["image_size"]
    temperature = float(cfg["temperature"])
    mean = cfg["normalization_mean"]
    std = cfg["normalization_std"]

    device = get_device()

    model = build_model(num_classes=len(class_names), dropout=0.2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    image = Image.open(image_path).convert("RGB")
    x = tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x).cpu().numpy() / temperature
        probs = softmax_np(logits)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "predicted_class": pred_label,
        "confidence": confidence,
        "class_probabilities": {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        }
    }