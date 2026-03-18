import json
import math
import os
from dataclasses import asdict

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from src.data import get_dataloaders
from src.jax_calibration import apply_temperature, fit_temperature_with_jax
from src.metrics import compute_ece, plot_confusion_matrix, plot_reliability_diagram
from src.model import build_model, freeze_backbone, unfreeze_all
from src.utils import ensure_dir, get_device, set_seed, softmax_np


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)

    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=float(loss.item()))

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }


@torch.no_grad()
def collect_logits_labels(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in tqdm(loader, desc="collect_logits", leave=False):
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def train_pipeline(cfg):
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    device = get_device()

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    train_loader, val_loader, test_loader, label_names, num_classes = get_dataloaders(
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = build_model(num_classes=num_classes, dropout=cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    freeze_backbone(model)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val_f1 = -math.inf
    best_model_path = os.path.join(cfg.output_dir, "best_model.pt")
    best_epoch = -1
    patience_counter = 0

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(asdict(cfg))
        mlflow.log_param("device", device)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("classes", json.dumps(label_names))

        for epoch in range(cfg.epochs):
            if epoch == cfg.freeze_backbone_epochs:
                unfreeze_all(model)
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=cfg.lr / 5,
                    weight_decay=cfg.weight_decay,
                )

            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            metrics_to_log = {
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }

            mlflow.log_metrics(metrics_to_log, step=epoch)

            print(f"Epoch {epoch + 1}/{cfg.epochs}")
            print(json.dumps(metrics_to_log, indent=2))

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_artifact(best_model_path, artifact_path="checkpoints")
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_logits, test_labels = collect_logits_labels(model, test_loader, device)
        val_logits, val_labels = collect_logits_labels(model, val_loader, device)

        test_probs_uncal = softmax_np(test_logits)
        test_preds_uncal = test_probs_uncal.argmax(axis=1)

        uncal_accuracy = accuracy_score(test_labels, test_preds_uncal)
        uncal_macro_f1 = f1_score(test_labels, test_preds_uncal, average="macro")
        uncal_ece = compute_ece(test_probs_uncal, test_labels)

        temperature = fit_temperature_with_jax(val_logits, val_labels, steps=300, lr=0.05)

        test_logits_cal = apply_temperature(test_logits, temperature)
        test_probs_cal = softmax_np(test_logits_cal)
        test_preds_cal = test_probs_cal.argmax(axis=1)

        cal_accuracy = accuracy_score(test_labels, test_preds_cal)
        cal_macro_f1 = f1_score(test_labels, test_preds_cal, average="macro")
        cal_ece = compute_ece(test_probs_cal, test_labels)

        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("temperature", temperature)
        mlflow.log_metric("test_accuracy_uncalibrated", uncal_accuracy)
        mlflow.log_metric("test_macro_f1_uncalibrated", uncal_macro_f1)
        mlflow.log_metric("test_ece_uncalibrated", uncal_ece)
        mlflow.log_metric("test_accuracy_calibrated", cal_accuracy)
        mlflow.log_metric("test_macro_f1_calibrated", cal_macro_f1)
        mlflow.log_metric("test_ece_calibrated", cal_ece)

        report = classification_report(
            test_labels,
            test_preds_cal,
            target_names=label_names,
            output_dict=True
        )
        report_path = os.path.join(cfg.output_dir, "classification_report_calibrated.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path, artifact_path="reports")

        cm = confusion_matrix(test_labels, test_preds_cal)
        cm_path = os.path.join(cfg.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, label_names, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        rel_uncal_path = os.path.join(cfg.output_dir, "reliability_uncalibrated.png")
        rel_cal_path = os.path.join(cfg.output_dir, "reliability_calibrated.png")
        plot_reliability_diagram(test_probs_uncal, test_labels, rel_uncal_path)
        plot_reliability_diagram(test_probs_cal, test_labels, rel_cal_path)
        mlflow.log_artifact(rel_uncal_path, artifact_path="plots")
        mlflow.log_artifact(rel_cal_path, artifact_path="plots")

        serving_bundle = {
            "class_names": label_names,
            "temperature": temperature,
            "image_size": cfg.image_size,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        }
        serving_bundle_path = os.path.join(cfg.output_dir, "serving_config.json")
        with open(serving_bundle_path, "w", encoding="utf-8") as f:
            json.dump(serving_bundle, f, indent=2)
        mlflow.log_artifact(serving_bundle_path, artifact_path="serving")

        mlflow.pytorch.log_model(model, artifact_path="model")

        summary = {
            "project": "Bean Leaf Disease Classifier",
            "dataset": "Hugging Face beans",
            "frameworks": ["PyTorch", "JAX", "MLflow"],
            "best_epoch": best_epoch,
            "temperature": temperature,
            "test_uncalibrated": {
                "accuracy": uncal_accuracy,
                "macro_f1": uncal_macro_f1,
                "ece": uncal_ece,
            },
            "test_calibrated": {
                "accuracy": cal_accuracy,
                "macro_f1": cal_macro_f1,
                "ece": cal_ece,
            },
        }
        summary_path = os.path.join(cfg.output_dir, "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path, artifact_path="summary")

        print("\nFinal summary:")
        print(json.dumps(summary, indent=2))