"""Train the final SisFall 1D CNN using the fixed subject split."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from .data import ROOT, SEED, load_windows
from .model import FallCNN1D

MODEL_PATH = ROOT / "models" / "final_cnn.pt"
RESULTS_DIR = ROOT / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    "filters": [16, 32, 64], "kernel_size": 9, "dropout": 0.30,
    "learning_rate": 0.003, "batch_size": 32, "weight_decay": 0.0001,
    "optimizer": "Adam", "max_epochs": 10, "patience": 3,
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_from_logits(logits: list[float], labels: list[int], recording_ids: list[int]) -> dict:
    """Average window logits per recording, then calculate fall-detection metrics."""
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    truths: dict[int, int] = {}
    for logit, label, recording_id in zip(logits, labels, recording_ids):
        sums[recording_id] += logit
        counts[recording_id] += 1
        truths[recording_id] = label
    ids = sorted(sums)
    record_logits = np.array([sums[index] / counts[index] for index in ids])
    y_true = np.array([truths[index] for index in ids])
    y_pred = (record_logits >= 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_records": int(len(y_true)), "adl_records": int((y_true == 0).sum()),
        "fall_records": int((y_true == 1).sum()), "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
        "record_ids": ids, "record_logits": record_logits, "record_labels": y_true,
    }


def run_epoch(model, loader, loss_function, optimizer=None) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_windows = 0
    logits_all, labels_all, recording_ids_all = [], [], []
    for x, labels, recording_ids in loader:
        x, labels = x.to(DEVICE), labels.to(DEVICE)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        total_windows += len(labels)
        logits_all.extend(logits.detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().to(torch.int64).tolist())
        recording_ids_all.extend(recording_ids.tolist())
    metrics = metrics_from_logits(logits_all, labels_all, recording_ids_all)
    metrics["loss"] = total_loss / total_windows
    return metrics["loss"], metrics


def main() -> None:
    set_seed()
    print(f"Using {DEVICE}")
    splits, normalization = load_windows()
    train_loader = DataLoader(splits["train"], batch_size=CONFIG["batch_size"], shuffle=True)
    validation_loader = DataLoader(splits["validation"], batch_size=CONFIG["batch_size"])
    train_labels = splits["train"].tensors[1]
    positive_weight = float((train_labels == 0).sum() / (train_labels == 1).sum())
    model = FallCNN1D(filters=tuple(CONFIG["filters"]), kernel_size=CONFIG["kernel_size"], dropout=CONFIG["dropout"]).to(DEVICE)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    best_f1, best_loss, best_epoch, patience_count = -1.0, float("inf"), 0, 0
    for epoch in range(1, CONFIG["max_epochs"] + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, loss_function, optimizer)
        validation_loss, validation_metrics = run_epoch(model, validation_loader, loss_function)
        improved = validation_metrics["f1"] > best_f1 or (validation_metrics["f1"] == best_f1 and validation_loss < best_loss)
        if improved:
            best_f1, best_loss, best_epoch, patience_count = validation_metrics["f1"], validation_loss, epoch, 0
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(), "config": CONFIG, "seed": SEED,
                "best_epoch": best_epoch, "pos_weight_train_windows": positive_weight,
                "window_samples": 3000, "sample_rate_hz": 200, "normalization": normalization,
                "label_mapping": {"ADL": 0, "Fall": 1}, "positive_class": "Fall",
            }, MODEL_PATH)
        else:
            patience_count += 1
        print(f"Epoch {epoch:02d}: train loss {train_loss:.4f}, validation loss {validation_loss:.4f}, validation F1 {validation_metrics['f1']:.4f}")
        if patience_count >= CONFIG["patience"]:
            break
    print(f"Best validation epoch: {best_epoch}. Checkpoint: {MODEL_PATH}")


if __name__ == "__main__":
    main()
