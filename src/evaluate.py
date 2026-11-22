"""Evaluate the saved CNN on the fixed held-out SisFall subjects."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader

from .data import ROOT, load_windows
from .model import FallCNN1D
from .train import MODEL_PATH, DEVICE, metrics_from_logits


def main() -> None:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    splits, _ = load_windows(checkpoint["normalization"])
    test_loader = DataLoader(splits["test"], batch_size=32)
    model = FallCNN1D(
        filters=tuple(checkpoint["config"]["filters"]),
        kernel_size=checkpoint["config"]["kernel_size"],
        dropout=checkpoint["config"]["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(checkpoint["pos_weight_train_windows"], device=DEVICE))
    logits_all, labels_all, recording_ids_all = [], [], []
    total_loss, total_windows = 0.0, 0
    with torch.no_grad():
        for x, labels, recording_ids in test_loader:
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            logits = model(x)
            loss = loss_function(logits, labels)
            total_loss += float(loss) * len(labels)
            total_windows += len(labels)
            logits_all.extend(logits.cpu().tolist())
            labels_all.extend(labels.to(torch.int64).cpu().tolist())
            recording_ids_all.extend(recording_ids.tolist())

    metrics = metrics_from_logits(logits_all, labels_all, recording_ids_all)
    report = {key: value for key, value in metrics.items() if key not in {"record_ids", "record_logits", "record_labels"}}
    report.update({
        "loss": total_loss / total_windows,
        "n_windows": total_windows,
        "positive_class": "Fall",
        "negative_class": "ADL",
        "evaluation_unit": "recordings; window logits averaged per recording",
        "decision_threshold": 0.5,
        "checkpoint": "models/final_cnn.pt",
    })
    RESULTS_DIR = ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "final_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    confusion = np.array([[report["tn"], report["fp"]], [report["fn"], report["tp"]]])
    display = ConfusionMatrixDisplay(confusion, display_labels=["ADL", "Fall"])
    display.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.title("Held-out test confusion matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
