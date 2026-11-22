"""Train and evaluate one simple SVM using the CNN's SisFall split."""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import ROOT, load_windows
from .train import metrics_from_logits

RESULTS_DIR = ROOT / "results"
METRICS = ("accuracy", "precision", "recall", "specificity", "f1")


def extract_features(windows: np.ndarray) -> np.ndarray:
    """Return six simple time-domain statistics for each of nine channels."""
    mean = windows.mean(axis=2)
    std = windows.std(axis=2)
    minimum = windows.min(axis=2)
    maximum = windows.max(axis=2)
    rms = np.sqrt(np.mean(np.square(windows), axis=2))
    value_range = maximum - minimum
    # Shape: (windows, channels, 6) -> (windows, 54 features).
    return np.stack((mean, std, minimum, maximum, rms, value_range), axis=2).reshape(len(windows), -1)


def main() -> None:
    splits, _ = load_windows()
    train_x, train_y, _ = splits["train"].tensors
    test_x, test_y, test_recording_ids = splits["test"].tensors

    train_features = extract_features(train_x.numpy())
    test_features = extract_features(test_x.numpy())
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # One standard RBF SVM; scaling is learned from training windows only.
    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(train_features, train_y.numpy().astype(int))
    window_scores = svm.decision_function(test_features)
    metrics = metrics_from_logits(
        window_scores.tolist(), test_y.numpy().astype(int).tolist(), test_recording_ids.numpy().tolist()
    )
    report = {key: value for key, value in metrics.items()
              if key not in {"record_ids", "record_logits", "record_labels"}}
    report.update({
        "model": "SVC(kernel='rbf', C=1.0, gamma='scale')",
        "feature_count": 54,
        "features_per_channel": ["mean", "standard deviation", "minimum", "maximum", "RMS", "range"],
        "scaler": "StandardScaler fitted on training windows only",
        "evaluation_unit": "recordings; window decision scores averaged per recording",
        "positive_class": "Fall",
        "negative_class": "ADL",
    })

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "svm_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cnn = json.loads((RESULTS_DIR / "final_metrics.json").read_text(encoding="utf-8"))
    with (RESULTS_DIR / "model_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Metric", "CNN", "SVM"])
        writer.writerows([[metric.capitalize(), cnn[metric], report[metric]] for metric in METRICS])

    confusion = np.array([[report["tn"], report["fp"]], [report["fn"], report["tp"]]])
    display = ConfusionMatrixDisplay(confusion, display_labels=["ADL", "Fall"])
    display.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.title("SVM held-out test confusion matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "svm_confusion_matrix.png", dpi=160)
    plt.close()
    print(json.dumps(report, indent=2))
    print("CNN vs SVM comparison:")
    print((RESULTS_DIR / "model_comparison.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
