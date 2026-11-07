
from pathlib import Path
from importlib import import_module
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from tqdm import tqdm

sys.path.insert(0, str(Path("scripts").resolve()))

dataset_module = import_module("14_pytorch_dataset")
model_module = import_module("15_build_resnet18")

LABELS = dataset_module.LABELS
device = model_module.get_device()

METRICS_DIR = Path("results/metrics")
FIG_DIR = Path("results/figures")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("LiverPath AI - Baseline vs Fine-Tuned Comparison")
print("=" * 60)
print("Device:", device)

val_dataset = dataset_module.LiverPathDataset(
    "data/processed/splits/val.csv",
    transform=dataset_module.eval_transform
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


def evaluate_checkpoint(name, checkpoint_path):
    model = model_module.build_model(
        pretrained=False,
        freeze_backbone=True
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    assert checkpoint["labels"] == LABELS

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_probs = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in tqdm(
            val_loader,
            desc=name
        ):
            images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_targets.extend(targets.tolist())

    probabilities = np.concatenate(all_probs)
    targets = np.asarray(all_targets)
    predictions = probabilities.argmax(axis=1)

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "model": name,
        "accuracy": accuracy_score(
            targets, predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            targets, predictions
        ),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro"
        ),
        "errors": int((targets != predictions).sum())
    }

    # Save per-class metrics.
    class_report = pd.DataFrame(report).T
    class_report.to_csv(
        METRICS_DIR / f"{name}_classification_report.csv"
    )

    # Save predictions in dataset row order.
    predictions_df = val_dataset.data.copy()
    predictions_df["true_index"] = targets
    predictions_df["pred_index"] = predictions
    predictions_df["predicted_label"] = [
        LABELS[i] for i in predictions
    ]
    predictions_df["confidence"] = (
        probabilities.max(axis=1)
    )
    predictions_df["correct"] = (
        predictions == targets
    )

    for i, label in enumerate(LABELS):
        predictions_df[f"prob_{label}"] = (
            probabilities[:, i]
        )

    predictions_df.to_csv(
        METRICS_DIR / f"{name}_validation_predictions.csv",
        index=False
    )

    # Confusion matrix.
    cm = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(LABELS)))
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(f"{name} - Validation Confusion Matrix")

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"{name}_confusion_matrix.png",
        dpi=200
    )
    plt.close()

    return metrics, class_report


results = []
reports = {}

checkpoints = {
    "baseline": "models/resnet18_baseline_best.pt",
    "finetuned": "models/resnet18_finetuned_best.pt"
}

for name, path in checkpoints.items():
    metrics, report = evaluate_checkpoint(name, path)

    results.append(metrics)
    reports[name] = report

summary = pd.DataFrame(results)

print("\nOverall comparison:")
print(summary.round(4).to_string(index=False))

summary.to_csv(
    METRICS_DIR / "resnet18_model_comparison.csv",
    index=False
)

print("\nPer-class F1 comparison:")

class_comparison = pd.DataFrame({
    "baseline_f1": reports["baseline"].loc[
        LABELS, "f1-score"
    ],
    "finetuned_f1": reports["finetuned"].loc[
        LABELS, "f1-score"
    ]
})

class_comparison["improvement"] = (
    class_comparison["finetuned_f1"]
    - class_comparison["baseline_f1"]
)

print(class_comparison.round(4).to_string())

class_comparison.to_csv(
    METRICS_DIR / "resnet18_per_class_comparison.csv"
)

print("\nSaved model comparison and prediction files.")
print("Step 30 completed.")
