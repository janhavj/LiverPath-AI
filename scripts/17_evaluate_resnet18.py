
from pathlib import Path
from importlib import import_module
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
from tqdm import tqdm

sys.path.insert(0, str(Path("scripts").resolve()))

dataset_module = import_module("14_pytorch_dataset")
model_module = import_module("15_build_resnet18")

LABELS = dataset_module.LABELS

OUT_DIR = Path("results/metrics")
FIG_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

device = model_module.get_device()

print("=" * 60)
print("LiverPath AI - Validation Error Analysis")
print("=" * 60)
print("Device:", device)

# Use the validation set for model development.
# The test set remains untouched.
val_dataset = dataset_module.LiverPathDataset(
    "data/processed/splits/val.csv",
    transform=dataset_module.eval_transform
)

loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

model = model_module.build_model(
    pretrained=False,
    freeze_backbone=True
).to(device)

checkpoint = torch.load(
    "models/resnet18_baseline_best.pt",
    map_location=device,
    weights_only=True
)

assert checkpoint["labels"] == LABELS

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_probs = []
all_preds = []
all_targets = []

with torch.inference_mode():
    for images, targets in tqdm(
        loader, desc="Evaluating"
    ):
        images = images.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.extend(
            probs.argmax(dim=1).cpu().tolist()
        )
        all_targets.extend(targets.tolist())

probs = np.concatenate(all_probs)
preds = np.asarray(all_preds)
targets = np.asarray(all_targets)

print("\nValidation results")
print("-" * 40)
print("Accuracy:", round(accuracy_score(targets, preds), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(targets, preds), 4)
)
print(
    "Macro-F1:",
    round(f1_score(targets, preds, average="macro"), 4)
)

report = classification_report(
    targets,
    preds,
    labels=list(range(len(LABELS))),
    target_names=LABELS,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).T
report_df.to_csv(
    OUT_DIR / "resnet18_validation_classification_report.csv"
)

print("\nPer-class performance")
print(report_df.loc[LABELS].round(4).to_string())

cm = confusion_matrix(
    targets,
    preds,
    labels=list(range(len(LABELS)))
)

fig, ax = plt.subplots(figsize=(9, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("ResNet-18 - Validation Confusion Matrix")
plt.tight_layout()
plt.savefig(
    FIG_DIR / "resnet18_validation_confusion_matrix.png",
    dpi=200
)
plt.close()

# Preserve alignment with validation CSV rows.
prediction_df = val_dataset.data.copy()
prediction_df["true_index"] = targets
prediction_df["pred_index"] = preds
prediction_df["predicted_label"] = [
    LABELS[i] for i in preds
]
prediction_df["confidence"] = probs.max(axis=1)
prediction_df["correct"] = preds == targets

for i, label in enumerate(LABELS):
    prediction_df[f"prob_{label}"] = probs[:, i]

prediction_df.to_csv(
    OUT_DIR / "resnet18_validation_predictions.csv",
    index=False
)

print("\nSaved:")
print("- Validation classification report")
print("- Validation confusion matrix")
print("- Validation predictions and probabilities")
print("\nStep 21 completed.")
