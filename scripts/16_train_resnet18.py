
from pathlib import Path
import sys
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from tqdm import tqdm

from importlib import import_module

# Import scripts whose filenames begin with numbers.
sys.path.insert(0, str(Path("scripts").resolve()))

dataset_module = import_module("14_pytorch_dataset")
model_module = import_module("15_build_resnet18")

create_dataloaders = dataset_module.create_dataloaders
LABELS = dataset_module.LABELS
build_model = model_module.build_model
get_device = model_module.get_device

SEED = 42
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 3

MODEL_DIR = Path("models")
METRICS_DIR = Path("results/metrics")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = get_device()

print("=" * 60)
print("LiverPath AI - ResNet-18 Baseline Training")
print("=" * 60)
print("Device:", device)

train_loader, val_loader, _ = create_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=0
)

model = build_model(
    pretrained=True,
    freeze_backbone=True
).to(device)

# Calculate class weights from training data only.
train_df = pd.read_csv("data/processed/splits/train.csv")

counts = (
    train_df["label"]
    .value_counts()
    .reindex(LABELS)
    .to_numpy()
)

weights = len(train_df) / (len(LABELS) * counts)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=device
)

print("\nClass weights:")
for label, weight in zip(LABELS, weights):
    print(f"{label}: {weight:.4f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


def train_one_epoch():
    model.eval()

    # Keep BatchNorm running statistics frozen while
    # training the new classification head.
    model.fc.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in tqdm(
        train_loader,
        desc="Training",
        leave=False
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate():
    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    for images, labels in tqdm(
        val_loader,
        desc="Validation",
        leave=False
    ):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return {
        "val_loss": total_loss / total_samples,
        "val_accuracy": accuracy_score(
            all_labels, all_predictions
        ),
        "val_balanced_accuracy": balanced_accuracy_score(
            all_labels, all_predictions
        ),
        "val_macro_f1": f1_score(
            all_labels,
            all_predictions,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0
        )
    }


history = []
best_f1 = -1.0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_loss = train_one_epoch()
    metrics = validate()

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        **metrics
    }

    history.append(record)

    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss: {metrics['val_loss']:.4f}")
    print(f"Val accuracy: {metrics['val_accuracy']:.4f}")
    print(f"Val balanced accuracy: "
          f"{metrics['val_balanced_accuracy']:.4f}")
    print(f"Val macro-F1: {metrics['val_macro_f1']:.4f}")

    pd.DataFrame(history).to_csv(
        METRICS_DIR / "resnet18_training_history.csv",
        index=False
    )

    if metrics["val_macro_f1"] > best_f1:
        best_f1 = metrics["val_macro_f1"]
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_macro_f1": best_f1,
                "labels": LABELS,
                "class_weights": weights.tolist(),
                "seed": SEED
            },
            MODEL_DIR / "resnet18_baseline_best.pt"
        )

        print("Best model saved.")

    else:
        epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

print("\nTraining completed.")
print(f"Best validation macro-F1: {best_f1:.4f}")
print("Step 20 completed.")
