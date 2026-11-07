
from pathlib import Path
from importlib import import_module
import sys
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path("scripts").resolve()))

dataset_module = import_module("14_pytorch_dataset")
model_module = import_module("15_build_resnet18")

LABELS = dataset_module.LABELS
get_device = model_module.get_device
build_model = model_module.build_model

SEED = 42
EPOCHS = 5
PATIENCE = 3
BATCH_SIZE = 32

MODEL_DIR = Path("models")
METRICS_DIR = Path("results/metrics")
MODEL_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = get_device()

print("=" * 60)
print("LiverPath AI - ResNet-18 Fine-Tuning")
print("=" * 60)
print("Device:", device)

train_loader, val_loader, _ = (
    dataset_module.create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0
    )
)

# Load our previously trained baseline.
model = build_model(
    pretrained=False,
    freeze_backbone=True
)

checkpoint = torch.load(
    MODEL_DIR / "resnet18_baseline_best.pt",
    map_location="cpu",
    weights_only=True
)

assert checkpoint["labels"] == LABELS
model.load_state_dict(checkpoint["model_state_dict"])

# Unfreeze the final residual block.
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(device)

trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", f"{trainable:,}")

train_df = pd.read_csv(
    "data/processed/splits/train.csv"
)

counts = (
    train_df["label"]
    .value_counts()
    .reindex(LABELS)
    .to_numpy()
)

weights = len(train_df) / (len(LABELS) * counts)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )
)

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.layer4.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.fc.parameters(),
            "lr": 1e-4
        }
    ],
    weight_decay=0.01
)


def train_epoch():
    # Keep frozen layers' BatchNorm statistics unchanged.
    model.eval()
    model.layer4.train()
    model.fc.train()

    total_loss = 0.0
    total_samples = 0

    for images, targets in tqdm(
        train_loader,
        desc="Training",
        leave=False
    ):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        n = targets.size(0)
        total_loss += loss.item() * n
        total_samples += n

    return total_loss / total_samples


@torch.no_grad()
def validate():
    model.eval()

    predictions = []
    targets_all = []

    for images, targets in tqdm(
        val_loader,
        desc="Validation",
        leave=False
    ):
        images = images.to(device)
        logits = model(images)

        predictions.extend(
            logits.argmax(dim=1).cpu().tolist()
        )
        targets_all.extend(targets.tolist())

    return f1_score(
        targets_all,
        predictions,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0
    )


best_f1 = -1.0
no_improvement = 0
history = []

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_loss = train_epoch()
    val_f1 = validate()

    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation macro-F1: {val_f1:.4f}")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_macro_f1": val_f1
    })

    pd.DataFrame(history).to_csv(
        METRICS_DIR / "resnet18_finetuning_history.csv",
        index=False
    )

    if val_f1 > best_f1:
        best_f1 = val_f1
        no_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_macro_f1": best_f1,
                "labels": LABELS,
                "seed": SEED
            },
            MODEL_DIR / "resnet18_finetuned_best.pt"
        )

        print("Best fine-tuned model saved.")

    else:
        no_improvement += 1

        if no_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

print("\nFine-tuning completed.")
print(f"Best validation macro-F1: {best_f1:.4f}")
print("Step 29 completed.")
