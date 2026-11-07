
from pathlib import Path
from importlib import import_module
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch

from scipy.special import logsumexp
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path("scripts").resolve()))

dataset_module = import_module("14_pytorch_dataset")
model_module = import_module("15_build_resnet18")

PROTOCOL_PATH = Path(
    "results/metrics/final_experiment_protocol.json"
)

OUTPUT_DIR = Path("results/final_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Never silently replace a previous final evaluation.
OUTPUT_FILES = [
    OUTPUT_DIR / "summary.json",
    OUTPUT_DIR / "predictions.csv",
    OUTPUT_DIR / "per_class_metrics.csv",
    OUTPUT_DIR / "confusion_matrix.csv",
    OUTPUT_DIR / "review_metrics.csv"
]

if any(path.exists() for path in OUTPUT_FILES):
    raise FileExistsError(
        "Final test outputs already exist. "
        "Do not overwrite the frozen evaluation."
    )

with open(PROTOCOL_PATH) as f:
    protocol = json.load(f)

LABELS = protocol["labels"]
K = len(LABELS)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


# Verify frozen files before evaluating.
model_path = Path(protocol["model"]["checkpoint"])

assert sha256_file(model_path) == (
    protocol["model"]["checkpoint_sha256"]
), "Checkpoint hash mismatch"

for name, info in protocol["splits"].items():
    assert sha256_file(Path(info["path"])) == (
        info["sha256"]
    ), f"{name} split hash mismatch"

assert protocol["review"]["primary_score"] == (
    "uncalibrated_predictive_entropy"
)

temperature = float(
    protocol["calibration"]["temperature"]
)

device = model_module.get_device()

print("=" * 60)
print("LiverPath AI - FINAL FROZEN TEST EVALUATION")
print("=" * 60)
print("Device:", device)
print("Temperature:", temperature)
print("Checkpoint and split hashes verified.")

test_path = protocol["splits"]["test"]["path"]

test_dataset = dataset_module.LiverPathDataset(
    test_path,
    transform=dataset_module.eval_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

model = model_module.build_model(
    pretrained=False,
    freeze_backbone=True
)

checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=True
)

assert checkpoint["labels"] == LABELS

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

probability_batches = []
target_batches = []

with torch.inference_mode():
    for images, targets in tqdm(
        test_loader,
        desc="Final test inference"
    ):
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)

        probability_batches.append(
            probs.cpu().numpy()
        )
        target_batches.extend(targets.tolist())

probs = np.concatenate(probability_batches)
y = np.asarray(target_batches, dtype=int)

assert len(y) == len(test_dataset)

pred = probs.argmax(axis=1)
errors = (pred != y).astype(int)

# Temperature scaling using frozen parameter.
logits = np.log(np.clip(probs, 1e-12, 1.0))
scaled_logits = logits / temperature
scaled_logits -= logsumexp(
    scaled_logits,
    axis=1,
    keepdims=True
)
calibrated = np.exp(scaled_logits)

assert np.array_equal(
    calibrated.argmax(axis=1), pred
)


def entropy(p):
    safe = np.clip(p, 1e-12, 1.0)
    return -np.sum(
        safe * np.log(safe), axis=1
    ) / np.log(K)


def calibration_metrics(p):
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1) == y

    bin_ids = np.minimum(
        np.digitize(
            confidence,
            np.linspace(0, 1, 16)[1:-1]
        ),
        14
    )

    ece = 0.0

    for i in range(15):
        mask = bin_ids == i

        if mask.any():
            ece += mask.mean() * abs(
                correct[mask].mean()
                - confidence[mask].mean()
            )

    one_hot = np.eye(K)[y]

    brier = np.mean(
        np.sum((p - one_hot) ** 2, axis=1)
    )

    nll = -np.mean(
        np.log(
            np.clip(
                p[np.arange(len(y)), y],
                1e-12,
                1.0
            )
        )
    )

    return {
        "ece": float(ece),
        "brier": float(brier),
        "nll": float(nll)
    }


scores = entropy(probs)

uncertainty_auroc = roc_auc_score(
    errors, scores
)
uncertainty_auprc = average_precision_score(
    errors, scores
)

# Stable ordering for ties.
ranking = np.argsort(-scores, kind="stable")

review_rows = []

fractions = [
    protocol["review"]["primary_review_fraction"],
    *protocol["review"]["secondary_review_fractions"]
]

for fraction in sorted(set(fractions)):
    n_review = max(
        1, int(len(y) * fraction)
    )

    flagged = ranking[:n_review]
    unflagged = ranking[n_review:]

    errors_found = int(errors[flagged].sum())

    review_rows.append({
        "review_fraction": fraction,
        "patches_reviewed": n_review,
        "errors_found": errors_found,
        "error_capture_rate": (
            errors_found / errors.sum()
            if errors.sum() else None
        ),
        "flagged_error_rate": (
            errors_found / n_review
        ),
        "unflagged_errors": int(
            errors[unflagged].sum()
        ),
        "unflagged_error_rate": (
            float(errors[unflagged].mean())
            if len(unflagged) else None
        )
    })

review_df = pd.DataFrame(review_rows)

report = classification_report(
    y,
    pred,
    labels=list(range(K)),
    target_names=LABELS,
    output_dict=True,
    zero_division=0
)

cm = confusion_matrix(
    y, pred, labels=list(range(K))
)

summary = {
    "protocol_version": protocol["protocol_version"],
    "checkpoint_sha256": (
        protocol["model"]["checkpoint_sha256"]
    ),
    "test_samples": int(len(y)),
    "errors": int(errors.sum()),
    "accuracy": float(accuracy_score(y, pred)),
    "balanced_accuracy": float(
        balanced_accuracy_score(y, pred)
    ),
    "macro_f1": float(
        f1_score(
            y, pred,
            labels=list(range(K)),
            average="macro",
            zero_division=0
        )
    ),
    "uncertainty_auroc": float(
        uncertainty_auroc
    ),
    "uncertainty_auprc": float(
        uncertainty_auprc
    ),
    "temperature": temperature,
    "uncalibrated": calibration_metrics(probs),
    "calibrated": calibration_metrics(calibrated)
}

# Save all outputs after successful evaluation.
predictions_df = pd.read_csv(test_path)

predictions_df["true_index"] = y
predictions_df["pred_index"] = pred
predictions_df["predicted_label"] = [
    LABELS[i] for i in pred
]
predictions_df["correct"] = errors == 0
predictions_df["confidence"] = probs.max(axis=1)
predictions_df["entropy"] = scores

for i, label in enumerate(LABELS):
    predictions_df[f"prob_{label}"] = probs[:, i]
    predictions_df[
        f"calibrated_prob_{label}"
    ] = calibrated[:, i]

predictions_df.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)

pd.DataFrame(report).T.to_csv(
    OUTPUT_DIR / "per_class_metrics.csv"
)

pd.DataFrame(
    cm,
    index=LABELS,
    columns=LABELS
).to_csv(
    OUTPUT_DIR / "confusion_matrix.csv"
)

review_df.to_csv(
    OUTPUT_DIR / "review_metrics.csv",
    index=False
)

with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nFINAL TEST RESULTS")
print("-" * 60)
print("Samples:", summary["test_samples"])
print("Errors:", summary["errors"])
print(f"Accuracy: {summary['accuracy']:.4f}")
print(
    "Balanced accuracy:",
    f"{summary['balanced_accuracy']:.4f}"
)
print(f"Macro-F1: {summary['macro_f1']:.4f}")
print(
    "Uncertainty AUROC:",
    f"{summary['uncertainty_auroc']:.4f}"
)
print(
    "Uncertainty AUPRC:",
    f"{summary['uncertainty_auprc']:.4f}"
)

print("\nCalibration:")
for name in ["uncalibrated", "calibrated"]:
    print(name, summary[name])

print("\nReview results:")
print(review_df.round(4).to_string(index=False))

print("\nPer-class classification report:")
print(
    pd.DataFrame(report).T.loc[
        LABELS,
        ["precision", "recall", "f1-score", "support"]
    ].round(4).to_string()
)

print("\nSaved results to:", OUTPUT_DIR)
print("Step 36 completed.")
