
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split

LABELS = [
    "FIB", "INF", "NEC",
    "NOR", "REA", "STE", "TUM"
]

INPUT = Path(
    "results/metrics/resnet18_validation_predictions.csv"
)

TEMP_PATH = Path(
    "results/metrics/resnet18_temperature_parameter.csv"
)

OUTPUT = Path(
    "results/metrics/resnet18_calibrated_review_comparison.csv"
)

df = pd.read_csv(INPUT)
temperature = float(
    pd.read_csv(TEMP_PATH)["temperature"].iloc[0]
)

probs = df[
    [f"prob_{label}" for label in LABELS]
].to_numpy(dtype=float)

targets = df["true_index"].to_numpy(dtype=int)

probs = np.clip(probs, 1e-12, 1.0)
probs /= probs.sum(axis=1, keepdims=True)

indices = np.arange(len(df))

_, assess_idx = train_test_split(
    indices,
    test_size=0.5,
    stratify=targets,
    random_state=42
)

original = probs[assess_idx]
y = targets[assess_idx]

# Reconstruct relative logits.
logits = np.log(original)

scaled_logits = logits / temperature
scaled_logits -= scaled_logits.max(
    axis=1, keepdims=True
)

exp_logits = np.exp(scaled_logits)
calibrated = exp_logits / exp_logits.sum(
    axis=1, keepdims=True
)

def entropy(probabilities):
    p = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(
        p * np.log(p), axis=1
    ) / np.log(p.shape[1])


errors = (
    original.argmax(axis=1) != y
).astype(int)

rows = []

for name, p in [
    ("Before calibration", original),
    ("After calibration", calibrated)
]:
    scores = entropy(p)

    auroc = roc_auc_score(errors, scores)
    auprc = average_precision_score(errors, scores)

    ranking = np.argsort(-scores)
    total_errors = errors.sum()

    for fraction in [0.05, 0.10, 0.20, 0.30]:
        n = max(1, int(len(y) * fraction))
        selected = ranking[:n]

        errors_found = int(errors[selected].sum())

        rows.append({
            "model": name,
            "review_fraction": fraction,
            "patches_reviewed": n,
            "errors_found": errors_found,
            "error_capture_rate": (
                errors_found / total_errors
                if total_errors else np.nan
            ),
            "flagged_error_rate": errors_found / n,
            "uncertainty_auroc": auroc,
            "uncertainty_auprc": auprc
        })

results = pd.DataFrame(rows)

print("=" * 60)
print("LiverPath AI - Calibrated Uncertainty Comparison")
print("=" * 60)

print("Assessment images:", len(y))
print("Assessment errors:", int(errors.sum()))
print("Temperature:", round(temperature, 4))

print("\nResults:")
print(results.round(4).to_string(index=False))

results.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("Step 27 completed.")
