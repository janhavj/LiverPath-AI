
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

INPUT = Path(
    "results/metrics/resnet18_validation_uncertainty.csv"
)

OUTPUT = Path(
    "results/metrics/resnet18_class_specific_uncertainty.csv"
)

df = pd.read_csv(INPUT)

LABELS = [
    "FIB", "INF", "NEC",
    "NOR", "REA", "STE", "TUM"
]

rows = []

for label in LABELS:
    subset = df[df["label"] == label].copy()

    errors = (
        subset["predicted_label"] != subset["label"]
    ).astype(int).to_numpy()

    scores = subset[
        "uncertainty_entropy"
    ].to_numpy()

    total_errors = int(errors.sum())

    if len(np.unique(errors)) == 2:
        auroc = roc_auc_score(errors, scores)
        auprc = average_precision_score(
            errors, scores
        )
    else:
        auroc = np.nan
        auprc = np.nan

    n_review = max(1, int(len(subset) * 0.10))
    ranked = np.argsort(-scores)

    errors_found = int(
        errors[ranked[:n_review]].sum()
    )

    rows.append({
        "class": label,
        "patches": len(subset),
        "errors": total_errors,
        "error_rate": total_errors / len(subset),
        "uncertainty_auroc": auroc,
        "uncertainty_auprc": auprc,
        "patches_reviewed_10pct": n_review,
        "errors_found_10pct": errors_found,
        "error_capture_10pct": (
            errors_found / total_errors
            if total_errors else np.nan
        )
    })

results = pd.DataFrame(rows)

print("=" * 60)
print("LiverPath AI - Class-Specific Uncertainty")
print("=" * 60)

print(results.round(4).to_string(index=False))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("Step 28 completed.")
