
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

LABELS = [
    "FIB", "INF", "NEC",
    "NOR", "REA", "STE", "TUM"
]

METRICS_DIR = Path("results/metrics")

files = {
    "baseline": METRICS_DIR / "baseline_validation_predictions.csv",
    "finetuned": METRICS_DIR / "finetuned_validation_predictions.csv"
}

results = []

for name, path in files.items():
    df = pd.read_csv(path)

    prob_columns = [
        f"prob_{label}" for label in LABELS
    ]

    probs = df[prob_columns].to_numpy(dtype=float)

    probs = np.clip(probs, 1e-12, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)

    targets = df["true_index"].to_numpy(dtype=int)
    predictions = probs.argmax(axis=1)

    errors = (predictions != targets).astype(int)

    entropy = -np.sum(
        probs * np.log(probs),
        axis=1
    ) / np.log(len(LABELS))

    auroc = roc_auc_score(errors, entropy)
    auprc = average_precision_score(errors, entropy)

    ranking = np.argsort(-entropy)

    total_errors = int(errors.sum())

    for fraction in [0.05, 0.10, 0.20, 0.30]:
        n_review = max(1, int(len(df) * fraction))

        selected = ranking[:n_review]
        errors_found = int(errors[selected].sum())

        results.append({
            "model": name,
            "validation_images": len(df),
            "total_errors": total_errors,
            "error_rate": total_errors / len(df),
            "uncertainty_auroc": auroc,
            "uncertainty_auprc": auprc,
            "review_fraction": fraction,
            "patches_reviewed": n_review,
            "errors_found": errors_found,
            "error_capture_rate": (
                errors_found / total_errors
                if total_errors else np.nan
            ),
            "flagged_error_rate": errors_found / n_review
        })

results_df = pd.DataFrame(results)

print("=" * 60)
print("LiverPath AI - Model Uncertainty Comparison")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

output = (
    METRICS_DIR / "resnet18_model_uncertainty_comparison.csv"
)

results_df.to_csv(output, index=False)

print("\nSaved:", output)
print("Step 31 completed.")
