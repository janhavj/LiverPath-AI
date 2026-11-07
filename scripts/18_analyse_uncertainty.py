
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

INPUT = Path(
    "results/metrics/resnet18_validation_predictions.csv"
)

OUT_DIR = Path("results/metrics")
FIG_DIR = Path("results/figures")

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "FIB", "INF", "NEC",
    "NOR", "REA", "STE", "TUM"
]

df = pd.read_csv(INPUT)

prob_cols = [f"prob_{label}" for label in LABELS]
probs = df[prob_cols].to_numpy(dtype=float)

# Numerical stability.
eps = 1e-12
probs = np.clip(probs, eps, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

# Confidence: highest class probability.
confidence = probs.max(axis=1)

# Normalised predictive entropy: 0 to 1.
entropy = -np.sum(
    probs * np.log(probs),
    axis=1
) / np.log(len(LABELS))

# Margin between the highest and second-highest
# predicted class probabilities.
sorted_probs = np.sort(probs, axis=1)
margin = sorted_probs[:, -1] - sorted_probs[:, -2]

df["confidence"] = confidence
df["uncertainty_entropy"] = entropy
df["uncertainty_margin"] = 1.0 - margin
df["error"] = (
    df["predicted_label"] != df["label"]
).astype(int)

print("=" * 60)
print("LiverPath AI - Uncertainty Analysis")
print("=" * 60)

print("Validation patches:", len(df))
print("Correct predictions:", int((df["error"] == 0).sum()))
print("Incorrect predictions:", int(df["error"].sum()))

print("\nMean uncertainty by prediction outcome:")
print(
    df.groupby("error")[
        ["confidence", "uncertainty_entropy",
         "uncertainty_margin"]
    ].mean().round(4)
)

print("\nError detection performance:")

for score in [
    "uncertainty_entropy",
    "uncertainty_margin"
]:
    auroc = roc_auc_score(df["error"], df[score])
    auprc = average_precision_score(
        df["error"], df[score]
    )

    print(
        f"{score}: AUROC={auroc:.4f}, "
        f"AUPRC={auprc:.4f}"
    )

# Examine the most uncertain patches.
most_uncertain = df.sort_values(
    "uncertainty_entropy",
    ascending=False
).head(100)

most_uncertain.to_csv(
    OUT_DIR / "resnet18_top100_uncertain_patches.csv",
    index=False
)

# Measure error rate at different review fractions.
rows = []

ranked = df.sort_values(
    "uncertainty_entropy",
    ascending=False
)

total_errors = df["error"].sum()

for fraction in [0.05, 0.10, 0.20, 0.30]:
    n_review = max(1, int(len(df) * fraction))
    reviewed = ranked.head(n_review)

    errors_found = reviewed["error"].sum()

    rows.append({
        "review_fraction": fraction,
        "patches_flagged": n_review,
        "errors_flagged": int(errors_found),
        "flagged_error_rate": errors_found / n_review,
        "fraction_of_all_errors_found": (
            errors_found / total_errors
            if total_errors else np.nan
        )
    })

review_df = pd.DataFrame(rows)

print("\nUncertainty-based review simulation:")
print(review_df.round(4).to_string(index=False))

review_df.to_csv(
    OUT_DIR / "resnet18_review_simulation.csv",
    index=False
)

# Plot entropy distributions.
fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(
    df.loc[df["error"] == 0, "uncertainty_entropy"],
    bins=40,
    alpha=0.6,
    density=True,
    label="Correct predictions"
)

ax.hist(
    df.loc[df["error"] == 1, "uncertainty_entropy"],
    bins=40,
    alpha=0.6,
    density=True,
    label="Incorrect predictions"
)

ax.set_xlabel("Normalised predictive entropy")
ax.set_ylabel("Density")
ax.set_title(
    "Prediction uncertainty: correct vs incorrect"
)
ax.legend()

plt.tight_layout()
plt.savefig(
    FIG_DIR / "resnet18_uncertainty_distribution.png",
    dpi=200
)
plt.close()

df.to_csv(
    OUT_DIR / "resnet18_validation_uncertainty.csv",
    index=False
)

print("\nSaved uncertainty analysis results.")
print("Step 22 completed.")
