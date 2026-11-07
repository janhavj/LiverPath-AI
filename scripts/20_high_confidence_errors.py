
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

INPUT = Path(
    "results/metrics/resnet18_validation_uncertainty.csv"
)

METRICS_DIR = Path("results/metrics")
FIG_DIR = Path("results/figures")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Select confidently incorrect predictions.
errors = df[
    (df["label"] != df["predicted_label"])
    & (df["confidence"] >= 0.90)
].copy()

errors = errors.sort_values(
    "confidence",
    ascending=False
)

print("=" * 60)
print("LiverPath AI - High-Confidence Error Analysis")
print("=" * 60)

total_errors = (
    df["label"] != df["predicted_label"]
).sum()

print("Total validation images:", len(df))
print("Total classification errors:", total_errors)
print("High-confidence errors:", len(errors))

print(
    "Percentage of all errors with confidence >= 0.90:",
    round(100 * len(errors) / total_errors, 2)
)

print("\nMost common high-confidence confusions:")

confusions = (
    errors.groupby(["label", "predicted_label"])
    .size()
    .sort_values(ascending=False)
)

print(confusions.head(15))

errors.to_csv(
    METRICS_DIR / "resnet18_high_confidence_errors.csv",
    index=False
)

# Display up to 24 high-confidence mistakes.
selected = errors.head(24)

fig, axes = plt.subplots(
    4, 6,
    figsize=(18, 14)
)

for ax in axes.flat:
    ax.axis("off")

for ax, (_, row) in zip(
    axes.flat,
    selected.iterrows()
):
    with Image.open(row["filepath"]) as image:
        rgb = np.asarray(image.convert("RGB"))

    ax.imshow(rgb)

    ax.set_title(
        f"True: {row['label']} | "
        f"Pred: {row['predicted_label']}\n"
        f"Confidence: {row['confidence']:.3f}",
        fontsize=9,
        color="red"
    )

    ax.axis("off")

fig.suptitle(
    "LiverPath AI - High-Confidence Classification Errors",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

output = (
    FIG_DIR / "resnet18_high_confidence_errors.png"
)

plt.savefig(output, dpi=200)
plt.close()

print("\nSaved:", output)
print("Step 24 completed.")
