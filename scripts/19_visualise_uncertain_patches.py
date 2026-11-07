
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

INPUT = Path(
    "results/metrics/resnet18_validation_uncertainty.csv"
)
OUTPUT = Path(
    "results/figures/resnet18_top24_uncertain_patches.png"
)

df = pd.read_csv(INPUT)

# Sort by highest predictive entropy.
df = df.sort_values(
    "uncertainty_entropy",
    ascending=False
).head(24)

fig, axes = plt.subplots(
    4, 6,
    figsize=(18, 14)
)

for ax, (_, row) in zip(axes.flat, df.iterrows()):
    with Image.open(row["filepath"]) as image:
        rgb = np.asarray(image.convert("RGB"))

    ax.imshow(rgb)

    true_label = row["label"]
    predicted = row["predicted_label"]
    confidence = row["confidence"]
    entropy = row["uncertainty_entropy"]

    correct = true_label == predicted

    ax.set_title(
        f"True: {true_label} | Pred: {predicted}\n"
        f"Conf: {confidence:.2f} | Ent: {entropy:.2f}",
        fontsize=9,
        color="green" if correct else "red"
    )

    ax.axis("off")

fig.suptitle(
    "LiverPath AI - Most Uncertain Validation Patches",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=200)
plt.close()

print("=" * 60)
print("LiverPath AI - Uncertain Patch Visualisation")
print("=" * 60)
print("Patches visualised:", len(df))
print("Incorrect predictions:", int(
    (df["label"] != df["predicted_label"]).sum()
))
print("Saved:", OUTPUT)
print("Step 23 completed.")
