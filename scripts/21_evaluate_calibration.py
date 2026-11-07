
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss

INPUT = Path(
    "results/metrics/resnet18_validation_predictions.csv"
)

METRICS_DIR = Path("results/metrics")
FIG_DIR = Path("results/figures")

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "FIB", "INF", "NEC",
    "NOR", "REA", "STE", "TUM"
]

df = pd.read_csv(INPUT)

probs = df[
    [f"prob_{label}" for label in LABELS]
].to_numpy(dtype=float)

targets = df["true_index"].to_numpy(dtype=int)

probs = np.clip(probs, 1e-12, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

predictions = probs.argmax(axis=1)
confidence = probs.max(axis=1)
correct = (predictions == targets).astype(float)

# Multiclass Brier score.
one_hot = np.eye(len(LABELS))[targets]

brier = np.mean(
    np.sum((probs - one_hot) ** 2, axis=1)
)

nll = log_loss(
    targets,
    probs,
    labels=list(range(len(LABELS)))
)

# Expected calibration error with 15 equal-width bins.
edges = np.linspace(0, 1, 16)
bin_ids = np.minimum(
    np.digitize(confidence, edges[1:-1]),
    14
)

rows = []
ece = 0.0

for i in range(15):
    mask = bin_ids == i
    count = int(mask.sum())

    if count == 0:
        continue

    mean_conf = float(confidence[mask].mean())
    mean_acc = float(correct[mask].mean())

    ece += (
        count / len(df)
    ) * abs(mean_acc - mean_conf)

    rows.append({
        "bin": i,
        "count": count,
        "mean_confidence": mean_conf,
        "observed_accuracy": mean_acc,
        "calibration_gap": mean_acc - mean_conf
    })

calibration_df = pd.DataFrame(rows)

print("=" * 60)
print("LiverPath AI - Probability Calibration")
print("=" * 60)

print("Validation images:", len(df))
print(f"Accuracy: {correct.mean():.4f}")
print(f"Mean confidence: {confidence.mean():.4f}")
print(f"Expected calibration error: {ece:.4f}")
print(f"Multiclass Brier score: {brier:.4f}")
print(f"Negative log-likelihood: {nll:.4f}")

print("\nCalibration by confidence bin:")
print(calibration_df.round(4).to_string(index=False))

calibration_df.to_csv(
    METRICS_DIR / "resnet18_calibration_bins.csv",
    index=False
)

# Reliability diagram.
fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="black",
    label="Perfect calibration"
)

ax.plot(
    calibration_df["mean_confidence"],
    calibration_df["observed_accuracy"],
    marker="o",
    label="ResNet-18"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Observed accuracy")
ax.set_title("ResNet-18 Validation Reliability Diagram")
ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(
    FIG_DIR / "resnet18_reliability_diagram.png",
    dpi=200
)
plt.close()

print("\nSaved calibration metrics and reliability diagram.")
print("Step 25 completed.")
