
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss

LABELS = [
    "FIB", "INF", "NEC",
    "NOR", "REA", "STE", "TUM"
]

METRICS_DIR = Path("results/metrics")
FIG_DIR = Path("results/figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "baseline": METRICS_DIR / "baseline_validation_predictions.csv",
    "finetuned": METRICS_DIR / "finetuned_validation_predictions.csv"
}


def compute_ece(probs, targets, n_bins=15):
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == targets).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.minimum(
        np.digitize(confidence, edges[1:-1]),
        n_bins - 1
    )

    ece = 0.0
    bin_rows = []

    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())

        if count == 0:
            continue

        mean_conf = float(confidence[mask].mean())
        mean_acc = float(correct[mask].mean())

        ece += mask.mean() * abs(mean_acc - mean_conf)

        bin_rows.append({
            "bin": i,
            "count": count,
            "mean_confidence": mean_conf,
            "observed_accuracy": mean_acc
        })

    return float(ece), pd.DataFrame(bin_rows)


results = []
calibration_curves = {}

for name, path in FILES.items():
    df = pd.read_csv(path)

    probs = df[
        [f"prob_{label}" for label in LABELS]
    ].to_numpy(dtype=float)

    targets = df["true_index"].to_numpy(dtype=int)

    probs = np.clip(probs, 1e-12, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)

    predictions = probs.argmax(axis=1)
    correct = predictions == targets

    one_hot = np.eye(len(LABELS))[targets]

    brier = np.mean(
        np.sum((probs - one_hot) ** 2, axis=1)
    )

    nll = log_loss(
        targets,
        probs,
        labels=list(range(len(LABELS)))
    )

    ece, curve = compute_ece(probs, targets)
    calibration_curves[name] = curve

    results.append({
        "model": name,
        "accuracy": correct.mean(),
        "mean_confidence": probs.max(axis=1).mean(),
        "ece": ece,
        "brier": brier,
        "nll": nll
    })

results_df = pd.DataFrame(results)

print("=" * 60)
print("LiverPath AI - Model Calibration Comparison")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    METRICS_DIR / "resnet18_model_calibration_comparison.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(
    [0, 1], [0, 1],
    linestyle="--",
    color="black",
    label="Perfect calibration"
)

for name, curve in calibration_curves.items():
    ax.plot(
        curve["mean_confidence"],
        curve["observed_accuracy"],
        marker="o",
        label=name
    )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Observed accuracy")
ax.set_title("ResNet-18 Calibration Comparison")
ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

output = FIG_DIR / "resnet18_model_calibration_comparison.png"
plt.savefig(output, dpi=200)
plt.close()

print("\nSaved:", output)
print("Step 32 completed.")
