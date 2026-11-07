
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

RESULTS = Path("results/final_test")
FIGURES = Path("results/figures/final")
FIGURES.mkdir(parents=True, exist_ok=True)

LABELS = ["FIB", "INF", "NEC", "NOR", "REA", "STE", "TUM"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300
})

pred = pd.read_csv(RESULTS / "predictions.csv")
cm = pd.read_csv(
    RESULTS / "confusion_matrix.csv",
    index_col=0
).loc[LABELS, LABELS]

metrics = pd.read_csv(
    RESULTS / "per_class_metrics.csv",
    index_col=0
).loc[LABELS]

review = pd.read_csv(
    RESULTS / "review_metrics.csv"
)


def save(name):
    plt.savefig(
        FIGURES / f"{name}.png",
        bbox_inches="tight"
    )
    plt.savefig(
        FIGURES / f"{name}.pdf",
        bbox_inches="tight"
    )
    plt.close()


# Figure 1: Row-normalised confusion matrix
matrix = cm.to_numpy(dtype=float)
normalised = matrix / matrix.sum(
    axis=1, keepdims=True
)

fig, ax = plt.subplots(figsize=(8, 6.5))

im = ax.imshow(
    normalised,
    cmap="Blues",
    vmin=0,
    vmax=1
)

ax.set_xticks(range(len(LABELS)), LABELS)
ax.set_yticks(range(len(LABELS)), LABELS)

ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("ResNet-18: Final Test Confusion Matrix")

for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        value = normalised[i, j] * 100
        ax.text(
            j, i,
            f"{value:.1f}%",
            ha="center",
            va="center",
            color=(
                "white" if normalised[i, j] > 0.5
                else "black"
            ),
            fontsize=9
        )

fig.colorbar(
    im,
    ax=ax,
    label="Proportion of true class"
)

save("01_confusion_matrix")


# Figure 2: Per-class classification metrics
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(LABELS))
width = 0.25

for offset, column, label in [
    (-width, "precision", "Precision"),
    (0, "recall", "Recall"),
    (width, "f1-score", "F1-score")
]:
    ax.bar(
        x + offset,
        metrics[column].values,
        width,
        label=label
    )

ax.set_xticks(x, LABELS)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_xlabel("Histopathology class")
ax.set_title("Per-Class Performance on the Final Test Set")
ax.legend(frameon=False, ncol=3)

save("02_per_class_performance")


# Figure 3: Uncertainty-based review
fig, ax = plt.subplots(figsize=(7.5, 5))

review_fraction = review["review_fraction"].to_numpy()
capture = review["error_capture_rate"].to_numpy()

ax.plot(
    review_fraction * 100,
    capture * 100,
    marker="o",
    linewidth=2.5,
    label="Entropy-based review"
)

ax.plot(
    [0, 100],
    [0, 100],
    linestyle="--",
    color="grey",
    label="Random review expectation"
)

ax.set_xlim(0, 35)
ax.set_ylim(0, 100)
ax.set_xlabel("Test patches reviewed (%)")
ax.set_ylabel("Classification errors captured (%)")
ax.set_title("Uncertainty-Guided Error Review")
ax.legend(frameon=False)

save("03_uncertainty_review")


# Figure 4: Confidence reliability diagram
def reliability_data(probabilities, true_labels, bins=15):
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == true_labels).astype(float)

    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(
        np.digitize(confidence, edges[1:-1]),
        0,
        bins - 1
    )

    mean_confidence = []
    mean_accuracy = []

    for i in range(bins):
        mask = indices == i

        if mask.any():
            mean_confidence.append(
                confidence[mask].mean()
            )
            mean_accuracy.append(
                correct[mask].mean()
            )

    return mean_confidence, mean_accuracy


true_labels = pred["true_index"].to_numpy()

uncalibrated = pred[
    [f"prob_{label}" for label in LABELS]
].to_numpy()

calibrated = pred[
    [f"calibrated_prob_{label}" for label in LABELS]
].to_numpy()

fig, ax = plt.subplots(figsize=(7, 6))

for probabilities, label in [
    (uncalibrated, "Before calibration"),
    (calibrated, "After calibration")
]:
    conf, acc = reliability_data(
        probabilities, true_labels
    )

    ax.plot(
        conf,
        acc,
        marker="o",
        label=label
    )

ax.plot(
    [0, 1],
    [0, 1],
    "--",
    color="grey",
    label="Perfect calibration"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted confidence")
ax.set_ylabel("Observed accuracy")
ax.set_title("Test-Set Reliability Diagram")
ax.legend(frameon=False)

save("04_calibration_reliability")

print("=" * 60)
print("LiverPath AI - Final Research Figures")
print("=" * 60)
print("Saved 4 figures as PNG and PDF:")
for path in sorted(FIGURES.glob("*.png")):
    print(" -", path)

print("\nStep 37 completed.")
