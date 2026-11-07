
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

INPUT = Path(
    "results/metrics/resnet18_validation_predictions.csv"
)

OUT = Path("results/metrics")
FIG = Path("results/figures")

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

LABELS = [
    "FIB", "INF", "NEC",
    "NOR", "REA", "STE", "TUM"
]

df = pd.read_csv(INPUT)

probs = df[
    [f"prob_{label}" for label in LABELS]
].to_numpy(dtype=float)

targets = df["true_index"].to_numpy(dtype=int)

# Reconstruct relative logits from saved probabilities.
# This is sufficient for temperature scaling, although
# original logits would be preferable.
probs = np.clip(probs, 1e-12, 1.0)
probs /= probs.sum(axis=1, keepdims=True)
logits = np.log(probs)

indices = np.arange(len(df))

fit_idx, assess_idx = train_test_split(
    indices,
    test_size=0.5,
    stratify=targets,
    random_state=42
)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)


def apply_temperature(logits_array, temperature):
    return softmax(logits_array / temperature)


def objective(log_temperature):
    temperature = np.exp(log_temperature)

    calibrated = apply_temperature(
        logits[fit_idx],
        temperature
    )

    return log_loss(
        targets[fit_idx],
        calibrated,
        labels=list(range(len(LABELS)))
    )


result = minimize_scalar(
    objective,
    bounds=(np.log(0.1), np.log(10.0)),
    method="bounded"
)

temperature = float(np.exp(result.x))


def compute_ece(probabilities, y, n_bins=15):
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correctness = (prediction == y).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.minimum(
        np.digitize(confidence, edges[1:-1]),
        n_bins - 1
    )

    ece = 0.0

    for i in range(n_bins):
        mask = bin_ids == i

        if not mask.any():
            continue

        ece += mask.mean() * abs(
            correctness[mask].mean()
            - confidence[mask].mean()
        )

    return float(ece)


def compute_brier(probabilities, y):
    one_hot = np.eye(len(LABELS))[y]
    return float(np.mean(
        np.sum((probabilities - one_hot) ** 2, axis=1)
    ))


before = probs[assess_idx]
after = apply_temperature(
    logits[assess_idx],
    temperature
)
y_assess = targets[assess_idx]

metrics = pd.DataFrame([
    {
        "model": "Before calibration",
        "nll": log_loss(
            y_assess, before,
            labels=list(range(len(LABELS)))
        ),
        "ece": compute_ece(before, y_assess),
        "brier": compute_brier(before, y_assess)
    },
    {
        "model": "After temperature scaling",
        "nll": log_loss(
            y_assess, after,
            labels=list(range(len(LABELS)))
        ),
        "ece": compute_ece(after, y_assess),
        "brier": compute_brier(after, y_assess)
    }
])

print("=" * 60)
print("LiverPath AI - Temperature Scaling")
print("=" * 60)

print("Temperature fit samples:", len(fit_idx))
print("Assessment samples:", len(assess_idx))
print(f"Optimal temperature: {temperature:.4f}")

print("\nAssessment metrics:")
print(metrics.round(4).to_string(index=False))

metrics.to_csv(
    OUT / "resnet18_temperature_scaling_metrics.csv",
    index=False
)

pd.DataFrame([{
    "temperature": temperature,
    "fit_samples": len(fit_idx),
    "assessment_samples": len(assess_idx),
    "random_state": 42
}]).to_csv(
    OUT / "resnet18_temperature_parameter.csv",
    index=False
)

print("\nSaved temperature scaling results.")
print("Step 26 completed.")
