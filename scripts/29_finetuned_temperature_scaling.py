
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split

LABELS = [
    "FIB", "INF", "NEC",
    "NOR", "REA", "STE", "TUM"
]

METRICS_DIR = Path("results/metrics")

df = pd.read_csv(
    METRICS_DIR / "finetuned_validation_predictions.csv"
)

prob_columns = [f"prob_{label}" for label in LABELS]

probs = df[prob_columns].to_numpy(dtype=float)
targets = df["true_index"].to_numpy(dtype=int)

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


def temperature_probs(logits_array, temperature):
    scaled = logits_array / temperature
    scaled -= logsumexp(
        scaled, axis=1, keepdims=True
    )
    return np.exp(scaled)


def nll_for_temperature(log_temperature):
    temperature = np.exp(log_temperature)

    scaled = logits[fit_idx] / temperature

    log_probs = scaled - logsumexp(
        scaled, axis=1, keepdims=True
    )

    return -np.mean(
        log_probs[
            np.arange(len(fit_idx)),
            targets[fit_idx]
        ]
    )


optimisation = minimize_scalar(
    nll_for_temperature,
    bounds=(np.log(0.1), np.log(10.0)),
    method="bounded"
)

if not optimisation.success:
    raise RuntimeError(
        f"Temperature optimisation failed: {optimisation.message}"
    )

temperature = float(np.exp(optimisation.x))


def compute_metrics(probabilities, y, n_bins=15):
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == y

    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.minimum(
        np.digitize(confidence, edges[1:-1]),
        n_bins - 1
    )

    ece = 0.0

    for bin_index in range(n_bins):
        mask = bin_ids == bin_index

        if not mask.any():
            continue

        ece += mask.mean() * abs(
            correct[mask].mean()
            - confidence[mask].mean()
        )

    nll = -np.log(
        np.clip(
            probabilities[np.arange(len(y)), y],
            1e-12,
            1.0
        )
    ).mean()

    one_hot = np.eye(len(LABELS))[y]

    brier = np.mean(
        np.sum(
            (probabilities - one_hot) ** 2,
            axis=1
        )
    )

    return {
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "ece": float(ece),
        "nll": float(nll),
        "brier": float(brier)
    }


original = probs[assess_idx]

calibrated = temperature_probs(
    logits[assess_idx],
    temperature
)

before = compute_metrics(
    original, targets[assess_idx]
)

after = compute_metrics(
    calibrated, targets[assess_idx]
)

results = pd.DataFrame([
    {"model": "Before calibration", **before},
    {"model": "After calibration", **after}
])

print("=" * 60)
print("LiverPath AI - Fine-Tuned Temperature Scaling")
print("=" * 60)

print("Fit samples:", len(fit_idx))
print("Assessment samples:", len(assess_idx))
print(f"Optimal temperature: {temperature:.4f}")

print("\nAssessment metrics:")
print(results.round(4).to_string(index=False))

results.to_csv(
    METRICS_DIR / "finetuned_temperature_scaling_metrics.csv",
    index=False
)

pd.DataFrame({
    "temperature": [temperature],
    "fit_samples": [len(fit_idx)],
    "assessment_samples": [len(assess_idx)]
}).to_csv(
    METRICS_DIR / "finetuned_temperature_parameter.csv",
    index=False
)

assessment = df.iloc[assess_idx].copy()
assessment["original_row_index"] = assess_idx

for i, label in enumerate(LABELS):
    assessment[f"calibrated_prob_{label}"] = calibrated[:, i]

assessment.to_csv(
    METRICS_DIR / "finetuned_calibrated_assessment.csv",
    index=False
)

print("\nSaved calibration metrics and assessment predictions.")
print("Step 33 completed.")
