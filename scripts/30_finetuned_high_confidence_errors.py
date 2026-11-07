
from pathlib import Path

import numpy as np
import pandas as pd

METRICS_DIR = Path("results/metrics")

df = pd.read_csv(
    METRICS_DIR / "finetuned_validation_predictions.csv"
)

df["error"] = (
    df["label"] != df["predicted_label"]
)

errors = df[df["error"]].copy()

print("=" * 60)
print("LiverPath AI - Fine-Tuned High-Confidence Errors")
print("=" * 60)

print("Validation patches:", len(df))
print("Total errors:", len(errors))
print("Overall error rate:", round(df["error"].mean(), 4))

rows = []

for threshold in [0.90, 0.95, 0.99]:
    high_conf = errors[
        errors["confidence"] >= threshold
    ]

    all_high_conf = df[
        df["confidence"] >= threshold
    ]

    rows.append({
        "confidence_threshold": threshold,
        "incorrect_predictions": len(high_conf),
        "fraction_of_all_errors": (
            len(high_conf) / len(errors)
            if len(errors) else np.nan
        ),
        "error_rate_among_high_confidence": (
            len(high_conf) / len(all_high_conf)
            if len(all_high_conf) else np.nan
        )
    })

threshold_results = pd.DataFrame(rows)

print("\nHigh-confidence error summary:")
print(threshold_results.round(4).to_string(index=False))

threshold_results.to_csv(
    METRICS_DIR / "finetuned_high_confidence_summary.csv",
    index=False
)

confusions = (
    errors.groupby(
        ["label", "predicted_label"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print("\nTop 15 misclassification patterns:")
print(confusions.head(15).to_string(index=False))

confusions.to_csv(
    METRICS_DIR / "finetuned_error_confusions.csv",
    index=False
)

high_conf_errors = errors[
    errors["confidence"] >= 0.95
].sort_values(
    "confidence", ascending=False
)

high_conf_errors.to_csv(
    METRICS_DIR / "finetuned_high_confidence_errors.csv",
    index=False
)

print("\nHigh-confidence errors by true class:")
print(
    high_conf_errors["label"]
    .value_counts()
    .to_string()
)

print("\nSaved error analysis files.")
print("Step 34 completed.")
