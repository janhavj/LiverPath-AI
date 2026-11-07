
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib

import pandas as pd

ROOT = Path(".")
OUTPUT = ROOT / "results/metrics/final_experiment_protocol.json"

MODEL = ROOT / "models/resnet18_finetuned_best.pt"
TEMPERATURE_FILE = (
    ROOT / "results/metrics/finetuned_temperature_parameter.csv"
)

SPLITS = {
    "train": ROOT / "data/processed/splits/train.csv",
    "validation": ROOT / "data/processed/splits/val.csv",
    "test": ROOT / "data/processed/splits/test.csv"
}

for path in [MODEL, TEMPERATURE_FILE, *SPLITS.values()]:
    if not path.exists():
        raise FileNotFoundError(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


temperature = float(
    pd.read_csv(TEMPERATURE_FILE)["temperature"].iloc[0]
)

# Read only CSV headers to verify schema, not test observations.
for name, path in SPLITS.items():
    columns = pd.read_csv(path, nrows=0).columns
    required = {"filepath", "label", "sha256"}

    if not required.issubset(columns):
        raise ValueError(
            f"{name} split missing required columns"
        )

protocol = {
    "project": "LiverPath AI",
    "protocol_version": "1.0",
    "frozen_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "dataset": {
        "name": "HepatoBench",
        "repository": "xtxx/HepatoBench",
        "revision": (
            "3fd9d3697fb63a84ea0635aed97f77ad3f1c5d97"
        ),
        "evaluation_unit": "image_patch",
        "patient_independent": False
    },
    "labels": [
        "FIB", "INF", "NEC",
        "NOR", "REA", "STE", "TUM"
    ],
    "model": {
        "architecture": "ResNet18",
        "checkpoint": str(MODEL),
        "checkpoint_sha256": sha256_file(MODEL),
        "input_size": [224, 224],
        "normalisation": "ImageNet"
    },
    "calibration": {
        "method": "temperature_scaling",
        "temperature": temperature,
        "fit_data": "validation_calibration_subset"
    },
    "review": {
        "primary_score": "uncalibrated_predictive_entropy",
        "primary_review_fraction": 0.10,
        "secondary_review_fractions": [0.05, 0.20, 0.30],
        "ranking": "descending_entropy"
    },
    "metrics": {
        "classification": [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "per_class_precision_recall_f1"
        ],
        "uncertainty": [
            "error_detection_auroc",
            "error_detection_auprc",
            "error_capture_rate",
            "flagged_error_rate",
            "unflagged_error_rate"
        ],
        "calibration": [
            "ece_15_bins",
            "multiclass_brier",
            "negative_log_likelihood"
        ]
    },
    "splits": {
        name: {
            "path": str(path),
            "sha256": sha256_file(path)
        }
        for name, path in SPLITS.items()
    },
    "limitations": [
        "Patch-level split without patient or slide identifiers",
        "No independent patient-level evaluation",
        "Validation set repeatedly used for model development",
        "No external or prospective clinical validation"
    ]
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT.exists():
    raise FileExistsError(
        "Protocol already exists. Do not overwrite a frozen protocol."
    )

with open(OUTPUT, "w") as f:
    json.dump(protocol, f, indent=2)

print("=" * 60)
print("LiverPath AI - Frozen Experimental Protocol")
print("=" * 60)
print("Model:", protocol["model"]["checkpoint"])
print("Temperature:", temperature)
print("Primary review fraction: 10%")
print("Model SHA256:", protocol["model"]["checkpoint_sha256"])
print("\nSaved:", OUTPUT)
print("Step 35 completed.")
