
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

INPUT = Path("results/metrics/image_hashes.csv")
OUTPUT_DIR = Path("data/processed/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("LiverPath AI - Reproducible Dataset Splitting")
print("=" * 60)

df = pd.read_csv(INPUT)

# Keep one representative per exact SHA-256 hash.
df = df.drop_duplicates(
    subset="sha256",
    keep="first"
).copy()

assert len(df) == 87220, (
    f"Unexpected number of unique images: {len(df)}"
)

# First reserve 15% for testing.
train_val, test = train_test_split(
    df,
    test_size=0.15,
    stratify=df["label"],
    random_state=SEED
)

# Reserve 15% of the full dataset for validation.
train, val = train_test_split(
    train_val,
    test_size=0.15 / 0.85,
    stratify=train_val["label"],
    random_state=SEED
)

splits = {
    "train": train,
    "val": val,
    "test": test
}

for name, subset in splits.items():
    subset = subset.sort_values("filepath").reset_index(drop=True)

    path = OUTPUT_DIR / f"{name}.csv"
    subset.to_csv(path, index=False)

    print(f"\n{name.upper()}: {len(subset):,} images")
    print(subset["label"].value_counts().sort_index())
    print("Saved:", path)

# Verify disjoint file hashes.
hash_sets = {
    name: set(subset["sha256"])
    for name, subset in splits.items()
}

assert hash_sets["train"].isdisjoint(hash_sets["val"])
assert hash_sets["train"].isdisjoint(hash_sets["test"])
assert hash_sets["val"].isdisjoint(hash_sets["test"])

assert sum(len(s) for s in splits.values()) == 87220

metadata = {
    "seed": SEED,
    "split_type": "stratified_patch_level",
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "total_unique_images": 87220,
    "patient_independent": False,
    "limitations": (
        "Patient and slide identifiers unavailable. "
        "Exact duplicates removed; pHash screening at "
        "Hamming distance <=5 found no candidate pairs. "
        "Clinical generalisation cannot be inferred."
    )
}

with (OUTPUT_DIR / "split_metadata.json").open("w") as f:
    json.dump(metadata, f, indent=2)

print("\nAll split integrity checks passed.")
print("Step 17 completed.")
