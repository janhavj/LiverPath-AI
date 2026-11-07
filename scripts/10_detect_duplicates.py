
from pathlib import Path
from collections import defaultdict
import hashlib

import pandas as pd
from tqdm import tqdm

QC_PATH = Path("results/metrics/image_qc_results.csv")
OUTPUT_DIR = Path("results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("LiverPath AI - Exact Duplicate Detection")
print("=" * 60)

qc = pd.read_csv(QC_PATH, low_memory=False)

# Exclude the confirmed invalid image.
valid = qc.loc[
    qc["valid"].astype(str).str.lower() == "true"
].copy()

print("Valid images:", len(valid))
print("Excluded invalid images:", len(qc) - len(valid))

hash_groups = defaultdict(list)
records = []

for row in tqdm(
    valid.itertuples(index=False),
    total=len(valid),
    desc="Hashing images"
):
    path = Path(row.filepath)

    if not path.exists():
        raise FileNotFoundError(path)

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(chunk)

    digest = sha256.hexdigest()

    hash_groups[digest].append(
        (str(path), row.label)
    )

    records.append({
        "filepath": str(path),
        "label": row.label,
        "sha256": digest
    })

hash_df = pd.DataFrame(records)

hash_df.to_csv(
    OUTPUT_DIR / "image_hashes.csv",
    index=False
)

# Groups with more than one identical file.
duplicate_groups = {
    digest: items
    for digest, items in hash_groups.items()
    if len(items) > 1
}

duplicate_records = []

for digest, items in duplicate_groups.items():
    labels = {label for _, label in items}

    for filepath, label in items:
        duplicate_records.append({
            "sha256": digest,
            "filepath": filepath,
            "label": label,
            "group_size": len(items),
            "cross_class": len(labels) > 1
        })

duplicate_df = pd.DataFrame(
    duplicate_records,
    columns=[
        "sha256",
        "filepath",
        "label",
        "group_size",
        "cross_class"
    ]
)

duplicate_df.to_csv(
    OUTPUT_DIR / "exact_duplicates.csv",
    index=False
)

print("\nResults")
print("-" * 40)

print("Total valid images:", len(hash_df))
print("Unique file hashes:", hash_df["sha256"].nunique())
print("Duplicate groups:", len(duplicate_groups))
print("Images in duplicate groups:", len(duplicate_df))

cross_class_groups = sum(
    len({label for _, label in items}) > 1
    for items in duplicate_groups.values()
)

print("Cross-class duplicate groups:", cross_class_groups)

print("\nSaved:")
print(OUTPUT_DIR / "image_hashes.csv")
print(OUTPUT_DIR / "exact_duplicates.csv")

print("\nStep 14 completed.")
