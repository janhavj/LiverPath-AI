
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import imagehash
import numpy as np

HASH_PATH = Path("results/metrics/image_hashes.csv")
OUTPUT_DIR = Path("results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(HASH_PATH)

print("=" * 60)
print("LiverPath AI - Perceptual Hash Analysis")
print("=" * 60)

# Hash only one representative of each exact duplicate.
unique = df.drop_duplicates("sha256").copy()

records = []

for row in tqdm(
    unique.itertuples(index=False),
    total=len(unique),
    desc="Computing perceptual hashes"
):
    with Image.open(row.filepath) as image:
        phash = imagehash.phash(image.convert("RGB"))

    records.append({
        "filepath": row.filepath,
        "label": row.label,
        "sha256": row.sha256,
        "phash": str(phash)
    })

result = pd.DataFrame(records)

result.to_csv(
    OUTPUT_DIR / "perceptual_hashes.csv",
    index=False
)

# Identify identical perceptual hashes.
groups = defaultdict(list)

for row in result.itertuples(index=False):
    groups[row.phash].append(
        (row.filepath, row.label)
    )

matching_groups = {
    key: values
    for key, values in groups.items()
    if len(values) > 1
}

print("\nResults")
print("-" * 40)
print("Unique images analysed:", len(result))
print("Unique perceptual hashes:", result["phash"].nunique())
print("Identical pHash groups:", len(matching_groups))

matching_records = []

for phash, images in matching_groups.items():
    for filepath, label in images:
        matching_records.append({
            "phash": phash,
            "filepath": filepath,
            "label": label,
            "group_size": len(images)
        })

pd.DataFrame(
    matching_records,
    columns=["phash", "filepath", "label", "group_size"]
).to_csv(
    OUTPUT_DIR / "matching_perceptual_hashes.csv",
    index=False
)

print("\nSaved perceptual hash results.")
print("Step 15 completed.")
