
from pathlib import Path
from collections import Counter
from PIL import Image, UnidentifiedImageError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

INDEX_PATH = Path("data/processed/image_index.csv")
OUTPUT_DIR = Path("results/metrics")
FIGURE_DIR = Path("results/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INDEX_PATH)

print("=" * 60)
print("LiverPath AI - Comprehensive Image QC")
print("=" * 60)
print("Total indexed images:", len(df))

records = []

for row in tqdm(
    df.itertuples(index=False),
    total=len(df),
    desc="Checking images"
):
    path = Path(row.filepath)

    record = {
        "filepath": str(path),
        "label": row.label,
        "valid": False,
        "width": np.nan,
        "height": np.nan,
        "mode": None,
        "mean_r": np.nan,
        "mean_g": np.nan,
        "mean_b": np.nan,
        "brightness": np.nan,
        "contrast": np.nan,
        "error": None,
    }

    try:
        with Image.open(path) as image:
            image.load()

            record["width"], record["height"] = image.size
            record["mode"] = image.mode

            rgb = np.asarray(
                image.convert("RGB"),
                dtype=np.float32
            )

            channel_means = rgb.mean(axis=(0, 1))

            record["mean_r"] = channel_means[0]
            record["mean_g"] = channel_means[1]
            record["mean_b"] = channel_means[2]

            gray = (
                0.299 * rgb[:, :, 0]
                + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]
            )

            record["brightness"] = gray.mean()
            record["contrast"] = gray.std()
            record["valid"] = True

    except (OSError, ValueError, UnidentifiedImageError) as exc:
        record["error"] = str(exc)

    records.append(record)

qc = pd.DataFrame(records)

qc_path = OUTPUT_DIR / "image_qc_results.csv"
qc.to_csv(qc_path, index=False)

print("\nImage validation")
print("----------------")
print("Valid images:", int(qc["valid"].sum()))
print("Invalid images:", int((~qc["valid"]).sum()))

valid = qc[qc["valid"]].copy()

print("\nImage dimensions:")
print(
    valid.groupby(["width", "height"])
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("\nImage modes:")
print(valid["mode"].value_counts())

print("\nPer-class summary:")
summary = valid.groupby("label").agg(
    images=("filepath", "count"),
    mean_brightness=("brightness", "mean"),
    mean_contrast=("contrast", "mean"),
    mean_red=("mean_r", "mean"),
    mean_green=("mean_g", "mean"),
    mean_blue=("mean_b", "mean"),
).round(2)

print(summary)

summary.to_csv(OUTPUT_DIR / "image_qc_summary.csv")

# Class distribution
fig, ax = plt.subplots(figsize=(9, 5))

counts = valid["label"].value_counts().sort_index()

ax.bar(counts.index, counts.values)
ax.set_xlabel("Tissue category")
ax.set_ylabel("Number of patches")
ax.set_title("HepatoBench Class Distribution")

for i, count in enumerate(counts.values):
    ax.text(i, count + 250, str(count), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "02_class_distribution.png",
    dpi=200
)
plt.close()

# Brightness distribution
fig, ax = plt.subplots(figsize=(9, 5))

for label, group in valid.groupby("label"):
    ax.hist(
        group["brightness"],
        bins=40,
        alpha=0.35,
        label=label,
        density=True
    )

ax.set_xlabel("Mean grayscale brightness")
ax.set_ylabel("Density")
ax.set_title("Brightness Distribution by Tissue Category")
ax.legend()

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "03_brightness_distribution.png",
    dpi=200
)
plt.close()

# Contrast distribution
fig, ax = plt.subplots(figsize=(9, 5))

valid.boxplot(
    column="contrast",
    by="label",
    ax=ax
)

ax.set_xlabel("Tissue category")
ax.set_ylabel("Grayscale standard deviation")
ax.set_title("Image Contrast by Tissue Category")
fig.suptitle("")

plt.tight_layout()
plt.savefig(
    FIGURE_DIR / "04_contrast_distribution.png",
    dpi=200
)
plt.close()

print("\nSaved:")
print(qc_path)
print(OUTPUT_DIR / "image_qc_summary.csv")
print(FIGURE_DIR / "02_class_distribution.png")
print(FIGURE_DIR / "03_brightness_distribution.png")
print(FIGURE_DIR / "04_contrast_distribution.png")

print("\nStep 11 completed.")
