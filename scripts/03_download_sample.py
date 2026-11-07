
from huggingface_hub import hf_hub_download
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
import matplotlib.pyplot as plt
import random

DATASET_ID = "xtxx/HepatoBench"
REVISION = "3fd9d3697fb63a84ea0635aed97f77ad3f1c5d97"
ARCHIVE = "06_REA.zip"

output_dir = Path("data/raw/REA")
figure_dir = Path("results/figures")

output_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

print("Downloading:", ARCHIVE)

zip_path = hf_hub_download(
    repo_id=DATASET_ID,
    filename=ARCHIVE,
    repo_type="dataset",
    revision=REVISION
)

print("Archive downloaded:", zip_path)

with ZipFile(zip_path) as archive:
    images = [
        name for name in archive.namelist()
        if name.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        )
    ]

    print("Images found:", len(images))
    print("First 10 filenames:")
    for name in images[:10]:
        print(" ", name)

    random.seed(42)
    selected = random.sample(images, min(12, len(images)))

    # Extract selected files only.
    for index, name in enumerate(selected):
        data = archive.read(name)
        destination = output_dir / f"REA_{index:03d}{Path(name).suffix.lower()}"
        destination.write_bytes(data)

# Create an image gallery.
files = sorted(output_dir.iterdir())
files = [p for p in files if p.is_file()][:12]

fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for ax, path in zip(axes.flat, files):
    with Image.open(path) as image:
        image = image.convert("RGB")
        ax.imshow(image)
        ax.set_title(path.name, fontsize=9)
    ax.axis("off")

for ax in axes.flat[len(files):]:
    ax.axis("off")

plt.suptitle("HepatoBench - Biliary Reaction", fontsize=15)
plt.tight_layout()

figure_path = figure_dir / "01_REA_sample_gallery.png"
plt.savefig(figure_path, dpi=200)
plt.close()

print("\nSaved images to:", output_dir)
print("Saved gallery to:", figure_path)
print("Step 5 completed.")
