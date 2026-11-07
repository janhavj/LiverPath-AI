
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from huggingface_hub import hf_hub_download
from collections import Counter
import csv

DATASET_ID = "xtxx/HepatoBench"
REVISION = "3fd9d3697fb63a84ea0635aed97f77ad3f1c5d97"

ARCHIVES = {
    "01_TUM.zip": ("TUM", 13200),
    "02_FIB.zip": ("FIB", 26364),
    "03_INF.zip": ("INF", 4776),
    "04_NEC.zip": ("NEC", 22023),
    "05_NOR.zip": ("NOR", 13200),
    "06_REA.zip": ("REA", 1540),
    "07_STE.zip": ("STE", 6198),
}

DATA_DIR = Path("data/raw/HepatoBench")
DATA_DIR.mkdir(parents=True, exist_ok=True)

all_records = []

for archive_name, (label, expected_count) in ARCHIVES.items():

    print(f"\n{'=' * 60}")
    print(f"Processing {archive_name}")
    print("=" * 60)

    # Hugging Face reuses previously downloaded files.
    archive_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=archive_name,
        repo_type="dataset",
        revision=REVISION
    )

    destination = DATA_DIR / label
    destination.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(archive_path) as archive:

            if archive.testzip() is not None:
                raise RuntimeError(
                    f"Corrupted archive: {archive_name}"
                )

            members = [
                item for item in archive.infolist()
                if not item.is_dir()
                and item.filename.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                )
            ]

            if len(members) != expected_count:
                raise RuntimeError(
                    f"{label}: Expected {expected_count} images, "
                    f"found {len(members)}"
                )

            for item in members:
                # Flatten archive paths and reject duplicates.
                filename = Path(item.filename).name
                output_path = destination / filename

                if not output_path.exists():
                    output_path.write_bytes(
                        archive.read(item)
                    )

                all_records.append({
                    "filename": filename,
                    "filepath": str(output_path),
                    "label": label
                })

    except BadZipFile as error:
        raise RuntimeError(
            f"Invalid ZIP archive: {archive_name}"
        ) from error

    print(f"Verified and extracted {len(members)} images")
    print(f"Saved to: {destination}")

# Save a dataset index.
index_path = Path("data/processed/image_index.csv")
index_path.parent.mkdir(parents=True, exist_ok=True)

with index_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["filename", "filepath", "label"]
    )
    writer.writeheader()
    writer.writerows(all_records)

counts = Counter(record["label"] for record in all_records)

print("\n" + "=" * 60)
print("DOWNLOAD AND EXTRACTION COMPLETE")
print("=" * 60)

for label, count in sorted(counts.items()):
    print(f"{label}: {count:,} images")

print(f"\nTotal images: {len(all_records):,}")
print(f"Dataset index: {index_path}")

assert len(all_records) == 87301
print("\nAll expected images accounted for!")
