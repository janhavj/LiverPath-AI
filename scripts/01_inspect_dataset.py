
from huggingface_hub import HfApi
from pathlib import Path
import json

DATASET_ID = "xtxx/HepatoBench"

api = HfApi()

print("=" * 55)
print("LiverPath AI - HepatoBench Dataset Inspection")
print("=" * 55)

# Retrieve dataset metadata without downloading images.
info = api.dataset_info(
    repo_id=DATASET_ID,
    files_metadata=True
)

print("\nDataset:", info.id)
print("Revision:", info.sha)

print("\nAvailable files:")
file_records = []

for file in info.siblings:
    size = getattr(file, "size", None)

    file_records.append({
        "filename": file.rfilename,
        "size_bytes": size
    })

    size_mb = size / (1024 ** 2) if size is not None else None
    size_label = (
        f"{size_mb:.2f} MB"
        if size_mb is not None
        else "Unknown size"
    )

    print(f"  {file.rfilename} ({size_label})")

# Save metadata for reproducibility.
output_dir = Path("results/metrics")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "dataset_manifest.json"

with open(output_file, "w") as f:
    json.dump(
        {
            "dataset_id": info.id,
            "revision": info.sha,
            "files": file_records
        },
        f,
        indent=2
    )

print("\nSaved manifest:", output_file)

print("\nInspection completed.")
