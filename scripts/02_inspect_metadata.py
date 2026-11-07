
from huggingface_hub import hf_hub_download
from pathlib import Path
import json

DATASET_ID = "xtxx/HepatoBench"
REVISION = "3fd9d3697fb63a84ea0635aed97f77ad3f1c5d97"

output_dir = Path("results/metrics")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("LiverPath AI - Dataset Metadata Inspection")
print("=" * 55)

for filename in ["README.md", "dataset_meta.json"]:

    print(f"\n{'=' * 55}")
    print(f"FILE: {filename}")
    print("=" * 55)

    file_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=filename,
        repo_type="dataset",
        revision=REVISION
    )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(content)

    # Save a local copy for reproducibility.
    destination = output_dir / filename
    destination.write_text(content, encoding="utf-8")

    print(f"\nSaved to: {destination}")

print("\nMetadata inspection completed.")
