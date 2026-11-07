
from pathlib import Path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download
from collections import Counter

DATASET_ID = "xtxx/HepatoBench"
REVISION = "3fd9d3697fb63a84ea0635aed97f77ad3f1c5d97"

archive_path = hf_hub_download(
    repo_id=DATASET_ID,
    filename="06_REA.zip",
    repo_type="dataset",
    revision=REVISION,
    local_files_only=True
)

with ZipFile(archive_path) as archive:
    files = [
        item for item in archive.infolist()
        if not item.is_dir()
    ]

    extensions = Counter(
        Path(item.filename).suffix.lower()
        for item in files
    )

    print("Total files:", len(files))
    print("\nFile types:")

    for extension, count in extensions.items():
        print(f"  {extension or '[none]'}: {count}")

    print("\nFirst 20 archive entries:")
    for item in files[:20]:
        print(" ", item.filename)

    metadata_files = [
        item.filename
        for item in files
        if Path(item.filename).suffix.lower()
        in {".csv", ".json", ".tsv", ".txt", ".xml"}
    ]

    print("\nPotential metadata files:")
    print(metadata_files if metadata_files else "None found")
