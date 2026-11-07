
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO

from huggingface_hub import hf_hub_download
from PIL import Image

DATASET_ID = "xtxx/HepatoBench"
REVISION = "3fd9d3697fb63a84ea0635aed97f77ad3f1c5d97"

FILENAME = "02_FIB_3355.png"
DESTINATION = Path("data/raw/HepatoBench/FIB") / FILENAME

archive_path = hf_hub_download(
    repo_id=DATASET_ID,
    filename="02_FIB.zip",
    repo_type="dataset",
    revision=REVISION,
    local_files_only=True
)

print("Inspecting original archive:", archive_path)

with ZipFile(archive_path) as archive:
    matches = [
        name for name in archive.namelist()
        if Path(name).name == FILENAME
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one archive entry, found {len(matches)}"
        )

    data = archive.read(matches[0])

print("Archived file size:", len(data), "bytes")

if not data:
    print(
        "Original archive entry is empty. "
        "The image cannot be recovered from this ZIP."
    )
else:
    # Validate before replacing the local file.
    with Image.open(BytesIO(data)) as image:
        image.load()
        print("Recovered image dimensions:", image.size)
        print("Recovered image mode:", image.mode)

    if DESTINATION.exists() and DESTINATION.stat().st_size > 0:
        raise RuntimeError(
            "Destination is not empty; refusing to overwrite."
        )

    DESTINATION.write_bytes(data)
    print("Image successfully restored:", DESTINATION)
