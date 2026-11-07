
from huggingface_hub import HfApi
from pathlib import Path
import json

DATASET_ID = "xtxx/HepatoBench"
REVISION = "3fd9d3697fb63a84ea0635aed97f77ad3f1c5d97"

api = HfApi()

print("=" * 55)
print("LiverPath AI - Dataset Provenance Audit")
print("=" * 55)

info = api.dataset_info(
    repo_id=DATASET_ID,
    revision=REVISION
)

metadata = json.loads(
    Path("results/metrics/dataset_meta.json").read_text()
)

readme = Path("results/metrics/README.md").read_text()

print("\nDataset:", info.id)
print("Revision:", info.sha)

print("\nDeclared characteristics:")
print("Classes:", metadata["dataset"]["num_classes"])
print("Patches:", metadata["dataset"]["num_samples"])
print("Dimensions:", metadata["dataset"]["pixel"])

print("\nDocumentation checks:")

checks = {
    "Patient identifiers": ["patient_id", "patient id"],
    "Slide identifiers": ["slide_id", "slide id"],
    "Data provenance": ["source", "origin", "provenance"],
    "Annotation methodology": ["annotat", "pathologist"],
    "Train/test split": ["train/test", "test split"],
    "External validation": ["external validation"],
}

documentation = readme.lower()

for item, keywords in checks.items():
    mentioned = any(
        keyword in documentation
        for keyword in keywords
    )
    status = "Mentioned" if mentioned else "Not documented"
    print(f"{item}: {status}")

print("\nAudit completed.")
print(
    "\nNote: These are documentation keyword checks, "
    "not verification of dataset provenance."
)
