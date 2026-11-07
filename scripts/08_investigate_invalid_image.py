
from pathlib import Path
from PIL import Image
import pandas as pd
import hashlib

qc = pd.read_csv("results/metrics/image_qc_results.csv")

invalid = qc[qc["valid"] == False].copy()

print("=" * 55)
print("LiverPath AI - Invalid Image Investigation")
print("=" * 55)

print("\nInvalid image records:")
print(invalid[["filepath", "label", "error"]].to_string(index=False))

for _, row in invalid.iterrows():
    path = Path(row["filepath"])

    print("\nFile:", path)
    print("Exists:", path.exists())

    if path.exists():
        print("Size:", path.stat().st_size, "bytes")

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print("SHA256:", digest)

        try:
            with Image.open(path) as image:
                print("Format:", image.format)
                print("Dimensions:", image.size)
                image.verify()
                print("Pillow verification passed")
        except Exception as exc:
            print("Verification failed:", repr(exc))

print("\nInvestigation completed.")
