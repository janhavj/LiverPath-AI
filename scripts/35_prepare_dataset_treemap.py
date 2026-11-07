
from pathlib import Path
import pandas as pd

split_dir = Path("data/processed/splits")
output_dir = Path("results/visualisation_data")
output_dir.mkdir(parents=True, exist_ok=True)

frames = []

for split in ["train", "val", "test"]:
    df = pd.read_csv(split_dir / f"{split}.csv")
    counts = df["label"].value_counts()

    for label, count in counts.items():
        frames.append({
            "class": label,
            "split": split,
            "patch_count": int(count)
        })

result = pd.DataFrame(frames)

output = output_dir / "dataset_composition.csv"
result.to_csv(output, index=False)

print("Saved:", output)
print("\nClass totals:")
print(result.groupby("class")["patch_count"].sum())
print("\nTotal:", result["patch_count"].sum())

assert result["patch_count"].sum() == 87220
