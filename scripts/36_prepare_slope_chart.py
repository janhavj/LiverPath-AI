
from pathlib import Path
import pandas as pd

source = Path(
    "results/metrics/resnet18_per_class_comparison.csv"
)

df = pd.read_csv(source)

# The first column contains the tissue class labels.
class_column = df.columns[0]

output = pd.DataFrame({
    "class": df[class_column],
    "baseline_f1": df["baseline_f1"].round(3),
    "finetuned_f1": df["finetuned_f1"].round(3)
})

destination = Path(
    "results/visualisation_data/model_f1_slope.csv"
)

destination.parent.mkdir(parents=True, exist_ok=True)
output.to_csv(destination, index=False)

print(output.to_string(index=False))
print("\nSaved:", destination)
