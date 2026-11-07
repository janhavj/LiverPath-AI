
from pathlib import Path
import pandas as pd

# Load the final test confusion matrix
source = Path("results/final_test/confusion_matrix.csv")

cm = pd.read_csv(source, index_col=0)

# Convert matrix into a format suitable for RAWGraphs
flow = (
    cm.rename_axis("true_class")
    .reset_index()
    .melt(
        id_vars="true_class",
        var_name="predicted_class",
        value_name="patch_count"
    )
)

# Keep misclassifications only
flow = flow[
    (flow["true_class"] != flow["predicted_class"])
    & (flow["patch_count"] > 0)
].copy()

flow = flow.sort_values(
    "patch_count",
    ascending=False
)

# Create output directory
output_dir = Path("results/visualisation_data")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "misclassification_flows.csv"

flow.to_csv(output_file, index=False)

print("=" * 55)
print("LiverPath AI - RAWGraphs Data Preparation")
print("=" * 55)
print("Saved:", output_file)
print("Misclassification patterns:", len(flow))
print("Total errors:", flow["patch_count"].sum())
print("\nTop 10 errors:")
print(flow.head(10).to_string(index=False))

assert flow["patch_count"].sum() == 926

print("\nStep completed successfully.")
