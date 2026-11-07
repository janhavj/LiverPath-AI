
from pathlib import Path
import pandas as pd

output_dir = Path("results/visualisation_data")
output_dir.mkdir(parents=True, exist_ok=True)

# Frozen final-test results, not validation estimates.
review_data = [
    (0, 0.0),
    (5, 30.67),
    (10, 56.80),
    (20, 81.53),
    (30, 91.58),
]

rows = []

for budget, captured in review_data:
    rows.append({
        "review_fraction": budget,
        "errors_captured": captured,
        "review_method": "Entropy-guided"
    })
    rows.append({
        "review_fraction": budget,
        "errors_captured": float(budget),
        "review_method": "Random expectation"
    })

df = pd.DataFrame(rows)

output = output_dir / "uncertainty_review_rawgraphs.csv"
df.to_csv(output, index=False)

print(df.to_string(index=False))
print("\nSaved:", output)
