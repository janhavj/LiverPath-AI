
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm

INPUT = Path("results/metrics/perceptual_hashes.csv")
OUTPUT = Path("results/metrics/phash_near_duplicate_pairs.csv")

THRESHOLD = 5

df = pd.read_csv(INPUT)
hashes = [int(value, 16) for value in df["phash"]]

def hamming_distance(a, b):
    return (a ^ b).bit_count()

class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, index, {}]
            return

        node = self.root

        while True:
            distance = hamming_distance(value, node[0])

            if distance == 0:
                return

            children = node[2]

            if distance in children:
                node = children[distance]
            else:
                children[distance] = [value, index, {}]
                return

    def search(self, value, radius):
        if self.root is None:
            return []

        results = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node[0])

            if distance <= radius:
                results.append((node[1], distance))

            lower = distance - radius
            upper = distance + radius

            for edge_distance, child in node[2].items():
                if lower <= edge_distance <= upper:
                    stack.append(child)

        return results

print("=" * 60)
print("LiverPath AI - Near-Duplicate Detection")
print("=" * 60)

tree = BKTree()
pairs = []

for i, value in enumerate(
    tqdm(hashes, desc="Searching perceptual neighbours")
):
    neighbours = tree.search(value, THRESHOLD)

    for j, distance in neighbours:
        pairs.append({
            "index_a": j,
            "index_b": i,
            "filepath_a": df.iloc[j]["filepath"],
            "filepath_b": df.iloc[i]["filepath"],
            "label_a": df.iloc[j]["label"],
            "label_b": df.iloc[i]["label"],
            "hamming_distance": distance
        })

    tree.add(value, i)

pairs_df = pd.DataFrame(
    pairs,
    columns=[
        "index_a", "index_b",
        "filepath_a", "filepath_b",
        "label_a", "label_b",
        "hamming_distance"
    ]
)

pairs_df.to_csv(OUTPUT, index=False)

print("\nResults")
print("-" * 40)
print("Images analysed:", len(df))
print("Candidate near-duplicate pairs:", len(pairs_df))

if not pairs_df.empty:
    print("\nHamming distance distribution:")
    print(
        pairs_df["hamming_distance"]
        .value_counts()
        .sort_index()
    )

    cross_class = (
        pairs_df["label_a"] != pairs_df["label_b"]
    ).sum()

    print("\nCross-class candidate pairs:", cross_class)

print("\nSaved:", OUTPUT)
print("Step 16 completed.")
