from pathlib import Path

import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent

scores_path = (
    MODEL_DIR
    / "reports"
    / "model6_mining_pool_scores.csv"
)

output_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_pre_qc.csv"
)

tile_size = 512
target_hard_negatives = 30
maximum_overlap_fraction = 0.25

# Load and rank candidates
scores = pd.read_csv(scores_path)

scores = scores.sort_values(
    by="max_confidence",
    ascending=False,
).reset_index(drop=True)

# Calculate overlap between two 512 x 512 windows
def overlap_fraction(candidate, selected):

    row_overlap = max(
        0,
        min(
            candidate["row_start"] + tile_size,
            selected["row_start"] + tile_size,
        )
        - max(
            candidate["row_start"],
            selected["row_start"],
        ),
    )

    col_overlap = max(
        0,
        min(
            candidate["col_start"] + tile_size,
            selected["col_start"] + tile_size,
        )
        - max(
            candidate["col_start"],
            selected["col_start"],
        ),
    )

    overlap_area = row_overlap * col_overlap

    tile_area = tile_size * tile_size

    return overlap_area / tile_area

# Select highest-scoring spatially diverse negatives
selected = []

for _, candidate in scores.iterrows():

    maximum_existing_overlap = 0.0

    acceptable = True

    for previous in selected:

        overlap = overlap_fraction(
            candidate,
            previous,
        )

        maximum_existing_overlap = max(
            maximum_existing_overlap,
            overlap,
        )

        if overlap > maximum_overlap_fraction:
            acceptable = False
            break

    if not acceptable:
        continue

    record = candidate.to_dict()

    record["max_overlap_with_selected"] = (
        maximum_existing_overlap
    )

    selected.append(record)

    if len(selected) == target_hard_negatives:
        break

# Save selection
selected_df = pd.DataFrame(selected)

selected_df.insert(
    0,
    "selection_rank",
    range(1, len(selected_df) + 1),
)

selected_df["selection_method"] = (
    "Model 6 confidence ranking with <=25% "
    "pairwise tile-area overlap"
)

selected_df.to_csv(
    output_path,
    index=False,
)

# Summary
print("\nHard-negative selection")
print("-----------------------")

print("Available candidates:", len(scores))
print("Requested:", target_hard_negatives)
print("Selected:", len(selected_df))

if len(selected_df) > 0:

    print(
        "Highest Model 6 confidence:",
        round(
            selected_df["max_confidence"].max(),
            3,
        ),
    )

    print(
        "Lowest selected confidence:",
        round(
            selected_df["max_confidence"].min(),
            3,
        ),
    )

    print(
        "Maximum selected overlap:",
        round(
            selected_df[
                "max_overlap_with_selected"
            ].max(),
            3,
        ),
    )

    print("\nSelected tiles:")

    print(
        selected_df[
            [
                "selection_rank",
                "tile_name",
                "max_confidence",
                "max_overlap_with_selected",
            ]
        ].to_string(index=False)
    )

print("\nSaved:")
print(output_path)

if len(selected_df) < target_hard_negatives:
    print(
        "\nWARNING: Spatial filtering prevented "
        "selection of the requested number."
    )
