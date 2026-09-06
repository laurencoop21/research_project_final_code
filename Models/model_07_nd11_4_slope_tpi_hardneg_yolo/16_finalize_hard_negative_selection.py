from pathlib import Path

import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent

scores_path = (
    MODEL_DIR
    / "reports"
    / "model6_mining_pool_scores.csv"
)

pre_qc_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_pre_qc.csv"
)

qc_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_qc.csv"
)

final_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_final.csv"
)

tile_size = 512
target_number = 30
maximum_overlap_fraction = 0.25

# QC exclusions
excluded_selection_ranks = {
    4,
    10,
    15,
    20,
    21,
    27,
}

exclusion_reason = (
    "Ambiguous compact anomaly; conservatively excluded "
    "because it could not confidently be treated as a "
    "true negative from the available imagery."
)

# Overlap function
def overlap_fraction(a, b):

    row_overlap = max(
        0,
        min(
            a["row_start"] + tile_size,
            b["row_start"] + tile_size,
        )
        - max(
            a["row_start"],
            b["row_start"],
        ),
    )

    col_overlap = max(
        0,
        min(
            a["col_start"] + tile_size,
            b["col_start"] + tile_size,
        )
        - max(
            a["col_start"],
            b["col_start"],
        ),
    )

    return (
        row_overlap * col_overlap
        / (tile_size * tile_size)
    )

# Load initial selection
pre_qc = pd.read_csv(pre_qc_path)

pre_qc["qc_decision"] = "keep"
pre_qc["qc_reason"] = (
    "Unambiguous non-target landscape feature."
)

mask = pre_qc["selection_rank"].isin(
    excluded_selection_ranks
)

pre_qc.loc[
    mask,
    "qc_decision"
] = "exclude"

pre_qc.loc[
    mask,
    "qc_reason"
] = exclusion_reason

pre_qc.to_csv(qc_path, index=False)

# Begin final list with retained candidates
kept = pre_qc[
    pre_qc["qc_decision"] == "keep"
].copy()

final_records = kept.to_dict("records")

selected_names = set(
    kept["tile_name"]
)

excluded_names = set(
    pre_qc.loc[
        pre_qc["qc_decision"] == "exclude",
        "tile_name",
    ]
)

# Search ranked pool for replacements
scores = pd.read_csv(scores_path)

scores = scores.sort_values(
    "max_confidence",
    ascending=False,
)

replacements = []

for _, candidate in scores.iterrows():

    name = candidate["tile_name"]

    if name in selected_names:
        continue

    if name in excluded_names:
        continue

    acceptable = True
    maximum_overlap = 0.0

    for previous in final_records:

        overlap = overlap_fraction(
            candidate,
            previous,
        )

        maximum_overlap = max(
            maximum_overlap,
            overlap,
        )

        if overlap > maximum_overlap_fraction:
            acceptable = False
            break

    if not acceptable:
        continue

    record = candidate.to_dict()

    record["max_overlap_with_selected"] = (
        maximum_overlap
    )

    record["qc_decision"] = (
        "replacement_pending_qc"
    )

    record["qc_reason"] = (
        "Automatically selected replacement "
        "after QC exclusion."
    )

    final_records.append(record)
    replacements.append(record)
    selected_names.add(name)

    if len(final_records) == target_number:
        break

# Save provisional final set
final_df = pd.DataFrame(final_records)

final_df = final_df.sort_values(
    "max_confidence",
    ascending=False,
).reset_index(drop=True)

final_df.insert(
    0,
    "final_rank",
    range(1, len(final_df) + 1),
)

final_df.to_csv(
    final_path,
    index=False,
)

# Print results
print("\nQC summary")
print("----------")
print("Initial candidates:", len(pre_qc))
print("Kept after QC:", len(kept))
print("Excluded after QC:", len(excluded_names))
print("Replacements selected:", len(replacements))
print("Provisional final total:", len(final_df))

print("\nExcluded:")
print(
    pre_qc.loc[
        pre_qc["qc_decision"] == "exclude",
        [
            "selection_rank",
            "tile_name",
            "max_confidence",
        ],
    ].to_string(index=False)
)

print("\nReplacement candidates:")
print(
    pd.DataFrame(replacements)[
        [
            "tile_name",
            "max_confidence",
            "max_overlap_with_selected",
        ]
    ].to_string(index=False)
)

print("\nSaved QC record:")
print(qc_path)

print("\nSaved provisional final selection:")
print(final_path)
