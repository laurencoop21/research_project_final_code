from pathlib import Path
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent

scores_path = (
    MODEL_DIR
    / "reports"
    / "model6_mining_pool_scores.csv"
)

qc_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_qc.csv"
)

current_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_final_round2.csv"
)

output_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_final_round3.csv"
)

tile_size = 512
maximum_overlap_fraction = 0.25
target_number = 30

# Candidates excluded in previous replacement QC
previous_replacement_exclusions = {
    "candidate_0162_r3214_c1451",
    "candidate_0004_r2569_c1199",
    "candidate_0212_r10373_c843",
}

# Candidates excluded in this QC round
new_exclusions = {
    "candidate_0332_r2631_c1079",
    "candidate_0188_r10362_c974",
}

# This replacement passed QC
newly_approved = {
    "candidate_0098_r2080_c4240",
}


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

# Load data
scores = pd.read_csv(scores_path)
current = pd.read_csv(current_path)
qc = pd.read_csv(qc_path)

historical_exclusions = set(
    qc.loc[
        qc["qc_decision"] == "exclude",
        "tile_name",
    ]
)

all_exclusions = (
    historical_exclusions
    | previous_replacement_exclusions
    | new_exclusions
)

# Record current QC decision
current.loc[
    current["tile_name"].isin(newly_approved),
    "qc_decision",
] = "keep"

current.loc[
    current["tile_name"].isin(newly_approved),
    "qc_reason",
] = (
    "Passed final replacement QC; false-positive "
    "response associated with interpretable "
    "non-target linear/agricultural landscape features."
)


# Remove the two ambiguous candidates
kept = current[
    ~current["tile_name"].isin(new_exclusions)
].copy()

final_records = kept.to_dict("records")

selected_names = set(
    kept["tile_name"]
)

# Select the next two eligible candidates
scores = scores.sort_values(
    "max_confidence",
    ascending=False,
)

new_replacements = []

for _, candidate in scores.iterrows():

    name = candidate["tile_name"]

    if name in selected_names:
        continue

    if name in all_exclusions:
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
        "Automatically selected after third-round "
        "visual QC exclusion."
    )

    final_records.append(record)
    new_replacements.append(record)
    selected_names.add(name)

    if len(final_records) == target_number:
        break

# Save
final_df = pd.DataFrame(final_records)

final_df = final_df.sort_values(
    "max_confidence",
    ascending=False,
).reset_index(drop=True)

final_df["final_rank"] = range(
    1,
    len(final_df) + 1,
)

final_df.to_csv(
    output_path,
    index=False,
)


print("\nThird-round QC")
print("--------------")
print("Approved before replacements:", len(kept))
print("New replacements:", len(new_replacements))
print("Provisional final total:", len(final_df))

print("\nNewly excluded:")
for name in sorted(new_exclusions):
    print(name)

print("\nNew replacement candidates:")

if new_replacements:
    print(
        pd.DataFrame(new_replacements)[
            [
                "tile_name",
                "max_confidence",
                "max_overlap_with_selected",
            ]
        ].to_string(index=False)
    )

print("\nSaved:")
print(output_path)
