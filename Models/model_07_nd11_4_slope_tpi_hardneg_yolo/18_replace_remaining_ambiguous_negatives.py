from pathlib import Path
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent

scores_path = MODEL_DIR / "reports" / "model6_mining_pool_scores.csv"
qc_path = MODEL_DIR / "reports" / "selected_hard_negatives_qc.csv"
current_path = MODEL_DIR / "reports" / "selected_hard_negatives_final.csv"

output_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_final_round2.csv"
)

tile_size = 512
maximum_overlap_fraction = 0.25
target_number = 30


# These three replacements were judged too ambiguous
# during the second visual QC step.
new_exclusions = {
    "candidate_0162_r3214_c1451",
    "candidate_0004_r2569_c1199",
    "candidate_0212_r10373_c843",
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

# Load existing records
current = pd.read_csv(current_path)
qc = pd.read_csv(qc_path)
scores = pd.read_csv(scores_path)

# Historical exclusions from first QC
historical_exclusions = set(
    qc.loc[
        qc["qc_decision"] == "exclude",
        "tile_name",
    ]
)

all_exclusions = historical_exclusions | new_exclusions

# Keep everything except the new ambiguous replacements
kept = current[
    ~current["tile_name"].isin(new_exclusions)
].copy()

# The three replacement candidates that passed visual QC
passed_replacements = {
    "candidate_0003_r9402_c3019",
    "candidate_0085_r10013_c3986",
    "candidate_0242_r6225_c3644",
}

kept.loc[
    kept["tile_name"].isin(passed_replacements),
    "qc_decision",
] = "keep"

kept.loc[
    kept["tile_name"].isin(passed_replacements),
    "qc_reason",
] = (
    "Passed replacement visual QC; response associated "
    "with interpretable non-target landscape features."
)

final_records = kept.to_dict("records")
selected_names = set(kept["tile_name"])

# Select next three eligible candidates
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

    record["max_overlap_with_selected"] = maximum_overlap
    record["qc_decision"] = "replacement_pending_qc"
    record["qc_reason"] = (
        "Automatically selected after second-round "
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

final_df.to_csv(output_path, index=False)


print("\nSecond-round QC")
print("---------------")
print("Kept before replacements:", len(kept))
print("New replacements:", len(new_replacements))
print("Provisional final total:", len(final_df))

print("\nNewly excluded:")
for name in sorted(new_exclusions):
    print(name)

print("\nNew replacement candidates:")
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
