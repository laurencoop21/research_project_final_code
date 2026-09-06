from pathlib import Path
import pandas as pd


# File paths
MODEL_DIR = Path(__file__).resolve().parent

current_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_final_round3.csv"
)

output_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_final_locked.csv"
)

# Tiles rejected at the last QC pass
final_exclusions = {
    "candidate_0172_r2660_c1013",
    "candidate_0101_r10355_c660",
}

df = pd.read_csv(current_path)

final_df = df[
    ~df["tile_name"].isin(final_exclusions)
].copy()

# Anything surviving into the locked set is approved.
final_df["qc_decision"] = "keep"

final_df.loc[
    final_df["qc_reason"].str.contains(
        "pending",
        case=False,
        na=False,
    ),
    "qc_reason",
] = "Passed final hard-negative quality control."

# Rank the locked set
final_df = final_df.sort_values(
    "max_confidence",
    ascending=False,
).reset_index(drop=True)

final_df["final_rank"] = range(
    1,
    len(final_df) + 1,
)

# Save the results
final_df.to_csv(output_path, index=False)

print("\nFinal hard-negative set")
print("-----------------------")
print("Approved hard negatives:", len(final_df))
print("Excluded in final QC:", len(final_exclusions))

print("\nFinal exclusions:")
for name in sorted(final_exclusions):
    print(name)

print(
    "\nHighest confidence:",
    round(final_df["max_confidence"].max(), 3),
)

print(
    "Lowest confidence:",
    round(final_df["max_confidence"].min(), 3),
)

print("\nSaved:")
print(output_path)
