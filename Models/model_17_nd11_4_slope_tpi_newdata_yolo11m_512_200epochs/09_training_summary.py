from pathlib import Path
import json

import pandas as pd


# File paths
MODEL_DIR = Path(__file__).resolve().parent
RESULTS_PATH = MODEL_DIR / "runs" / "baseline" / "results.csv"
REPORTS_DIR = MODEL_DIR / "reports"

REPORTS_DIR.mkdir(exist_ok=True)

# Read the Ultralytics training log
df = pd.read_csv(RESULTS_PATH)
df.columns = df.columns.str.strip()

# Metrics to summarise
metrics = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

summary = {}

# Find the best and final epoch for each metric
for metric in metrics:
    best_idx = df[metric].idxmax()

    summary[metric] = {
        "best_epoch": int(df.loc[best_idx, "epoch"]),
        "best_value": float(df.loc[best_idx, metric]),
        "final_epoch": int(df["epoch"].iloc[-1]),
        "final_value": float(df.iloc[-1][metric]),
    }

# Save the results
output_path = REPORTS_DIR / "model_17_training_summary.json"

with open(output_path, "w") as file:
    json.dump(summary, file, indent=4)

# Print the results
print("\nModel 17 training summary")
print("-------------------------")

for metric, values in summary.items():
    print(f"\n{metric}")
    print("Best epoch:", values["best_epoch"])
    print("Best value:", values["best_value"])
    print("Final epoch:", values["final_epoch"])
    print("Final value:", values["final_value"])

print("\nSaved to:")
print(output_path)
