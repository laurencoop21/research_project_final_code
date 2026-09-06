from pathlib import Path
import json

import torch
from ultralytics import YOLO

# File paths
MODEL_DIR = Path(__file__).resolve().parent

dataset_yaml = MODEL_DIR / "dataset" / "dataset.yaml"
best_model_path = (
    MODEL_DIR / "checkpoints" / "model_03_best.pt"
)

runs_dir = MODEL_DIR / "runs"
# Save the results
reports_dir = MODEL_DIR / "reports"

reports_dir.mkdir(exist_ok=True)

# Check that the required files exist
if not dataset_yaml.exists():
    raise FileNotFoundError(
        f"Missing dataset file: {dataset_yaml}"
    )

if not best_model_path.exists():
    raise FileNotFoundError(
        f"Missing trained model: {best_model_path}"
    )

# Choose the device
if torch.cuda.is_available():
    device = 0
else:
    device = "cpu"

# Load the best validation checkpoint
model = YOLO(str(best_model_path))

# Evaluate on the untouched test split
results = model.val(
    data=str(dataset_yaml),
    split="test",
    imgsz=512,
    batch=8,
    device=device,
    workers=0,
    plots=True,
    project=str(runs_dir),
    name="test_evaluation",
)

# Save the test metrics
metrics = {
    "precision": float(results.box.mp),
    "recall": float(results.box.mr),
    "mAP50": float(results.box.map50),
    "mAP50_95": float(results.box.map),
}

metrics_path = reports_dir / "model_03_test_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

# Print the results
print("\nModel 3 test metrics")
print("--------------------")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"mAP50:     {metrics['mAP50']:.4f}")
print(f"mAP50-95:  {metrics['mAP50_95']:.4f}")

print("\nSaved test metrics to:")
print(metrics_path)

print("\nSaved test plots to:")
print(runs_dir / "test_evaluation")