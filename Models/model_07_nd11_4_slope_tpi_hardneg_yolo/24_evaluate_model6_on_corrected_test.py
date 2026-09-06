from pathlib import Path
import json

import torch
from ultralytics import YOLO

# File paths
MODEL_DIR = Path(__file__).resolve().parent

dataset_yaml = MODEL_DIR / "dataset" / "dataset.yaml"

model6_path = (
    MODEL_DIR.parent
    / "model_06_nd11_4_slope_tpi_yolo"
    / "checkpoints"
    / "model_06_best.pt"
)

runs_dir = MODEL_DIR / "runs"
reports_dir = MODEL_DIR / "reports"

reports_dir.mkdir(exist_ok=True)

# Checks
if not dataset_yaml.exists():
    raise FileNotFoundError(dataset_yaml)

if not model6_path.exists():
    raise FileNotFoundError(model6_path)

# Device
device = 0 if torch.cuda.is_available() else "cpu"

# Load Model 6
model = YOLO(str(model6_path))

# Evaluate Model 6 using the corrected Model 7
# test dataset
results = model.val(
    data=str(dataset_yaml),
    split="test",
    imgsz=512,
    batch=8,
    device=device,
    workers=0,
    plots=True,
    project=str(runs_dir),
    name="model6_corrected_test_evaluation",
)

# Save metrics
metrics = {
    "precision": float(results.box.mp),
    "recall": float(results.box.mr),
    "mAP50": float(results.box.map50),
    "mAP50_95": float(results.box.map),
}

output_path = (
    reports_dir
    / "model_06_corrected_test_metrics.json"
)

with open(output_path, "w") as file:
    json.dump(metrics, file, indent=4)


print("\nModel 6 on corrected test set")
print("-----------------------------")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"mAP50:     {metrics['mAP50']:.4f}")
print(f"mAP50-95:  {metrics['mAP50_95']:.4f}")

print("\nSaved to:")
print(output_path)
