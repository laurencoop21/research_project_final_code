from pathlib import Path
import json

from ultralytics import YOLO


# File paths
MODEL_DIR = Path(__file__).resolve().parent
MODELS_DIR = MODEL_DIR.parent

# Model 9 best checkpoint
checkpoint_path = MODEL_DIR / "checkpoints" / "model_09_best.pt"

# Corrected common test dataset used for the fair Model 6/7/8 comparison
data_path = (
    MODELS_DIR
    / "model_07_nd11_4_slope_tpi_hardneg_yolo"
    / "dataset"
    / "dataset.yaml"
)

print("Checkpoint:", checkpoint_path)
print("Dataset:", data_path)

# Load the best checkpoint
model = YOLO(str(checkpoint_path))

# Run validation on the corrected test split
metrics = model.val(
    data=str(data_path),
    split="test",
    imgsz=512,
    batch=8,
    device=0,
    project=str(MODEL_DIR / "runs"),
    name="corrected_test_evaluation",
    exist_ok=True,
    plots=True,
)

# Collect the headline metrics
results = {
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "mAP50": float(metrics.box.map50),
    "mAP50_95": float(metrics.box.map),
}

# Save the results
reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

output_path = reports_dir / "model_09_corrected_test_metrics.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

# Print the results
print("\nModel 9 corrected test metrics")
print("------------------------------")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall:    {results['recall']:.4f}")
print(f"mAP50:     {results['mAP50']:.4f}")
print(f"mAP50-95:  {results['mAP50_95']:.4f}")

print(f"\nSaved metrics to:\n{output_path}")
