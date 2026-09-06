from pathlib import Path
import json

from ultralytics import YOLO


# File paths
MODEL_DIR = Path(__file__).resolve().parent

checkpoint_path = MODEL_DIR / "checkpoints" / "model_10_best.pt"
data_path = MODEL_DIR / "dataset" / "dataset.yaml"

print("Checkpoint:", checkpoint_path)
print("Dataset:", data_path)

# Load the best checkpoint
model = YOLO(str(checkpoint_path))

# Run validation on the held-out test split
metrics = model.val(
    data=str(data_path),
    split="test",
    imgsz=512,
    batch=8,
    device=0,
    workers=0,
    project=str(MODEL_DIR / "runs"),
    name="test_evaluation",
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

output_path = reports_dir / "model_10_test_metrics.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

# Print the results
print("\nModel 10 test metrics")
print("---------------------")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall:    {results['recall']:.4f}")
print(f"mAP50:     {results['mAP50']:.4f}")
print(f"mAP50-95:  {results['mAP50_95']:.4f}")

print(f"\nSaved metrics to:\n{output_path}")
