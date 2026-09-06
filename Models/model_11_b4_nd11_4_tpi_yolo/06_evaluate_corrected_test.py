from pathlib import Path
import json

from ultralytics import YOLO


# File paths
MODEL_DIR = Path(__file__).resolve().parent

checkpoint = MODEL_DIR / "checkpoints" / "model_11_best.pt"
data = MODEL_DIR / "dataset_corrected_test" / "dataset.yaml"

print("Checkpoint:", checkpoint)
print("Dataset:", data)

model = YOLO(str(checkpoint))

# Run validation on the corrected test split
metrics = model.val(
    data=str(data),
    split="test",
    imgsz=512,
    batch=8,
    device=0,
    workers=0,
    plots=True,
    project=str(MODEL_DIR / "runs"),
    name="corrected_test_evaluation",
    exist_ok=True,
)

# Collect the headline metrics
results = {
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "mAP50": float(metrics.box.map50),
    "mAP50_95": float(metrics.box.map),
}

reports = MODEL_DIR / "reports"
reports.mkdir(exist_ok=True)

output = reports / "model_11_corrected_test_metrics.json"

with open(output, "w") as f:
    json.dump(results, f, indent=4)

# Print the results
print("\nModel 11 corrected test metrics")
print("--------------------------------")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall:    {results['recall']:.4f}")
print(f"mAP50:     {results['mAP50']:.4f}")
print(f"mAP50-95:  {results['mAP50_95']:.4f}")
print("\nSaved to:", output)
