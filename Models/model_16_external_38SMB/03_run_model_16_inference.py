from pathlib import Path
import json

import pandas as pd
import torch
from ultralytics import YOLO

# File paths
MODEL_DIR = Path(__file__).resolve().parent

model_path = MODEL_DIR / "checkpoints" / "model_16_best.pt"
images_dir = MODEL_DIR / "tiles"

reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

predictions_path = (
    reports_dir / "model_16_38SMB_tile_predictions.csv"
)

summary_path = (
    reports_dir / "model_16_38SMB_inference_summary.json"
)

# Model 16 inference settings
# These settings are retained from the original
# Model 16 error-analysis prediction code.

confidence_threshold = 0.10
prediction_iou = 0.70
image_size = 320

# Check inputs
if not model_path.exists():
    raise FileNotFoundError(f"Missing model: {model_path}")

if not images_dir.exists():
    raise FileNotFoundError(f"Missing inference tiles: {images_dir}")

# Set prediction device
if torch.cuda.is_available():
    device = 0
else:
    device = "cpu"

print("Prediction device:", device)
print("Confidence threshold:", confidence_threshold)
print("Prediction IoU:", prediction_iou)
print("Image size:", image_size)

# Load Model 16
model = YOLO(str(model_path))

# Run inference
results_stream = model.predict(
    source=str(images_dir),
    imgsz=image_size,
    conf=confidence_threshold,
    iou=prediction_iou,
    device=device,
    stream=True,
    verbose=False,
)

# Save tile-space predictions
prediction_rows = []
tiles_processed = 0

for result in results_stream:
    image_path = Path(result.path)
    tile_name = image_path.stem

    tiles_processed += 1

    if result.boxes is None or len(result.boxes) == 0:
        continue

    predicted_boxes = result.boxes.xyxy.cpu().numpy()
    predicted_confidences = result.boxes.conf.cpu().numpy()

    for prediction_index, (
        predicted_box,
        confidence,
    ) in enumerate(
        zip(
            predicted_boxes,
            predicted_confidences,
        )
    ):
        prediction_rows.append(
            {
                "tile_name": tile_name,
                "prediction_index": prediction_index,
                "confidence": float(confidence),
                "x_min": float(predicted_box[0]),
                "y_min": float(predicted_box[1]),
                "x_max": float(predicted_box[2]),
                "y_max": float(predicted_box[3]),
            }
        )

# Save prediction table
prediction_results = pd.DataFrame(
    prediction_rows,
    columns=[
        "tile_name",
        "prediction_index",
        "confidence",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ],
)

prediction_results.to_csv(
    predictions_path,
    index=False,
)

# Save inference summary
summary = {
    "model": str(model_path),
    "image_directory": str(images_dir),
    "image_size": image_size,
    "confidence_threshold": confidence_threshold,
    "prediction_iou": prediction_iou,
    "tiles_processed": int(tiles_processed),
    "predictions": int(len(prediction_results)),
}

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)

# Print results
print("\nModel 16 external inference")
print("---------------------------")
print("Tiles processed:", tiles_processed)
print("Predictions:", len(prediction_results))

print("\nSaved tile-space predictions to:")
print(predictions_path)

print("\nSaved inference summary to:")
print(summary_path)
