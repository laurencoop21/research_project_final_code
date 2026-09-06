from pathlib import Path
import shutil

import torch
from ultralytics import YOLO

# File paths
MODEL_DIR = Path(__file__).resolve().parent

dataset_yaml = MODEL_DIR / "dataset" / "dataset.yaml"
runs_dir = MODEL_DIR / "runs"
checkpoints_dir = MODEL_DIR / "checkpoints"

runs_dir.mkdir(exist_ok=True)
checkpoints_dir.mkdir(exist_ok=True)

if not dataset_yaml.exists():
    raise FileNotFoundError(f"Missing dataset file: {dataset_yaml}")

# Choose the computer device
if torch.cuda.is_available():
    device = 0
else:
    device = "cpu"

# Load a small pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train the model
model.train(
    data=str(dataset_yaml),
    epochs=100,
    imgsz=512,
    batch=8,
    device=device,
    patience=15,
    seed=456,
    workers=0,
    project=str(runs_dir),
    name="seed456",
    plots=True,

    # Appropriate augmentation for overhead imagery
    degrees=180,
    fliplr=0.5,
    flipud=0.5,

    # Do not alter the false-colour band relationships
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
)

# Copy the saved model weights
best_model = Path(model.trainer.best)
last_model = Path(model.trainer.last)

shutil.copy2(
    best_model,
    checkpoints_dir / "model_06_seed456_best.pt",
)

shutil.copy2(
    last_model,
    checkpoints_dir / "model_06_seed456_last.pt",
)

print("\nTraining complete.")
print("Best model saved to:")
print(checkpoints_dir / "model_06_seed456_best.pt")

print("\nLast model saved to:")
print(checkpoints_dir / "model_06_seed456_last.pt")