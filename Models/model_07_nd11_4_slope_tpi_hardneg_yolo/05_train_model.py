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

# Choose device
if torch.cuda.is_available():
    device = 0
else:
    device = "cpu"

# Load pretrained YOLO11m
weights_path = MODEL_DIR / "yolo11m.pt"

if not weights_path.exists():
    raise FileNotFoundError(
        f"Missing pretrained weights: {weights_path}"
    )

model = YOLO(str(weights_path))

# Train Model 7
model.train(
    data=str(dataset_yaml),

    # Training duration
    epochs=150,
    patience=50,

    # Image / GPU settings
    imgsz=512,
    batch=8,
    device=device,
    workers=0,

    # Reproducibility
    seed=42,

    # Output
    project=str(runs_dir),
    name="model_07_yolo11m",
    plots=True,

    # Augmentation
    # Mosaic augmentation
    mosaic=1.0,

    # Appropriate for overhead imagery where there is
    # no meaningful fixed orientation
    degrees=180,
    fliplr=0.5,
    flipud=0.5,

    # Do not alter the relationships between the
    # ND11-4, slope and TPI channels
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
)

# Copy final weights to checkpoints/
best_model = Path(model.trainer.best)
last_model = Path(model.trainer.last)

shutil.copy2(
    best_model,
    checkpoints_dir / "model_07_best.pt",
)

shutil.copy2(
    last_model,
    checkpoints_dir / "model_07_last.pt",
)

print("\nTraining complete.")

print("\nBest model saved to:")
print(checkpoints_dir / "model_07_best.pt")

print("\nLast model saved to:")
print(checkpoints_dir / "model_07_last.pt")
