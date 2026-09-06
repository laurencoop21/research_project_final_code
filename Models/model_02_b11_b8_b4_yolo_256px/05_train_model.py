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
    raise FileNotFoundError(
        f"Missing dataset file: {dataset_yaml}"
    )

# Choose the training device
# Use an NVIDIA GPU when running on the NCC.
if torch.cuda.is_available():
    device = 0

    print("CUDA available:", True)
    print("Training device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

# This fallback allows the script to run on an
# Apple silicon Mac if needed.
elif torch.backends.mps.is_available():
    device = "mps"

    print("CUDA available:", False)
    print("Training device:", device)

else:
    device = "cpu"

    print("CUDA available:", False)
    print("Training device:", device)

# Load the pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train the model
model.train(
    data=str(dataset_yaml),

    # Training settings
    epochs=100,
    imgsz=256,
    batch=8,
    device=device,
    patience=15,
    seed=42,
    workers=0,

    # Output settings
    project=str(runs_dir),
    name="baseline_256px",
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

if not best_model.exists():
    raise FileNotFoundError(
        f"Best model was not found: {best_model}"
    )

if not last_model.exists():
    raise FileNotFoundError(
        f"Last model was not found: {last_model}"
    )

best_output = checkpoints_dir / "model_02_best.pt"
last_output = checkpoints_dir / "model_02_last.pt"

shutil.copy2(
    best_model,
    best_output,
)

shutil.copy2(
    last_model,
    last_output,
)

# Print the results
print("\nTraining complete.")

print("\nBest model saved to:")
print(best_output)

print("\nLast model saved to:")
print(last_output)