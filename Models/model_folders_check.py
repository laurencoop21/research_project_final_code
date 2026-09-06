from pathlib import Path
import shutil


# The folder containing this script:
# .../iraq_tells_gis/Models
MODELS_DIR = Path(__file__).resolve().parent

model_01_dir = (
    MODELS_DIR / "model_01_b11_b8_b4_yolo"
)

model_02_dir = (
    MODELS_DIR / "model_02_b11_b8_b4_yolo_256px"
)


# Create the Model 2 folder structure
folders = [
    model_02_dir,
    model_02_dir / "dataset",
    model_02_dir / "dataset" / "images" / "train",
    model_02_dir / "dataset" / "images" / "val",
    model_02_dir / "dataset" / "images" / "test",
    model_02_dir / "dataset" / "labels" / "train",
    model_02_dir / "dataset" / "labels" / "val",
    model_02_dir / "dataset" / "labels" / "test",
    model_02_dir / "reports",
    model_02_dir / "figures",
    model_02_dir / "predictions",
    model_02_dir / "runs",
    model_02_dir / "checkpoints",
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)


# Copy the existing geographic split
split_file = (
    model_01_dir
    / "reports"
    / "geographic_split.gpkg"
)

split_summary = (
    model_01_dir
    / "reports"
    / "geographic_split_summary.csv"
)

if split_file.exists():
    shutil.copy2(
        split_file,
        model_02_dir / "reports" / split_file.name,
    )

if split_summary.exists():
    shutil.copy2(
        split_summary,
        model_02_dir / "reports" / split_summary.name,
    )

print("Model 2 folder created:")
print(model_02_dir)