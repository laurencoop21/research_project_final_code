from pathlib import Path
import shutil

import pandas as pd
import rasterio
from rasterio.windows import Window, bounds


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

raster_path = (
    PROJECT_DIR
    / "Iraq_ND11_4_slope_TPI_uint8.tif"
)

manifest_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_final_locked.csv"
)

tile_index_path = (
    MODEL_DIR
    / "reports"
    / "tile_index.csv"
)

train_images = (
    MODEL_DIR
    / "dataset"
    / "images"
    / "train"
)

train_labels = (
    MODEL_DIR
    / "dataset"
    / "labels"
    / "train"
)

tile_size = 512

# Load approved hard negatives
hard_negatives = pd.read_csv(manifest_path)

print("\nApproved hard negatives:", len(hard_negatives))

if len(hard_negatives) != 28:
    raise ValueError(
        f"Expected 28 approved hard negatives, "
        f"found {len(hard_negatives)}."
    )

# Copy images and create empty YOLO labels
for _, row in hard_negatives.iterrows():

    source_image = Path(row["image_path"])

    if not source_image.exists():
        raise FileNotFoundError(source_image)

    tile_name = row["tile_name"]

    destination_image = (
        train_images
        / f"{tile_name}.png"
    )

    destination_label = (
        train_labels
        / f"{tile_name}.txt"
    )

    shutil.copy2(
        source_image,
        destination_image,
    )

    # Empty label = valid YOLO background image.
    destination_label.write_text("")

# Add hard negatives to tile_index.csv
tile_index = pd.read_csv(tile_index_path)

# Make rerunning this script safe:
# remove any existing rows for these hard negatives
# before appending them again.
hard_negative_names = set(
    hard_negatives["tile_name"]
)

tile_index = tile_index[
    ~tile_index["tile_name"].isin(
        hard_negative_names
    )
].copy()


new_rows = []

with rasterio.open(raster_path) as raster:

    for _, row in hard_negatives.iterrows():

        row_start = int(row["row_start"])
        col_start = int(row["col_start"])

        window = Window(
            col_start,
            row_start,
            tile_size,
            tile_size,
        )

        tile_bounds = bounds(
            window,
            raster.transform,
        )

        new_rows.append({
            "tile_name": row["tile_name"],
            "split": "train",
            "row_start": row_start,
            "col_start": col_start,
            "tell_count": 0,
            "left": tile_bounds[0],
            "bottom": tile_bounds[1],
            "right": tile_bounds[2],
            "top": tile_bounds[3],
        })


tile_index = pd.concat(
    [
        tile_index,
        pd.DataFrame(new_rows),
    ],
    ignore_index=True,
)

tile_index.to_csv(
    tile_index_path,
    index=False,
)

# Final counts
image_count = len(
    list(train_images.glob("*.png"))
)

label_count = len(
    list(train_labels.glob("*.txt"))
)

empty_labels = sum(
    1
    for path in train_labels.glob("*.txt")
    if path.stat().st_size == 0
)

print("\nModel 7 training dataset")
print("------------------------")
print("Training images:", image_count)
print("Training labels:", label_count)
print("Background-only tiles:", empty_labels)
print("Hard negatives added:", len(hard_negatives))

print("\nUpdated:")
print(tile_index_path)
