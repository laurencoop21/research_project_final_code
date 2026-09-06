from pathlib import Path
import pandas as pd

# File paths
MODEL_DIR = Path(__file__).resolve().parent

images_dir = MODEL_DIR / "dataset" / "images"
labels_dir = MODEL_DIR / "dataset" / "labels"
tile_index_path = MODEL_DIR / "reports" / "tile_index.csv"

splits = ["train", "val", "test"]

# Load the valid tile names
tile_index = pd.read_csv(tile_index_path)

# Remove files not listed in tile_index.csv
for split in splits:
    valid_names = set(
        tile_index.loc[
            tile_index["split"] == split,
            "tile_name"
        ]
    )

    removed_images = 0
    removed_labels = 0

    for image_path in (images_dir / split).glob("*"):
        if image_path.is_file() and image_path.stem not in valid_names:
            image_path.unlink()
            removed_images += 1

    for label_path in (labels_dir / split).glob("*"):
        if label_path.is_file() and label_path.suffix == ".txt":
            if label_path.stem not in valid_names:
                label_path.unlink()
                removed_labels += 1

    print(
        f"{split}: removed {removed_images} extra images "
        f"and {removed_labels} extra labels"
    )

# Remove old Ultralytics cache files
for split in splits:
    cache_path = labels_dir / f"{split}.cache"

    if cache_path.exists():
        cache_path.unlink()
        print(f"Deleted cache: {cache_path}")

# Print the final counts
print("\nFinal dataset counts")
print("--------------------")

for split in splits:
    image_count = len(list((images_dir / split).glob("*.png")))
    label_count = len(list((labels_dir / split).glob("*.txt")))

    print(
        f"{split}: {image_count} images, "
        f"{label_count} labels"
    )