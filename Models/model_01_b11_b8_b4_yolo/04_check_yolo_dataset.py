from pathlib import Path

import geopandas as gpd
import pandas as pd
from PIL import Image, ImageDraw
from shapely.geometry import box


# File paths and settings
MODEL_DIR = Path(__file__).resolve().parent

dataset_dir = MODEL_DIR / "dataset"
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

reports_dir = MODEL_DIR / "reports"
figures_dir = MODEL_DIR / "figures" / "label_checks"
figures_dir.mkdir(parents=True, exist_ok=True)

split_path = reports_dir / "geographic_split.gpkg"
tile_index_path = reports_dir / "tile_index.csv"

coverage_path = reports_dir / "tell_coverage.csv"
summary_path = reports_dir / "dataset_check_summary.csv"

minimum_visible_fraction = 0.50
splits = ["train", "val", "test"]


# Load the saved tile and tell information
tile_index = pd.read_csv(tile_index_path)

tells = gpd.read_file(
    split_path,
    layer="tells_with_split",
)

usable_tells = tells[tells["split"].isin(splits)].copy()


# Check that every image has a matching label
file_check_rows = []

for split in splits:
    image_files = sorted((images_dir / split).glob("*.png"))
    label_files = sorted((labels_dir / split).glob("*.txt"))

    image_names = {path.stem for path in image_files}
    label_names = {path.stem for path in label_files}

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    file_check_rows.append(
        {
            "split": split,
            "images": len(image_files),
            "labels": len(label_files),
            "missing_labels": len(missing_labels),
            "missing_images": len(missing_images),
        }
    )


# Check the values inside every YOLO label file
invalid_label_lines = 0

for split in splits:
    for label_path in (labels_dir / split).glob("*.txt"):
        with open(label_path) as file:
            lines = [line.strip() for line in file if line.strip()]

        for line in lines:
            values = line.split()

            if len(values) != 5:
                invalid_label_lines += 1
                continue

            class_id = values[0]

            try:
                x_center, y_center, width, height = map(
                    float,
                    values[1:],
                )
            except ValueError:
                invalid_label_lines += 1
                continue

            valid = (
                class_id == "0"
                and 0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            )

            if not valid:
                invalid_label_lines += 1

# Check which unique tells appear in the tiles
coverage_rows = []

for split in splits:
    split_tells = usable_tells[
        usable_tells["split"] == split
    ].copy()

    split_tiles = tile_index[
        tile_index["split"] == split
    ]

    covered_ids = set()

    for _, tile in split_tiles.iterrows():
        tile_polygon = box(
            tile["left"],
            tile["bottom"],
            tile["right"],
            tile["top"],
        )

        intersecting_tells = split_tells[
            split_tells.geometry.intersects(tile_polygon)
        ]

        for _, tell in intersecting_tells.iterrows():
            visible_geometry = tell.geometry.intersection(
                tile_polygon
            )

            visible_fraction = (
                visible_geometry.area / tell.geometry.area
            )

            if visible_fraction >= minimum_visible_fraction:
                covered_ids.add(tell["polygon_id"])

    for _, tell in split_tells.iterrows():
        coverage_rows.append(
            {
                "polygon_id": tell["polygon_id"],
                "split": split,
                "covered_by_tile": (
                    tell["polygon_id"] in covered_ids
                ),
            }
        )

coverage = pd.DataFrame(coverage_rows)
coverage.to_csv(coverage_path, index=False)

# Create sample images with the boxes drawn on
for split in splits:
    split_tiles = tile_index[
        tile_index["split"] == split
    ]

    positive_tiles = split_tiles[
        split_tiles["tell_count"] > 0
    ].sample(
        n=min(4, (split_tiles["tell_count"] > 0).sum()),
        random_state=42,
    )

    background_tiles = split_tiles[
        split_tiles["tell_count"] == 0
    ].sample(
        n=min(2, (split_tiles["tell_count"] == 0).sum()),
        random_state=42,
    )

    sample_tiles = pd.concat(
        [positive_tiles, background_tiles]
    )

    for _, tile in sample_tiles.iterrows():
        tile_name = tile["tile_name"]

        image_path = images_dir / split / f"{tile_name}.png"
        label_path = labels_dir / split / f"{tile_name}.txt"

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        with open(label_path) as file:
            lines = [line.strip() for line in file if line.strip()]

        for line in lines:
            _, x_center, y_center, width, height = map(
                float,
                line.split(),
            )

            image_width, image_height = image.size

            x_min = (
                x_center - width / 2
            ) * image_width

            y_min = (
                y_center - height / 2
            ) * image_height

            x_max = (
                x_center + width / 2
            ) * image_width

            y_max = (
                y_center + height / 2
            ) * image_height

            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                outline="red",
                width=3,
            )

        output_path = figures_dir / f"{tile_name}_check.png"
        image.save(output_path)

# Print and save the summary
file_checks = pd.DataFrame(file_check_rows)

coverage_summary = (
    coverage.groupby("split")
    .agg(
        assigned_tells=("polygon_id", "count"),
        covered_tells=("covered_by_tile", "sum"),
    )
    .reset_index()
)

coverage_summary["missing_tells"] = (
    coverage_summary["assigned_tells"]
    - coverage_summary["covered_tells"]
)

summary = file_checks.merge(
    coverage_summary,
    on="split",
)

summary["invalid_label_lines"] = invalid_label_lines

summary.to_csv(summary_path, index=False)

print("Dataset check summary")
print("---------------------")
print(summary.to_string(index=False))

missing_tells = coverage[
    coverage["covered_by_tile"] == False
]

if len(missing_tells) > 0:
    print("\nTells missing from the tiles")
    print("----------------------------")
    print(missing_tells.to_string(index=False))
else:
    print("\nEvery usable tell appears in at least one tile.")

print("\nSaved tell coverage to:")
print(coverage_path)

print("\nSaved sample box images to:")
print(figures_dir)
