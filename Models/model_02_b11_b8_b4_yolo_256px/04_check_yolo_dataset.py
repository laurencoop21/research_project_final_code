from pathlib import Path
import math

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

figures_dir = (
    MODEL_DIR
    / "figures"
    / "label_checks"
)

figures_dir.mkdir(
    parents=True,
    exist_ok=True,
)

split_path = (
    reports_dir
    / "geographic_split.gpkg"
)

tile_index_path = (
    reports_dir
    / "tile_index.csv"
)

coverage_path = (
    reports_dir
    / "tell_coverage.csv"
)

summary_path = (
    reports_dir
    / "dataset_check_summary.csv"
)

tile_size = 256
minimum_visible_fraction = 0.50

splits = ["train", "val", "test"]

# Check that required files exist
required_files = [
    split_path,
    tile_index_path,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

for split in splits:
    image_folder = images_dir / split
    label_folder = labels_dir / split

    if not image_folder.exists():
        raise FileNotFoundError(
            f"Missing image folder: {image_folder}"
        )

    if not label_folder.exists():
        raise FileNotFoundError(
            f"Missing label folder: {label_folder}"
        )

print("All required dataset files were found.")

# Load the saved tile and tell information
tile_index = pd.read_csv(tile_index_path)

tells = gpd.read_file(
    split_path,
    layer="tells_with_split",
)

usable_tells = tells[
    tells["split"].isin(splits)
].copy()

# Check image and label files
file_check_rows = []

for split in splits:

    image_files = sorted(
        (images_dir / split).glob("*.png")
    )

    label_files = sorted(
        (labels_dir / split).glob("*.txt")
    )

    image_names = {
        path.stem
        for path in image_files
    }

    label_names = {
        path.stem
        for path in label_files
    }

    expected_names = set(
        tile_index.loc[
            tile_index["split"] == split,
            "tile_name",
        ].astype(str)
    )

    missing_labels = (
        image_names - label_names
    )

    missing_images = (
        label_names - image_names
    )

    images_missing_from_dataset = (
        expected_names - image_names
    )

    labels_missing_from_dataset = (
        expected_names - label_names
    )

    extra_images = (
        image_names - expected_names
    )

    extra_labels = (
        label_names - expected_names
    )

    invalid_image_sizes = 0

    for image_path in image_files:
        with Image.open(image_path) as image:
            if image.size != (
                tile_size,
                tile_size,
            ):
                invalid_image_sizes += 1

    file_check_rows.append(
        {
            "split": split,
            "tile_index_rows": len(
                expected_names
            ),
            "images": len(image_files),
            "labels": len(label_files),
            "missing_labels": len(
                missing_labels
            ),
            "missing_images": len(
                missing_images
            ),
            "images_missing_from_dataset": len(
                images_missing_from_dataset
            ),
            "labels_missing_from_dataset": len(
                labels_missing_from_dataset
            ),
            "extra_images": len(
                extra_images
            ),
            "extra_labels": len(
                extra_labels
            ),
            "invalid_image_sizes": (
                invalid_image_sizes
            ),
        }
    )

# Check every YOLO label file
label_check_rows = []

for split in splits:

    invalid_label_lines = 0
    label_count_mismatches = 0
    total_label_lines = 0

    split_tile_index = (
        tile_index[
            tile_index["split"] == split
        ]
        .set_index("tile_name")
    )

    for label_path in (
        labels_dir / split
    ).glob("*.txt"):

        with open(label_path) as file:
            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]

        total_label_lines += len(lines)

        tile_name = label_path.stem

        if tile_name in split_tile_index.index:
            expected_count = int(
                split_tile_index.loc[
                    tile_name,
                    "tell_count",
                ]
            )

            if len(lines) != expected_count:
                label_count_mismatches += 1

        for line in lines:

            values = line.split()

            if len(values) != 5:
                invalid_label_lines += 1
                continue

            class_id = values[0]

            try:
                (
                    x_center,
                    y_center,
                    width,
                    height,
                ) = map(
                    float,
                    values[1:],
                )

            except ValueError:
                invalid_label_lines += 1
                continue

            all_finite = all(
                math.isfinite(value)
                for value in [
                    x_center,
                    y_center,
                    width,
                    height,
                ]
            )

            x_min = x_center - width / 2
            x_max = x_center + width / 2

            y_min = y_center - height / 2
            y_max = y_center + height / 2

            valid = (
                class_id == "0"
                and all_finite
                and 0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
                and 0 <= x_min <= 1
                and 0 <= x_max <= 1
                and 0 <= y_min <= 1
                and 0 <= y_max <= 1
            )

            if not valid:
                invalid_label_lines += 1

    label_check_rows.append(
        {
            "split": split,
            "total_label_lines": (
                total_label_lines
            ),
            "invalid_label_lines": (
                invalid_label_lines
            ),
            "label_count_mismatches": (
                label_count_mismatches
            ),
        }
    )

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
            split_tells.geometry.intersects(
                tile_polygon
            )
        ]

        for _, tell in (
            intersecting_tells.iterrows()
        ):

            visible_geometry = (
                tell.geometry.intersection(
                    tile_polygon
                )
            )

            if (
                visible_geometry.is_empty
                or visible_geometry.area <= 0
                or tell.geometry.area <= 0
            ):
                continue

            visible_fraction = (
                visible_geometry.area
                / tell.geometry.area
            )

            if (
                visible_fraction
                >= minimum_visible_fraction
            ):
                covered_ids.add(
                    tell["polygon_id"]
                )

    for _, tell in split_tells.iterrows():

        coverage_rows.append(
            {
                "polygon_id": (
                    tell["polygon_id"]
                ),
                "split": split,
                "covered_by_tile": (
                    tell["polygon_id"]
                    in covered_ids
                ),
            }
        )

coverage = pd.DataFrame(coverage_rows)

coverage.to_csv(
    coverage_path,
    index=False,
)

# Create sample images with boxes drawn on
for split in splits:

    split_tiles = tile_index[
        tile_index["split"] == split
    ]

    positive_tiles = split_tiles[
        split_tiles["tell_count"] > 0
    ]

    background_tiles = split_tiles[
        split_tiles["tell_count"] == 0
    ]

    positive_sample = positive_tiles.sample(
        n=min(
            4,
            len(positive_tiles),
        ),
        random_state=42,
    )

    background_sample = (
        background_tiles.sample(
            n=min(
                2,
                len(background_tiles),
            ),
            random_state=42,
        )
    )

    sample_tiles = pd.concat(
        [
            positive_sample,
            background_sample,
        ],
        ignore_index=True,
    )

    for _, tile in sample_tiles.iterrows():

        tile_name = tile["tile_name"]

        image_path = (
            images_dir
            / split
            / f"{tile_name}.png"
        )

        label_path = (
            labels_dir
            / split
            / f"{tile_name}.txt"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        draw = ImageDraw.Draw(image)

        with open(label_path) as file:
            lines = [
                line.strip()
                for line in file
                if line.strip()
            ]

        for line in lines:

            values = line.split()

            if len(values) != 5:
                continue

            try:
                (
                    _,
                    x_center,
                    y_center,
                    width,
                    height,
                ) = map(float, values)

            except ValueError:
                continue

            image_width, image_height = (
                image.size
            )

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
                [
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ],
                outline="red",
                width=3,
            )

        output_path = (
            figures_dir
            / f"{tile_name}_check.png"
        )

        image.save(output_path)

# Create and save the summary
file_checks = pd.DataFrame(
    file_check_rows
)

label_checks = pd.DataFrame(
    label_check_rows
)

coverage_summary = (
    coverage
    .groupby("split")
    .agg(
        assigned_tells=(
            "polygon_id",
            "count",
        ),
        covered_tells=(
            "covered_by_tile",
            "sum",
        ),
    )
    .reset_index()
)

coverage_summary["missing_tells"] = (
    coverage_summary["assigned_tells"]
    - coverage_summary["covered_tells"]
)

summary = file_checks.merge(
    label_checks,
    on="split",
)

summary = summary.merge(
    coverage_summary,
    on="split",
)

summary.to_csv(
    summary_path,
    index=False,
)

# Print the results
print("\nDataset check summary")
print("---------------------")
print(summary.to_string(index=False))

missing_tells = coverage[
    coverage["covered_by_tile"] == False
]

if len(missing_tells) > 0:

    print("\nTells missing from the tiles")
    print("----------------------------")

    print(
        missing_tells.to_string(
            index=False
        )
    )

else:
    print(
        "\nEvery usable tell appears "
        "in at least one tile."
    )

print("\nSaved dataset check summary to:")
print(summary_path)

print("\nSaved tell coverage to:")
print(coverage_path)

print("\nSaved sample box images to:")
print(figures_dir)