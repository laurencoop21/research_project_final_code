from pathlib import Path
import json
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window, bounds
from shapely.geometry import box

# Settings and file paths
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

raster_path = PROJECT_DIR / "Iraq_11_8_4.tif"

reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

split_path = reports_dir / "geographic_split.gpkg"

dataset_dir = MODEL_DIR / "dataset"
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

tile_size = 256
overlap = 64
stride = tile_size - overlap
minimum_visible_fraction = 0.50

splits = ["train", "val", "test"]

# Check that the required files exist
required_files = [
    raster_path,
    split_path,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

print("All required files were found.")

print("\nTiling settings")
print("---------------")
print("Tile size:", tile_size)
print("Overlap:", overlap)
print("Stride:", stride)
print("Minimum visible fraction:", minimum_visible_fraction)

# Reset the generated dataset folders
# This only deletes the generated dataset inside the
# current model folder.

for split in splits:
    image_folder = images_dir / split
    label_folder = labels_dir / split

    if image_folder.exists():
        shutil.rmtree(image_folder)

    if label_folder.exists():
        shutil.rmtree(label_folder)

    image_folder.mkdir(parents=True)
    label_folder.mkdir(parents=True)

# Load the geographic split data
split_regions = gpd.read_file(
    split_path,
    layer="split_regions",
)

tells = gpd.read_file(
    split_path,
    layer="tells_with_split",
)

required_split_region_columns = {"split", "geometry"}
required_tell_columns = {"split", "polygon_id", "geometry"}

missing_region_columns = (
    required_split_region_columns - set(split_regions.columns)
)

missing_tell_columns = (
    required_tell_columns - set(tells.columns)
)

if missing_region_columns:
    raise ValueError(
        "Missing columns from split_regions: "
        f"{sorted(missing_region_columns)}"
    )

if missing_tell_columns:
    raise ValueError(
        "Missing columns from tells_with_split: "
        f"{sorted(missing_tell_columns)}"
    )

excluded_tells = tells[
    tells["split"] == "exclude_boundary"
].copy()

usable_tells = tells[
    tells["split"].isin(splits)
].copy()

print("\nSplit information")
print("-----------------")
print("Usable tells:", len(usable_tells))
print("Excluded boundary tells:", len(excluded_tells))

print(
    usable_tells["split"]
    .value_counts()
    .reindex(splits)
    .fillna(0)
    .astype(int)
)

# Make tile starting positions
def make_starts(start, end, size, step):
    """
    Create the pixel positions where tiles begin.

    The final tile is moved so that it finishes at the
    edge of the requested area, even when the normal
    stride does not divide the area exactly.
    """

    starts = list(
        range(
            start,
            end - size + 1,
            step,
        )
    )

    last_start = end - size

    if last_start >= start and last_start not in starts:
        starts.append(last_start)

    return sorted(starts)

# Create the image tiles and YOLO labels
tile_records = []

with rasterio.open(raster_path) as raster:

    # Make sure all geometries use the raster CRS.
    if split_regions.crs != raster.crs:
        split_regions = split_regions.to_crs(raster.crs)

    if usable_tells.crs != raster.crs:
        usable_tells = usable_tells.to_crs(raster.crs)

    if excluded_tells.crs != raster.crs:
        excluded_tells = excluded_tells.to_crs(raster.crs)

    # Row positions cover the full raster from north to south.
    row_starts = make_starts(
        start=0,
        end=raster.height,
        size=tile_size,
        step=stride,
    )

    for split in splits:

        matching_regions = split_regions[
            split_regions["split"] == split
        ]

        if len(matching_regions) != 1:
            raise ValueError(
                f"Expected one region for {split}, "
                f"but found {len(matching_regions)}."
            )

        region = matching_regions.geometry.iloc[0]

        region_left, _, region_right, _ = region.bounds

        # Convert the geographic split boundaries into
        # raster column positions.
        start_col = max(
            0,
            int(
                (region_left - raster.bounds.left)
                / raster.res[0]
            ),
        )

        end_col = min(
            raster.width,
            int(
                np.ceil(
                    (region_right - raster.bounds.left)
                    / raster.res[0]
                )
            ),
        )

        col_starts = make_starts(
            start=start_col,
            end=end_col,
            size=tile_size,
            step=stride,
        )

        split_tells = usable_tells[
            usable_tells["split"] == split
        ].copy()

        print(f"\nCreating {split} tiles...")

        tile_number = 0

        for row_start in row_starts:
            for col_start in col_starts:

                window = Window(
                    col_off=col_start,
                    row_off=row_start,
                    width=tile_size,
                    height=tile_size,
                )

                tile_bounds = bounds(
                    window,
                    raster.transform,
                )

                tile_polygon = box(*tile_bounds)

                intersecting_tells = split_tells[
                    split_tells.geometry.intersects(
                        tile_polygon
                    )
                ]

                # These tells were deliberately excluded from
                # the geographic train/validation/test split.
                # They are counted for auditing but are not labelled.
                intersecting_excluded_tells = excluded_tells[
                    excluded_tells.geometry.intersects(
                        tile_polygon
                    )
                ]

                labels = []
                labelled_polygon_ids = []

                for _, tell in intersecting_tells.iterrows():

                    visible_geometry = (
                        tell.geometry.intersection(
                            tile_polygon
                        )
                    )

                    if (
                        visible_geometry.is_empty
                        or visible_geometry.area <= 0
                    ):
                        continue

                    visible_fraction = (
                        visible_geometry.area
                        / tell.geometry.area
                    )

                    # Do not label a tell when less than half
                    # of the polygon is visible in this tile.
                    #
                    # Keep the tile because it may contain
                    # another tell that is sufficiently visible.
                    if (
                        visible_fraction
                        < minimum_visible_fraction
                    ):
                        continue

                    minx, miny, maxx, maxy = (
                        visible_geometry.bounds
                    )

                    left, bottom, right, top = tile_bounds

                    x_min = (
                        (minx - left)
                        / (right - left)
                    )

                    x_max = (
                        (maxx - left)
                        / (right - left)
                    )

                    y_min = (
                        (top - maxy)
                        / (top - bottom)
                    )

                    y_max = (
                        (top - miny)
                        / (top - bottom)
                    )

                    # Protect against tiny floating-point values
                    # outside the valid YOLO range.
                    x_min = float(np.clip(x_min, 0, 1))
                    x_max = float(np.clip(x_max, 0, 1))
                    y_min = float(np.clip(y_min, 0, 1))
                    y_max = float(np.clip(y_max, 0, 1))

                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min

                    if width <= 0 or height <= 0:
                        continue

                    labels.append(
                        f"0 "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}"
                    )

                    labelled_polygon_ids.append(
                        str(tell["polygon_id"])
                    )

                # Read the three raster bands for this window.
                image_data = raster.read(window=window)

                # Rasterio returns:
                # bands, rows, columns
                #
                # PIL requires:
                # rows, columns, bands
                image_data = np.moveaxis(
                    image_data,
                    0,
                    -1,
                )

                tile_name = (
                    f"{split}_{tile_number:04d}_"
                    f"r{row_start}_c{col_start}"
                )

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

                Image.fromarray(image_data).save(
                    image_path
                )

                with open(
                    label_path,
                    "w",
                ) as label_file:
                    label_file.write(
                        "\n".join(labels)
                    )

                tile_records.append(
                    {
                        "tile_name": tile_name,
                        "split": split,
                        "row_start": row_start,
                        "col_start": col_start,
                        "tell_count": len(labels),
                        "polygon_ids": ",".join(
                            labelled_polygon_ids
                        ),
                        "excluded_tell_count": len(
                            intersecting_excluded_tells
                        ),
                        "left": tile_bounds[0],
                        "bottom": tile_bounds[1],
                        "right": tile_bounds[2],
                        "top": tile_bounds[3],
                    }
                )

                tile_number += 1

        print(f"Saved {tile_number} {split} tiles.")

# Save the tile index
tile_index = pd.DataFrame(tile_records)

tile_index_path = reports_dir / "tile_index.csv"

tile_index.to_csv(
    tile_index_path,
    index=False,
)

# Create and save the tile summary
summary = (
    tile_index
    .groupby("split")
    .agg(
        tiles=(
            "tile_name",
            "count",
        ),
        positive_tiles=(
            "tell_count",
            lambda x: (x > 0).sum(),
        ),
        background_tiles=(
            "tell_count",
            lambda x: (x == 0).sum(),
        ),
        label_instances=(
            "tell_count",
            "sum",
        ),
        tiles_with_excluded_tells=(
            "excluded_tell_count",
            lambda x: (x > 0).sum(),
        ),
    )
    .reset_index()
)

summary_path = reports_dir / "tile_summary.csv"

summary.to_csv(
    summary_path,
    index=False,
)

# Save the tiling settings
settings = {
    "model": (
        "Model 2: B11-B8-B4 YOLO "
        "with 256-pixel tiles"
    ),
    "raster_path": str(raster_path),
    "split_path": str(split_path),
    "tile_size": tile_size,
    "overlap": overlap,
    "stride": stride,
    "minimum_visible_fraction": (
        minimum_visible_fraction
    ),
}

settings_path = reports_dir / "tile_settings.json"

with open(settings_path, "w") as settings_file:
    json.dump(
        settings,
        settings_file,
        indent=4,
    )

# Create the YOLO dataset YAML file
yaml_text = f"""path: "{dataset_dir.resolve()}"
train: images/train
val: images/val
test: images/test

names:
  0: tell
"""

yaml_path = dataset_dir / "dataset.yaml"

with open(yaml_path, "w") as yaml_file:
    yaml_file.write(yaml_text)

# Print the results
print("\nTile summary")
print("------------")
print(summary.to_string(index=False))

print("\nSaved tile index to:")
print(tile_index_path)

print("\nSaved tile summary to:")
print(summary_path)

print("\nSaved tiling settings to:")
print(settings_path)

print("\nSaved dataset settings to:")
print(yaml_path)