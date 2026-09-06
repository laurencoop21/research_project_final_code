from pathlib import Path
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

raster_path = PROJECT_DIR / "Iraq_ND11_4_slope_TPI_uint8.tif"
split_path = MODEL_DIR / "reports" / "geographic_split.gpkg"

dataset_dir = MODEL_DIR / "dataset"
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

tile_size = 512
overlap = 128
stride = tile_size - overlap
minimum_visible_fraction = 0.50

splits = ["train", "val", "test"]

# Reset the generated dataset folders
for split in splits:
    image_folder = images_dir / split
    label_folder = labels_dir / split

    if image_folder.exists():
        shutil.rmtree(image_folder)

    if label_folder.exists():
        shutil.rmtree(label_folder)

    image_folder.mkdir(parents=True)
    label_folder.mkdir(parents=True)

# Load the split data
split_regions = gpd.read_file(
    split_path,
    layer="split_regions",
)

tells = gpd.read_file(
    split_path,
    layer="tells_with_split",
)

excluded_tells = tells[tells["split"] == "exclude_boundary"].copy()
usable_tells = tells[tells["split"].isin(splits)].copy()

# Make tile starting positions
def make_starts(start, end, size, step):
    starts = list(range(start, end - size + 1, step))

    last_start = end - size

    if last_start >= start and last_start not in starts:
        starts.append(last_start)

    return sorted(starts)

# Create the image tiles and YOLO labels
tile_records = []

with rasterio.open(raster_path) as raster:
    if split_regions.crs != raster.crs:
        split_regions = split_regions.to_crs(raster.crs)

    if usable_tells.crs != raster.crs:
        usable_tells = usable_tells.to_crs(raster.crs)
        excluded_tells = excluded_tells.to_crs(raster.crs)

    row_starts = make_starts(
        0,
        raster.height,
        tile_size,
        stride,
    )

    for split in splits:
        region = split_regions[
            split_regions["split"] == split
        ].geometry.iloc[0]

        region_left, _, region_right, _ = region.bounds

        start_col = max(
            0,
            int((region_left - raster.bounds.left) / raster.res[0]),
        )

        end_col = min(
            raster.width,
            int(np.ceil(
                (region_right - raster.bounds.left) / raster.res[0]
            )),
        )

        col_starts = make_starts(
            start_col,
            end_col,
            tile_size,
            stride,
        )

        split_tells = usable_tells[
            usable_tells["split"] == split
        ].copy()

        print(f"\nCreating {split} tiles...")

        tile_number = 0

        for row_start in row_starts:
            for col_start in col_starts:
                window = Window(
                    col_start,
                    row_start,
                    tile_size,
                    tile_size,
                )

                tile_bounds = bounds(window, raster.transform)
                tile_polygon = box(*tile_bounds)

                intersecting_tells = split_tells[
                    split_tells.geometry.intersects(tile_polygon)
                ]

                labels = []

                for _, tell in intersecting_tells.iterrows():
                    visible_geometry = tell.geometry.intersection(
                        tile_polygon
                    )

                    visible_fraction = (
                        visible_geometry.area / tell.geometry.area
                    )

                    # Do not label a tell when less than half of it is
                    # visible in this tile. The overlap should capture it
                    # more completely in another tile.
                    #
                    # Importantly, keep the tile itself. The old version
                    # skipped the whole tile here, which removed other
                    # valid tells from the dataset.
                    if visible_fraction < minimum_visible_fraction:
                        continue

                    minx, miny, maxx, maxy = visible_geometry.bounds
                    left, bottom, right, top = tile_bounds

                    x_min = (minx - left) / (right - left)
                    x_max = (maxx - left) / (right - left)

                    y_min = (top - maxy) / (top - bottom)
                    y_max = (top - miny) / (top - bottom)

                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min

                    labels.append(
                        f"0 {x_center:.6f} {y_center:.6f} "
                        f"{width:.6f} {height:.6f}"
                    )

                image_data = raster.read(window=window)

                # Rasterio reads data as bands, rows, columns.
                # PIL expects rows, columns, bands.
                image_data = np.moveaxis(image_data, 0, -1)

                tile_name = (
                    f"{split}_{tile_number:04d}_"
                    f"r{row_start}_c{col_start}"
                )

                image_path = (
                    images_dir / split / f"{tile_name}.png"
                )

                label_path = (
                    labels_dir / split / f"{tile_name}.txt"
                )

                Image.fromarray(image_data).save(image_path)

                with open(label_path, "w") as label_file:
                    label_file.write("\n".join(labels))

                tile_records.append(
                    {
                        "tile_name": tile_name,
                        "split": split,
                        "row_start": row_start,
                        "col_start": col_start,
                        "tell_count": len(labels),
                        "left": tile_bounds[0],
                        "bottom": tile_bounds[1],
                        "right": tile_bounds[2],
                        "top": tile_bounds[3],
                    }
                )

                tile_number += 1

        print(f"Saved {tile_number} {split} tiles.")

# Save the tile index and dataset file
tile_index = pd.DataFrame(tile_records)

tile_index_path = MODEL_DIR / "reports" / "tile_index.csv"
tile_index.to_csv(tile_index_path, index=False)

summary = (
    tile_index.groupby("split")
    .agg(
        tiles=("tile_name", "count"),
        positive_tiles=("tell_count", lambda x: (x > 0).sum()),
        background_tiles=("tell_count", lambda x: (x == 0).sum()),
        labelled_tells=("tell_count", "sum"),
    )
    .reset_index()
)

summary_path = MODEL_DIR / "reports" / "tile_summary.csv"
summary.to_csv(summary_path, index=False)

yaml_text = f"""path: {dataset_dir}
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

print("\nSaved dataset settings to:")
print(yaml_path)