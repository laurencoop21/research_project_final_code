from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window, bounds
from shapely.geometry import box

# Settings
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

raster_path = PROJECT_DIR / "Iraq_ND11_4_slope_TPI_uint8.tif"
split_path = MODEL_DIR / "reports" / "geographic_split.gpkg"
tile_index_path = MODEL_DIR / "reports" / "tile_index.csv"

pool_dir = MODEL_DIR / "background_mining_pool"
images_dir = pool_dir / "images"
output_csv = MODEL_DIR / "reports" / "background_mining_pool.csv"

tile_size = 512
target_candidates = 400
max_attempts = 50000
random_seed = 42

# Prepare output folder
if pool_dir.exists():
    shutil.rmtree(pool_dir)

images_dir.mkdir(parents=True, exist_ok=True)

# Load geographic split and tells
split_regions = gpd.read_file(
    split_path,
    layer="split_regions",
)

tells = gpd.read_file(
    split_path,
    layer="tells_with_split",
)

train_region = split_regions[
    split_regions["split"] == "train"
].geometry.iloc[0]

usable_tells = tells[
    tells["split"].isin(["train", "val", "test"])
].copy()

# Existing training tile coordinates
tile_index = pd.read_csv(tile_index_path)

existing_train_positions = set(
    zip(
        tile_index.loc[
            tile_index["split"] == "train",
            "row_start"
        ].astype(int),
        tile_index.loc[
            tile_index["split"] == "train",
            "col_start"
        ].astype(int),
    )
)

# Generate random clean background windows
rng = np.random.default_rng(random_seed)

records = []
used_positions = set()

with rasterio.open(raster_path) as raster:

    if split_regions.crs != raster.crs:
        split_regions = split_regions.to_crs(raster.crs)
        train_region = split_regions[
            split_regions["split"] == "train"
        ].geometry.iloc[0]

    if usable_tells.crs != raster.crs:
        usable_tells = usable_tells.to_crs(raster.crs)

    tells_sindex = usable_tells.sindex

    attempts = 0

    while (
        len(records) < target_candidates
        and attempts < max_attempts
    ):
        attempts += 1

        row_start = int(
            rng.integers(
                0,
                raster.height - tile_size + 1
            )
        )

        col_start = int(
            rng.integers(
                0,
                raster.width - tile_size + 1
            )
        )

        position = (row_start, col_start)

        # Do not duplicate a window already in the
        # real Model 7 training dataset.
        if position in existing_train_positions:
            continue

        if position in used_positions:
            continue

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

        tile_polygon = box(*tile_bounds)

        # Candidate must be completely inside
        # the training geographic region.
        if not train_region.covers(tile_polygon):
            continue

        # Candidate must not intersect ANY known tell.
        possible_indices = list(
            tells_sindex.query(
                tile_polygon,
                predicate="intersects",
            )
        )

        if possible_indices:
            continue

        image_data = raster.read(window=window)

        image_data = np.moveaxis(
            image_data,
            0,
            -1,
        )

        candidate_number = len(records)

        tile_name = (
            f"candidate_{candidate_number:04d}"
            f"_r{row_start}_c{col_start}"
        )

        image_path = images_dir / f"{tile_name}.png"

        Image.fromarray(image_data).save(image_path)

        records.append({
            "tile_name": tile_name,
            "row_start": row_start,
            "col_start": col_start,
            "left": tile_bounds[0],
            "bottom": tile_bounds[1],
            "right": tile_bounds[2],
            "top": tile_bounds[3],
            "image_path": str(image_path),
        })

        used_positions.add(position)

# Save manifest
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)

print("\nBackground mining pool")
print("----------------------")
print("Requested:", target_candidates)
print("Created:", len(df))
print("Attempts:", attempts)

print("\nSaved images to:")
print(images_dir)

print("\nSaved manifest to:")
print(output_csv)

if len(df) < target_candidates:
    print(
        "\nWARNING: Could not create the full "
        "requested candidate pool."
    )
