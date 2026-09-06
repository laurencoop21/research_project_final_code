from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window, bounds

# Settings and file paths
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

raster_path = PROJECT_DIR / "38SMB_ND11_4_slope_TPI_uint8.tif"

tiles_dir = MODEL_DIR / "tiles"

reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# Same tile settings used to construct the Model 16 dataset.
tile_size = 320
overlap = 80
stride = tile_size - overlap

# Reset the generated tile folder
if tiles_dir.exists():
    shutil.rmtree(tiles_dir)

tiles_dir.mkdir(parents=True)

# Make tile starting positions
def make_starts(start, end, size, step):
    starts = list(range(start, end - size + 1, step))

    last_start = end - size

    if last_start >= start and last_start not in starts:
        starts.append(last_start)

    return sorted(starts)

# Create the external inference tiles
tile_records = []

with rasterio.open(raster_path) as raster:
    if raster.count != 3:
        raise ValueError(
            f"Expected a 3-band raster, but found {raster.count} bands."
        )

    row_starts = make_starts(
        0,
        raster.height,
        tile_size,
        stride,
    )

    col_starts = make_starts(
        0,
        raster.width,
        tile_size,
        stride,
    )

    print("Raster shape:", (raster.height, raster.width))
    print("Tile size:", tile_size)
    print("Overlap:", overlap)
    print("Stride:", stride)
    print("Row positions:", len(row_starts))
    print("Column positions:", len(col_starts))
    print("Expected tiles:", len(row_starts) * len(col_starts))

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

            image_data = raster.read(window=window)

            # Rasterio reads data as bands, rows, columns.
            # PIL expects rows, columns, bands.
            image_data = np.moveaxis(image_data, 0, -1)

            tile_name = (
                f"38SMB_{tile_number:05d}_"
                f"r{row_start}_c{col_start}"
            )

            image_path = tiles_dir / f"{tile_name}.png"

            Image.fromarray(image_data).save(image_path)

            tile_records.append(
                {
                    "tile_name": tile_name,
                    "row_start": row_start,
                    "col_start": col_start,
                    "left": tile_bounds[0],
                    "bottom": tile_bounds[1],
                    "right": tile_bounds[2],
                    "top": tile_bounds[3],
                }
            )

            tile_number += 1

# Save the tile index
tile_index = pd.DataFrame(tile_records)

tile_index_path = reports_dir / "tile_index.csv"
tile_index.to_csv(tile_index_path, index=False)

# Print the results
print("\nInference tile summary")
print("----------------------")
print("Tiles saved:", len(tile_index))
print("Tile size:", tile_size)
print("Overlap:", overlap)
print("Stride:", stride)

print("\nSaved tiles to:")
print(tiles_dir)

print("\nSaved tile index to:")
print(tile_index_path)
