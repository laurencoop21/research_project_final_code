from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

# File paths
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent
REPORTS_DIR = MODEL_DIR / "reports"

files = [
    PROJECT_DIR / "elevation_utm.tif",
    PROJECT_DIR / "iraq_fabdem_slope_30m.tif",
    PROJECT_DIR / "iraq_fabdem_tpi_30m.tif",
    PROJECT_DIR / "Iraq_ND11_4_slope_TPI_uint8.tif",
]

# Save raster metadata
metadata_records = []

for path in files:
    if not path.exists():
        raise FileNotFoundError(f"Missing raster: {path}")

    with rasterio.open(path) as src:
        metadata_records.append(
            {
                "file": path.name,
                "crs": str(src.crs),
                "resolution_x": src.res[0],
                "resolution_y": src.res[1],
                "height": src.height,
                "width": src.width,
                "bands": src.count,
                "dtype": ", ".join(src.dtypes),
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            }
        )

metadata = pd.DataFrame(metadata_records)

metadata_path = (
    REPORTS_DIR / "topography_raster_metadata.csv"
)

metadata.to_csv(metadata_path, index=False)

# Inspect final TPI band at the QGIS check location
final_raster = (
    PROJECT_DIR / "Iraq_ND11_4_slope_TPI_uint8.tif"
)

x = 553999
y = 3545614

with rasterio.open(final_raster) as src:
    row, col = src.index(x, y)

    tpi_patch = src.read(
        3,
        window=Window(
            col - 6,
            row - 6,
            12,
            12,
        ),
    )

    large_patch = src.read(
        3,
        window=Window(
            col - 75,
            row - 75,
            150,
            150,
        ),
    )

# Save the 12 x 12 TPI patch
patch_path = (
    REPORTS_DIR / "topography_tpi_12x12_patch.csv"
)

pd.DataFrame(tpi_patch).to_csv(
    patch_path,
    index=False,
    header=False,
)

# Check for constant 3 x 3 blocks
block_records = []

for row_offset in range(3):
    for col_offset in range(3):

        total = 0
        constant = 0

        for r in range(
            row_offset,
            large_patch.shape[0] - 2,
            3,
        ):
            for c in range(
                col_offset,
                large_patch.shape[1] - 2,
                3,
            ):

                block = large_patch[
                    r:r + 3,
                    c:c + 3,
                ]

                if block.shape != (3, 3):
                    continue

                total += 1

                if np.all(block == block[0, 0]):
                    constant += 1

        percentage = (
            100 * constant / total
            if total
            else 0
        )

        block_records.append(
            {
                "row_offset": row_offset,
                "col_offset": col_offset,
                "constant_blocks": constant,
                "total_blocks": total,
                "constant_percentage": percentage,
            }
        )

block_results = pd.DataFrame(block_records)

block_path = (
    REPORTS_DIR / "topography_3x3_block_check.csv"
)

block_results.to_csv(
    block_path,
    index=False,
)

# Save summary
summary = {
    "check_location_x": x,
    "check_location_y": y,
    "final_raster": final_raster.name,
    "band_checked": 3,
    "band_meaning": "TPI",
    "patch_unique_values": int(
        len(np.unique(tpi_patch))
    ),
    "minimum_constant_block_percentage": float(
        block_results["constant_percentage"].min()
    ),
    "maximum_constant_block_percentage": float(
        block_results["constant_percentage"].max()
    ),
}

summary_path = (
    REPORTS_DIR / "topography_resampling_summary.json"
)

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)

# Print results
print("\nRaster metadata")
print("---------------")
print(metadata.to_string(index=False))

print("\n12 x 12 TPI values")
print("------------------")
print(tpi_patch)

print(
    "\nUnique TPI values:",
    len(np.unique(tpi_patch)),
)

print("\nConstant 3 x 3 block check")
print("--------------------------")
print(block_results.to_string(index=False))

print("\nSaved reports:")
print(metadata_path)
print(patch_path)
print(block_path)
print(summary_path)
