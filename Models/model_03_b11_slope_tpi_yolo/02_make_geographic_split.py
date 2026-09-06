from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

# File paths
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

raster_path = PROJECT_DIR / "Iraq_B11_slope_TPI_uint8.tif"
tells_path = PROJECT_DIR / "tells.shp"

reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

split_gpkg_path = reports_dir / "geographic_split.gpkg"
split_csv_path = reports_dir / "geographic_split_summary.csv"

# Load the raster and tell polygons
with rasterio.open(raster_path) as raster:
    raster_crs = raster.crs
    left = raster.bounds.left
    right = raster.bounds.right
    bottom = raster.bounds.bottom
    top = raster.bounds.top

tells = gpd.read_file(tells_path)

if tells.crs != raster_crs:
    tells = tells.to_crs(raster_crs)

# Find the split boundaries
# The tells are split from west to east.
# The boundaries are based on tell centroid locations so that
# approximately 70% are in train, 15% in validation, and 15% in test.

tells["centroid_x"] = tells.geometry.centroid.x

train_boundary = tells["centroid_x"].quantile(0.70)
val_boundary = tells["centroid_x"].quantile(0.85)

print("Geographic split boundaries")
print("---------------------------")
print("Train/validation boundary:", train_boundary)
print("Validation/test boundary:", val_boundary)

# Create the three split regions
split_regions = gpd.GeoDataFrame(
    {
        "split": ["train", "val", "test"],
        "geometry": [
            box(left, bottom, train_boundary, top),
            box(train_boundary, bottom, val_boundary, top),
            box(val_boundary, bottom, right, top),
        ],
    },
    crs=raster_crs,
)

# Assign each tell to a split
def assign_split(x):
    if x < train_boundary:
        return "train"
    elif x < val_boundary:
        return "val"
    else:
        return "test"


tells["split"] = tells["centroid_x"].apply(assign_split)

# Find tells that cross a split boundary
crosses_train_boundary = (
    (tells.geometry.bounds["minx"] < train_boundary)
    & (tells.geometry.bounds["maxx"] > train_boundary)
)

crosses_val_boundary = (
    (tells.geometry.bounds["minx"] < val_boundary)
    & (tells.geometry.bounds["maxx"] > val_boundary)
)

tells["crosses_boundary"] = (
    crosses_train_boundary | crosses_val_boundary
)

# Boundary-crossing tells will be excluded later so that the same
# archaeological site cannot appear in more than one data split.
tells.loc[tells["crosses_boundary"], "split"] = "exclude_boundary"

# Print the split counts
summary = (
    tells["split"]
    .value_counts()
    .rename_axis("split")
    .reset_index(name="tell_count")
)

print("\nTell counts")
print("-----------")
print(summary.to_string(index=False))

# Save the split files
if split_gpkg_path.exists():
    split_gpkg_path.unlink()

split_regions.to_file(
    split_gpkg_path,
    layer="split_regions",
    driver="GPKG",
)

tells.to_file(
    split_gpkg_path,
    layer="tells_with_split",
    driver="GPKG",
)

summary.to_csv(split_csv_path, index=False)

print("\nSaved geographic split to:")
print(split_gpkg_path)

print("\nSaved split summary to:")
print(split_csv_path)