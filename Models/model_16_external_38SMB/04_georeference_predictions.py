from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# File paths and settings
MODEL_DIR = Path(__file__).resolve().parent
reports_dir = MODEL_DIR / "reports"
predictions_dir = MODEL_DIR / "predictions"

predictions_dir.mkdir(parents=True, exist_ok=True)

tile_index_path = reports_dir / "tile_index.csv"

tile_predictions_path = (
    reports_dir / "model_16_38SMB_tile_predictions.csv"
)

output_csv = (
    reports_dir / "model_16_38SMB_georeferenced_predictions.csv"
)

output_gpkg = (
    predictions_dir / "model_16_38SMB_raw_predictions.gpkg"
)

tile_size = 320
output_crs = "EPSG:32638"

# Load tile index and Model 16 predictions
tile_index = pd.read_csv(tile_index_path)
predictions = pd.read_csv(tile_predictions_path)

print("Tile-index rows:", len(tile_index))
print("Raw predictions:", len(predictions))

# Check tile names
if tile_index["tile_name"].duplicated().any():
    raise ValueError("tile_index.csv contains duplicate tile names.")

missing_tiles = set(predictions["tile_name"]) - set(
    tile_index["tile_name"]
)

if missing_tiles:
    raise ValueError(
        f"{len(missing_tiles)} prediction tile names "
        "were not found in tile_index.csv."
    )

# Join each prediction to its tile geography
tile_columns = [
    "tile_name",
    "row_start",
    "col_start",
    "left",
    "bottom",
    "right",
    "top",
]

georeferenced = predictions.merge(
    tile_index[tile_columns],
    on="tile_name",
    how="left",
    validate="many_to_one",
)

# Convert tile-pixel coordinates to map coordinates
# The original 320 x 320 tile represents the geographic
# extent stored in tile_index.csv.
#
# Image x increases from left to right.
# Image y increases from top to bottom.
#
# Map easting increases from left to right.
# Map northing increases from bottom to top.

pixel_width = (
    georeferenced["right"] - georeferenced["left"]
) / tile_size

pixel_height = (
    georeferenced["top"] - georeferenced["bottom"]
) / tile_size

georeferenced["map_x_min"] = (
    georeferenced["left"]
    + georeferenced["x_min"] * pixel_width
)

georeferenced["map_x_max"] = (
    georeferenced["left"]
    + georeferenced["x_max"] * pixel_width
)

georeferenced["map_y_max"] = (
    georeferenced["top"]
    - georeferenced["y_min"] * pixel_height
)

georeferenced["map_y_min"] = (
    georeferenced["top"]
    - georeferenced["y_max"] * pixel_height
)

# Basic geographic checks
if (
    georeferenced["map_x_min"]
    > georeferenced["map_x_max"]
).any():
    raise ValueError("One or more predictions have invalid x bounds.")

if (
    georeferenced["map_y_min"]
    > georeferenced["map_y_max"]
).any():
    raise ValueError("One or more predictions have invalid y bounds.")

# Create prediction polygons
geometries = [
    box(
        row.map_x_min,
        row.map_y_min,
        row.map_x_max,
        row.map_y_max,
    )
    for row in georeferenced.itertuples()
]

prediction_gdf = gpd.GeoDataFrame(
    georeferenced,
    geometry=geometries,
    crs=output_crs,
)

# Add physical box dimensions
prediction_gdf["width_m"] = (
    prediction_gdf["map_x_max"]
    - prediction_gdf["map_x_min"]
)

prediction_gdf["height_m"] = (
    prediction_gdf["map_y_max"]
    - prediction_gdf["map_y_min"]
)

prediction_gdf["box_area_m2"] = (
    prediction_gdf["width_m"]
    * prediction_gdf["height_m"]
)

# Save outputs
prediction_gdf.drop(
    columns="geometry"
).to_csv(
    output_csv,
    index=False,
)

prediction_gdf.to_file(
    output_gpkg,
    layer="raw_predictions",
    driver="GPKG",
)

# Final audit
print("\nGeoreferenced Model 16 predictions")
print("----------------------------------")
print("Predictions:", len(prediction_gdf))
print("CRS:", prediction_gdf.crs)

print(
    "Easting range:",
    prediction_gdf["map_x_min"].min(),
    "to",
    prediction_gdf["map_x_max"].max(),
)

print(
    "Northing range:",
    prediction_gdf["map_y_min"].min(),
    "to",
    prediction_gdf["map_y_max"].max(),
)

print(
    "Confidence range:",
    prediction_gdf["confidence"].min(),
    "to",
    prediction_gdf["confidence"].max(),
)

print("\nSaved CSV to:")
print(output_csv)

print("\nSaved GeoPackage to:")
print(output_gpkg)
