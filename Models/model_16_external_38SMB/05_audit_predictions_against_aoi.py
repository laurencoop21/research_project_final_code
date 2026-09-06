from pathlib import Path

import geopandas as gpd

# File paths
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

predictions_path = (
    MODEL_DIR
    / "predictions"
    / "model_16_38SMB_raw_predictions.gpkg"
)

aoi_path = PROJECT_DIR / "38SMB_external_test_AOI.shp"

reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

output_path = (
    reports_dir
    / "model_16_38SMB_aoi_prediction_audit.csv"
)

# Check inputs
if not predictions_path.exists():
    raise FileNotFoundError(
        f"Missing predictions: {predictions_path}"
    )

if not aoi_path.exists():
    raise FileNotFoundError(
        f"Missing external AOI: {aoi_path}"
    )

# Load spatial data
predictions = gpd.read_file(
    predictions_path,
    layer="raw_predictions",
)

aoi = gpd.read_file(aoi_path)

print("Raw predictions:", len(predictions))
print("Prediction CRS:", predictions.crs)
print("AOI features:", len(aoi))
print("AOI CRS:", aoi.crs)

# Put AOI into prediction CRS
if aoi.crs != predictions.crs:
    aoi = aoi.to_crs(predictions.crs)

aoi_geometry = aoi.geometry.union_all()

# Audit relationship between predictions and AOI
predictions["prediction_area_m2"] = (
    predictions.geometry.area
)

predictions["intersects_aoi"] = (
    predictions.geometry.intersects(aoi_geometry)
)

predictions["centroid_inside_aoi"] = (
    predictions.geometry.centroid.within(aoi_geometry)
)

intersection_area = predictions.geometry.apply(
    lambda geom: geom.intersection(aoi_geometry).area
)

predictions["area_inside_aoi_m2"] = intersection_area

predictions["fraction_inside_aoi"] = (
    predictions["area_inside_aoi_m2"]
    / predictions["prediction_area_m2"]
)

predictions["at_least_50pct_inside"] = (
    predictions["fraction_inside_aoi"] >= 0.50
)

predictions["fully_inside_aoi"] = (
    predictions.geometry.apply(aoi_geometry.covers)
)

# Save audit table
audit_columns = [
    "tile_name",
    "prediction_index",
    "confidence",
    "width_m",
    "height_m",
    "box_area_m2",
    "intersects_aoi",
    "centroid_inside_aoi",
    "fraction_inside_aoi",
    "at_least_50pct_inside",
    "fully_inside_aoi",
]

predictions[audit_columns].to_csv(
    output_path,
    index=False,
)

# Print summary
print("\nAOI prediction audit")
print("--------------------")
print("Raw predictions:", len(predictions))

print(
    "Intersect AOI:",
    int(predictions["intersects_aoi"].sum()),
)

print(
    "Centroid inside AOI:",
    int(predictions["centroid_inside_aoi"].sum()),
)

print(
    "At least 50% inside AOI:",
    int(predictions["at_least_50pct_inside"].sum()),
)

print(
    "Fully inside AOI:",
    int(predictions["fully_inside_aoi"].sum()),
)

print(
    "Completely outside AOI:",
    int((~predictions["intersects_aoi"]).sum()),
)

print("\nFraction-inside summary")
print("-----------------------")
print(
    predictions["fraction_inside_aoi"].describe()
)

print("\nSaved audit to:")
print(output_path)
