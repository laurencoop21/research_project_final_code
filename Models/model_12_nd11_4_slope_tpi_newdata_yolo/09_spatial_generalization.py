from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# File paths
MODEL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = MODEL_DIR / "reports"

detection_path = (
    REPORTS_DIR / "model_12_unique_tell_detection.csv"
)

split_path = (
    REPORTS_DIR / "geographic_split.gpkg"
)

# Load Model 12 unique-tell detections
detections = pd.read_csv(detection_path)

tells = gpd.read_file(
    split_path,
    layer="tells_with_split",
)

regions = gpd.read_file(
    split_path,
    layer="split_regions",
)

test_tells = tells[
    tells["split"] == "test"
].copy()

# Join tell location to detection result
data = test_tells[
    ["polygon_id", "centroid_x"]
].merge(
    detections,
    on="polygon_id",
    how="left",
    validate="one_to_one",
)

if len(data) != 201:
    raise ValueError(
        f"Expected 201 test tells, found {len(data)}"
    )

if data["detected"].isna().any():
    raise ValueError(
        "Some test tells are missing detection results."
    )

# Convert detection field to 0 / 1
data["detected_numeric"] = (
    data["detected"]
    .astype(str)
    .str.lower()
    .map(
        {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
        }
    )
)

if data["detected_numeric"].isna().any():
    raise ValueError(
        "Could not interpret all detected values."
    )

# Position within the east-west test region
test_region = regions[
    regions["split"] == "test"
].geometry.iloc[0]

west = test_region.bounds[0]
east = test_region.bounds[2]
width = east - west

data["distance_from_west_km"] = (
    data["centroid_x"] - west
) / 1000

data["relative_east_position"] = (
    data["centroid_x"] - west
) / width

# Divide the test region into equal-width thirds
data["test_third"] = pd.cut(
    data["relative_east_position"],
    bins=[
        -np.inf,
        1 / 3,
        2 / 3,
        np.inf,
    ],
    labels=[
        "western third",
        "middle third",
        "eastern third",
    ],
)

# Recall by geographic third
third_summary = (
    data.groupby(
        "test_third",
        observed=False,
    )
    .agg(
        tells=("polygon_id", "count"),
        detected=("detected_numeric", "sum"),
    )
    .reset_index()
)

third_summary["recall"] = (
    third_summary["detected"]
    / third_summary["tells"]
)

# Recall by geographic third and tell size
third_size_summary = (
    data.groupby(
        ["test_third", "size_class"],
        observed=False,
    )
    .agg(
        tells=("polygon_id", "count"),
        detected=("detected_numeric", "sum"),
    )
    .reset_index()
)

third_size_summary["recall"] = np.where(
    third_size_summary["tells"] > 0,
    (
        third_size_summary["detected"]
        / third_size_summary["tells"]
    ),
    np.nan,
)

# Compare detected and missed tell locations
location_summary = (
    data.groupby(
        "detected_numeric"
    )
    .agg(
        tells=("polygon_id", "count"),
        mean_distance_km=(
            "distance_from_west_km",
            "mean",
        ),
        median_distance_km=(
            "distance_from_west_km",
            "median",
        ),
        mean_relative_position=(
            "relative_east_position",
            "mean",
        ),
    )
    .reset_index()
)

location_summary["status"] = (
    location_summary["detected_numeric"]
    .map(
        {
            0: "missed",
            1: "detected",
        }
    )
)

# Save results
third_summary.to_csv(
    REPORTS_DIR
    / "model_12_recall_by_test_third.csv",
    index=False,
)

third_size_summary.to_csv(
    REPORTS_DIR
    / "model_12_recall_by_test_third_and_size.csv",
    index=False,
)

location_summary.to_csv(
    REPORTS_DIR
    / "model_12_detected_vs_missed_location.csv",
    index=False,
)

data.to_csv(
    REPORTS_DIR
    / "model_12_test_tells_spatial_detection.csv",
    index=False,
)

# Print results
print(
    "\nModel 12 unique-tell recall "
    "by test-region third"
)
print("-----------------------------------------")
print(third_summary.to_string(index=False))

print(
    "\nRecall by test-region third "
    "and size class"
)
print("-----------------------------------------")
print(third_size_summary.to_string(index=False))

print("\nDetected versus missed locations")
print("---------------------------------")
print(location_summary.to_string(index=False))

print("\nSaved spatial-generalization reports.")
