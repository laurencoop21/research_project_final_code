from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# File paths and settings
MODEL_DIR = Path(__file__).resolve().parent
reports_dir = MODEL_DIR / "reports"

split_path = reports_dir / "geographic_split.gpkg"
tile_index_path = reports_dir / "tile_index.csv"
matches_path = reports_dir / "model_14_test_ground_truth_matches.csv"

instance_output = reports_dir / "model_14_test_instance_to_tell.csv"
tell_output = reports_dir / "model_14_unique_tell_detection.csv"
summary_output = reports_dir / "model_14_unique_tell_recall_by_size.csv"

# Visibility threshold used when labelling
minimum_visible_fraction = 0.50

# Fixed size classes in square metres
small_max_area = 16718
large_min_area = 129300

# Load the Model 14 test data
tells = gpd.read_file(
    split_path,
    layer="tells_with_split",
)

test_tells = tells[
    tells["split"] == "test"
].copy()

test_tells["polygon_id"] = (
    test_tells["polygon_id"].astype(int)
)

test_tells["area_m2"] = (
    test_tells.geometry.area
)

tile_index = pd.read_csv(tile_index_path)

test_tiles = tile_index[
    tile_index["split"] == "test"
].copy()

matches = pd.read_csv(matches_path)

# Reconstruct which tell produced each label
instance_rows = []

for _, tile in test_tiles.iterrows():

    tile_polygon = box(
        tile["left"],
        tile["bottom"],
        tile["right"],
        tile["top"],
    )

    intersecting_tells = test_tells[
        test_tells.geometry.intersects(tile_polygon)
    ]

    visible_ids = []

    for _, tell in intersecting_tells.iterrows():

        visible_geometry = tell.geometry.intersection(
            tile_polygon
        )

        visible_fraction = (
            visible_geometry.area / tell.geometry.area
        )

        if visible_fraction >= minimum_visible_fraction:
            visible_ids.append(
                int(tell["polygon_id"])
            )

    tile_matches = matches[
        matches["tile_name"] == tile["tile_name"]
    ].sort_values("ground_truth_index")

    expected_count = int(tile["tell_count"])

    if len(visible_ids) != expected_count:
        raise ValueError(
            f"{tile['tile_name']}: "
            f"{len(visible_ids)} reconstructed labels "
            f"but tile index says {expected_count}"
        )

    if len(tile_matches) != expected_count:
        raise ValueError(
            f"{tile['tile_name']}: "
            f"{len(tile_matches)} match rows "
            f"but tile index says {expected_count}"
        )

    for ground_truth_index, polygon_id in enumerate(
        visible_ids
    ):

        match_row = tile_matches.iloc[
            ground_truth_index
        ]

        matched_value = str(
            match_row["matched"]
        ).strip().lower()

        matched = matched_value == "true"

        instance_rows.append(
            {
                "polygon_id": polygon_id,
                "tile_name": tile["tile_name"],
                "ground_truth_index": (
                    ground_truth_index
                ),
                "matched": matched,
                "matched_confidence": (
                    match_row["matched_confidence"]
                ),
                "matched_iou": (
                    match_row["matched_iou"]
                ),
            }
        )


instances = pd.DataFrame(instance_rows)

instances.to_csv(
    instance_output,
    index=False,
)

# Collapse overlapping tile instances to unique tells
tell_detection = (
    instances.groupby("polygon_id")
    .agg(
        tile_instances=("matched", "count"),
        matched_instances=("matched", "sum"),
        detected=("matched", "any"),
        maximum_matched_confidence=(
            "matched_confidence",
            "max",
        ),
        maximum_matched_iou=(
            "matched_iou",
            "max",
        ),
    )
    .reset_index()
)

# Add physical tell area and size class
tell_info = test_tells[
    ["polygon_id", "area_m2"]
].copy()

tell_results = tell_info.merge(
    tell_detection,
    on="polygon_id",
    how="left",
)

tell_results["tile_instances"] = (
    tell_results["tile_instances"].fillna(0).astype(int)
)

tell_results["matched_instances"] = (
    tell_results["matched_instances"].fillna(0).astype(int)
)

tell_results["detected"] = (
    tell_results["detected"].fillna(False)
)


def assign_size_class(area):
    if area <= small_max_area:
        return "small"
    if area < large_min_area:
        return "medium"
    return "large"


tell_results["size_class"] = (
    tell_results["area_m2"].apply(
        assign_size_class
    )
)

tell_results.to_csv(
    tell_output,
    index=False,
)

# Calculate recall by unique tell
size_order = ["small", "medium", "large"]

tell_results["size_class"] = pd.Categorical(
    tell_results["size_class"],
    categories=size_order,
    ordered=True,
)

summary = (
    tell_results.groupby(
        "size_class",
        observed=False,
    )
    .agg(
        test_tells=("polygon_id", "count"),
        detected_tells=("detected", "sum"),
    )
    .reset_index()
)

summary["missed_tells"] = (
    summary["test_tells"]
    - summary["detected_tells"]
)

summary["recall"] = (
    summary["detected_tells"]
    / summary["test_tells"]
)

overall = pd.DataFrame(
    {
        "size_class": ["overall"],
        "test_tells": [len(tell_results)],
        "detected_tells": [
            int(tell_results["detected"].sum())
        ],
        "missed_tells": [
            int((~tell_results["detected"]).sum())
        ],
        "recall": [
            tell_results["detected"].mean()
        ],
    }
)

summary = pd.concat(
    [summary, overall],
    ignore_index=True,
)

summary.to_csv(
    summary_output,
    index=False,
)

# Print results
print("Model 14 unique-tell recall")
print("---------------------------")
print("Confidence threshold: 0.10")
print("IoU threshold:        0.50")
print()
print(summary.to_string(index=False))

print("\nInstance-to-tell rows:", len(instances))
print("Unique test tells:", len(tell_results))

print("\nSaved unique-tell results to:")
print(tell_output)

print("\nSaved size recall summary to:")
print(summary_output)
