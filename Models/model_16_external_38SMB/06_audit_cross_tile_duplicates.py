from pathlib import Path

import geopandas as gpd
import numpy as np

# File paths
MODEL_DIR = Path(__file__).resolve().parent

predictions_path = (
    MODEL_DIR
    / "predictions"
    / "model_16_38SMB_raw_predictions.gpkg"
)

# Load predictions
predictions = gpd.read_file(
    predictions_path,
    layer="raw_predictions",
)

print("Raw predictions:", len(predictions))

# IoU function
def calculate_iou(geometry_a, geometry_b):
    intersection_area = (
        geometry_a.intersection(geometry_b).area
    )

    if intersection_area == 0:
        return 0.0

    union_area = (
        geometry_a.area
        + geometry_b.area
        - intersection_area
    )

    if union_area == 0:
        return 0.0

    return intersection_area / union_area

# Find overlapping predictions from different tiles
spatial_index = predictions.sindex

overlap_ious = []

for i, row in predictions.iterrows():

    candidate_indices = list(
        spatial_index.intersection(
            row.geometry.bounds
        )
    )

    for j in candidate_indices:

        if j <= i:
            continue

        other = predictions.loc[j]

        # We are interested specifically in duplicate
        # detections produced by different overlapping tiles.
        if row["tile_name"] == other["tile_name"]:
            continue

        if not row.geometry.intersects(
            other.geometry
        ):
            continue

        iou = calculate_iou(
            row.geometry,
            other.geometry,
        )

        if iou > 0:
            overlap_ious.append(iou)

# Summarise overlap
overlap_ious = np.array(
    overlap_ious,
    dtype=float,
)

print("\nCross-tile overlap audit")
print("------------------------")
print(
    "Overlapping prediction pairs:",
    len(overlap_ious),
)

for threshold in [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]:
    count = int(
        (overlap_ious >= threshold).sum()
    )

    print(
        f"Pairs with IoU >= {threshold:.2f}:",
        count,
    )

if len(overlap_ious) > 0:
    print("\nIoU distribution")
    print("----------------")
    print(
        "Minimum:",
        float(overlap_ious.min()),
    )
    print(
        "Median:",
        float(np.median(overlap_ious)),
    )
    print(
        "Maximum:",
        float(overlap_ious.max()),
    )
