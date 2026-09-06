from pathlib import Path

import geopandas as gpd
import pandas as pd

# File paths
MODEL_DIR = Path(__file__).resolve().parent

predictions_path = (
    MODEL_DIR
    / "predictions"
    / "model_16_38SMB_raw_predictions.gpkg"
)

reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

output_path = (
    reports_dir
    / "model_16_38SMB_deduplication_threshold_audit.csv"
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

# Cross-tile non-maximum suppression
def cross_tile_nms(predictions, iou_threshold):
    """
    Keep the highest-confidence prediction.

    Suppress a lower-confidence prediction from a
    different tile when its geographic box overlaps
    the retained prediction by at least the selected
    IoU threshold.

    Predictions from the same tile are not compared
    because YOLO already performs within-image NMS.
    """

    ordered = predictions.sort_values(
        "confidence",
        ascending=False,
    )

    # Rank allows us to compare only predictions
    # below the current prediction in confidence order.
    rank = {
        index: position
        for position, index
        in enumerate(ordered.index)
    }

    spatial_index = predictions.sindex

    kept_indices = []
    suppressed_indices = set()

    for index, prediction in ordered.iterrows():

        if index in suppressed_indices:
            continue

        kept_indices.append(index)

        # Only inspect predictions whose bounding boxes
        # could spatially overlap this prediction.
        candidate_indices = list(
            spatial_index.intersection(
                prediction.geometry.bounds
            )
        )

        for other_index in candidate_indices:

            if other_index == index:
                continue

            if other_index in suppressed_indices:
                continue

            # Only consider lower-ranked predictions.
            if rank[other_index] <= rank[index]:
                continue

            other = predictions.loc[other_index]

            # Cross-tile duplicates only.
            if (
                prediction["tile_name"]
                == other["tile_name"]
            ):
                continue

            if not prediction.geometry.intersects(
                other.geometry
            ):
                continue

            iou = calculate_iou(
                prediction.geometry,
                other.geometry,
            )

            if iou >= iou_threshold:
                suppressed_indices.add(
                    other_index
                )

    return (
        predictions.loc[kept_indices].copy(),
        suppressed_indices,
    )

# Test several thresholds
thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

results = []

print("\nDeduplication threshold audit")
print("-----------------------------")

for threshold in thresholds:

    kept, suppressed = cross_tile_nms(
        predictions,
        threshold,
    )

    results.append(
        {
            "iou_threshold": threshold,
            "raw_predictions": len(predictions),
            "kept_predictions": len(kept),
            "suppressed_predictions": len(suppressed),
            "percent_suppressed": (
                100.0
                * len(suppressed)
                / len(predictions)
            ),
        }
    )

    print(
        f"IoU >= {threshold:.2f}: "
        f"kept {len(kept)}, "
        f"suppressed {len(suppressed)}"
    )

# Save audit
results_df = pd.DataFrame(results)

results_df.to_csv(
    output_path,
    index=False,
)

print("\nSaved threshold audit to:")
print(output_path)