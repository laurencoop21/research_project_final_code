from pathlib import Path
import json

import geopandas as gpd
import pandas as pd

# File paths and settings
MODEL_DIR = Path(__file__).resolve().parent

predictions_path = (
    MODEL_DIR
    / "predictions"
    / "model_16_38SMB_raw_predictions.gpkg"
)

predictions_dir = MODEL_DIR / "predictions"
reports_dir = MODEL_DIR / "reports"

predictions_dir.mkdir(exist_ok=True)
reports_dir.mkdir(exist_ok=True)

candidate_gpkg_path = (
    predictions_dir
    / "model_16_38SMB_review_candidates.gpkg"
)

candidate_csv_path = (
    reports_dir
    / "model_16_38SMB_review_candidates.csv"
)

suppressed_csv_path = (
    reports_dir
    / "model_16_38SMB_suppressed_duplicates.csv"
)

summary_path = (
    reports_dir
    / "model_16_38SMB_deduplication_summary.json"
)

iou_threshold = 0.70

# Load raw georeferenced predictions
predictions = gpd.read_file(
    predictions_path,
    layer="raw_predictions",
)

predictions = predictions.reset_index(drop=True)

predictions["raw_detection_id"] = [
    f"R{i:04d}"
    for i in range(1, len(predictions) + 1)
]

print("Raw predictions:", len(predictions))
print("Deduplication IoU threshold:", iou_threshold)

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

# Conservative cross-tile NMS
ordered_indices = (
    predictions
    .sort_values(
        "confidence",
        ascending=False,
    )
    .index
    .tolist()
)

suppressed_indices = set()
kept_indices = []

# Map each suppressed raw detection to the retained
# representative that caused it to be suppressed.
suppression_records = []

for index in ordered_indices:

    if index in suppressed_indices:
        continue

    kept_indices.append(index)

    representative = predictions.loc[index]

    for other_index in ordered_indices:

        if other_index == index:
            continue

        if other_index in suppressed_indices:
            continue

        if other_index in kept_indices:
            continue

        other = predictions.loc[other_index]

        # Do not repeat within-tile NMS.
        # YOLO already performed NMS within each tile.
        if (
            representative["tile_name"]
            == other["tile_name"]
        ):
            continue

        if not representative.geometry.intersects(
            other.geometry
        ):
            continue

        iou = calculate_iou(
            representative.geometry,
            other.geometry,
        )

        if iou >= iou_threshold:
            suppressed_indices.add(other_index)

            suppression_records.append(
                {
                    "suppressed_index": other_index,
                    "kept_index": index,
                    "iou": float(iou),
                }
            )

# Build retained candidate table
candidates = predictions.loc[
    kept_indices
].copy()

candidates = candidates.sort_values(
    "confidence",
    ascending=False,
).reset_index(drop=True)

candidates["candidate_id"] = [
    f"C{i:04d}"
    for i in range(1, len(candidates) + 1)
]

# Map original dataframe index to candidate ID.
index_to_candidate_id = {
    original_index: candidate_id
    for original_index, candidate_id in zip(
        candidates.index,
        candidates["candidate_id"],
    )
}

# Because reset_index changed the dataframe index,
# rebuild the mapping using raw_detection_id.
raw_to_candidate = dict(
    zip(
        candidates["raw_detection_id"],
        candidates["candidate_id"],
    )
)

original_index_to_raw = predictions[
    "raw_detection_id"
].to_dict()

suppressed_rows = []

for record in suppression_records:
    kept_raw_id = original_index_to_raw[
        record["kept_index"]
    ]

    suppressed_raw_id = original_index_to_raw[
        record["suppressed_index"]
    ]

    suppressed_detection = predictions.loc[
        record["suppressed_index"]
    ]

    suppressed_rows.append(
        {
            "candidate_id": raw_to_candidate[
                kept_raw_id
            ],
            "kept_raw_detection_id": kept_raw_id,
            "suppressed_raw_detection_id": (
                suppressed_raw_id
            ),
            "suppressed_tile_name": (
                suppressed_detection["tile_name"]
            ),
            "suppressed_confidence": float(
                suppressed_detection["confidence"]
            ),
            "iou_with_kept_candidate": (
                record["iou"]
            ),
        }
    )

suppressed = pd.DataFrame(
    suppressed_rows,
    columns=[
        "candidate_id",
        "kept_raw_detection_id",
        "suppressed_raw_detection_id",
        "suppressed_tile_name",
        "suppressed_confidence",
        "iou_with_kept_candidate",
    ],
)

# Count source detections per retained candidate
if len(suppressed) > 0:
    duplicate_counts = (
        suppressed
        .groupby("candidate_id")
        .size()
        .to_dict()
    )
else:
    duplicate_counts = {}

candidates["source_detection_count"] = (
    candidates["candidate_id"].map(
        lambda candidate_id:
        1 + duplicate_counts.get(candidate_id, 0)
    )
)

# Add blank advisor-review fields
candidates["advisor_label"] = ""
candidates["advisor_notes"] = ""

# Save outputs
candidate_csv_columns = [
    "candidate_id",
    "raw_detection_id",
    "confidence",
    "source_detection_count",
    "tile_name",
    "prediction_index",
    "width_m",
    "height_m",
    "box_area_m2",
    "map_x_min",
    "map_y_min",
    "map_x_max",
    "map_y_max",
    "advisor_label",
    "advisor_notes",
]

candidates[candidate_csv_columns].to_csv(
    candidate_csv_path,
    index=False,
)

candidates.to_file(
    candidate_gpkg_path,
    layer="review_candidates",
    driver="GPKG",
)

suppressed.to_csv(
    suppressed_csv_path,
    index=False,
)

# Save summary
summary = {
    "raw_predictions": int(len(predictions)),
    "deduplication_iou_threshold": iou_threshold,
    "review_candidates": int(len(candidates)),
    "suppressed_duplicates": int(len(suppressed)),
    "candidates_with_multiple_source_detections": int(
        (candidates["source_detection_count"] > 1).sum()
    ),
    "maximum_source_detection_count": int(
        candidates["source_detection_count"].max()
    ),
}

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)

# Final audit
print("\n38SMB review candidate deduplication")
print("-----------------------------------")
print("Raw predictions:", len(predictions))
print("Review candidates:", len(candidates))
print("Suppressed duplicates:", len(suppressed))

print(
    "Candidates with multiple source detections:",
    int(
        (
            candidates["source_detection_count"] > 1
        ).sum()
    ),
)

print(
    "Maximum source detections for one candidate:",
    int(
        candidates["source_detection_count"].max()
    ),
)

print("\nSaved review candidates to:")
print(candidate_gpkg_path)

print("\nSaved review CSV to:")
print(candidate_csv_path)

print("\nSaved suppressed-duplicate audit to:")
print(suppressed_csv_path)

print("\nSaved summary to:")
print(summary_path)
