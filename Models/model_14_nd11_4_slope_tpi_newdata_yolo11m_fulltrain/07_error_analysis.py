from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO


MODEL_DIR = Path(__file__).resolve().parent

model_path = MODEL_DIR / "checkpoints" / "model_14_best.pt"
images_dir = MODEL_DIR / "dataset" / "images" / "test"
labels_dir = MODEL_DIR / "dataset" / "labels" / "test"

reports_dir = MODEL_DIR / "reports"
output_dir = MODEL_DIR / "figures" / "test_error_analysis"

all_tiles_dir = output_dir / "all_annotated_tiles"
tp_examples_dir = output_dir / "true_positive_examples"
fp_examples_dir = output_dir / "false_positive_examples"
fn_examples_dir = output_dir / "false_negative_examples"

# Prediction and matching thresholds
confidence_threshold = 0.10
iou_threshold = 0.50
examples_per_group = 12


if not model_path.exists():
    raise FileNotFoundError(f"Missing model: {model_path}")

if not images_dir.exists():
    raise FileNotFoundError(f"Missing test images: {images_dir}")

if not labels_dir.exists():
    raise FileNotFoundError(f"Missing test labels: {labels_dir}")


if output_dir.exists():
    shutil.rmtree(output_dir)

for folder in [
    all_tiles_dir,
    tp_examples_dir,
    fp_examples_dir,
    fn_examples_dir,
]:
    folder.mkdir(parents=True, exist_ok=True)

reports_dir.mkdir(exist_ok=True)


# Convert YOLO labels to pixel coordinates
def read_ground_truth(label_path, image_width, image_height):
    boxes = []

    with open(label_path) as file:
        lines = [line.strip() for line in file if line.strip()]

    for line in lines:
        _, x_center, y_center, width, height = map(
            float,
            line.split(),
        )

        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height

        boxes.append(
            [
                x_center - width / 2,
                y_center - height / 2,
                x_center + width / 2,
                y_center + height / 2,
            ]
        )

    return np.array(boxes, dtype=float).reshape(-1, 4)


# Intersection over union of two boxes
def calculate_iou(box_a, box_b):
    intersection_x_min = max(box_a[0], box_b[0])
    intersection_y_min = max(box_a[1], box_b[1])
    intersection_x_max = min(box_a[2], box_b[2])
    intersection_y_max = min(box_a[3], box_b[3])

    intersection_width = max(
        0,
        intersection_x_max - intersection_x_min,
    )

    intersection_height = max(
        0,
        intersection_y_max - intersection_y_min,
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = (
        max(0, box_a[2] - box_a[0])
        * max(0, box_a[3] - box_a[1])
    )

    area_b = (
        max(0, box_b[2] - box_b[0])
        * max(0, box_b[3] - box_b[1])
    )

    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def draw_box(draw, box, outline, text):
    x_min, y_min, x_max, y_max = box

    draw.rectangle(
        [x_min, y_min, x_max, y_max],
        outline=outline,
        width=3,
    )

    draw.text(
        (max(0, x_min), max(0, y_min - 12)),
        text,
        fill=outline,
    )


if torch.cuda.is_available():
    device = 0
else:
    device = "cpu"
    
print("Prediction device:", device)
print("Confidence threshold:", confidence_threshold)
print("IoU matching threshold:", iou_threshold)

model = YOLO(str(model_path))

results_stream = model.predict(
    source=str(images_dir),
    imgsz=512,
    conf=confidence_threshold,
    iou=0.70,
    device=device,
    stream=True,
    verbose=False,
)


tile_rows = []
prediction_rows = []
ground_truth_rows = []

for result in results_stream:
    image_path = Path(result.path)
    tile_name = image_path.stem
    label_path = labels_dir / f"{tile_name}.txt"

    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    ground_truth_boxes = read_ground_truth(
        label_path,
        image_width,
        image_height,
    )

    if result.boxes is None or len(result.boxes) == 0:
        predicted_boxes = np.empty((0, 4), dtype=float)
        predicted_confidences = np.empty((0,), dtype=float)
    else:
        predicted_boxes = result.boxes.xyxy.cpu().numpy()
        predicted_confidences = result.boxes.conf.cpu().numpy()

    # Match predictions to labels, highest confidence first
    prediction_order = np.argsort(-predicted_confidences)

    unmatched_ground_truth = set(
        range(len(ground_truth_boxes))
    )

    prediction_statuses = []

    ground_truth_matches = {
        index: None
        for index in range(len(ground_truth_boxes))
    }

    for prediction_index in prediction_order:
        predicted_box = predicted_boxes[prediction_index]

        best_ground_truth_index = None
        best_iou = 0.0

        for ground_truth_index in unmatched_ground_truth:
            current_iou = calculate_iou(
                predicted_box,
                ground_truth_boxes[ground_truth_index],
            )

            if current_iou > best_iou:
                best_iou = current_iou
                best_ground_truth_index = ground_truth_index

        if (
            best_ground_truth_index is not None
            and best_iou >= iou_threshold
        ):
            status = "true_positive"

            unmatched_ground_truth.remove(
                best_ground_truth_index
            )

            ground_truth_matches[
                best_ground_truth_index
            ] = {
                "prediction_index": int(prediction_index),
                "confidence": float(
                    predicted_confidences[prediction_index]
                ),
                "iou": float(best_iou),
            }
        else:
            status = "false_positive"
            best_ground_truth_index = None

        prediction_statuses.append(
            (
                int(prediction_index),
                status,
                best_ground_truth_index,
                float(best_iou),
            )
        )

    true_positive_count = sum(
        status == "true_positive"
        for _, status, _, _ in prediction_statuses
    )

    false_positive_count = sum(
        status == "false_positive"
        for _, status, _, _ in prediction_statuses
    )

    false_negative_count = len(
        unmatched_ground_truth
    )

    maximum_true_positive_confidence = 0.0
    maximum_false_positive_confidence = 0.0

    for (
        prediction_index,
        status,
        matched_ground_truth_index,
        best_iou,
    ) in prediction_statuses:
        confidence = float(
            predicted_confidences[prediction_index]
        )

        if status == "true_positive":
            maximum_true_positive_confidence = max(
                maximum_true_positive_confidence,
                confidence,
            )
        else:
            maximum_false_positive_confidence = max(
                maximum_false_positive_confidence,
                confidence,
            )

        predicted_box = predicted_boxes[prediction_index]

        prediction_rows.append(
            {
                "tile_name": tile_name,
                "prediction_index": prediction_index,
                "status": status,
                "confidence": confidence,
                "best_iou": best_iou,
                "matched_ground_truth_index": (
                    matched_ground_truth_index
                ),
                "x_min": predicted_box[0],
                "y_min": predicted_box[1],
                "x_max": predicted_box[2],
                "y_max": predicted_box[3],
            }
        )

    for ground_truth_index, ground_truth_box in enumerate(
        ground_truth_boxes
    ):
        width_pixels = (
            ground_truth_box[2] - ground_truth_box[0]
        )

        height_pixels = (
            ground_truth_box[3] - ground_truth_box[1]
        )

        area_pixels = width_pixels * height_pixels

        match = ground_truth_matches[ground_truth_index]

        ground_truth_rows.append(
            {
                "tile_name": tile_name,
                "ground_truth_index": ground_truth_index,
                "matched": match is not None,
                "matched_confidence": (
                    match["confidence"]
                    if match is not None
                    else np.nan
                ),
                "matched_iou": (
                    match["iou"]
                    if match is not None
                    else np.nan
                ),
                "x_min": ground_truth_box[0],
                "y_min": ground_truth_box[1],
                "x_max": ground_truth_box[2],
                "y_max": ground_truth_box[3],
                "width_pixels": width_pixels,
                "height_pixels": height_pixels,
                "area_pixels": area_pixels,
            }
        )

    tile_rows.append(
        {
            "tile_name": tile_name,
            "ground_truth_boxes": len(
                ground_truth_boxes
            ),
            "predicted_boxes": len(
                predicted_boxes
            ),
            "true_positives": true_positive_count,
            "false_positives": false_positive_count,
            "false_negatives": false_negative_count,
            "maximum_true_positive_confidence": (
                maximum_true_positive_confidence
            ),
            "maximum_false_positive_confidence": (
                maximum_false_positive_confidence
            ),
        }
    )

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    for ground_truth_index, ground_truth_box in enumerate(
        ground_truth_boxes
    ):
        if ground_truth_matches[ground_truth_index] is None:
            draw_box(
                draw,
                ground_truth_box,
                "yellow",
                "FN ground truth",
            )
        else:
            draw_box(
                draw,
                ground_truth_box,
                "white",
                "ground truth",
            )

    for (
        prediction_index,
        status,
        _,
        best_iou,
    ) in prediction_statuses:
        predicted_box = predicted_boxes[prediction_index]
        confidence = float(
            predicted_confidences[prediction_index]
        )

        if status == "true_positive":
            draw_box(
                draw,
                predicted_box,
                "lime",
                (
                    f"TP {confidence:.2f} "
                    f"IoU {best_iou:.2f}"
                ),
            )
        else:
            draw_box(
                draw,
                predicted_box,
                "red",
                f"FP {confidence:.2f}",
            )

    summary_text = (
        f"TP={true_positive_count}  "
        f"FP={false_positive_count}  "
        f"FN={false_negative_count}"
    )

    draw.rectangle(
        [0, 0, 220, 20],
        fill="black",
    )

    draw.text(
        (5, 5),
        summary_text,
        fill="white",
    )

    annotated_image.save(
        all_tiles_dir / f"{tile_name}_errors.png"
    )


tile_results = pd.DataFrame(tile_rows)
prediction_results_table = pd.DataFrame(prediction_rows)
ground_truth_results = pd.DataFrame(ground_truth_rows)

tile_results.to_csv(
    reports_dir / "model_14_error_analysis_by_tile.csv",
    index=False,
)

prediction_results_table.to_csv(
    reports_dir / "model_14_test_predictions.csv",
    index=False,
)

total_true_positives = int(
    tile_results["true_positives"].sum()
)

total_false_positives = int(
    tile_results["false_positives"].sum()
)

total_false_negatives = int(
    tile_results["false_negatives"].sum()
)

precision_denominator = (
    total_true_positives + total_false_positives
)

recall_denominator = (
    total_true_positives + total_false_negatives
)

threshold_precision = (
    total_true_positives / precision_denominator
    if precision_denominator > 0
    else 0.0
)

threshold_recall = (
    total_true_positives / recall_denominator
    if recall_denominator > 0
    else 0.0
)

# Group labelled boxes into size quartiles
area_values = ground_truth_results["area_pixels"]

quartile_1 = float(area_values.quantile(0.25))
quartile_2 = float(area_values.quantile(0.50))
quartile_3 = float(area_values.quantile(0.75))


def assign_size_group(area):
    if area <= quartile_1:
        return "smallest_25_percent"
    if area <= quartile_2:
        return "small_to_medium"
    if area <= quartile_3:
        return "medium_to_large"
    return "largest_25_percent"


ground_truth_results["size_group"] = (
    ground_truth_results["area_pixels"].apply(
        assign_size_group
    )
)

ground_truth_results.to_csv(
    reports_dir / "model_14_test_ground_truth_matches.csv",
    index=False,
)

size_summary = (
    ground_truth_results.groupby(
        "size_group",
        sort=False,
    )
    .agg(
        labelled_instances=("matched", "count"),
        detected_instances=("matched", "sum"),
        mean_width_pixels=("width_pixels", "mean"),
        mean_height_pixels=("height_pixels", "mean"),
        mean_area_pixels=("area_pixels", "mean"),
    )
    .reset_index()
)

size_summary["recall"] = (
    size_summary["detected_instances"]
    / size_summary["labelled_instances"]
)

size_summary.to_csv(
    reports_dir / "model_14_recall_by_box_size.csv",
    index=False,
)


# Save example tiles for each outcome
def copy_examples(table, sort_columns, output_folder):
    selected = (
        table.sort_values(
            sort_columns,
            ascending=False,
        )
        .head(examples_per_group)
    )

    for tile_name in selected["tile_name"]:
        source_path = (
            all_tiles_dir / f"{tile_name}_errors.png"
        )

        shutil.copy2(
            source_path,
            output_folder / source_path.name,
        )


copy_examples(
    tile_results[tile_results["true_positives"] > 0],
    [
        "maximum_true_positive_confidence",
        "true_positives",
    ],
    tp_examples_dir,
)

copy_examples(
    tile_results[tile_results["false_positives"] > 0],
    [
        "maximum_false_positive_confidence",
        "false_positives",
    ],
    fp_examples_dir,
)

copy_examples(
    tile_results[tile_results["false_negatives"] > 0],
    [
        "false_negatives",
        "ground_truth_boxes",
    ],
    fn_examples_dir,
)

summary = {
    "confidence_threshold": confidence_threshold,
    "iou_threshold": iou_threshold,
    "test_tiles": int(len(tile_results)),
    "ground_truth_instances": int(
        len(ground_truth_results)
    ),
    "predicted_instances": int(
        len(prediction_results_table)
    ),
    "true_positives": total_true_positives,
    "false_positives": total_false_positives,
    "false_negatives": total_false_negatives,
    "precision_at_selected_threshold": (
        threshold_precision
    ),
    "recall_at_selected_threshold": (
        threshold_recall
    ),
    "ground_truth_area_quartiles_pixels": {
        "q1": quartile_1,
        "median": quartile_2,
        "q3": quartile_3,
    },
}

summary_path = (
    reports_dir / "model_14_error_analysis_summary.json"
)

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)

print("\nModel 12 test error analysis")
print("---------------------------")
print(f"Test tiles:       {len(tile_results)}")
print(f"Ground truths:    {len(ground_truth_results)}")
print(f"Predictions:      {len(prediction_results_table)}")
print(f"True positives:   {total_true_positives}")
print(f"False positives:  {total_false_positives}")
print(f"False negatives:  {total_false_negatives}")
print(
    f"Precision at conf {confidence_threshold:.2f}: "
    f"{threshold_precision:.4f}"
)
print(
    f"Recall at conf {confidence_threshold:.2f}:    "
    f"{threshold_recall:.4f}"
)

print("\nRecall by ground-truth box size")
print("--------------------------------")
print(size_summary.to_string(index=False))

print("\nSaved summary to:")
print(summary_path)

print("\nSaved annotated examples to:")
print(output_dir)