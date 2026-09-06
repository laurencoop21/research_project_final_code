from pathlib import Path

import pandas as pd
from ultralytics import YOLO


# File paths
MODEL_DIR = Path(__file__).resolve().parent

background_csv = MODEL_DIR / "reports" / "clean_training_backgrounds.csv"

model6_weights = (
    MODEL_DIR.parent
    / "model_06_nd11_4_slope_tpi_yolo"
    / "checkpoints"
    / "model_06_best.pt"
)

output_csv = MODEL_DIR / "reports" / "model6_background_scores.csv"


print("Loading Model 6:")
print(model6_weights)

# Load Model 6 to score the background tiles
model = YOLO(model6_weights)

backgrounds = pd.read_csv(background_csv)

records = []

# Score how strongly Model 6 predicts a tell in each tile
for i, row in backgrounds.iterrows():

    image_path = MODEL_DIR / row["image_path"]

    result = model.predict(
        source=str(image_path),
        imgsz=512,
        conf=0.05,
        verbose=False,
    )[0]

    if result.boxes is not None and len(result.boxes) > 0:
        confidences = result.boxes.conf.cpu().numpy()

        n_predictions = len(confidences)
        max_confidence = float(confidences.max())

        n_above_025 = int((confidences >= 0.25).sum())
        n_above_050 = int((confidences >= 0.50).sum())

    else:
        n_predictions = 0
        max_confidence = 0.0
        n_above_025 = 0
        n_above_050 = 0

    records.append({
        "tile_name": row["tile_name"],
        "row_start": row["row_start"],
        "col_start": row["col_start"],
        "image_path": row["image_path"],
        "n_predictions_conf_005": n_predictions,
        "n_predictions_conf_025": n_above_025,
        "n_predictions_conf_050": n_above_050,
        "max_confidence": max_confidence,
    })

    print(
        f"{i + 1:02d}/{len(backgrounds)} "
        f"{row['tile_name']} "
        f"max_conf={max_confidence:.3f}"
    )


scores = pd.DataFrame(records)

# Rank by strongest false signal
scores = scores.sort_values(
    by=["max_confidence", "n_predictions_conf_005"],
    ascending=False,
)

# Save the results
scores.to_csv(output_csv, index=False)


print("\nModel 6 background scoring")
print("--------------------------")
print("Background tiles:", len(scores))
print(
    "Tiles with prediction >= 0.05:",
    (scores["max_confidence"] >= 0.05).sum(),
)
print(
    "Tiles with prediction >= 0.25:",
    (scores["max_confidence"] >= 0.25).sum(),
)
print(
    "Tiles with prediction >= 0.50:",
    (scores["max_confidence"] >= 0.50).sum(),
)

print("\nTop 15 hardest backgrounds:")
print(
    scores[
        [
            "tile_name",
            "max_confidence",
            "n_predictions_conf_025",
            "n_predictions_conf_050",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nSaved:")
print(output_csv)
