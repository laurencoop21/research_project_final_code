from pathlib import Path
import shutil

import pandas as pd
from PIL import Image
from ultralytics import YOLO


MODEL_DIR = Path(__file__).resolve().parent

scores_path = (
    MODEL_DIR
    / "reports"
    / "model6_mining_pool_scores.csv"
)

model6_weights = (
    MODEL_DIR.parent
    / "model_06_nd11_4_slope_tpi_yolo"
    / "checkpoints"
    / "model_06_best.pt"
)

review_dir = MODEL_DIR / "hard_negative_review"

top_n = 50

# Prepare review folder
if review_dir.exists():
    shutil.rmtree(review_dir)

review_dir.mkdir(parents=True)

# Load scores and Model 6
scores = pd.read_csv(scores_path)

scores = scores.sort_values(
    "max_confidence",
    ascending=False,
).head(top_n)

model = YOLO(model6_weights)

# Render predictions
records = []

for rank, (_, row) in enumerate(
    scores.iterrows(),
    start=1,
):

    result = model.predict(
        source=row["image_path"],
        imgsz=512,
        conf=0.05,
        verbose=False,
    )[0]

    plotted = result.plot()

    # result.plot() returns BGR
    plotted = plotted[:, :, ::-1]

    output_name = (
        f"{rank:02d}_"
        f"conf_{row['max_confidence']:.3f}_"
        f"{row['tile_name']}.png"
    )

    output_path = review_dir / output_name

    Image.fromarray(plotted).save(output_path)

    records.append({
        "rank": rank,
        "tile_name": row["tile_name"],
        "max_confidence": row["max_confidence"],
        "row_start": row["row_start"],
        "col_start": row["col_start"],
        "review_image": str(output_path),
    })

    print(
        f"{rank:02d}/{top_n} "
        f"{row['tile_name']} "
        f"conf={row['max_confidence']:.3f}"
    )


review = pd.DataFrame(records)

review.to_csv(
    MODEL_DIR / "reports" / "hard_negative_review.csv",
    index=False,
)

print("\nCreated hard-negative review set:")
print(review_dir)

print("\nCandidates:", len(review))
