from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO


# File paths
MODEL_DIR = Path(__file__).resolve().parent

selection_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_pre_qc.csv"
)

qc_csv = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_qc.csv"
)

qc_dir = MODEL_DIR / "selected_hard_negative_qc"
qc_dir.mkdir(exist_ok=True)

model6_weights = (
    MODEL_DIR.parent
    / "model_06_nd11_4_slope_tpi_yolo"
    / "checkpoints"
    / "model_06_best.pt"
)

df = pd.read_csv(selection_path)

# Load Model 6 to draw its predictions
model = YOLO(model6_weights)

# Render each selected tile for manual review
review_images = []

# Render Model 6 predictions for the selected 30
for _, row in df.iterrows():

    result = model.predict(
        source=row["image_path"],
        imgsz=512,
        conf=0.05,
        verbose=False,
    )[0]

    plotted = result.plot()
    plotted = plotted[:, :, ::-1]

    rank = int(row["selection_rank"])

    output_path = (
        qc_dir
        / (
            f"{rank:02d}_"
            f"conf_{row['max_confidence']:.3f}_"
            f"{row['tile_name']}.png"
        )
    )

    Image.fromarray(plotted).save(output_path)

    review_images.append(str(output_path))


df["qc_image"] = review_images
df["qc_decision"] = "pending"
df["qc_reason"] = ""

df.to_csv(qc_csv, index=False)

# Make contact sheets: 10 candidates per sheet
per_sheet = 10
columns = 2
rows = 5

thumb_width = 512
thumb_height = 512
label_height = 45

for sheet_number in range(
    math.ceil(len(df) / per_sheet)
):

    subset = df.iloc[
        sheet_number * per_sheet:
        (sheet_number + 1) * per_sheet
    ]

    sheet = Image.new(
        "RGB",
        (
            columns * thumb_width,
            rows * (thumb_height + label_height),
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for position, (_, row) in enumerate(
        subset.iterrows()
    ):

        image = Image.open(
            row["qc_image"]
        ).convert("RGB")

        col = position % columns
        row_num = position // columns

        x = col * thumb_width
        y = row_num * (thumb_height + label_height)

        sheet.paste(image, (x, y))

        label = (
            f"Selected {int(row['selection_rank'])} | "
            f"conf={row['max_confidence']:.3f} | "
            f"{row['tile_name']}"
        )

        draw.text(
            (x + 5, y + thumb_height + 5),
            label,
            fill="black",
        )

    output_path = (
        qc_dir
        / f"selected_qc_sheet_{sheet_number + 1}.jpg"
    )

    sheet.save(output_path, quality=90)

    print("Saved:", output_path)


print("\nQC manifest:")
print(qc_csv)

print("\nSelected candidates:", len(df))
