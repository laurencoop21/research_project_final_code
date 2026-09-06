from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO


# File paths
MODEL_DIR = Path(__file__).resolve().parent

final_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_final.csv"
)

output_dir = (
    MODEL_DIR
    / "replacement_hard_negative_qc"
)

model6_weights = (
    MODEL_DIR.parent
    / "model_06_nd11_4_slope_tpi_yolo"
    / "checkpoints"
    / "model_06_best.pt"
)

output_dir.mkdir(exist_ok=True)

df = pd.read_csv(final_path)

replacements = df[
    df["qc_decision"] == "replacement_pending_qc"
].copy()

# Load Model 6 to draw its predictions
model = YOLO(model6_weights)

# Render each replacement tile for manual review
rendered = []

for number, (_, row) in enumerate(
    replacements.iterrows(),
    start=1,
):

    result = model.predict(
        source=row["image_path"],
        imgsz=512,
        conf=0.05,
        verbose=False,
    )[0]

    image = result.plot()
    image = image[:, :, ::-1]

    path = (
        output_dir
        / f"{number:02d}_{row['tile_name']}.png"
    )

    Image.fromarray(image).save(path)

    rendered.append({
        "number": number,
        "tile_name": row["tile_name"],
        "max_confidence": row["max_confidence"],
        "image": str(path),
    })

# Make one contact sheet
columns = 2
rows = 3
width = 512
height = 512
label_height = 45

sheet = Image.new(
    "RGB",
    (
        columns * width,
        rows * (height + label_height),
    ),
    "white",
)

draw = ImageDraw.Draw(sheet)

for position, item in enumerate(rendered):

    image = Image.open(item["image"]).convert("RGB")

    col = position % columns
    row = position // columns

    x = col * width
    y = row * (height + label_height)

    sheet.paste(image, (x, y))

    label = (
        f"Replacement {item['number']} | "
        f"conf={item['max_confidence']:.3f} | "
        f"{item['tile_name']}"
    )

    draw.text(
        (x + 5, y + height + 5),
        label,
        fill="black",
    )


sheet_path = (
    output_dir
    / "replacement_qc_sheet.jpg"
)

sheet.save(sheet_path, quality=90)

print("\nReplacement QC")
print("--------------")
print("Candidates:", len(replacements))

for item in rendered:
    print(
        item["number"],
        item["tile_name"],
        f"conf={item['max_confidence']:.3f}",
    )

print("\nSaved contact sheet:")
print(sheet_path)
