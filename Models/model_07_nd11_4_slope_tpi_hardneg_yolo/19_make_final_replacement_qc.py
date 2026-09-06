from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO


# File paths
MODEL_DIR = Path(__file__).resolve().parent

selection_path = (
    MODEL_DIR
    / "reports"
    / "selected_hard_negatives_final_round2.csv"
)

output_dir = (
    MODEL_DIR
    / "final_replacement_qc"
)

model6_weights = (
    MODEL_DIR.parent
    / "model_06_nd11_4_slope_tpi_yolo"
    / "checkpoints"
    / "model_06_best.pt"
)

output_dir.mkdir(exist_ok=True)

df = pd.read_csv(selection_path)

pending = df[
    df["qc_decision"] == "replacement_pending_qc"
].copy()

print("\nPending replacements:", len(pending))

# Load Model 6 to draw its predictions
model = YOLO(model6_weights)

# Render each replacement tile for manual review
rendered = []

for number, (_, row) in enumerate(
    pending.iterrows(),
    start=1,
):

    result = model.predict(
        source=row["image_path"],
        imgsz=512,
        conf=0.05,
        verbose=False,
    )[0]

    plotted = result.plot()
    plotted = plotted[:, :, ::-1]

    image_path = (
        output_dir
        / f"{number:02d}_{row['tile_name']}.png"
    )

    Image.fromarray(plotted).save(image_path)

    rendered.append({
        "number": number,
        "tile_name": row["tile_name"],
        "max_confidence": row["max_confidence"],
        "image_path": image_path,
    })

# Make one horizontal contact sheet
width = 512
height = 512
label_height = 50

sheet = Image.new(
    "RGB",
    (
        len(rendered) * width,
        height + label_height,
    ),
    "white",
)

draw = ImageDraw.Draw(sheet)

for i, item in enumerate(rendered):

    image = Image.open(
        item["image_path"]
    ).convert("RGB")

    x = i * width

    sheet.paste(
        image,
        (x, 0),
    )

    label = (
        f"Replacement {item['number']} | "
        f"conf={item['max_confidence']:.3f} | "
        f"{item['tile_name']}"
    )

    draw.text(
        (x + 5, height + 5),
        label,
        fill="black",
    )


sheet_path = (
    output_dir
    / "final_replacement_qc_sheet.jpg"
)

sheet.save(
    sheet_path,
    quality=90,
)

print("\nCandidates:")
for item in rendered:
    print(
        item["number"],
        item["tile_name"],
        f"conf={item['max_confidence']:.3f}",
    )

print("\nSaved:")
print(sheet_path)
