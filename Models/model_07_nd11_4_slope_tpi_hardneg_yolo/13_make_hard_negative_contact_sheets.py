from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
import math

# File paths
MODEL_DIR = Path(__file__).resolve().parent

review_csv = MODEL_DIR / "reports" / "hard_negative_review.csv"
output_dir = MODEL_DIR / "hard_negative_contact_sheets"

output_dir.mkdir(exist_ok=True)

df = pd.read_csv(review_csv)

# Contact sheet layout
per_sheet = 10
columns = 2
rows = 5

thumb_width = 512
thumb_height = 512
label_height = 45

# Build one contact sheet per batch of tiles
for sheet_number in range(math.ceil(len(df) / per_sheet)):

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

    for position, (_, row) in enumerate(subset.iterrows()):

        image = Image.open(row["review_image"]).convert("RGB")
        image = image.resize((thumb_width, thumb_height))

        col = position % columns
        row_num = position // columns

        x = col * thumb_width
        y = row_num * (thumb_height + label_height)

        sheet.paste(image, (x, y))

        label = (
            f"Rank {int(row['rank'])} | "
            f"conf={row['max_confidence']:.3f} | "
            f"{row['tile_name']}"
        )

        draw.text(
            (x + 5, y + thumb_height + 5),
            label,
            fill="black",
        )

    output_path = (
        output_dir
        / f"hard_negatives_{sheet_number + 1}.jpg"
    )

    sheet.save(
        output_path,
        quality=90,
    )

    print("Saved:", output_path)

print("\nFinished.")
