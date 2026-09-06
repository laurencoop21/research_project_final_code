from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import Window, bounds
from shapely.geometry import box


# File paths
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

raster_path = PROJECT_DIR / "Iraq_ND11_4_slope_TPI_uint8.tif"
split_path = MODEL_DIR / "reports" / "geographic_split.gpkg"

# Tiling settings, matching the dataset build
tile_size = 512
overlap = 128
stride = tile_size - overlap
minimum_visible_fraction = 0.50


split_regions = gpd.read_file(split_path, layer="split_regions")
tells = gpd.read_file(split_path, layer="tells_with_split")

usable_tells = tells[tells["split"].isin(["train", "val", "test"])].copy()


# Tile start positions along one axis
def make_starts(start, end, size, step):
    starts = list(range(start, end - size + 1, step))

    last_start = end - size

    if last_start >= start and last_start not in starts:
        starts.append(last_start)

    return sorted(starts)


records = []

with rasterio.open(raster_path) as raster:

    if split_regions.crs != raster.crs:
        split_regions = split_regions.to_crs(raster.crs)

    if usable_tells.crs != raster.crs:
        usable_tells = usable_tells.to_crs(raster.crs)

    row_starts = make_starts(
        0,
        raster.height,
        tile_size,
        stride,
    )

    # Count, per tile, how many tells are labelled and how many fall below the threshold
    for split in ["train", "val", "test"]:

        region = split_regions[
            split_regions["split"] == split
        ].geometry.iloc[0]

        region_left, _, region_right, _ = region.bounds

        start_col = max(
            0,
            int((region_left - raster.bounds.left) / raster.res[0]),
        )

        end_col = min(
            raster.width,
            int(
                (region_right - raster.bounds.left) /
                raster.res[0]
            ),
        )

        col_starts = make_starts(
            start_col,
            end_col,
            tile_size,
            stride,
        )

        split_tells = usable_tells[
            usable_tells["split"] == split
        ]

        for row_start in row_starts:
            for col_start in col_starts:

                window = Window(
                    col_start,
                    row_start,
                    tile_size,
                    tile_size,
                )

                tile_bounds = bounds(window, raster.transform)
                tile_polygon = box(*tile_bounds)

                intersecting = split_tells[
                    split_tells.geometry.intersects(tile_polygon)
                ]

                labelled = 0
                partial_unlabelled = 0

                for _, tell in intersecting.iterrows():

                    visible = tell.geometry.intersection(tile_polygon)

                    visible_fraction = (
                        visible.area / tell.geometry.area
                    )

                    if visible_fraction >= minimum_visible_fraction:
                        labelled += 1
                    else:
                        partial_unlabelled += 1

                records.append({
                    "split": split,
                    "row_start": row_start,
                    "col_start": col_start,
                    "intersecting_tells": len(intersecting),
                    "labelled_tells": labelled,
                    "partial_unlabelled_tells": partial_unlabelled,
                    "hidden_positive_tile": (
                        labelled == 0 and partial_unlabelled > 0
                    ),
                })


# Save the results
df = pd.DataFrame(records)

# Compare candidate tiles with tiles actually saved
# by 03_make_yolo_dataset.py
tile_index_path = MODEL_DIR / "reports" / "tile_index.csv"
tile_index = pd.read_csv(tile_index_path)

required_columns = {"split", "row_start", "col_start"}

if not required_columns.issubset(tile_index.columns):
    raise ValueError(
        "tile_index.csv does not contain the expected columns. "
        f"Found: {tile_index.columns.tolist()}"
    )

saved_keys = set(
    zip(
        tile_index["split"].astype(str),
        tile_index["row_start"].astype(int),
        tile_index["col_start"].astype(int),
    )
)

df["saved_to_dataset"] = [
    (str(split), int(row), int(col)) in saved_keys
    for split, row, col in zip(
        df["split"],
        df["row_start"],
        df["col_start"],
    )
]


print("\nCandidate tile grid")
print("-------------------")

for split in ["train", "val", "test"]:

    sub = df[df["split"] == split]

    print(f"\n{split}")
    print("Candidate tiles:", len(sub))
    print(
        "Tiles with labels:",
        (sub["labelled_tells"] > 0).sum()
    )
    print(
        "True background tiles:",
        (
            (sub["labelled_tells"] == 0) &
            (sub["partial_unlabelled_tells"] == 0)
        ).sum()
    )
    print(
        "Hidden-positive candidates:",
        sub["hidden_positive_tile"].sum()
    )


print("\nActual saved dataset")
print("--------------------")

for split in ["train", "val", "test"]:

    sub = df[
        (df["split"] == split) &
        (df["saved_to_dataset"])
    ]

    print(f"\n{split}")
    print("Saved tiles:", len(sub))
    print(
        "Saved positive tiles:",
        (sub["labelled_tells"] > 0).sum()
    )
    print(
        "Saved true backgrounds:",
        (
            (sub["labelled_tells"] == 0) &
            (sub["partial_unlabelled_tells"] == 0)
        ).sum()
    )
    print(
        "Saved hidden-positive tiles:",
        sub["hidden_positive_tile"].sum()
    )


output = MODEL_DIR / "reports" / "partial_tell_tile_audit.csv"
df.to_csv(output, index=False)

print("\nSaved audit:")
print(output)
