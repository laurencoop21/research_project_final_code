from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MODELS_DIR.parent

RASTER_PATH = (
    PROJECT_DIR
    / "Iraq_ND11_4_slope_TPI_uint8.tif"
)

SPLIT_PATH = (
    MODELS_DIR
    / "model_12_nd11_4_slope_tpi_newdata_yolo"
    / "reports"
    / "geographic_split.gpkg"
)

REPORTS_DIR = SCRIPT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BAND_NAMES = {
    1: "ND11_4",
    2: "Slope",
    3: "TPI",
}


# Reproduce the actual geographic split boundaries
tells = gpd.read_file(
    SPLIT_PATH,
    layer="tells_with_split",
)

train_boundary = float(
    tells["centroid_x"].quantile(0.70)
)

val_boundary = float(
    tells["centroid_x"].quantile(0.85)
)

# Helper to summarize a raster band without loading
# the whole geographic region into memory
def summarize_band(
    src,
    band_number,
    col_start,
    col_stop,
    chunk_rows=512,
):
    n = 0
    total = 0.0
    total_squared = 0.0

    for row_start in range(
        0,
        src.height,
        chunk_rows,
    ):
        height = min(
            chunk_rows,
            src.height - row_start,
        )

        window = Window(
            col_start,
            row_start,
            col_stop - col_start,
            height,
        )

        values = src.read(
            band_number,
            window=window,
        ).ravel()

        if src.nodata is not None:
            values = values[
                values != src.nodata
            ]

        values = values[
            np.isfinite(values)
        ]

        n_chunk = len(values)

        if n_chunk == 0:
            continue

        values_float = values.astype(
            np.float64,
            copy=False,
        )

        n += n_chunk

        total += values_float.sum(
            dtype=np.float64
        )

        total_squared += np.square(
            values_float
        ).sum(dtype=np.float64)

    mean = total / n

    variance = (
        total_squared
        - n * mean ** 2
    ) / (n - 1)

    variance = max(
        variance,
        0.0,
    )

    sd = np.sqrt(variance)

    return {
        "n_pixels": int(n),
        "mean": float(mean),
        "sd": float(sd),
    }


# Determine raster columns belonging to the exact
# train and test geographic regions
records = []

with rasterio.open(RASTER_PATH) as src:
    x_resolution = src.res[0]

    x_centres = (
        src.bounds.left
        + (
            np.arange(src.width)
            + 0.5
        )
        * x_resolution
    )

    train_columns = np.where(
        x_centres < train_boundary
    )[0]

    test_columns = np.where(
        x_centres >= val_boundary
    )[0]

    train_col_start = int(
        train_columns.min()
    )

    train_col_stop = int(
        train_columns.max() + 1
    )

    test_col_start = int(
        test_columns.min()
    )

    test_col_stop = int(
        test_columns.max() + 1
    )

    for band_number, band_name in BAND_NAMES.items():
        train_stats = summarize_band(
            src,
            band_number,
            train_col_start,
            train_col_stop,
        )

        test_stats = summarize_band(
            src,
            band_number,
            test_col_start,
            test_col_stop,
        )

        mean_difference = (
            test_stats["mean"]
            - train_stats["mean"]
        )

        pooled_sd = np.sqrt(
            (
                train_stats["sd"] ** 2
                + test_stats["sd"] ** 2
            )
            / 2
        )

        standardized_difference = (
            mean_difference / pooled_sd
            if pooled_sd > 0
            else np.nan
        )

        records.append(
            {
                "band": band_name,
                "train_n_pixels":
                    train_stats["n_pixels"],
                "test_n_pixels":
                    test_stats["n_pixels"],
                "train_mean":
                    train_stats["mean"],
                "test_mean":
                    test_stats["mean"],
                "train_sd":
                    train_stats["sd"],
                "test_sd":
                    test_stats["sd"],
                "mean_difference_test_minus_train":
                    mean_difference,
                "standardized_mean_difference":
                    standardized_difference,
            }
        )

    metadata = {
        "train_boundary_x":
            train_boundary,
        "validation_test_boundary_x":
            val_boundary,
        "raster_left":
            float(src.bounds.left),
        "raster_right":
            float(src.bounds.right),
        "raster_bottom":
            float(src.bounds.bottom),
        "raster_top":
            float(src.bounds.top),
        "pixel_size_x":
            float(src.res[0]),
        "pixel_size_y":
            float(src.res[1]),
        "train_first_column":
            train_col_start,
        "train_last_column":
            train_col_stop - 1,
        "test_first_column":
            test_col_start,
        "test_last_column":
            test_col_stop - 1,
        "pixel_assignment":
            "Pixel centre x-coordinate",
    }


# Save results
results = pd.DataFrame(records)

csv_path = (
    REPORTS_DIR
    / "train_test_band_distribution_comparison.csv"
)

json_path = (
    REPORTS_DIR
    / "train_test_band_distribution_comparison.json"
)

metadata_path = (
    REPORTS_DIR
    / "train_test_band_distribution_metadata.json"
)

results.to_csv(
    csv_path,
    index=False,
)

with open(json_path, "w") as f:
    json.dump(
        results.replace(
            {np.nan: None}
        ).to_dict(
            orient="records"
        ),
        f,
        indent=2,
    )

with open(metadata_path, "w") as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


# Print results
print("\nExact geographic split boundaries")
print("---------------------------------")
print(
    f"Train/validation boundary: "
    f"{train_boundary:.3f}"
)
print(
    f"Validation/test boundary: "
    f"{val_boundary:.3f}"
)

print(
    "\nTrain/test input-band distribution comparison"
)
print(
    "---------------------------------------------"
)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nSaved reports to:")
print(REPORTS_DIR)
