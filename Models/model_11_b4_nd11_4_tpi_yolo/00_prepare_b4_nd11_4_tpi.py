from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# File paths
# Save this script inside:
# iraq_tells_gis/Models/model_11_b4_nd11_4_tpi_yolo/

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

# The Model 1 composite is used only as the reference grid.
reference_path = PROJECT_DIR / "Iraq_11_8_4.tif"

nd11_4_path = PROJECT_DIR / "iraq_nd_11_4_10m.tif"
b4_path = PROJECT_DIR / "iraq_b4_10m.tif"
tpi_path = PROJECT_DIR / "iraq_fabdem_tpi_30m.tif"

output_path = PROJECT_DIR / "Iraq_B4_ND11_4_TPI_uint8.tif"

reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

scaling_report_path = reports_dir / "channel_scaling.json"

# Settings
# WD's suggested approach:
# 1. Standardise each channel using its own mean and standard deviation.
# 2. Clip extreme values to -3 and +3 standard deviations.
# 3. Map that range to 0-255.
#
# A value equal to the channel mean is therefore mapped to 127.5,
# which is approximately the midpoint of an 8-bit image.

clip_standard_deviations = 3.0
neutral_nodata_value = 128

# Check that the required files exist
required_paths = [
    reference_path,
    b4_path,
    nd11_4_path,
    tpi_path,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

print("All required raster files were found.")

# Align one raster to the Model 1 grid
def align_to_reference(input_path, reference):
    """Resample one continuous raster onto the reference raster grid."""

    aligned = np.full(
        (reference.height, reference.width),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(input_path) as source:
        if source.count != 1:
            raise ValueError(
                f"{input_path.name} should have one band, "
                f"but it has {source.count}."
            )

        source_data = source.read(1).astype(np.float32)

        if source.nodata is not None:
            source_data[source_data == source.nodata] = np.nan

        reproject(
            source=source_data,
            destination=aligned,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            num_threads=2,
        )

    return aligned

# Standardise and scale one aligned channel
def standardise_to_uint8(data, clip_sd=3.0, nodata_value=128):
    """
    Standardise one raster channel and convert it to unsigned 8-bit.

    z = (value - mean) / standard_deviation

    Values are clipped to +/- clip_sd standard deviations and then
    mapped to 0-255. A value equal to the channel mean maps to 127.5.
    """

    valid_mask = np.isfinite(data)
    valid_pixels = int(valid_mask.sum())
    missing_pixels = int(data.size - valid_pixels)

    if valid_pixels == 0:
        raise ValueError("The aligned raster contains no valid values.")

    mean = float(np.nanmean(data))
    standard_deviation = float(np.nanstd(data))

    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise ValueError(
            "The aligned raster has an invalid or zero standard deviation."
        )

    original_minimum = float(np.nanmin(data))
    original_maximum = float(np.nanmax(data))

    data -= mean
    data /= standard_deviation

    z_minimum_before_clipping = float(np.nanmin(data))
    z_maximum_before_clipping = float(np.nanmax(data))

    np.clip(data, -clip_sd, clip_sd, out=data)

    data += clip_sd
    data *= 255.0 / (2.0 * clip_sd)

    # Missing pixels receive the neutral midpoint rather than black.
    data[~valid_mask] = float(nodata_value)

    scaled = np.rint(data).astype(np.uint8)

    statistics = {
        "original_mean": mean,
        "original_standard_deviation": standard_deviation,
        "original_minimum": original_minimum,
        "original_maximum": original_maximum,
        "clip_standard_deviations": clip_sd,
        "z_minimum_before_clipping": z_minimum_before_clipping,
        "z_maximum_before_clipping": z_maximum_before_clipping,
        "mean_value_maps_to": 127.5,
        "missing_value_maps_to": nodata_value,
        "valid_pixels": valid_pixels,
        "missing_pixels": missing_pixels,
        "scaled_minimum": int(scaled.min()),
        "scaled_maximum": int(scaled.max()),
        "scaled_mean": float(scaled[valid_mask].mean()),
        "scaled_standard_deviation": float(scaled[valid_mask].std()),
    }

    return scaled, statistics

# Prepare and write the three-channel raster
input_channels = [
    ("B4", b4_path),
    ("ND11_4", nd11_4_path),
    ("TPI", tpi_path),
]

scaling_report = {
    "reference_raster": str(reference_path),
    "output_raster": str(output_path),
    "channel_order": {
        "band_1": "B4",
        "band_2": "ND11_4",
        "band_3": "TPI",
    },
    "scaling_method": (
        "Per-channel z-score standardisation, clipping to +/-3 standard "
        "deviations, then linear scaling to 0-255"
    ),
    "mean_value_maps_to": 127.5,
    "channels": {},
}

with rasterio.open(reference_path) as reference:
    output_profile = reference.profile.copy()
    output_profile.update(
        count=3,
        dtype="uint8",
        nodata=None,
        compress="lzw",
    )

    print("\nReference grid")
    print("--------------")
    print("CRS:", reference.crs)
    print("Shape:", (reference.height, reference.width))
    print("Resolution:", reference.res)
    print("Bounds:", reference.bounds)

    with rasterio.open(output_path, "w", **output_profile) as output:
        for band_number, (channel_name, input_path) in enumerate(
            input_channels,
            start=1,
        ):
            print(f"\nPreparing {channel_name}...")

            aligned = align_to_reference(input_path, reference)
            scaled, statistics = standardise_to_uint8(
                aligned,
                clip_sd=clip_standard_deviations,
                nodata_value=neutral_nodata_value,
            )

            output.write(scaled, band_number)
            output.set_band_description(band_number, channel_name)

            scaling_report["channels"][channel_name] = {
                "source_path": str(input_path),
                **statistics,
            }

            print(f"Written as output band {band_number}.")
            print(f"Original mean: {statistics['original_mean']:.6f}")
            print(
                "Original standard deviation:",
                f"{statistics['original_standard_deviation']:.6f}",
            )
            print(
                "Scaled valid-pixel mean and standard deviation:",
                f"{statistics['scaled_mean']:.2f}",
                f"{statistics['scaled_standard_deviation']:.2f}",
            )

# Save the scaling report
with open(scaling_report_path, "w") as report_file:
    json.dump(scaling_report, report_file, indent=2)

# Final audit
with rasterio.open(output_path) as output:
    print("\nCompleted composite")
    print("-------------------")
    print("Path:", output_path)
    print("CRS:", output.crs)
    print("Bands:", output.count)
    print("Band descriptions:", output.descriptions)
    print("Shape:", (output.height, output.width))
    print("Resolution:", output.res)
    print("Data types:", output.dtypes)
    print("Bounds:", output.bounds)

print("\nSaved scaling report to:")
print(scaling_report_path)

print("\nB4-ND11_4-TPI raster preparation is complete.")