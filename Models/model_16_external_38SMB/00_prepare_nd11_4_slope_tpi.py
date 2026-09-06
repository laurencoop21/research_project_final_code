from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# File paths
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

# The external 10 m ND11_4 raster is used as the
# reference grid for the new composite.
reference_path = PROJECT_DIR / "38SMB_nd11_4_10m.tif"

nd11_4_path = PROJECT_DIR / "38SMB_nd11_4_10m.tif"
slope_path = PROJECT_DIR / "38SMB_fabdem_slope_30m.tif"
tpi_path = PROJECT_DIR / "38SMB_fabdem_tpi_30m.tif"

output_path = PROJECT_DIR / "38SMB_ND11_4_slope_TPI_uint8.tif"


reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# Copy of the scaling report from the original SNA
# raster used to construct the modelling dataset.
original_scaling_path = (
    reports_dir / "original_SNA_channel_scaling.json"
)

# Report describing the scaling applied to 38SMB.
scaling_report_path = (
    reports_dir / "channel_scaling.json"
)

# Settings
# Keep exactly the same transformation used for the
# original modelling raster:
#
# 1. Standardise using the ORIGINAL SNA mean and SD.
# 2. Clip to +/-3 ORIGINAL SNA standard deviations.
# 3. Map that range to 0-255.
#
# The external AOI does not determine new scaling
# parameters.

clip_standard_deviations = 3.0
neutral_nodata_value = 128

# Check required files
required_paths = [
    reference_path,
    nd11_4_path,
    slope_path,
    tpi_path,
    original_scaling_path,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

print("All required files were found.")

# Load original SNA scaling values
with open(original_scaling_path) as report_file:
    original_scaling = json.load(report_file)


print("\nOriginal SNA scaling report")
print("---------------------------")
print(original_scaling_path)

for channel_name in [
    "ND11_4",
    "Slope",
    "TPI",
]:
    channel = original_scaling["channels"][channel_name]

    mean = float(channel["original_mean"])
    standard_deviation = float(
        channel["original_standard_deviation"]
    )

    lower_bound = (
        mean
        - clip_standard_deviations
        * standard_deviation
    )

    upper_bound = (
        mean
        + clip_standard_deviations
        * standard_deviation
    )

    print(
        f"{channel_name}: "
        f"mean={mean:.6f}, "
        f"SD={standard_deviation:.6f}, "
        f"bounds=({lower_bound:.6f}, "
        f"{upper_bound:.6f})"
    )

# Align one raster to the external reference grid
def align_to_reference(input_path, reference):
    """
    Resample one continuous raster onto the
    external 10 m reference grid.
    """

    aligned = np.full(
        (
            reference.height,
            reference.width,
        ),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(input_path) as source:

        if source.count != 1:
            raise ValueError(
                f"{input_path.name} should have "
                f"one band, but it has "
                f"{source.count}."
            )

        source_data = source.read(1).astype(
            np.float32
        )

        if source.nodata is not None:
            source_data[
                source_data == source.nodata
            ] = np.nan

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

# Apply the ORIGINAL SNA scaling
def standardise_to_uint8(
    data,
    mean,
    standard_deviation,
    clip_sd=3.0,
    nodata_value=128,
):
    """
    Convert one external raster channel to uint8
    using the original SNA scaling parameters.

    z = (value - SNA mean) / SNA standard deviation

    Values are clipped to +/-3 SNA standard
    deviations and then mapped to 0-255.
    """

    valid_mask = np.isfinite(data)

    valid_pixels = int(
        valid_mask.sum()
    )

    missing_pixels = int(
        data.size - valid_pixels
    )

    if valid_pixels == 0:
        raise ValueError(
            "The aligned raster contains "
            "no valid values."
        )

    if (
        not np.isfinite(standard_deviation)
        or standard_deviation == 0
    ):
        raise ValueError(
            "The SNA standard deviation is "
            "invalid or zero."
        )

    # Record the external distribution for auditing.
    external_mean = float(
        np.nanmean(data)
    )

    external_standard_deviation = float(
        np.nanstd(data)
    )

    external_minimum = float(
        np.nanmin(data)
    )

    external_maximum = float(
        np.nanmax(data)
    )


    # Physical clipping bounds represented by
    # +/-3 SD in the ORIGINAL SNA raster.
    lower_clip_value = (
        mean
        - clip_sd
        * standard_deviation
    )

    upper_clip_value = (
        mean
        + clip_sd
        * standard_deviation
    )


    # Standardise using ORIGINAL SNA statistics.
    data -= mean
    data /= standard_deviation


    z_minimum_before_clipping = float(
        np.nanmin(data)
    )

    z_maximum_before_clipping = float(
        np.nanmax(data)
    )


    # Apply the same +/-3 SD limits used when
    # preparing the original modelling raster.
    np.clip(
        data,
        -clip_sd,
        clip_sd,
        out=data,
    )


    # Map -3 ... +3 onto 0 ... 255.
    data += clip_sd

    data *= (
        255.0
        / (2.0 * clip_sd)
    )


    # Missing pixels receive the same neutral
    # midpoint used in the original raster.
    data[~valid_mask] = float(
        nodata_value
    )


    scaled = np.rint(
        data
    ).astype(np.uint8)


    statistics = {
        "sna_mean_used": mean,
        "sna_standard_deviation_used":
            standard_deviation,
        "sna_lower_clip_value":
            lower_clip_value,
        "sna_upper_clip_value":
            upper_clip_value,
        "external_mean_observed":
            external_mean,
        "external_standard_deviation_observed":
            external_standard_deviation,
        "external_minimum":
            external_minimum,
        "external_maximum":
            external_maximum,
        "clip_standard_deviations":
            clip_sd,
        "z_minimum_before_clipping":
            z_minimum_before_clipping,
        "z_maximum_before_clipping":
            z_maximum_before_clipping,
        "mean_value_maps_to":
            127.5,
        "missing_value_maps_to":
            nodata_value,
        "valid_pixels":
            valid_pixels,
        "missing_pixels":
            missing_pixels,
        "scaled_minimum":
            int(scaled.min()),
        "scaled_maximum":
            int(scaled.max()),
        "scaled_mean":
            float(
                scaled[valid_mask].mean()
            ),
        "scaled_standard_deviation":
            float(
                scaled[valid_mask].std()
            ),
    }

    return scaled, statistics

# Prepare the three-channel raster
input_channels = [
    ("ND11_4", nd11_4_path),
    ("Slope", slope_path),
    ("TPI", tpi_path),
]


scaling_report = {
    "reference_raster":
        str(reference_path),

    "output_raster":
        str(output_path),

    "scaling_source":
        str(original_scaling_path),

    "channel_order": {
        "band_1": "ND11_4",
        "band_2": "Slope",
        "band_3": "TPI",
    },

    "scaling_method": (
        "External channels standardised using "
        "the original SNA per-channel mean and "
        "standard deviation, clipped to +/-3 "
        "original SNA standard deviations, then "
        "linearly scaled to 0-255"
    ),

    "mean_value_maps_to":
        127.5,

    "channels": {},
}


with rasterio.open(
    reference_path
) as reference:

    output_profile = (
        reference.profile.copy()
    )

    output_profile.update(
        count=3,
        dtype="uint8",
        nodata=None,
        compress="lzw",
    )


    print("\nReference grid")
    print("--------------")
    print("CRS:", reference.crs)
    print(
        "Shape:",
        (
            reference.height,
            reference.width,
        ),
    )
    print(
        "Resolution:",
        reference.res,
    )
    print(
        "Bounds:",
        reference.bounds,
    )


    with rasterio.open(
        output_path,
        "w",
        **output_profile,
    ) as output:

        for (
            band_number,
            (
                channel_name,
                input_path,
            ),
        ) in enumerate(
            input_channels,
            start=1,
        ):

            print(
                f"\nPreparing "
                f"{channel_name}..."
            )


            aligned = align_to_reference(
                input_path,
                reference,
            )


            sna_channel = (
                original_scaling[
                    "channels"
                ][channel_name]
            )

            sna_mean = float(
                sna_channel[
                    "original_mean"
                ]
            )

            sna_standard_deviation = float(
                sna_channel[
                    "original_standard_deviation"
                ]
            )


            scaled, statistics = (
                standardise_to_uint8(
                    aligned,
                    mean=sna_mean,
                    standard_deviation=
                        sna_standard_deviation,
                    clip_sd=
                        clip_standard_deviations,
                    nodata_value=
                        neutral_nodata_value,
                )
            )


            output.write(
                scaled,
                band_number,
            )

            output.set_band_description(
                band_number,
                channel_name,
            )


            scaling_report[
                "channels"
            ][channel_name] = {
                "source_path":
                    str(input_path),
                **statistics,
            }


            print(
                f"Written as output "
                f"band {band_number}."
            )

            print(
                "SNA mean used:",
                f"{sna_mean:.6f}",
            )

            print(
                "SNA standard deviation used:",
                f"{sna_standard_deviation:.6f}",
            )

            print(
                "SNA physical clip bounds:",
                f"{statistics['sna_lower_clip_value']:.6f}",
                "to",
                f"{statistics['sna_upper_clip_value']:.6f}",
            )

            print(
                "Observed external mean:",
                f"{statistics['external_mean_observed']:.6f}",
            )

            print(
                "Observed external SD:",
                f"{statistics['external_standard_deviation_observed']:.6f}",
            )

            print(
                "Scaled valid-pixel mean and SD:",
                f"{statistics['scaled_mean']:.2f}",
                f"{statistics['scaled_standard_deviation']:.2f}",
            )

# Save scaling report
with open(
    scaling_report_path,
    "w",
) as report_file:

    json.dump(
        scaling_report,
        report_file,
        indent=2,
    )

# Final audit
with rasterio.open(
    output_path
) as output:

    print(
        "\nCompleted composite"
    )

    print(
        "-------------------"
    )

    print(
        "Path:",
        output_path,
    )

    print(
        "CRS:",
        output.crs,
    )

    print(
        "Bands:",
        output.count,
    )

    print(
        "Band descriptions:",
        output.descriptions,
    )

    print(
        "Shape:",
        (
            output.height,
            output.width,
        ),
    )

    print(
        "Resolution:",
        output.res,
    )

    print(
        "Data types:",
        output.dtypes,
    )

    print(
        "Bounds:",
        output.bounds,
    )


print(
    "\nSaved scaling report to:"
)

print(
    scaling_report_path
)

print(
    "\n38SMB ND11_4-Slope-TPI "
    "raster preparation is complete."
)
