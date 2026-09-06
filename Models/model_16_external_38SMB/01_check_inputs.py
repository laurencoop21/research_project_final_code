from pathlib import Path
import json

import geopandas as gpd
import rasterio
from shapely.geometry import box

# File paths
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

raster_path = (
    PROJECT_DIR
    / "38SMB_ND11_4_slope_TPI_uint8.tif"
)

aoi_path = (
    PROJECT_DIR
    / "38SMB_external_test_AOI.shp"
)

reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(
    parents=True,
    exist_ok=True,
)

report_path = (
    reports_dir
    / "input_audit.json"
)

# Check required files
required_files = [
    raster_path,
    aoi_path,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

print(
    "All required files were found."
)

# Read raster
with rasterio.open(
    raster_path
) as raster:

    raster_crs = raster.crs
    raster_bounds = raster.bounds

    raster_shape = (
        raster.height,
        raster.width,
    )

    raster_bands = raster.count
    raster_pixel_size = raster.res
    raster_dtypes = list(raster.dtypes)

    raster_descriptions = list(
        raster.descriptions
    )


print(
    "\nRaster information"
)

print(
    "------------------"
)

print(
    "CRS:",
    raster_crs,
)

print(
    "Bands:",
    raster_bands,
)

print(
    "Band descriptions:",
    raster_descriptions,
)

print(
    "Shape:",
    raster_shape,
)

print(
    "Pixel size:",
    raster_pixel_size,
)

print(
    "Data types:",
    raster_dtypes,
)

print(
    "Bounds:",
    raster_bounds,
)

# Read external AOI
aoi = gpd.read_file(
    aoi_path
)

if aoi.crs is None:
    raise ValueError(
        "The 38SMB AOI has no CRS. "
        "Do not assign one manually "
        "until its source CRS is verified."
    )

original_aoi_crs = aoi.crs

if aoi.crs != raster_crs:
    aoi = aoi.to_crs(
        raster_crs
    )


print(
    "\nAOI information"
)

print(
    "---------------"
)

print(
    "AOI features:",
    len(aoi),
)

print(
    "Original AOI CRS:",
    original_aoi_crs,
)

print(
    "AOI CRS used for audit:",
    aoi.crs,
)

print(
    "AOI bounds:",
    aoi.total_bounds,
)

# AOI geometry checks
invalid_aoi = int(
    (~aoi.geometry.is_valid).sum()
)

empty_aoi = int(
    aoi.geometry.is_empty.sum()
)

missing_aoi = int(
    aoi.geometry.isna().sum()
)


# Raster footprint as a polygon.
raster_polygon = box(
    *raster_bounds
)

# Merge AOI features if necessary.
aoi_geometry = (
    aoi.geometry.union_all()
)

raster_area = float(
    raster_polygon.area
)

aoi_area = float(
    aoi_geometry.area
)

intersection_area = float(
    aoi_geometry
    .intersection(
        raster_polygon
    )
    .area
)

# Fraction of raster covered by AOI.
raster_overlap_fraction = (
    intersection_area
    / raster_area
)

# Fraction of AOI covered by raster.
aoi_overlap_fraction = (
    intersection_area
    / aoi_area
)


print(
    "\nSpatial checks"
)

print(
    "--------------"
)

print(
    "Invalid AOI geometries:",
    invalid_aoi,
)

print(
    "Empty AOI geometries:",
    empty_aoi,
)

print(
    "Missing AOI geometries:",
    missing_aoi,
)

print(
    "Raster area (km²):",
    f"{raster_area / 1_000_000:.2f}",
)

print(
    "AOI area (km²):",
    f"{aoi_area / 1_000_000:.2f}",
)

print(
    "Raster covered by AOI:",
    f"{raster_overlap_fraction * 100:.4f}%",
)

print(
    "AOI covered by raster:",
    f"{aoi_overlap_fraction * 100:.4f}%",
)

# Final checks
expected_descriptions = [
    "ND11_4",
    "Slope",
    "TPI",
]

checks = {
    "raster_has_3_bands":
        raster_bands == 3,

    "raster_is_epsg_32638":
        (
            raster_crs is not None
            and raster_crs.to_epsg()
            == 32638
        ),

    "raster_has_10m_pixels":
        (
            abs(
                raster_pixel_size[0]
            ) == 10
            and abs(
                raster_pixel_size[1]
            ) == 10
        ),

    "raster_is_10000_by_10000":
        raster_shape
        == (10000, 10000),

    "all_bands_are_uint8":
        all(
            dtype == "uint8"
            for dtype
            in raster_dtypes
        ),

    "band_order_is_correct":
        raster_descriptions
        == expected_descriptions,

    "no_invalid_aoi_geometries":
        invalid_aoi == 0,

    "no_empty_aoi_geometries":
        empty_aoi == 0,

    "no_missing_aoi_geometries":
        missing_aoi == 0,

    # Allow a tiny tolerance for vector
    # reprojection/export differences.
    "aoi_matches_raster":
        (
            raster_overlap_fraction
            >= 0.999
            and aoi_overlap_fraction
            >= 0.999
        ),
}

audit_passed = bool(
    all(
        checks.values()
    )
)


print(
    "\nFinal checks"
)

print(
    "------------"
)

for name, passed in checks.items():
    print(
        f"{name}: {passed}"
    )

print(
    "\nAUDIT PASSED:",
    audit_passed,
)

# Save report
report = {
    "deployment":
        "Model 16 external 38SMB",

    "raster": {
        "path":
            str(raster_path),

        "crs":
            str(raster_crs),

        "bands":
            raster_bands,

        "band_descriptions":
            raster_descriptions,

        "shape":
            raster_shape,

        "pixel_size":
            raster_pixel_size,

        "data_types":
            raster_dtypes,

        "bounds":
            list(raster_bounds),

        "area_km2":
            raster_area
            / 1_000_000,
    },

    "external_aoi": {
        "path":
            str(aoi_path),

        "original_crs":
            str(original_aoi_crs),

        "audit_crs":
            str(aoi.crs),

        "feature_count":
            len(aoi),

        "area_km2":
            aoi_area
            / 1_000_000,

        "invalid_geometries":
            invalid_aoi,

        "empty_geometries":
            empty_aoi,

        "missing_geometries":
            missing_aoi,

        "raster_overlap_fraction":
            raster_overlap_fraction,

        "aoi_overlap_fraction":
            aoi_overlap_fraction,
    },

    "checks":
        checks,

    "audit_passed":
        audit_passed,
}


with open(
    report_path,
    "w",
) as file:

    json.dump(
        report,
        file,
        indent=4,
    )


print(
    "\nSaved report to:"
)

print(
    report_path
)


