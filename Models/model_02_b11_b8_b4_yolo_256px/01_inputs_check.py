from pathlib import Path
import json
import math

import geopandas as gpd
import rasterio
from shapely.geometry import box

# File paths
# This script should be saved inside:
# iraq_tells_gis/Models/model_02_b11_b8_b4_yolo_256/

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

raster_path = PROJECT_DIR / "Iraq_11_8_4.tif"
tells_path = PROJECT_DIR / "tells.shp"
aoi_path = PROJECT_DIR / "aoi_sentinel.shp"

reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

report_path = reports_dir / "input_audit.json"

# Check that the required files exist
required_files = [raster_path, tells_path, aoi_path]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

print("All required files were found.")

# Read the raster
with rasterio.open(raster_path) as raster:
    raster_crs = raster.crs
    raster_bounds = raster.bounds
    raster_shape = (raster.height, raster.width)
    raster_bands = raster.count
    raster_pixel_size = raster.res
    raster_dtypes = list(raster.dtypes)

print("\nRaster information")
print("------------------")
print("CRS:", raster_crs)
print("Bands:", raster_bands)
print("Shape:", raster_shape)
print("Pixel size:", raster_pixel_size)
print("Data types:", raster_dtypes)
print("Bounds:", raster_bounds)

# Read the shapefiles
tells = gpd.read_file(tells_path)
aoi = gpd.read_file(aoi_path)

# A missing CRS would prevent a reliable spatial comparison.
if raster_crs is None:
    raise ValueError("The raster does not have a CRS.")

if tells.crs is None:
    raise ValueError("The tell shapefile does not have a CRS.")

if aoi.crs is None:
    raise ValueError("The AOI shapefile does not have a CRS.")

# Reproject the vectors in memory if needed.
if tells.crs != raster_crs:
    tells = tells.to_crs(raster_crs)

if aoi.crs != raster_crs:
    aoi = aoi.to_crs(raster_crs)

print("\nVector information")
print("------------------")
print("Tell polygons:", len(tells))
print("AOI features:", len(aoi))
print("Tell CRS:", tells.crs)
print("AOI CRS:", aoi.crs)

# Check the geometries
missing_tells = tells.geometry.isna().sum()
empty_tells = tells.geometry.is_empty.sum()

# Exclude missing geometries from the invalid-geometry count
# because missing geometries are reported separately.
invalid_tells = (
    (~tells.geometry.is_valid)
    & tells.geometry.notna()
).sum()

# Rectangular polygon covering the full raster extent.
raster_polygon = box(*raster_bounds)

# Count tells that overlap the raster at all.
tells_intersecting_raster = tells.geometry.intersects(
    raster_polygon
).sum()

# Count tells that are completely contained within the raster.
tells_inside_raster = tells.geometry.apply(
    raster_polygon.covers
).sum()

# Check whether the AOI overlaps the raster.
aoi_intersects_raster = aoi.geometry.intersects(
    raster_polygon
).any()

# Check the polygon ID field.
polygon_id_exists = "polygon_id" in tells.columns

if polygon_id_exists:
    polygon_id_unique = tells["polygon_id"].is_unique
    polygon_id_not_missing = tells["polygon_id"].notna().all()
else:
    polygon_id_unique = False
    polygon_id_not_missing = False

print("\nGeometry checks")
print("---------------")
print("Invalid tells:", invalid_tells)
print("Empty tells:", empty_tells)
print("Missing tells:", missing_tells)

print(
    "Tells intersecting raster:",
    f"{tells_intersecting_raster} / {len(tells)}"
)

print(
    "Tells fully inside raster:",
    f"{tells_inside_raster} / {len(tells)}"
)

print("AOI intersects raster:", aoi_intersects_raster)
print("polygon_id field exists:", polygon_id_exists)
print("polygon_id values are unique:", polygon_id_unique)
print("polygon_id values are not missing:", polygon_id_not_missing)

# Decide whether the inputs pass
#
# Rules:
# - Raster contains three bands
# - Raster uses EPSG:32638
# - Raster has 10 m pixels
# - Tell count is 1,242
# - AOI contains one feature
# - No invalid, empty, or missing tell geometries
# - Every tell intersects the raster
# - AOI intersects the raster
# - polygon_id exists, is unique, and has no missing values
raster_epsg = raster_crs.to_epsg()

checks = {
    "raster_has_3_bands": bool(raster_bands == 3),

    "raster_is_epsg_32638": bool(
        raster_epsg == 32638
    ),

    "raster_has_10m_pixels": bool(
        math.isclose(
            abs(raster_pixel_size[0]),
            10.0,
            abs_tol=0.000001,
        )
        and math.isclose(
            abs(raster_pixel_size[1]),
            10.0,
            abs_tol=0.000001,
        )
    ),

    "tell_count_is_1242": bool(
        len(tells) == 1242
    ),

    "aoi_count_is_1": bool(
        len(aoi) == 1
    ),

    "no_invalid_tells": bool(
        invalid_tells == 0
    ),

    "no_empty_tells": bool(
        empty_tells == 0
    ),

    "no_missing_tells": bool(
        missing_tells == 0
    ),

    "all_tells_intersect_raster": bool(
        tells_intersecting_raster == len(tells)
    ),

    "aoi_intersects_raster": bool(
        aoi_intersects_raster
    ),

    "polygon_id_exists": bool(
        polygon_id_exists
    ),

    "polygon_id_is_unique": bool(
        polygon_id_unique
    ),

    "polygon_id_has_no_missing_values": bool(
        polygon_id_not_missing
    ),
}

audit_passed = all(checks.values())

print("\nFinal checks")
print("------------")

for name, passed in checks.items():
    print(f"{name}: {passed}")

print("\nAUDIT PASSED:", audit_passed)

# Save the results
report = {
    "model": (
        "Model 2: B11-B8-B4 YOLO "
        "with 256-pixel source tiles"
    ),

    "raster": {
        "path": str(raster_path),
        "crs": str(raster_crs),
        "epsg": raster_epsg,
        "bands": raster_bands,
        "shape": raster_shape,
        "pixel_size": raster_pixel_size,
        "data_types": raster_dtypes,
        "bounds": list(raster_bounds),
    },

    "vectors": {
        "tells_path": str(tells_path),
        "aoi_path": str(aoi_path),
        "tell_count": len(tells),
        "aoi_count": len(aoi),
        "invalid_tells": int(invalid_tells),
        "empty_tells": int(empty_tells),
        "missing_tells": int(missing_tells),
        "tells_intersecting_raster": int(
            tells_intersecting_raster
        ),
        "tells_inside_raster": int(
            tells_inside_raster
        ),
        "aoi_intersects_raster": bool(
            aoi_intersects_raster
        ),
        "polygon_id_exists": bool(
            polygon_id_exists
        ),
        "polygon_id_is_unique": bool(
            polygon_id_unique
        ),
        "polygon_id_has_no_missing_values": bool(
            polygon_id_not_missing
        ),
    },

    "checks": checks,
    "audit_passed": audit_passed,
}

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

print("\nSaved report to:")
print(report_path)