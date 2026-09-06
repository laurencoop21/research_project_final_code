from pathlib import Path
import json

import geopandas as gpd
import rasterio
from shapely.geometry import box

# File paths
# This script should be saved inside:
# iraq_tells_gis/Models/model_12_nd11_4_slope_tpi_newdata_yolo/

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent.parent

raster_path = PROJECT_DIR / "Iraq_ND11_4_slope_TPI_uint8.tif"
tells_path = PROJECT_DIR / "tells_final_II.shp"
aoi_path = PROJECT_DIR / "aoi_sentinel.shp"

reports_dir = MODEL_DIR / "reports"
reports_dir.mkdir(exist_ok=True)

report_path = reports_dir / "input_audit.json"

# Check that the files exist
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

# Make sure both shapefiles use the same CRS as the raster.
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
invalid_tells = (~tells.geometry.is_valid).sum()
empty_tells = tells.geometry.is_empty.sum()
missing_tells = tells.geometry.isna().sum()

raster_polygon = box(*raster_bounds)

tells_intersecting_raster = tells.geometry.intersects(raster_polygon).sum()
tells_inside_raster = tells.geometry.apply(raster_polygon.covers).sum()

fid_exists = "fid" in tells.columns

if fid_exists:
    fid_unique = tells["fid"].is_unique
    fid_missing = tells["fid"].isna().sum()
else:
    fid_unique = False
    fid_missing = len(tells)

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
print("fid field exists:", fid_exists)
print("Missing fid values:", fid_missing)
print("fid values are unique:", fid_unique)

# Decide whether the inputs pass
checks = {
    "raster_has_3_bands": bool(raster_bands == 3),
    "raster_is_epsg_32638": bool(raster_crs.to_epsg() == 32638),
    "raster_has_10m_pixels": bool(
        abs(raster_pixel_size[0]) == 10
        and abs(raster_pixel_size[1]) == 10
    ),
    "tell_count_is_1362": bool(len(tells) == 1362),
    "no_invalid_tells": bool(invalid_tells == 0),
    "no_empty_tells": bool(empty_tells == 0),
    "no_missing_tells": bool(missing_tells == 0),
    "all_tells_intersect_raster": bool(
        tells_intersecting_raster == len(tells)
    ),
    "fid_exists": bool(fid_exists),
    "no_missing_fid": bool(fid_missing == 0),
    "fid_is_unique": bool(fid_unique),
}

audit_passed = all(checks.values())

print("\nFinal checks")
print("------------")

for name, passed in checks.items():
    print(f"{name}: {passed}")

print("\nAUDIT PASSED:", audit_passed)

# Save the results
report = {
    "model": "Model 12: ND11_4-Slope-TPI YOLO with updated tell dataset",
    "raster": {
        "path": str(raster_path),
        "crs": str(raster_crs),
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
        "missing_fid": int(fid_missing),
        "tells_intersecting_raster": int(tells_intersecting_raster),
        "tells_inside_raster": int(tells_inside_raster),
    },
    "checks": checks,
    "audit_passed": audit_passed,
}

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

print("\nSaved report to:")
print(report_path)