from pathlib import Path

import geopandas as gpd

# File paths
MODEL_DIR = Path(__file__).resolve().parent

input_path = (
    MODEL_DIR
    / "predictions"
    / "model_16_38SMB_review_candidates.gpkg"
)

output_path = (
    MODEL_DIR
    / "predictions"
    / "38SMB_candidates_for_advisor_review.gpkg"
)

# Load review candidates
candidates = gpd.read_file(
    input_path,
    layer="review_candidates",
)

print("Review candidates:", len(candidates))
print("CRS:", candidates.crs)

# Add useful review information
candidates["easting"] = (
    candidates.geometry.centroid.x
)

candidates["northing"] = (
    candidates.geometry.centroid.y
)

# Keep the review fields blank for the advisor.
candidates["advisor_label"] = ""
candidates["advisor_notes"] = ""

# Create centroid layer
centroids = candidates.copy()
centroids.geometry = candidates.geometry.centroid

# Save both layers
candidates.to_file(
    output_path,
    layer="candidate_boxes",
    driver="GPKG",
)

centroids.to_file(
    output_path,
    layer="candidate_centroids",
    driver="GPKG",
)

# Final audit
print("\nAdvisor review package")
print("----------------------")
print("Candidate boxes:", len(candidates))
print("Candidate centroids:", len(centroids))

print("\nConfidence range:")
print(
    candidates["confidence"].min(),
    "to",
    candidates["confidence"].max(),
)

print("\nSaved GeoPackage to:")
print(output_path)
