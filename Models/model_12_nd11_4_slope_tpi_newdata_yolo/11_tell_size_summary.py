from pathlib import Path
import json

import geopandas as gpd
import pandas as pd

# File paths
MODEL_DIR = Path(__file__).resolve().parent
REPORTS_DIR = MODEL_DIR / "reports"

split_path = REPORTS_DIR / "geographic_split.gpkg"

# Fixed project tell-size thresholds
SMALL_MAX_M2 = 16718
LARGE_MIN_M2 = 129300

# Load split tells
tells = gpd.read_file(
    split_path,
    layer="tells_with_split",
)

usable = tells[
    tells["split"].isin(["train", "val", "test"])
].copy()

test = tells[
    tells["split"] == "test"
].copy()

usable["area_m2"] = usable.geometry.area
test["area_m2"] = test.geometry.area

# Area quartiles
usable_quartiles = (
    usable["area_m2"]
    .quantile([0.25, 0.50, 0.75])
)

test_quartiles = (
    test["area_m2"]
    .quantile([0.25, 0.50, 0.75])
)

quartile_summary = pd.DataFrame(
    {
        "dataset": [
            "usable",
            "usable",
            "usable",
            "test",
            "test",
            "test",
        ],
        "quantile": [
            0.25,
            0.50,
            0.75,
            0.25,
            0.50,
            0.75,
        ],
        "area_m2": [
            usable_quartiles.loc[0.25],
            usable_quartiles.loc[0.50],
            usable_quartiles.loc[0.75],
            test_quartiles.loc[0.25],
            test_quartiles.loc[0.50],
            test_quartiles.loc[0.75],
        ],
    }
)

# Assign fixed size classes
def assign_size_class(area):
    if area <= SMALL_MAX_M2:
        return "small"

    if area >= LARGE_MIN_M2:
        return "large"

    return "medium"


usable["size_class"] = (
    usable["area_m2"].apply(assign_size_class)
)

test["size_class"] = (
    test["area_m2"].apply(assign_size_class)
)

# Size-class counts
usable_counts = (
    usable["size_class"]
    .value_counts()
    .reindex(["small", "medium", "large"], fill_value=0)
)

test_counts = (
    test["size_class"]
    .value_counts()
    .reindex(["small", "medium", "large"], fill_value=0)
)

count_summary = pd.DataFrame(
    {
        "size_class": [
            "small",
            "medium",
            "large",
        ],
        "usable_count": [
            int(usable_counts["small"]),
            int(usable_counts["medium"]),
            int(usable_counts["large"]),
        ],
        "test_count": [
            int(test_counts["small"]),
            int(test_counts["medium"]),
            int(test_counts["large"]),
        ],
    }
)

# Save reports
quartile_path = (
    REPORTS_DIR / "tell_area_quartiles.csv"
)

counts_path = (
    REPORTS_DIR / "tell_size_class_counts.csv"
)

summary_path = (
    REPORTS_DIR / "tell_size_summary.json"
)

quartile_summary.to_csv(
    quartile_path,
    index=False,
)

count_summary.to_csv(
    counts_path,
    index=False,
)

summary = {
    "small_definition": (
        f"area <= {SMALL_MAX_M2} m2"
    ),
    "medium_definition": (
        f"{SMALL_MAX_M2} < area < "
        f"{LARGE_MIN_M2} m2"
    ),
    "large_definition": (
        f"area >= {LARGE_MIN_M2} m2"
    ),
    "usable_tell_count": int(len(usable)),
    "test_tell_count": int(len(test)),
    "usable_area_quartiles_m2": {
        "q25": float(usable_quartiles.loc[0.25]),
        "median": float(usable_quartiles.loc[0.50]),
        "q75": float(usable_quartiles.loc[0.75]),
    },
    "test_area_quartiles_m2": {
        "q25": float(test_quartiles.loc[0.25]),
        "median": float(test_quartiles.loc[0.50]),
        "q75": float(test_quartiles.loc[0.75]),
    },
    "test_size_counts": {
        "small": int(test_counts["small"]),
        "medium": int(test_counts["medium"]),
        "large": int(test_counts["large"]),
    },
}

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)

# Print results
print("\nUsable dataset area quartiles")
print("-----------------------------")
print(usable_quartiles)

print("\nTest-set area quartiles")
print("-----------------------")
print(test_quartiles)

print("\nSize-class counts")
print("-----------------")
print(count_summary.to_string(index=False))

print("\nSaved reports:")
print(quartile_path)
print(counts_path)
print(summary_path)
