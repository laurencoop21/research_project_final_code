from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


# Paths and settings
MODELS_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Visibility threshold used when labelling
MINIMUM_VISIBLE_FRACTION = 0.50

# The two tile sizes behind the final candidate models
PIPELINES = {
    "512": {
        "tile_index": (
            MODELS_DIR
            / "model_17_nd11_4_slope_tpi_newdata_yolo11m_512_200epochs"
            / "reports"
            / "tile_index.csv"
        ),
        "split_file": (
            MODELS_DIR
            / "model_17_nd11_4_slope_tpi_newdata_yolo11m_512_200epochs"
            / "reports"
            / "geographic_split.gpkg"
        ),
    },
    "320": {
        "tile_index": (
            MODELS_DIR
            / "model_16_nd11_4_slope_tpi_newdata_yolo11m_320_150epochs"
            / "reports"
            / "tile_index.csv"
        ),
        "split_file": (
            MODELS_DIR
            / "model_16_nd11_4_slope_tpi_newdata_yolo11m_320_150epochs"
            / "reports"
            / "geographic_split.gpkg"
        ),
    },
}


# Audit one pipeline
# Check what each background tile actually contains
def audit_pipeline(name, paths):
    tile_index = pd.read_csv(paths["tile_index"])

    tells = gpd.read_file(
        paths["split_file"],
        layer="tells_with_split",
    )

    background_tiles = tile_index[
        tile_index["tell_count"] == 0
    ].copy()

    records = []

    for _, tile in background_tiles.iterrows():
        tile_polygon = box(
            tile["left"],
            tile["bottom"],
            tile["right"],
            tile["top"],
        )

        candidate_positions = list(
            tells.sindex.query(
                tile_polygon,
                predicate="intersects",
            )
        )

        same_split_partial_ids = []
        same_split_labelable_ids = []
        other_split_ids = []
        excluded_boundary_ids = []

        same_split_visible_fractions = []
        all_visible_fractions = []

        for position in candidate_positions:
            tell = tells.iloc[position]

            intersection = tell.geometry.intersection(tile_polygon)

            if intersection.is_empty or intersection.area <= 0:
                continue

            visible_fraction = (
                intersection.area / tell.geometry.area
            )

            all_visible_fractions.append(visible_fraction)

            tell_split = tell["split"]
            tell_id = int(tell["polygon_id"])

            if tell_split == tile["split"]:
                same_split_visible_fractions.append(
                    visible_fraction
                )

                if visible_fraction < MINIMUM_VISIBLE_FRACTION:
                    same_split_partial_ids.append(tell_id)
                else:
                    same_split_labelable_ids.append(tell_id)

            elif tell_split == "exclude_boundary":
                excluded_boundary_ids.append(tell_id)

            else:
                other_split_ids.append(tell_id)

        partial_contamination = (
            len(same_split_partial_ids) > 0
        )

        cross_split_contamination = (
            len(other_split_ids) > 0
        )

        excluded_boundary_contamination = (
            len(excluded_boundary_ids) > 0
        )

        any_unlabelled_tell = (
            partial_contamination
            or cross_split_contamination
            or excluded_boundary_contamination
        )

        records.append(
            {
                "pipeline": name,
                "tile_name": tile["tile_name"],
                "split": tile["split"],
                "row_start": int(tile["row_start"]),
                "col_start": int(tile["col_start"]),
                "same_split_partial_tells":
                    len(same_split_partial_ids),
                "same_split_labelable_tells":
                    len(same_split_labelable_ids),
                "other_split_tells":
                    len(other_split_ids),
                "excluded_boundary_tells":
                    len(excluded_boundary_ids),
                "partial_contamination":
                    partial_contamination,
                "cross_split_contamination":
                    cross_split_contamination,
                "excluded_boundary_contamination":
                    excluded_boundary_contamination,
                "any_unlabelled_tell":
                    any_unlabelled_tell,
                "max_same_split_visible_fraction":
                    max(same_split_visible_fractions)
                    if same_split_visible_fractions
                    else np.nan,
                "max_any_visible_fraction":
                    max(all_visible_fractions)
                    if all_visible_fractions
                    else np.nan,
                "same_split_partial_ids":
                    ";".join(map(str, same_split_partial_ids)),
                "other_split_ids":
                    ";".join(map(str, other_split_ids)),
                "excluded_boundary_ids":
                    ";".join(map(str, excluded_boundary_ids)),
            }
        )

    audit = pd.DataFrame(records)

    summaries = []

    for split_name in ["train", "val", "test", "ALL"]:
        if split_name == "ALL":
            subset = audit
        else:
            subset = audit[audit["split"] == split_name]

        n_background = len(subset)

        n_partial = int(
            subset["partial_contamination"].sum()
        )

        n_cross = int(
            subset["cross_split_contamination"].sum()
        )

        n_boundary = int(
            subset[
                "excluded_boundary_contamination"
            ].sum()
        )

        n_any = int(
            subset["any_unlabelled_tell"].sum()
        )

        n_labelable = int(
            (subset["same_split_labelable_tells"] > 0).sum()
        )

        summaries.append(
            {
                "pipeline": name,
                "split": split_name,
                "background_tiles": n_background,
                "partial_tell_background_tiles": n_partial,
                "partial_tell_percent":
                    100 * n_partial / n_background
                    if n_background else np.nan,
                "cross_split_background_tiles": n_cross,
                "cross_split_percent":
                    100 * n_cross / n_background
                    if n_background else np.nan,
                "excluded_boundary_background_tiles":
                    n_boundary,
                "excluded_boundary_percent":
                    100 * n_boundary / n_background
                    if n_background else np.nan,
                "any_unlabelled_tell_background_tiles":
                    n_any,
                "any_unlabelled_tell_percent":
                    100 * n_any / n_background
                    if n_background else np.nan,
                "unexpected_labelable_background_tiles":
                    n_labelable,
            }
        )

    summary = pd.DataFrame(summaries)

    audit_path = (
        REPORTS_DIR
        / f"background_contamination_{name}.csv"
    )

    summary_path = (
        REPORTS_DIR
        / f"background_contamination_summary_{name}.csv"
    )

    audit.to_csv(audit_path, index=False)
    summary.to_csv(summary_path, index=False)

    return audit, summary


# Run both tiling pipeliness
all_summaries = []

for pipeline_name, pipeline_paths in PIPELINES.items():
    audit, summary = audit_pipeline(
        pipeline_name,
        pipeline_paths,
    )

    all_summaries.append(summary)

combined_summary = pd.concat(
    all_summaries,
    ignore_index=True,
)

combined_path = (
    REPORTS_DIR
    / "background_contamination_summary_all.csv"
)

combined_summary.to_csv(
    combined_path,
    index=False,
)

json_path = (
    REPORTS_DIR
    / "background_contamination_summary_all.json"
)

with open(json_path, "w") as f:
    json.dump(
        combined_summary.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        f,
        indent=2,
    )


# Printresults
print("\nBackground-tile contamination audit")
print("-----------------------------------")
print(
    combined_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

print("\nSaved reports to:")
print(REPORTS_DIR)
