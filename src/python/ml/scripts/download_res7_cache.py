"""
Merge and clean the two GEE terrain exports into a single res-7 cache.

Prerequisites:
    The two Export tasks kicked off by ml.scripts.export_res7_terrain must
    have completed. Copy the resulting CSVs into <INPUT_DIR>:
        res7_terrain_30m.csv
        res7_terrain_coarse.csv

    If the exports went to a GEE asset, use:
        earthengine table export --asset_id=projects/renewably/assets/res7_terrain_30m \\
            --file_format=CSV --destination=drive --folder=renewably_exports
    then sync the Drive folder to <INPUT_DIR>.

Output:
    <DATA_DIR>/renewably_exports/terrain_res7_cache.csv
      columns: h3_index, elevation_m, slope_deg, land_type, impervious,
               soil_type, wind_speed, protected_area, in_wdpa, pop_density

Run:
    python -m ml.scripts.download_res7_cache
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = ML_ROOT / "data"

DATA_DIR = Path(os.getenv("RENEWABLY_DATA_DIR", DEFAULT_DATA_DIR))
INPUT_DIR = Path(os.getenv("RES7_INPUT_DIR", DATA_DIR / "renewably_exports"))
OUTPUT_PATH = Path(
    os.getenv("RES7_OUTPUT_PATH", DATA_DIR / "renewably_exports" / "terrain_res7_cache.csv")
)

# GEE writes columns with these raw names; we normalize to the model schema.
FINE_COLS = [
    "h3_index",
    "elevation_m",
    "slope_deg",
    "land_type",
    "impervious",
    "soil_type",
    "wind_speed",
]
COARSE_COLS = [
    "h3_index",
    "protected_area",
    "in_wdpa",
    "pop_density",
]
FINAL_COLS = [
    "h3_index",
    "elevation_m",
    "slope_deg",
    "land_type",
    "impervious",
    "soil_type",
    "wind_speed",
    "protected_area",
    "in_wdpa",
    "pop_density",
]


def _load_gee_csv(path: Path, keep_cols: list[str]) -> pd.DataFrame:
    print(f"Loading {path}")
    df = pd.read_csv(path, dtype={"h3_index": str})
    # Drop GEE metadata columns if present.
    for junk in ("system:index", ".geo"):
        if junk in df.columns:
            df = df.drop(columns=[junk])
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} missing expected columns {missing}. "
            f"Present columns: {list(df.columns)}"
        )
    return df[keep_cols]


def run() -> None:
    fine_path = INPUT_DIR / "res7_terrain_30m.csv"
    coarse_path = INPUT_DIR / "res7_terrain_coarse.csv"
    if not fine_path.exists():
        raise FileNotFoundError(f"Missing {fine_path}")
    if not coarse_path.exists():
        raise FileNotFoundError(f"Missing {coarse_path}")

    fine = _load_gee_csv(fine_path, FINE_COLS)
    coarse = _load_gee_csv(coarse_path, COARSE_COLS)
    print(f"  30m rows: {len(fine):,}")
    print(f"  coarse rows: {len(coarse):,}")

    merged = fine.merge(coarse, on="h3_index", how="left")

    # Normalize dtypes / clip sentinels.
    numeric_cols = [c for c in FINAL_COLS if c != "h3_index"]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["pop_density"] = merged["pop_density"].clip(lower=0)
    merged["protected_area"] = merged["protected_area"].fillna(0).clip(0, 1)
    merged["in_wdpa"] = merged["in_wdpa"].fillna(0).clip(0, 1)

    # Drop rows whose h3_index didn't round-trip (shouldn't happen, but defensive).
    merged = merged.dropna(subset=["h3_index"])
    merged = merged.drop_duplicates(subset=["h3_index"], keep="last")
    merged = merged[FINAL_COLS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} ({len(merged):,} rows, {len(merged.columns)} cols)")

    # Summary — helpful for catching silent export bugs.
    print("\nColumn summary:")
    print(merged[numeric_cols].describe().T[["count", "min", "max", "mean"]].to_string())
    nulls = merged.isna().sum()
    if nulls.any():
        print("\nNull counts:")
        print(nulls[nulls > 0].to_string())


if __name__ == "__main__":
    run()
