"""
Generate H3 resolution-7 cells covering the continental US.

Outputs:
  - <DATA_DIR>/res7_cells.csv          columns: h3_index, lat, lon
  - <DATA_DIR>/res7_points.geojson     FeatureCollection<Point> for GEE upload

The GeoJSON can be uploaded to Earth Engine as a Table asset:
    earthengine upload table --asset_id=projects/renewably/assets/res7_points \\
        <DATA_DIR>/res7_points.geojson

Run:
    python -m ml.scripts.generate_res7_cells
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

RESOLUTION = 7

SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = ML_ROOT / "data"
DEFAULT_BOUNDARY_PATH = ML_ROOT / "notebooks" / "data" / "US_StateBoundaries.geojson"

DATA_DIR = Path(os.getenv("RENEWABLY_DATA_DIR", DEFAULT_DATA_DIR))
BOUNDARY_PATH = Path(os.getenv("US_BOUNDARY_PATH", DEFAULT_BOUNDARY_PATH))

# Drop non-CONUS states/territories (same list used in prior notebooks).
EXCLUDED_STATES = {
    "Alaska",
    "Hawaii",
    "Puerto Rico",
    "United States Virgin Islands",
    "American Samoa",
    "Guam",
    "Commonwealth of the Northern Mariana Islands",
}


def _load_conus_polygons() -> list[Polygon]:
    gdf = gpd.read_file(BOUNDARY_PATH)
    name_col = next(
        (c for c in ("NAME", "name", "STATE_NAME", "State") if c in gdf.columns),
        None,
    )
    if name_col is not None:
        gdf = gdf[~gdf[name_col].isin(EXCLUDED_STATES)]
    gdf = gdf.to_crs("EPSG:4326")

    polys: list[Polygon] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            polys.append(geom)
        elif isinstance(geom, MultiPolygon):
            polys.extend(geom.geoms)
    return polys


def _polygon_to_cells(poly: Polygon, res: int) -> set[str]:
    outer = [(lat, lng) for lng, lat in poly.exterior.coords]
    holes = [
        [(lat, lng) for lng, lat in ring.coords] for ring in poly.interiors
    ]
    shape = h3.LatLngPoly(outer, *holes) if holes else h3.LatLngPoly(outer)
    return set(h3.polygon_to_cells(shape, res))


def generate() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    polys = _load_conus_polygons()
    print(f"Loaded {len(polys)} CONUS polygons from {BOUNDARY_PATH}")

    cells: set[str] = set()
    for i, poly in enumerate(polys):
        cells |= _polygon_to_cells(poly, RESOLUTION)
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(polys)} polygons, {len(cells):,} cells so far")

    print(f"Total unique res-{RESOLUTION} cells: {len(cells):,}")

    records = []
    for cell in cells:
        lat, lon = h3.cell_to_latlng(cell)
        records.append({"h3_index": cell, "lat": lat, "lon": lon})

    df = pd.DataFrame.from_records(records)
    csv_path = DATA_DIR / "res7_cells.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    geojson_path = DATA_DIR / "res7_points.geojson"
    features = [
        {
            "type": "Feature",
            "properties": {"h3_index": rec["h3_index"]},
            "geometry": {"type": "Point", "coordinates": [rec["lon"], rec["lat"]]},
        }
        for rec in df.to_dict(orient="records")
    ]
    with geojson_path.open("w") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh)
    print(f"Wrote {geojson_path}")
    print()
    print("Next step — upload to Earth Engine as a Table asset:")
    print(
        f"  earthengine upload table --asset_id=projects/renewably/assets/res7_points "
        f"{geojson_path}"
    )
    return df


if __name__ == "__main__":
    generate()
