"""
Feature engine for the wind-turbine viability backend.

Responsibilities:
  - Load the H3 res-7 terrain cache produced by ml.scripts.download_res7_cache.
  - Look up KNN wind speed from the training-time wind dataset (same pipeline
    the model was trained on — not the ERA5 GEE band, which differs slightly).
  - Compute proximity features (road / transmission line / airport distances in
    km) from local shapefile/geojson via projected STRtree queries.
  - Derive the log + interaction features the v2 model expects.
  - Apply hard siting constraints that zero out probability for obviously
    unsuitable terrain (open water, ice, protected land, steep slope,
    dense urban).

All heavy assets are loaded exactly once via ``functools.lru_cache`` and reused
across requests within a warm Modal container.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Resolution used for inference (res-8 ≈ 0.74 km² per cell).
H3_INFERENCE_RES = 8
# Terrain features are cached at this coarser resolution.
H3_TERRAIN_RES = 7

# Paths. These are set to match the Modal image layout; when running locally we
# fall back to the repo-relative locations.
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent

TERRAIN_CACHE_CANDIDATES = (
    Path("/seed_feature_cache/terrain_res7_cache.csv"),
    REPO_ROOT / "ml" / "data" / "renewably_exports" / "terrain_res7_cache.csv",
)
WIND_DATASET_CANDIDATES = (
    Path("/seed_data/us_wind_speed_dataset_2022.csv"),
    REPO_ROOT / "ml" / "data" / "us_wind_speed_dataset_2022.csv",
)
TRANSMISSION_CANDIDATES = (
    Path("/seed_data/us_transmission_lines.geojson"),
    BACKEND_DIR / "data" / "us_transmission_lines.geojson",
)
AIRPORT_CANDIDATES = (
    Path("/seed_data/airports.geojson"),
    BACKEND_DIR / "data" / "airports.geojson",
)
ROAD_CANDIDATES = (
    Path("/seed_data/roads/tl_2023_us_primaryroads.shp"),
    BACKEND_DIR / "data" / "tl_2023_us_primaryroads.shp",
)
FEATURE_COLUMNS_CANDIDATES = (
    Path("/seed_models/wind_v2_feature_columns.json"),
    REPO_ROOT / "ml" / "models" / "wind_v2_feature_columns.json",
)

TERRAIN_COLUMNS = [
    "elevation_m",
    "slope_deg",
    "land_type",
    "impervious",
    "soil_type",
    "protected_area",
    "in_wdpa",
    "pop_density",
]


def _resolve(candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"None of the expected paths exist: {searched}")


@lru_cache(maxsize=1)
def load_feature_columns() -> list[str]:
    path = _resolve(FEATURE_COLUMNS_CANDIDATES)
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def _load_terrain_cache() -> pd.DataFrame:
    path = _resolve(TERRAIN_CACHE_CANDIDATES)
    df = pd.read_csv(path, dtype={"h3_index": str})
    for col in TERRAIN_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["h3_index"], keep="last").set_index("h3_index")
    return df


@lru_cache(maxsize=1)
def _terrain_medians() -> dict[str, float]:
    df = _load_terrain_cache()
    medians = df[TERRAIN_COLUMNS].median(numeric_only=True).to_dict()
    return {col: float(medians.get(col, 0.0) or 0.0) for col in TERRAIN_COLUMNS}


@lru_cache(maxsize=1)
def _terrain_spatial_index():
    """Fallback KD-tree over res-7 centroids for cells not in the cache."""
    from scipy.spatial import cKDTree
    import h3

    df = _load_terrain_cache()
    cells = df.index.tolist()
    centroids = np.array([h3.cell_to_latlng(c) for c in cells], dtype=float)
    tree = cKDTree(np.radians(centroids))
    return cells, tree


@lru_cache(maxsize=1)
def _load_wind_knn():
    from sklearn.neighbors import KNeighborsRegressor

    path = _resolve(WIND_DATASET_CANDIDATES)
    df = pd.read_csv(path)
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance")
    knn.fit(df[["lat", "lon"]].to_numpy(), df["annual_mean_wind_speed"].to_numpy())
    return knn


@lru_cache(maxsize=1)
def _load_transmission_tree():
    import geopandas as gpd
    from shapely.strtree import STRtree

    path = _resolve(TRANSMISSION_CANDIDATES)
    gdf = gpd.read_file(path).to_crs(epsg=5070)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    geoms = list(gdf.geometry.values)
    return STRtree(geoms), geoms


@lru_cache(maxsize=1)
def _load_airport_tree():
    import geopandas as gpd
    from shapely.strtree import STRtree

    path = _resolve(AIRPORT_CANDIDATES)
    gdf = gpd.read_file(path).to_crs(epsg=5070)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    geoms = list(gdf.geometry.values)
    return STRtree(geoms), geoms


@lru_cache(maxsize=1)
def _load_road_tree():
    import geopandas as gpd
    from shapely.strtree import STRtree

    path = _resolve(ROAD_CANDIDATES)
    # TIGER shapefiles are in EPSG:4269 by default.
    gdf = gpd.read_file(path).set_crs(epsg=4269, allow_override=True).to_crs(epsg=5070)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    geoms = list(gdf.geometry.values)
    return STRtree(geoms), geoms


def _nearest_distances_km(
    points_5070, tree_and_geoms
) -> np.ndarray:
    """Return nearest distance in km from each projected point to any geometry."""
    tree, geoms = tree_and_geoms
    nearest_idx = tree.nearest(points_5070)
    # shapely 2.x STRtree.nearest returns an array of indices aligned with the
    # query geometries. Distance from each point to the matched geometry.
    distances_m = np.fromiter(
        (pt.distance(geoms[int(idx)]) for pt, idx in zip(points_5070, nearest_idx)),
        dtype=float,
        count=len(points_5070),
    )
    return distances_m / 1000.0


def _terrain_for_parent(parent_cells: list[str]) -> pd.DataFrame:
    """Look up terrain rows for each parent res-7 cell, with nearest-centroid fallback."""
    import h3

    cache = _load_terrain_cache()
    medians = _terrain_medians()

    present_mask = np.array([c in cache.index for c in parent_cells])
    missing = [c for c, ok in zip(parent_cells, present_mask) if not ok]

    # Nearest-centroid fallback for cells missing from the cache.
    fallback_map: dict[str, str] = {}
    if missing:
        cells, tree = _terrain_spatial_index()
        centroids = np.array([h3.cell_to_latlng(c) for c in missing], dtype=float)
        _, nearest_idx = tree.query(np.radians(centroids), k=1)
        fallback_map = {c: cells[int(i)] for c, i in zip(missing, nearest_idx)}

    records: list[dict] = []
    for cell, present in zip(parent_cells, present_mask):
        lookup_cell = cell if present else fallback_map[cell]
        row = cache.loc[lookup_cell]
        record = {"feature_source": "exact_cache" if present else "nearest_cache"}
        for col in TERRAIN_COLUMNS:
            value = row.get(col)
            record[col] = (
                float(value) if pd.notna(value) else medians[col]
            )
        records.append(record)

    return pd.DataFrame(records)


def compute_features_for_cells(h3_indices: list[str]) -> pd.DataFrame:
    """Build the full feature frame for a batch of H3 res-8 cells.

    Returned columns: h3_index, lat, lon, feature_source, plus the 8 raw terrain
    columns, wind_speed, road/transmission/airport distances, and the 6
    engineered features the v2 model expects. The caller selects the final
    model-input columns via ``load_feature_columns()``.
    """
    import h3

    if not h3_indices:
        raise ValueError("compute_features_for_cells requires at least one cell.")

    parent_cells = [h3.cell_to_parent(c, H3_TERRAIN_RES) for c in h3_indices]
    terrain = _terrain_for_parent(parent_cells)

    centroids = np.array([h3.cell_to_latlng(c) for c in h3_indices], dtype=float)
    lat = centroids[:, 0]
    lon = centroids[:, 1]

    df = pd.DataFrame({"h3_index": h3_indices, "lat": lat, "lon": lon})
    df = pd.concat([df, terrain], axis=1)

    # Wind speed from KNN (matches training pipeline).
    knn = _load_wind_knn()
    df["wind_speed"] = knn.predict(np.column_stack([lat, lon]))

    # Project points once to EPSG:5070 for distance queries.
    from pyproj import Transformer
    from shapely.geometry import Point

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xs, ys = transformer.transform(lon, lat)
    points_5070 = np.array([Point(x, y) for x, y in zip(xs, ys)], dtype=object)

    df["road_dist_km"] = _nearest_distances_km(points_5070, _load_road_tree())
    df["transmission_line_dist_km"] = _nearest_distances_km(
        points_5070, _load_transmission_tree()
    )
    df["airport_dist_km"] = _nearest_distances_km(points_5070, _load_airport_tree())

    # Engineered features (exact match to training notebook).
    df["log_road_dist_km"] = np.log1p(df["road_dist_km"])
    df["log_transmission_line_dist_km"] = np.log1p(df["transmission_line_dist_km"])
    df["log_road_dist_km_x_transmission_line_dist_km"] = (
        df["log_road_dist_km"] * df["log_transmission_line_dist_km"]
    )
    df["slope_x_elevation_m"] = df["slope_deg"] * df["elevation_m"]
    df["wind_speed_x_elevation_m"] = df["wind_speed"] * df["elevation_m"]
    df["wind_speed_x_slope_deg"] = df["wind_speed"] * df["slope_deg"]

    return df


# NLCD developed-land classes.
# 21 = developed open space, 22 low-, 23 medium-, 24 high-intensity.
# We include 21 because suburban/city-park cells still fail turbine siting
# (setbacks, zoning) even if the model likes the surrounding terrain.
URBAN_LAND_TYPES = (21, 22, 23, 24)
# Hard cap on residential density (people/km²) regardless of model output.
# Low enough to catch suburban Denver where the res-7 terrain cache smooths
# the land-type signal below the developed-land classes.
URBAN_POP_DENSITY_THRESHOLD = 75.0
# NLCD impervious-surface percentage. Urban cores are typically >25%;
# this is a backstop when land_type aggregation at res-7 washes out.
URBAN_IMPERVIOUS_THRESHOLD = 15.0


def apply_siting_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """Zero out probability for obviously unsuitable sites.

    Also annotates each row with a ``siting_exclusion_reason`` string so the
    caller can explain to users why candidate cells were dropped (in particular
    urban / high-density cells that the model might otherwise have scored
    favorably).
    """
    water_ice = df["land_type"].isin([11, 12])
    protected = df["protected_area"] == 1
    steep = df["slope_deg"] > 20
    urban_land = df["land_type"].isin(URBAN_LAND_TYPES)
    high_pop = df["pop_density"] > URBAN_POP_DENSITY_THRESHOLD
    impervious = df["impervious"] > URBAN_IMPERVIOUS_THRESHOLD

    reasons = np.array([""] * len(df), dtype=object)
    # Order matters: later assignments override earlier ones; urban/pop win
    # so we always surface the people-facing reason when applicable.
    reasons[steep.to_numpy()] = "steep_slope"
    reasons[protected.to_numpy()] = "protected_area"
    reasons[water_ice.to_numpy()] = "water_or_ice"
    reasons[impervious.to_numpy()] = "impervious_surface"
    reasons[urban_land.to_numpy()] = "urban_land_cover"
    reasons[high_pop.to_numpy()] = "high_population_density"

    unsuitable = (
        water_ice | protected | steep | urban_land | high_pop | impervious
    )
    df = df.copy()
    df["siting_exclusion_reason"] = reasons
    if "turbine_probability" in df.columns:
        df.loc[unsuitable, "turbine_probability"] = 0.0
    return df


def warm() -> None:
    """Force-load every cached asset (call at Modal container startup)."""
    load_feature_columns()
    _load_terrain_cache()
    _terrain_medians()
    _terrain_spatial_index()
    _load_wind_knn()
    _load_transmission_tree()
    _load_airport_tree()
    _load_road_tree()
