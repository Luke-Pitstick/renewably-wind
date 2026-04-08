"""
H3 Exogenous Variable Enrichment Script
========================================
Paste cells into your notebook after you have a Polars df
with columns: h3_index, lat, lng, has_turbine (or has_solar).

Adds: elevation, slope, aspect, roughness, wind speed,
land cover, population density, distance to transmission lines,
distance to roads.
"""

# %% Cell 0: Install dependencies (run once)
# !pip install rasterio scipy requests geopandas polars h3

import glob
import hashlib
import json
import os
import time

# %% Cell 1: Imports & setup
import ee
import geopandas as gpd
import h3
import numpy as np
import polars as pl
import requests
from scipy.spatial import cKDTree
from shapely.geometry import Point
from tqdm import tqdm

ee.Initialize()

DATA_DIR = os.getenv(
    "RENEWABLY_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(""), "..", "data")),
)
os.makedirs(DATA_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(DATA_DIR, "enrich_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
GEE_DRIVE_FOLDER = os.getenv("GEE_DRIVE_FOLDER", "renewably_exports")
GEE_DRIVE_MOUNT_ROOT = os.getenv("GEE_DRIVE_MOUNT_ROOT", "/content/drive/MyDrive")
LOCAL_GEE_EXPORT_DIR = os.path.join(DATA_DIR, GEE_DRIVE_FOLDER)
GEE_EXPORT_POLL_SECONDS = int(os.getenv("GEE_EXPORT_POLL_SECONDS", "15"))
GEE_EXPORT_TIMEOUT_SECONDS = int(os.getenv("GEE_EXPORT_TIMEOUT_SECONDS", "7200"))
GEE_DIRECT_SAMPLE_MAX_ROWS = int(os.getenv("GEE_DIRECT_SAMPLE_MAX_ROWS", "2000"))
GEE_EXPORT_BATCH_SIZE = int(os.getenv("GEE_EXPORT_BATCH_SIZE", "2000"))
GEE_RESUME_FROM_BATCH = int(os.getenv("GEE_RESUME_FROM_BATCH", "1"))
GEE_STOP_AFTER_BATCH = int(os.getenv("GEE_STOP_AFTER_BATCH", "0"))
FOREST_CODES = {41, 42, 43}
CROP_CODES = {81, 82}
URBAN_CODES = {21, 22, 23, 24}
GRASS_CODES = {71}
WATER_CODES = {11}

# Assumes wind_df or solar_df already exists as a Polars DataFrame
# with columns: h3_index, lat, lng, has_turbine/has_solar
# We'll call it `df` below — reassign as needed:
# df = wind_df  # or solar_df


# %% Cell 2: Helper — upload centroids to GEE as FeatureCollection
def centroids_to_ee_fc(df: pl.DataFrame) -> ee.FeatureCollection:
    """Convert H3 centroid lat/lng to an ee.FeatureCollection of points."""
    features = []
    for row in df.select("h3_index", "lat", "lng").iter_rows(named=True):
        feat = ee.Feature(
            ee.Geometry.Point([row["lng"], row["lat"]]),
            {"h3_index": row["h3_index"]},
        )
        features.append(feat)
    return ee.FeatureCollection(features)


def unique_h3_centroids(df: pl.DataFrame) -> pl.DataFrame:
    """Keep one centroid row per H3 cell for expensive enrichment work."""
    return df.select("h3_index", "lat", "lng").unique(subset=["h3_index"], keep="first")


# %% Cell 3: Build one combined GEE image and sample it once per batch
def build_gee_feature_image() -> ee.Image:
    """Build a single image containing all GEE-derived bands."""
    dem = ee.Image("USGS/SRTMGL1_003")
    terrain = (
        dem.rename("elevation")
        .addBands(ee.Terrain.slope(dem).rename("slope"))
        .addBands(ee.Terrain.aspect(dem).rename("aspect"))
    )

    # Roughness: std dev of elevation in a 3x3 kernel (~90m)
    roughness = dem.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(3, "pixels"),
    ).rename("roughness")

    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").select(
        ["u_component_of_wind_10m", "v_component_of_wind_10m"]
    )

    # Compute wind speed magnitude for each month, then reduce
    def wind_speed(img):
        u = img.select("u_component_of_wind_10m")
        v = img.select("v_component_of_wind_10m")
        speed = u.pow(2).add(v.pow(2)).sqrt().rename("wind_speed_10m")
        return speed

    speeds = era5.map(wind_speed)
    mean_speed = speeds.mean().rename("wind_10m_avg")
    std_speed = speeds.reduce(ee.Reducer.stdDev()).rename("wind_10m_std")
    nlcd = ee.Image("USGS/NLCD_RELEASES/2021_REL/NLCD/2021").select("landcover")
    pop = (
        ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
        .sort("system:time_start", False)
        .first()
        .select("population_density")
    )

    return (
        terrain.addBands(roughness)
        .addBands(mean_speed)
        .addBands(std_speed)
        .addBands(nlcd)
        .addBands(pop)
    )


def sample_gee_features(fc: ee.FeatureCollection, feature_image: ee.Image) -> dict:
    """Sample all GEE features for one batch with retry/backoff."""
    sampled = feature_image.sampleRegions(collection=fc, scale=30, geometries=False)

    last_error = None
    for attempt in range(4):
        try:
            results = sampled.getInfo()
            break
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                raise
            sleep_s = 2**attempt
            print(
                f"GEE request failed on attempt {attempt + 1}/4; retrying in {sleep_s}s"
            )
            time.sleep(sleep_s)
    else:
        raise last_error

    out = {}
    for feat in results["features"]:
        p = feat["properties"]
        lc = p.get("landcover", 0)
        out[p["h3_index"]] = {
            "h3_elev_mean": p.get("elevation"),
            "h3_slope_mean_deg": p.get("slope"),
            "h3_aspect_mean_deg": p.get("aspect"),
            "h3_roughness": p.get("roughness"),
            "h3_wind_10m_avg": p.get("wind_10m_avg"),
            "h3_wind_10m_std": p.get("wind_10m_std"),
            "h3_land_cover_code": lc,
            "h3_cover_is_forest": int(lc in FOREST_CODES),
            "h3_cover_is_crop": int(lc in CROP_CODES),
            "h3_cover_is_urban": int(lc in URBAN_CODES),
            "h3_cover_is_grass": int(lc in GRASS_CODES),
            "h3_cover_is_water": int(lc in WATER_CODES),
            "h3_pop_density_km2": p.get("population_density"),
        }
    return out


def normalize_gee_feature_df(gee_df: pl.DataFrame) -> pl.DataFrame:
    """Normalize either export CSV columns or direct-sample columns to one schema."""
    if gee_df.is_empty():
        return gee_df

    rename_map = {
        "elevation": "h3_elev_mean",
        "slope": "h3_slope_mean_deg",
        "aspect": "h3_aspect_mean_deg",
        "roughness": "h3_roughness",
        "wind_10m_avg": "h3_wind_10m_avg",
        "wind_10m_std": "h3_wind_10m_std",
        "landcover": "h3_land_cover_code",
        "population_density": "h3_pop_density_km2",
    }
    gee_df = gee_df.rename(
        {src: dst for src, dst in rename_map.items() if src in gee_df.columns}
    )

    if "h3_land_cover_code" not in gee_df.columns:
        return gee_df

    lc = pl.col("h3_land_cover_code").cast(pl.Int64, strict=False).fill_null(0)
    return gee_df.with_columns(
        lc.alias("h3_land_cover_code"),
        lc.is_in(sorted(FOREST_CODES)).cast(pl.Int8).alias("h3_cover_is_forest"),
        lc.is_in(sorted(CROP_CODES)).cast(pl.Int8).alias("h3_cover_is_crop"),
        lc.is_in(sorted(URBAN_CODES)).cast(pl.Int8).alias("h3_cover_is_urban"),
        lc.is_in(sorted(GRASS_CODES)).cast(pl.Int8).alias("h3_cover_is_grass"),
        lc.is_in(sorted(WATER_CODES)).cast(pl.Int8).alias("h3_cover_is_water"),
    )


def gee_feature_dict_to_df(feature_data: dict) -> pl.DataFrame:
    rows = [
        {"h3_index": h3_index, **values} for h3_index, values in feature_data.items()
    ]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def dataset_signature(df: pl.DataFrame) -> str:
    """Stable identifier so exports can be reused safely on reruns."""
    digest = hashlib.sha1()
    digest.update(str(len(df)).encode("utf-8"))
    for h3_index in df["h3_index"].to_list():
        digest.update(str(h3_index).encode("utf-8"))
    return digest.hexdigest()[:12]


def export_prefix(df: pl.DataFrame) -> str:
    return f"gee_features_{dataset_signature(df)}"


def export_batch_prefix(df: pl.DataFrame, batch_num: int) -> str:
    return f"{export_prefix(df)}_batch_{batch_num:04d}"


def export_batch_csv_glob(df: pl.DataFrame, batch_num: int) -> str:
    return os.path.join(
        GEE_DRIVE_MOUNT_ROOT,
        GEE_DRIVE_FOLDER,
        f"{export_batch_prefix(df, batch_num)}*.csv",
    )


def local_export_batch_csv_glob(df: pl.DataFrame, batch_num: int) -> str:
    return os.path.join(
        LOCAL_GEE_EXPORT_DIR,
        f"{export_batch_prefix(df, batch_num)}*.csv",
    )


def find_exported_batch_csv(df: pl.DataFrame, batch_num: int) -> str | None:
    matches = sorted(
        set(
            glob.glob(local_export_batch_csv_glob(df, batch_num))
            + glob.glob(export_batch_csv_glob(df, batch_num))
        )
    )
    return matches[0] if matches else None


def list_all_export_batch_csvs() -> list[str]:
    local_matches = glob.glob(os.path.join(LOCAL_GEE_EXPORT_DIR, "gee_features_*_batch_*.csv"))
    drive_matches = glob.glob(
        os.path.join(GEE_DRIVE_MOUNT_ROOT, GEE_DRIVE_FOLDER, "gee_features_*_batch_*.csv")
    )
    return sorted(set(local_matches + drive_matches))


def iter_df_batches(df: pl.DataFrame, batch_size: int):
    total_batches = max(1, (len(df) + batch_size - 1) // batch_size)
    for batch_num, offset in enumerate(range(0, len(df), batch_size), start=1):
        yield batch_num, total_batches, df.slice(offset, batch_size)


def export_gee_features_to_drive(
    fc: ee.FeatureCollection,
    feature_image: ee.Image,
    description: str,
    file_name_prefix: str,
) -> object:
    selectors = [
        "h3_index",
        "elevation",
        "slope",
        "aspect",
        "roughness",
        "wind_10m_avg",
        "wind_10m_std",
        "landcover",
        "population_density",
    ]
    sampled = feature_image.sampleRegions(collection=fc, scale=30, geometries=False)
    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description=description,
        folder=GEE_DRIVE_FOLDER,
        fileNamePrefix=file_name_prefix,
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    return task


def wait_for_ee_task(task: object, description: str) -> None:
    started_at = time.time()
    while True:
        status = task.status()
        state = status.get("state")

        if state == "COMPLETED":
            return
        if state in {"FAILED", "CANCELLED"}:
            raise RuntimeError(
                f"Earth Engine export {description} {state.lower()}: {status}"
            )
        if time.time() - started_at > GEE_EXPORT_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Timed out waiting for Earth Engine export {description}"
            )

        print(f"Export {description} status: {state}")
        time.sleep(GEE_EXPORT_POLL_SECONDS)


def find_active_export_task(description: str) -> object | None:
    try:
        tasks = ee.batch.Task.list()
    except Exception:
        return None

    for task in tasks:
        status = task.status()
        if status.get("description") != description:
            continue
        if status.get("state") in {"READY", "RUNNING"}:
            return task
    return None


def load_matching_local_export_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Load any locally downloaded Earth Engine batch CSVs that contain the target H3 cells.
    This is resilient to prefix changes across resumed export runs because matching is done
    by h3_index rather than only by dataset signature.
    """
    csv_paths = list_all_export_batch_csvs()
    if not csv_paths:
        return pl.DataFrame()

    target_indices = df["h3_index"].to_list()
    target_set = set(target_indices)
    frames = []
    coverage_by_prefix = {}

    for csv_path in csv_paths:
        batch_df = normalize_gee_feature_df(pl.read_csv(csv_path))
        if "h3_index" not in batch_df.columns:
            continue

        batch_df = batch_df.unique(subset=["h3_index"], keep="last")
        matched_df = batch_df.filter(pl.col("h3_index").is_in(target_indices))
        if matched_df.is_empty():
            continue

        prefix = os.path.basename(csv_path).split("_batch_")[0]
        coverage_by_prefix[prefix] = coverage_by_prefix.get(prefix, 0) + len(matched_df)
        frames.append(matched_df)

    if not frames:
        return pl.DataFrame()

    pooled_df = pl.concat(frames, how="vertical").unique(subset=["h3_index"], keep="last")
    matched_count = len(pooled_df)
    top_prefixes = sorted(coverage_by_prefix.items(), key=lambda item: item[1], reverse=True)[:3]
    prefix_summary = ", ".join(f"{prefix} ({count})" for prefix, count in top_prefixes)
    print(
        f"Loaded {matched_count}/{len(target_set)} H3 rows from downloaded GEE batch CSVs"
        + (f" using {prefix_summary}" if prefix_summary else "")
    )
    return pooled_df


def load_gee_direct_batch_df(df: pl.DataFrame, feature_image: ee.Image) -> pl.DataFrame:
    """Directly sample GEE in smaller batches when exports are incomplete or unavailable."""
    batch_frames = []
    for batch_num, total_batches, batch_df in iter_df_batches(df, GEE_DIRECT_SAMPLE_MAX_ROWS):
        print(
            f"[GEE direct {batch_num}/{total_batches}] Sampling {len(batch_df)} H3 cells directly"
        )
        fc = centroids_to_ee_fc(batch_df)
        batch_result = normalize_gee_feature_df(
            gee_feature_dict_to_df(sample_gee_features(fc, feature_image))
        )
        batch_result = batch_result.unique(subset=["h3_index"], keep="last")
        batch_frames.append(batch_result)

    return pl.concat(batch_frames, how="vertical") if batch_frames else pl.DataFrame()


def load_gee_export_df(df: pl.DataFrame, feature_image: ee.Image) -> pl.DataFrame:
    if len(df) <= GEE_DIRECT_SAMPLE_MAX_ROWS:
        fc = centroids_to_ee_fc(df)
        print(f"Using direct Earth Engine sampling for {len(df)} unique H3 cells")
        return normalize_gee_feature_df(
            gee_feature_dict_to_df(sample_gee_features(fc, feature_image))
        )

    pooled_local_df = load_matching_local_export_df(df)
    if not pooled_local_df.is_empty():
        matched_indices = set(pooled_local_df["h3_index"].to_list())
        target_indices = set(df["h3_index"].to_list())
        if matched_indices >= target_indices:
            print("Using downloaded GEE batch CSVs matched by h3_index.")
            return pooled_local_df
        missing_indices = [h3_index for h3_index in df["h3_index"].to_list() if h3_index not in matched_indices]
        print(
            f"Downloaded GEE batch CSVs cover {len(matched_indices)}/{len(target_indices)} "
            "target H3 cells."
        )
        if not os.path.exists(GEE_DRIVE_MOUNT_ROOT):
            print(
                f"Drive mount {GEE_DRIVE_MOUNT_ROOT} not found. Sampling the remaining "
                f"{len(missing_indices)} H3 cells directly from Earth Engine."
            )
            missing_df = df.filter(pl.col("h3_index").is_in(missing_indices))
            missing_gee_df = load_gee_direct_batch_df(missing_df, feature_image)
            return pl.concat([pooled_local_df, missing_gee_df], how="vertical").unique(
                subset=["h3_index"], keep="last"
            )
        print(
            "Falling back to exact batch matching/export for the remaining H3 cells."
        )

    elif not os.path.exists(GEE_DRIVE_MOUNT_ROOT):
        print(
            f"No downloaded GEE batch CSVs matched and Drive mount {GEE_DRIVE_MOUNT_ROOT} "
            "is unavailable. Sampling all H3 cells directly from Earth Engine."
        )
        return load_gee_direct_batch_df(df, feature_image)

    batch_frames = []
    for batch_num, total_batches, batch_df in iter_df_batches(
        df, GEE_EXPORT_BATCH_SIZE
    ):
        checkpoint_df = load_batch_checkpoint(df, batch_num)
        if checkpoint_df is not None:
            print(
                f"[GEE {batch_num}/{total_batches}] Loaded local checkpoint "
                f"{batch_checkpoint_path(df, batch_num)}"
            )
            batch_frames.append(checkpoint_df)
            continue

        exported_csv = find_exported_batch_csv(df, batch_num)
        if exported_csv:
            print(
                f"[GEE {batch_num}/{total_batches}] Loaded existing Drive export from {exported_csv}"
            )
            batch_result = normalize_gee_feature_df(pl.read_csv(exported_csv))
            batch_result = batch_result.unique(subset=["h3_index"], keep="last")
            save_batch_checkpoint(df, batch_num, batch_result)
            batch_frames.append(batch_result)
            continue

        if batch_num < GEE_RESUME_FROM_BATCH:
            print(
                f"[GEE {batch_num}/{total_batches}] Skipping missing batch before "
                f"GEE_RESUME_FROM_BATCH={GEE_RESUME_FROM_BATCH}"
            )
            continue
        if GEE_STOP_AFTER_BATCH and batch_num > GEE_STOP_AFTER_BATCH:
            print(
                f"[GEE] Stopping before batch {batch_num} due to GEE_STOP_AFTER_BATCH={GEE_STOP_AFTER_BATCH}"
            )
            break

        if not os.path.exists(GEE_DRIVE_MOUNT_ROOT):
            raise FileNotFoundError(
                f"No local checkpoint or exported CSV found for batch {batch_num}. "
                f"Expected a downloaded CSV in {LOCAL_GEE_EXPORT_DIR} or a Drive mount at "
                f"{GEE_DRIVE_MOUNT_ROOT}."
            )

        description = export_batch_prefix(df, batch_num)
        fc = centroids_to_ee_fc(batch_df)
        task = find_active_export_task(description)
        if task is None:
            print(
                f"[GEE {batch_num}/{total_batches}] Submitting Earth Engine export "
                f"{description} ({len(batch_df)} H3 cells) to Drive folder {GEE_DRIVE_FOLDER}"
            )
            task = export_gee_features_to_drive(
                fc, feature_image, description, description
            )
        else:
            print(
                f"[GEE {batch_num}/{total_batches}] Reusing active Earth Engine export task {description}"
            )
        wait_for_ee_task(task, description)

        started_at = time.time()
        while True:
            exported_csv = find_exported_batch_csv(df, batch_num)
            if exported_csv:
                print(
                    f"[GEE {batch_num}/{total_batches}] Found exported CSV at {exported_csv}"
                )
                batch_result = normalize_gee_feature_df(pl.read_csv(exported_csv))
                batch_result = batch_result.unique(subset=["h3_index"], keep="last")
                save_batch_checkpoint(df, batch_num, batch_result)
                batch_frames.append(batch_result)
                break
            if time.time() - started_at > 600:
                raise TimeoutError(
                    f"Earth Engine batch {description} completed but exported CSV did not appear in Drive"
                )
            time.sleep(5)

    return pl.concat(batch_frames, how="vertical") if batch_frames else pl.DataFrame()


def batch_checkpoint_path(df: pl.DataFrame, batch_num: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{export_batch_prefix(df, batch_num)}.parquet")


def save_batch_checkpoint(
    df: pl.DataFrame, batch_num: int, batch_df: pl.DataFrame
) -> None:
    batch_df.write_parquet(batch_checkpoint_path(df, batch_num))


def load_batch_checkpoint(df: pl.DataFrame, batch_num: int) -> pl.DataFrame | None:
    path = batch_checkpoint_path(df, batch_num)
    if not os.path.exists(path):
        return None
    return pl.read_parquet(path)


def materialize_gee_dicts(gee_df: pl.DataFrame) -> tuple[dict, dict, dict, dict]:
    elev_data = {}
    wind_data = {}
    lc_data = {}
    pop_data = {}

    if gee_df.is_empty():
        return elev_data, wind_data, lc_data, pop_data

    for row in gee_df.iter_rows(named=True):
        h3_index = row["h3_index"]
        elev_data[h3_index] = {
            "h3_elev_mean": row.get("h3_elev_mean"),
            "h3_slope_mean_deg": row.get("h3_slope_mean_deg"),
            "h3_aspect_mean_deg": row.get("h3_aspect_mean_deg"),
            "h3_roughness": row.get("h3_roughness"),
        }
        wind_data[h3_index] = {
            "h3_wind_10m_avg": row.get("h3_wind_10m_avg"),
            "h3_wind_10m_std": row.get("h3_wind_10m_std"),
        }
        lc_data[h3_index] = {
            "h3_land_cover_code": row.get("h3_land_cover_code"),
            "h3_cover_is_forest": row.get("h3_cover_is_forest"),
            "h3_cover_is_crop": row.get("h3_cover_is_crop"),
            "h3_cover_is_urban": row.get("h3_cover_is_urban"),
            "h3_cover_is_grass": row.get("h3_cover_is_grass"),
            "h3_cover_is_water": row.get("h3_cover_is_water"),
        }
        pop_data[h3_index] = {
            "h3_pop_density_km2": row.get("h3_pop_density_km2"),
        }

    return elev_data, wind_data, lc_data, pop_data


def single_checkpoint_path(name: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{name}.parquet")


def save_single_checkpoint(name: str, data: dict) -> None:
    rows = [{"h3_index": h3_index, **values} for h3_index, values in data.items()]
    pl.DataFrame(rows).write_parquet(single_checkpoint_path(name))


def load_single_checkpoint(name: str) -> dict | None:
    path = single_checkpoint_path(name)
    if not os.path.exists(path):
        return None

    df = pl.read_parquet(path)
    out = {}
    for row in df.iter_rows(named=True):
        h3_index = row["h3_index"]
        out[h3_index] = {k: v for k, v in row.items() if k != "h3_index"}
    return out


def load_or_compute_checkpoint(
    name: str,
    df: pl.DataFrame,
    compute_fn,
    label: str,
) -> dict:
    cached = load_single_checkpoint(name) or {}
    h3_indices = df["h3_index"].to_list()
    missing_indices = [h3_index for h3_index in h3_indices if h3_index not in cached]

    if missing_indices:
        print(f"Computing {label} for {len(missing_indices)} missing H3 cells")
        missing_df = df.filter(pl.col("h3_index").is_in(missing_indices))
        cached.update(compute_fn(missing_df))
        save_single_checkpoint(name, cached)
    else:
        print(f"Loaded {label} checkpoint from {single_checkpoint_path(name)}")

    return {h3_index: cached[h3_index] for h3_index in h3_indices if h3_index in cached}


def fetch_json_with_retries(url: str, label: str) -> dict:
    last_error = None
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            try:
                return resp.json()
            except requests.exceptions.JSONDecodeError as exc:
                snippet = resp.text[:200].strip().replace("\n", " ")
                raise RuntimeError(
                    f"{label} returned non-JSON response "
                    f"(status {resp.status_code}): {snippet or '<empty response>'}"
                ) from exc
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                raise
            sleep_s = 2**attempt
            print(f"{label} request failed on attempt {attempt + 1}/4; retrying in {sleep_s}s")
            time.sleep(sleep_s)
    raise last_error


# %% Cell 7: Distance to transmission lines (HIFLD — local download + KDTree)
def fetch_transmission_distances(df: pl.DataFrame) -> dict:
    """
    Download HIFLD transmission lines GeoJSON and compute distance
    from each H3 centroid to the nearest line vertex using a KDTree.
    """
    hifld_path = os.path.join(DATA_DIR, "transmission_lines.geojson")

    if not os.path.exists(hifld_path):
        print("Downloading HIFLD transmission lines (this may take a minute)...")
        url = (
            "https://services1.arcgis.com/Hp6G80Pky0om6HgA/arcgis/rest/services/"
            "Transmission_Lines/FeatureServer/0/query"
            "?where=1%3D1&outFields=*&f=geojson&resultRecordCount=50000"
        )
        # Paginate to get all features
        all_features = []
        offset = 0
        while True:
            data = fetch_json_with_retries(
                url + f"&resultOffset={offset}",
                "HIFLD transmission lines",
            )
            feats = data.get("features", [])
            if not feats:
                break
            all_features.extend(feats)
            print(f"  Downloaded {len(all_features)} features...")
            offset += len(feats)
            if len(feats) < 50000:
                break

        if not all_features:
            raise RuntimeError(
                "HIFLD transmission line download returned zero features. "
                f"If you already downloaded the dataset manually, place it at {hifld_path} and rerun."
            )

        geojson = {"type": "FeatureCollection", "features": all_features}
        with open(hifld_path, "w") as f:
            json.dump(geojson, f)
        print(f"Saved {len(all_features)} transmission line features.")

    print("Loading transmission lines...")
    gdf = gpd.read_file(hifld_path)

    # Extract all line vertices as points
    coords = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                coords.extend(line.coords)
        elif geom.geom_type == "LineString":
            coords.extend(geom.coords)
    coords = np.array(coords)  # shape (N, 2) in (lng, lat)
    if len(coords) == 0:
        raise ValueError(f"No transmission line coordinates were found in {hifld_path}")
    print(f"  {len(coords)} transmission line vertices")

    # Build KDTree in radians for haversine-approximate distances
    coords_rad = np.radians(coords[:, ::-1])  # (lat, lng) in radians
    tree = cKDTree(coords_rad)

    # Query all H3 centroids
    lats = df["lat"].to_numpy()
    lngs = df["lng"].to_numpy()
    query_rad = np.radians(np.column_stack([lats, lngs]))
    dists, _ = tree.query(query_rad)

    # Convert radian distance to km (Earth radius ~6371 km)
    dists_km = dists * 6371.0

    h3_indices = df["h3_index"].to_list()
    return {
        h3_indices[i]: {"h3_dist_to_transmission_km": float(dists_km[i])}
        for i in range(len(h3_indices))
    }


# %% Cell 8: Distance to major roads (TIGER primary roads — local download + KDTree)
def fetch_road_distances(df: pl.DataFrame) -> dict:
    """
    Download TIGER/Line primary roads shapefile and compute distance
    from each H3 centroid to the nearest road vertex.
    """
    roads_dir = os.path.join(DATA_DIR, "tiger_roads")
    roads_shp = os.path.join(roads_dir, "tl_2023_us_primaryroads.shp")

    if not os.path.exists(roads_shp):
        print("Downloading TIGER primary roads shapefile...")
        os.makedirs(roads_dir, exist_ok=True)
        url = "https://www2.census.gov/geo/tiger/TIGER2023/PRIMARYROADS/tl_2023_us_primaryroads.zip"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        zip_path = os.path.join(roads_dir, "roads.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(roads_dir)
        print("Extracted roads shapefile.")

    print("Loading roads...")
    gdf = gpd.read_file(roads_shp)

    coords = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                coords.extend(line.coords)
        elif geom.geom_type == "LineString":
            coords.extend(geom.coords)
    coords = np.array(coords)
    print(f"  {len(coords)} road vertices")

    coords_rad = np.radians(coords[:, ::-1])  # (lat, lng) in radians
    tree = cKDTree(coords_rad)

    lats = df["lat"].to_numpy()
    lngs = df["lng"].to_numpy()
    query_rad = np.radians(np.column_stack([lats, lngs]))
    dists, _ = tree.query(query_rad)
    dists_km = dists * 6371.0

    h3_indices = df["h3_index"].to_list()
    return {
        h3_indices[i]: {"h3_dist_to_major_road_km": float(dists_km[i])}
        for i in range(len(h3_indices))
    }


# %% Cell 9: Run all enrichments and join to DataFrame


def enrich_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Run all enrichment steps and join results into the DataFrame."""
    unique_df = unique_h3_centroids(df)
    feature_image = build_gee_feature_image()
    print(
        f"=== GEE enrichment: {len(unique_df)} unique H3 cells from {len(df)} rows ==="
    )
    gee_df = load_gee_export_df(unique_df, feature_image)
    gee_df = gee_df.unique(subset=["h3_index"], keep="last")
    elev_data, wind_data, lc_data, pop_data = materialize_gee_dicts(gee_df)

    # --- Local enrichments (no batching needed) ---
    print("\n=== 5/6 Distance to Transmission Lines (HIFLD) ===")
    trans_data = load_or_compute_checkpoint(
        "transmission",
        unique_df,
        fetch_transmission_distances,
        "transmission distances",
    )

    print("\n=== 6/6 Distance to Major Roads (TIGER) ===")
    road_data = load_or_compute_checkpoint(
        "roads",
        unique_df,
        fetch_road_distances,
        "road distances",
    )

    # Merge all dicts into one row per unique h3_index, then join back to the full frame.
    all_indices = unique_df["h3_index"].to_list()
    records = []
    for idx in tqdm(all_indices, desc="Merging features"):
        row = {"h3_index": idx}
        for d in [elev_data, wind_data, lc_data, pop_data, trans_data, road_data]:
            row.update(d.get(idx, {}))
        records.append(row)

    exog_df = pl.DataFrame(records)

    # Join to original df
    enriched = df.join(exog_df, on="h3_index", how="left")
    print(f"\nDone! Shape: {enriched.shape}")
    print(f"Columns: {enriched.columns}")
    return enriched


# %% Cell 10: Execute
# Uncomment and run:
# df = wind_df  # or solar_df
# enriched_df = enrich_dataframe(df)
# enriched_df.head()
