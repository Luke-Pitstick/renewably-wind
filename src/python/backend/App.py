"""Wind-turbine viability backend (Modal + FastAPI).

Single-model pipeline:
    res-8 H3 cell -> feature_engine.compute_features_for_cells
                  -> wind_xgboost_v2.predict_proba
                  -> apply_siting_constraints (zeroes out obviously bad land)

Exposed endpoints:
    GET  /health
    POST /predict/wind        body: {lat, lon}
    POST /optimize            body: {mode, target_value, bounding_box, polygon?}
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import modal
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import feature_engine

app = modal.App(os.getenv("MODAL_APP_NAME", "energy-predictor-v2"))

# ---------- Paths & Modal image ----------
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
MODEL_DIR = REPO_ROOT / "ml" / "models"
ML_DATA_DIR = REPO_ROOT / "ml" / "data"

WIND_MODEL_FILE = MODEL_DIR / "wind_xgboost_v2.pkl"
FEATURE_COLUMNS_FILE = MODEL_DIR / "wind_v2_feature_columns.json"
TERRAIN_CACHE_FILE = ML_DATA_DIR / "renewably_exports" / "terrain_res7_cache.csv"
WIND_DATASET_FILE = ML_DATA_DIR / "us_wind_speed_dataset_2022.csv"
TRANSMISSION_FILE = BACKEND_DIR / "data" / "us_transmission_lines.geojson"
AIRPORT_FILE = BACKEND_DIR / "data" / "airports.geojson"
ROAD_SHP = BACKEND_DIR / "data" / "tl_2023_us_primaryroads.shp"
ROAD_SHX = BACKEND_DIR / "data" / "tl_2023_us_primaryroads.shx"
ROAD_DBF = BACKEND_DIR / "data" / "tl_2023_us_primaryroads.dbf"
ROAD_PRJ = BACKEND_DIR / "data" / "tl_2023_us_primaryroads.prj"

VOLUME_WIND_MODEL_PATH = "/models/wind_xgboost_v2.pkl"
FALLBACK_WIND_MODEL_PATH = "/seed_models/wind_xgboost_v2.pkl"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "pydantic",
        "joblib",
        "xgboost",
        "scikit-learn",
        "pandas",
        "numpy",
        "scipy",
        "h3",
        "geopandas",
        "shapely",
        "pyproj",
    )
    .add_local_file(
        local_path=WIND_MODEL_FILE, remote_path=FALLBACK_WIND_MODEL_PATH
    )
    .add_local_file(
        local_path=FEATURE_COLUMNS_FILE,
        remote_path="/seed_models/wind_v2_feature_columns.json",
    )
    .add_local_file(
        local_path=TERRAIN_CACHE_FILE,
        remote_path="/seed_feature_cache/terrain_res7_cache.csv",
    )
    .add_local_file(
        local_path=WIND_DATASET_FILE,
        remote_path="/seed_data/us_wind_speed_dataset_2022.csv",
    )
    .add_local_file(
        local_path=TRANSMISSION_FILE,
        remote_path="/seed_data/us_transmission_lines.geojson",
    )
    .add_local_file(
        local_path=AIRPORT_FILE, remote_path="/seed_data/airports.geojson"
    )
    .add_local_file(
        local_path=ROAD_SHP, remote_path="/seed_data/roads/tl_2023_us_primaryroads.shp"
    )
    .add_local_file(
        local_path=ROAD_SHX, remote_path="/seed_data/roads/tl_2023_us_primaryroads.shx"
    )
    .add_local_file(
        local_path=ROAD_DBF, remote_path="/seed_data/roads/tl_2023_us_primaryroads.dbf"
    )
    .add_local_file(
        local_path=ROAD_PRJ, remote_path="/seed_data/roads/tl_2023_us_primaryroads.prj"
    )
    .add_local_python_source("feature_engine")
)

volume = modal.Volume.from_name(
    os.getenv("MODAL_VOLUME_NAME", "energy-models-v2"),
    create_if_missing=True,
)

# ---------- FastAPI ----------
web_app = FastAPI()
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://renewably-wind.onrender.com",
]
extra_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
if extra_cors_origins:
    allowed_origins.extend(
        origin.strip()
        for origin in extra_cors_origins.split(",")
        if origin.strip()
    )
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(set(allowed_origins)),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Constants ----------
# NREL ATB land-based wind representative cost.
WIND_CAPEX_USD_PER_KW = 1370.0
WIND_FIXED_OM_USD_PER_KW_YEAR = 39.0
WIND_LIFETIME_YEARS = 30
WIND_LIFETIME_COST_USD_PER_KW = (
    WIND_CAPEX_USD_PER_KW + WIND_FIXED_OM_USD_PER_KW_YEAR * WIND_LIFETIME_YEARS
)
WIND_TURBINE_RATED_KW = 3000.0
HOURS_PER_YEAR = 8760.0
MIN_MODEL_PROBABILITY = 0.05
H3_RESOLUTION = feature_engine.H3_INFERENCE_RES

# ---------- Weibull / power curve helpers (unchanged) ----------
def _weibull_pdf(v: np.ndarray, k: float, c: float) -> np.ndarray:
    return (k / c) * (v / c) ** (k - 1) * np.exp(-(v / c) ** k)


def _adjust_to_hub_height(v_ref: float, z_ref: float = 10, z_hub: float = 100,
                          alpha: float = 0.14) -> float:
    return v_ref * (z_hub / z_ref) ** alpha


def _simple_power_curve(v: float, cut_in: float = 3.0, rated: float = 12.0,
                        cut_out: float = 25.0,
                        rated_power: float = WIND_TURBINE_RATED_KW) -> float:
    if v < cut_in or v >= cut_out:
        return 0.0
    if v < rated:
        return rated_power * ((v**3 - cut_in**3) / (rated**3 - cut_in**3))
    return rated_power


def estimate_aep_from_mean_speed(mean_speed_10m: float, k: float = 2.0,
                                 z_ref: float = 10, z_hub: float = 100,
                                 alpha: float = 0.14,
                                 rated_power_kw: float = WIND_TURBINE_RATED_KW) -> dict[str, float]:
    mean_speed_hub = _adjust_to_hub_height(mean_speed_10m, z_ref, z_hub, alpha)
    c = mean_speed_hub / math.gamma(1 + 1 / k)
    v = np.linspace(0, 40, 4001)
    pdf = _weibull_pdf(v, k, c)
    power = np.array(
        [_simple_power_curve(float(x), rated_power=rated_power_kw) for x in v]
    )
    avg_power_kw = np.trapezoid(power * pdf, v)
    aep_kwh = avg_power_kw * 8760
    return {
        "mean_speed_hub_mps": mean_speed_hub,
        "average_power_kw": float(avg_power_kw),
        "annual_energy_kwh": float(aep_kwh),
        "capacity_factor": float(avg_power_kw / rated_power_kw),
    }


@lru_cache(maxsize=4096)
def _wind_energy_from_speed(mean_speed_10m: float) -> float:
    return float(
        estimate_aep_from_mean_speed(mean_speed_10m=max(mean_speed_10m, 0.0))[
            "annual_energy_kwh"
        ]
    )


def _convert_wind_to_power_kwh(wind_values: np.ndarray) -> np.ndarray:
    """Convert array of 10m mean wind speeds (m/s) to avg hourly kWh per turbine."""
    rounded = np.round(np.maximum(wind_values, 0.0), 2)
    return np.array(
        [
            _wind_energy_from_speed(float(v)) / HOURS_PER_YEAR
            for v in rounded
        ],
        dtype=float,
    )


# ---------- Request models ----------
class BoundingBoxRequest(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class PolygonRequest(BaseModel):
    rings: list[list[list[float]]]


class WindRequest(BaseModel):
    lat: float
    lon: float


class OptimizationRequest(BaseModel):
    mode: Literal["cash", "power"]
    target_value: float = Field(gt=0)
    bounding_box: BoundingBoxRequest
    polygon: PolygonRequest | None = None


# ---------- H3 selection helpers ----------
def _load_h3():
    import h3
    return h3


def _bbox_to_h3_cells(bbox: BoundingBoxRequest) -> list[str]:
    h3 = _load_h3()
    outer_ring = [
        (bbox.ymin, bbox.xmin),
        (bbox.ymin, bbox.xmax),
        (bbox.ymax, bbox.xmax),
        (bbox.ymax, bbox.xmin),
    ]
    return sorted(h3.polygon_to_cells(h3.LatLngPoly(outer_ring), res=H3_RESOLUTION))


def _selection_to_h3_cells(bbox: BoundingBoxRequest,
                           polygon: PolygonRequest | None = None) -> list[str]:
    h3 = _load_h3()
    if polygon is None or not polygon.rings:
        return _bbox_to_h3_cells(bbox)
    exterior = [(lat, lon) for lon, lat, *_ in polygon.rings[0]]
    holes = [
        [(lat, lon) for lon, lat, *_ in ring]
        for ring in polygon.rings[1:]
    ]
    return sorted(
        h3.polygon_to_cells(h3.LatLngPoly(exterior, *holes), res=H3_RESOLUTION)
    )


# ---------- Scoring / selection ----------
def _score_cells(df: pd.DataFrame, feature_columns: list[str],
                 wind_model) -> pd.DataFrame:
    """Adds turbine_probability, wind_power_kwh, cost_usd columns to df."""
    proba = wind_model.predict_proba(df[feature_columns])[:, 1]
    df = df.copy()
    df["turbine_probability"] = proba.astype(float)
    df = feature_engine.apply_siting_constraints(df)
    df["wind_power_kwh"] = _convert_wind_to_power_kwh(df["wind_speed"].to_numpy())
    df["cost_usd"] = WIND_LIFETIME_COST_USD_PER_KW * WIND_TURBINE_RATED_KW
    df["expected_power_kwh"] = df["turbine_probability"] * df["wind_power_kwh"]
    return df


def _select_budget(df: pd.DataFrame, budget_usd: float) -> tuple[pd.DataFrame, dict]:
    candidates = df[df["expected_power_kwh"] > 0].copy()
    candidates["score"] = candidates["expected_power_kwh"] / candidates["cost_usd"]
    candidates = candidates.sort_values(
        by=["score", "expected_power_kwh"], ascending=[False, False]
    )

    remaining = budget_usd
    chosen_idx = []
    total_cost = 0.0
    total_expected = 0.0
    total_raw = 0.0
    for idx, row in candidates.iterrows():
        cost = float(row["cost_usd"])
        if cost > remaining:
            continue
        remaining -= cost
        total_cost += cost
        total_expected += float(row["expected_power_kwh"])
        total_raw += float(row["wind_power_kwh"])
        chosen_idx.append(idx)

    chosen = candidates.loc[chosen_idx]
    totals = {
        "total_cost_usd": total_cost,
        "total_expected_power_kwh": total_expected,
        "total_raw_power_kwh": total_raw,
    }
    return chosen, totals


def _select_power(df: pd.DataFrame, target_power_kwh: float) -> tuple[pd.DataFrame, dict]:
    candidates = df[df["wind_power_kwh"] > 0].copy()
    floored_prob = candidates["turbine_probability"].clip(lower=MIN_MODEL_PROBABILITY)
    candidates["effective_cost_usd"] = candidates["cost_usd"] / floored_prob
    candidates["score"] = candidates["effective_cost_usd"] / candidates["wind_power_kwh"]
    candidates = candidates.sort_values(
        by=["score", "wind_power_kwh"], ascending=[True, False]
    )

    chosen_idx: list = []
    accumulated_power = 0.0
    total_effective = 0.0
    total_actual = 0.0
    for idx, row in candidates.iterrows():
        chosen_idx.append(idx)
        accumulated_power += float(row["wind_power_kwh"])
        total_effective += float(row["effective_cost_usd"])
        total_actual += float(row["cost_usd"])
        if accumulated_power >= target_power_kwh:
            break

    chosen = candidates.loc[chosen_idx]
    totals = {
        "total_power_kwh": accumulated_power,
        "total_effective_cost_usd": total_effective,
        "total_actual_cost_usd": total_actual,
    }
    return chosen, totals


def _points_payload(df: pd.DataFrame) -> list[dict]:
    df = df.sort_values(by="turbine_probability", ascending=False).copy()
    df["device_type"] = "wind"
    df["solar_power_kwh"] = 0.0
    df["wind_probability"] = df["turbine_probability"]
    df["device_cost_usd"] = df["cost_usd"]
    payload_cols = [
        "h3_index", "lat", "lon",
        "device_type",
        "turbine_probability", "wind_probability",
        "wind_speed",
        "solar_power_kwh", "wind_power_kwh",
        "cost_usd", "device_cost_usd",
        "feature_source",
    ]
    for optional in ("expected_power_kwh", "effective_cost_usd"):
        if optional in df.columns:
            payload_cols.append(optional)
    return df[payload_cols].to_dict(orient="records")


# ---------- ModelService ----------
@app.cls(image=image, volumes={"/models": volume}, scaledown_window=300)
class ModelService:
    @modal.enter()
    def load_models(self):
        model_path = (
            VOLUME_WIND_MODEL_PATH
            if Path(VOLUME_WIND_MODEL_PATH).exists()
            else FALLBACK_WIND_MODEL_PATH
        )
        self.wind_model = joblib.load(model_path)
        self.feature_columns = feature_engine.load_feature_columns()
        feature_engine.warm()

    @modal.method()
    def predict_wind(self, lat: float, lon: float):
        import h3
        cell = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        df = feature_engine.compute_features_for_cells([cell])
        scored = _score_cells(df, self.feature_columns, self.wind_model)
        row = scored.iloc[0]
        return {
            "h3_index": str(row["h3_index"]),
            "turbine_probability": float(row["turbine_probability"]),
            "wind_speed": float(row["wind_speed"]),
            "feature_source": str(row["feature_source"]),
        }

    @modal.method()
    def optimize(
        self,
        mode: Literal["cash", "power"],
        target_value: float,
        bounding_box: dict,
        polygon: dict | None = None,
    ):
        bbox = BoundingBoxRequest(**bounding_box)
        sel_polygon = PolygonRequest(**polygon) if polygon else None
        cells = _selection_to_h3_cells(bbox, sel_polygon)
        if not cells:
            raise HTTPException(
                status_code=400, detail="Selection produced zero H3 cells."
            )

        df = feature_engine.compute_features_for_cells(cells)
        scored = _score_cells(df, self.feature_columns, self.wind_model)

        if mode == "cash":
            chosen, totals = _select_budget(scored, target_value)
        else:
            chosen, totals = _select_power(scored, target_value)

        exclusion_counts = {
            reason: int(count)
            for reason, count in scored["siting_exclusion_reason"]
            .value_counts()
            .items()
            if reason
        }
        urban_excluded = (
            exclusion_counts.get("urban_land_cover", 0)
            + exclusion_counts.get("high_population_density", 0)
            + exclusion_counts.get("impervious_surface", 0)
        )
        siting_notes: list[str] = []
        if urban_excluded:
            siting_notes.append(
                f"{urban_excluded} candidate site(s) were removed because they "
                "fall in urban or densely populated areas. Wind turbines are not "
                "sited near residential populations due to noise, safety, and "
                "zoning restrictions, even when the model scored them favorably."
            )

        debug = {
            "hex_count": len(cells),
            "probability_summary": {
                "min": float(scored["turbine_probability"].min()),
                "max": float(scored["turbine_probability"].max()),
                "mean": float(scored["turbine_probability"].mean()),
                "zeroed_by_siting": int(
                    (scored["turbine_probability"] == 0.0).sum()
                ),
            },
            "wind_speed_summary": {
                "min": float(scored["wind_speed"].min()),
                "max": float(scored["wind_speed"].max()),
                "mean": float(scored["wind_speed"].mean()),
            },
            "siting_exclusions": exclusion_counts,
        }
        print(
            "Optimization debug:",
            json.dumps({"mode": mode, "target_value": target_value, "debug": debug}),
            flush=True,
        )

        return {
            "mode": mode,
            "hex_resolution": H3_RESOLUTION,
            "hex_count": len(cells),
            "selected_count": len(chosen),
            "device_cost_usd": WIND_LIFETIME_COST_USD_PER_KW * WIND_TURBINE_RATED_KW,
            "device_rated_kw": WIND_TURBINE_RATED_KW,
            "points": _points_payload(chosen),
            "siting_notes": siting_notes,
            "siting_exclusions": exclusion_counts,
            "debug": debug,
            **totals,
        }


model_service = ModelService()

# ---------- FastAPI endpoints ----------
@web_app.get("/health")
async def health():
    return {"status": "ok"}


@web_app.post("/predict/wind")
async def predict_wind(req: WindRequest):
    return await model_service.predict_wind.remote.aio(req.lat, req.lon)


@web_app.post("/optimize")
async def optimize(req: OptimizationRequest):
    return await model_service.optimize.remote.aio(
        req.mode,
        req.target_value,
        req.bounding_box.model_dump(),
        req.polygon.model_dump() if req.polygon else None,
    )


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
