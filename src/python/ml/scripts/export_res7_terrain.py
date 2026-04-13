from __future__ import annotations

import os
import time
import ee


POINTS_ASSET = os.getenv(
    "RES7_POINTS_ASSET",
    "projects/renewably/assets/res7_points",
)

ASSET_30M = os.getenv(
    "RES7_TERRAIN_30M_ASSET",
    "projects/renewably/assets/res7_terrain_30m",
)

ASSET_COARSE = os.getenv(
    "RES7_TERRAIN_COARSE_ASSET",
    "projects/renewably/assets/res7_terrain_coarse",
)

EE_PROJECT = os.getenv("EE_PROJECT", "renewably")
DRIVE_EXPORT = os.getenv("DRIVE_EXPORT", "0") == "1"
N_SHARDS = int(os.getenv("RES7_N_SHARDS", "12"))


def build_stack_30m() -> ee.Image:
    nlcd = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").first()

    land_cover = nlcd.select("landcover").rename("land_type")
    impervious = nlcd.select("impervious").rename("impervious")

    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("elevation_m")
    slope = ee.Terrain.slope(elevation).rename("slope_deg")

    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate("2020-01-01", "2023-12-31")
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
    )

    def to_wind_speed(img: ee.Image) -> ee.Image:
        return img.expression(
            "sqrt(u*u + v*v)",
            {
                "u": img.select("u_component_of_wind_10m"),
                "v": img.select("v_component_of_wind_10m"),
            },
        ).rename("wind_speed")

    mean_wind = era5.map(to_wind_speed).mean()

    soil = (
        ee.Image("OpenLandMap/SOL/SOL_GRTGROUP_USDA-SOILTAX_C/v01")
        .select("grtgroup")
        .rename("soil_type")
    )

    return ee.Image.cat([
        land_cover,
        impervious,
        elevation,
        slope,
        soil,
        mean_wind,
    ])


def _fc_indicator_image(
    fc: ee.FeatureCollection,
    prop_name: str,
    band_name: str,
    crs: ee.Projection,
    scale: int = 250,
) -> ee.Image:
    fc = fc.map(lambda f: ee.Feature(f).set(prop_name, 1))
    return (
        fc.reduceToImage(properties=[prop_name], reducer=ee.Reducer.first())
        .unmask(0)
        .gt(0)
        .rename(band_name)
        .reproject(crs=crs, scale=scale)
    )


def build_stack_coarse() -> ee.Image:
    nlcd = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").first()
    crs = nlcd.select("landcover").projection()

    padus = ee.FeatureCollection("USGS/GAP/PAD-US/v20/designation")
    protected_area = _fc_indicator_image(
        padus, prop_name="pad_flag", band_name="protected_area", crs=crs, scale=250
    )

    wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons")
    in_wdpa = _fc_indicator_image(
        wdpa, prop_name="wdpa_flag", band_name="in_wdpa", crs=crs, scale=250
    )

    pop_img = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020")
    pop_density = (
        pop_img.select("population_count")
        .rename("pop_density")
        .updateMask(pop_img.select("population_count").gte(0))
        .reproject(crs=crs, scale=250)
    )

    return ee.Image.cat([protected_area, in_wdpa, pop_density])


def shard_points(fc: ee.FeatureCollection, n_shards: int) -> ee.FeatureCollection:
    return (
        fc.randomColumn(columnName="rand", seed=42, rowKeys=["system:index"])
        .map(
            lambda f: f.set(
                "shard",
                ee.Number(f.get("rand")).multiply(n_shards).floor().int()
            )
        )
    )


def start_task(
    stack: ee.Image,
    points: ee.FeatureCollection,
    scale: int,
    description: str,
    asset_id: str,
):
    sampled = stack.sampleRegions(
        collection=points,
        scale=scale,
        geometries=False,
        tileScale=16,
    )

    if DRIVE_EXPORT:
        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=description,
            folder="renewably_exports",
            fileNamePrefix=description,
            fileFormat="CSV",
        )
    else:
        task = ee.batch.Export.table.toAsset(
            collection=sampled,
            description=description,
            assetId=asset_id,
        )

    task.start()
    return task


def run() -> None:
    ee.Initialize(project=EE_PROJECT)

    points = ee.FeatureCollection(POINTS_ASSET)
    points = shard_points(points, N_SHARDS)

    print("Points asset:", POINTS_ASSET)
    print("Project:", EE_PROJECT)
    print("Shards:", N_SHARDS)
    print("Drive export:", DRIVE_EXPORT)

    stack_30m = build_stack_30m()
    stack_coarse = build_stack_coarse()

    tasks = []

    for i in range(N_SHARDS):
        shard = points.filter(ee.Filter.eq("shard", i))
        print(f"Starting shard {i}")

        tasks.append(
            start_task(
                stack_30m,
                shard,
                30,
                f"res7_terrain_30m_shard{i}",
                f"{ASSET_30M}_shard{i}",
            )
        )

        tasks.append(
            start_task(
                stack_coarse,
                shard,
                250,
                f"res7_terrain_coarse_shard{i}",
                f"{ASSET_COARSE}_shard{i}",
            )
        )

    print("\nAll tasks submitted.")
    print("Run: earthengine task list")

    time.sleep(2)

    for task in tasks:
        status = task.status()
        print(f"{status.get('description')} : {status.get('state')}")


if __name__ == "__main__":
    run()