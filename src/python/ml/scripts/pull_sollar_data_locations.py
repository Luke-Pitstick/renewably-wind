import ee
import geemap
import json

ee.Authenticate()
ee.Initialize()

# Load datasets
wind = ee.FeatureCollection("projects/sat-io/open-datasets/GRW/WIND_V1")
solar = ee.FeatureCollection("projects/sat-io/open-datasets/GRW/SOLAR_V1")

# Filter to continental US
us_fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(
    ee.Filter.eq("country_na", "United States")
)
continental_us = us_fc.geometry()
wind_us = wind.filterBounds(continental_us)
solar_us = solar.filterBounds(continental_us)

# --- Option 1: Direct download (works if dataset is small enough) ---
try:
    print("Attempting direct download of solar data...")
    solar_geojson = solar_us.getInfo()
    with open("../data/solar_us_farms.geojson", "w") as f:
        json.dump(solar_geojson, f)
    print("Saved solar_us_farms.geojson")

    print("Attempting direct download of wind data...")
    wind_geojson = wind_us.getInfo()
    with open("../data/wind_us_farms.geojson", "w") as f:
        json.dump(wind_geojson, f)
    print("Saved wind_us_farms.geojson")

except Exception as e:
    print(f"Direct download failed ({e}), falling back to Google Drive export...")

    # --- Option 2: Export to Google Drive ---
    task_solar = ee.batch.Export.table.toDrive(
        collection=solar_us,
        description="solar_us_farms",
        fileFormat="GeoJSON",
    )
    task_solar.start()
    print("Started export: solar_us_farms -> Google Drive")

    task_wind = ee.batch.Export.table.toDrive(
        collection=wind_us,
        description="wind_us_farms",
        fileFormat="GeoJSON",
    )
    task_wind.start()
    print("Started export: wind_us_farms -> Google Drive")

    print("Check your Google Drive for the exported files once tasks complete.")
    print("Monitor at: https://code.earthengine.google.com/tasks")
