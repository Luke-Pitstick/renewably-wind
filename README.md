# Renewably Wind

Renewably is a renewable energy optimization platform for exploring wind and solar suitability, selecting candidate development areas, and generating optimized site recommendations.

Live app: [https://renewably-wind.onrender.com/](https://renewably-wind.onrender.com/)

![Renewably app screenshot](./src/app/frontend/public/picofapp.png)

## Repository layout

- `src/app/frontend`: Vite + React frontend for the interactive map and optimization workflow
- `src/python/backend`: FastAPI and Modal-powered optimization API
- `src/python/ml`: training assets, models, notebooks, and supporting data pipelines

## Frontend highlights

- Search by city, state, or address
- Toggle solar, wind, topography, and infrastructure layers
- Draw a polygon and optimize for budget or target power
- Review recommended sites and export results as GeoJSON
- Frontend created with Claude Code and some CSS/layout tuning by me

## How the wind model works

**Both models were created from scratch by me**
Renewably's wind pipeline has two models that work together to score potential wind sites:

### 1. Wind speed regressor (`wind_xgboost_v2.pkl`)

Predicts annual mean wind speed (m/s) at ~10 m for any point in the contiguous U.S.

- **Source data**: ERA5-Land hourly `u`/`v` wind components pulled from Google Earth Engine, averaged to annual mean wind speed for 2022. Sampled at random CONUS points on a ~9 km grid.
- **Features**: `lat`, `lon`, `elevation` (SRTM).
- **Algorithm**: XGBoost regression (`n_estimators=500`, `max_depth=6`, `learning_rate=0.05`).
- **Test performance**: MAE 0.2562 m/s, RMSE 0.3874 m/s, R² 0.8801.

### 2. Site viability classifier (`viability_model.pkl`)

Predicts the probability that an H3 res-7 cell is suitable for a utility-scale wind turbine (trained on presence/absence of real turbines from the USGS U.S. Wind Turbine Database).

- **Spatial unit**: H3 resolution 7 cells covering CONUS.
- **Exogenous features** (built via the `scripts/` pipeline + Google Earth Engine):
  - Terrain: `elevation_m`, `slope_deg`
  - Land cover: `land_type`, `impervious`, `soil_type`
  - Protection status: `protected_area`, `in_wdpa`
  - Population: `pop_density`
  - Infrastructure distances: `airport_dist_km`, `log_road_dist_km`, `log_transmission_line_dist_km`
  - Predicted `wind_speed` from model 1
  - Engineered interactions: `log_road_dist_km × log_transmission_line_dist_km`, `slope × elevation`, `wind_speed × elevation`, `wind_speed × slope`
- **Algorithm**: XGBoost classifier, tuned with `RandomizedSearchCV` (100 iters, 5-fold CV, `f1_weighted` scoring).
- **Best params**: `max_depth=10`, `learning_rate=0.2`, `n_estimators=500`, `min_child_weight=5`, `reg_lambda=10`, `reg_alpha=0.1`, `subsample=1.0`, `colsample_bytree=1.0`.
- **Test performance** (n=52,375):
  - Accuracy: **0.96** (5-fold CV mean 0.9552)
  - Best CV F1 (weighted): **0.9555**
  - Turbine-class precision 0.90, recall 0.94, F1 0.92
  - Non-turbine precision 0.98, recall 0.97, F1 0.97

### Optimization flow

1. User draws a polygon in the frontend.
2. Backend tiles the polygon into H3 res-7 cells and pulls cached terrain/infrastructure features from `data/renewably_exports/`.
3. Wind speed is predicted for each cell, then fed into the viability classifier.
4. Cells are ranked by viability probability and filtered/packed to meet the user's budget or target-power constraint via a Modal-hosted optimizer.

### Training pipeline (ML assets)

- `ml/notebooks/wind_model_creation.ipynb` — wind speed regressor
- `ml/notebooks/viability/data_collection.ipynb` — GEE feature collection
- `ml/notebooks/viability/generate_h3_cells.ipynb` — H3 cell generation + turbine labelling
- `ml/notebooks/viability/model.ipynb` — classifier training + evaluation
- `ml/scripts/` — batch exporters for caching res-7 terrain and exogenous features

## Development

Frontend:

```bash
cd src/app/frontend
npm install
npm run dev
```

Backend:

```bash
cd src/python
uv sync
uv run python backend/main.py
```
