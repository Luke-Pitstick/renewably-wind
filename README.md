# Renewably

Renewably is a renewable energy optimization platform for exploring wind and solar suitability, selecting candidate development areas, and generating optimized site recommendations.

Live app: [https://renewably-1.onrender.com](https://renewably-1.onrender.com)

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
