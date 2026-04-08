# Renewably Frontend

Renewably is an interactive renewable energy optimization studio for exploring solar and wind resource layers, selecting target areas, and generating recommended build sites.

Live site: [renewably-1.onrender.com](https://renewably-1.onrender.com)

![Renewably application screenshot](./public/picofapp.png)

## Features

- Search for a city, state, or address and jump the map to that area.
- Toggle solar, wind, farms, topography, and transmission context layers.
- Draw a polygon and optimize for either budget or power output.
- Review selected sites and export the result as GeoJSON.

## Local development

```bash
npm install
npm run dev
```

## Production build

```bash
npm run build
```

## Render deployment

This frontend is configured for Render as a static site.

- Service name: `renewably-frontend`
- Root directory: `src/app/frontend`
- Build command: `npm ci --include=dev && npm run build`
- Publish directory: `dist`

The Render blueprint for this app lives in [`render.yaml`](./render.yaml).
