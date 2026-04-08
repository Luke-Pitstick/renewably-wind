import { memo, useCallback, useEffect, useRef, useState } from 'react'

type ArcGISMapProps = {
  topographyVisible: boolean
  solarVisible: boolean
  windVisible: boolean
  windParticlesVisible: boolean
  solarFarmsVisible: boolean
  windFarmsVisible: boolean
  powerLinesVisible: boolean
  optimizationSites: OptimizationSite[]
  optimizationFocusRequest: {
    id: number
    boundingBox: NumericBbox
  } | null
  boundingBox: {
    xmin: number
    ymin: number
    xmax: number
    ymax: number
  } | null
  boundingBoxSelectionActive: boolean
  editSelectionRequest: {
    id: number
  } | null
  locationSearchRequest: {
    id: number
    query: string
  } | null
  onBoundingBoxSelectionChange: (active: boolean) => void
  onBoundingBoxSelect: (boundingBox: {
    xmin: number
    ymin: number
    xmax: number
    ymax: number
  } | null) => void
  onSelectionPolygonSelect: (polygon: {
    rings: number[][][]
  } | null) => void
}

type ThemeLayerKind =
  | 'topography'
  | 'solar'
  | 'wind'
  | 'solarFarms'
  | 'windFarms'
  | 'powerLines'

type OptimizationSite = {
  lat: number
  lon: number
  device_type: 'solar' | 'wind'
  h3_index?: string
  solar_power_kwh: number
  wind_power_kwh: number
  solar_probability?: number
  wind_probability?: number
  none_probability?: number
  expected_power_kwh?: number
  selected_power_kwh?: number
  device_cost_usd?: number
  effective_cost_usd?: number
  installed_capacity_kw?: number
  score?: number
  chance_feature_source?: string
}

type LayerHandle = {
  visible: boolean
}

type MapHandle = {
  add: (layer: unknown, index?: number) => void
}

type GraphicsLayerHandle = {
  add: (graphic: unknown) => void
  removeAll: () => void
}

type GraphicHandle = {
  geometry?: {
    extent?: NumericBbox
    rings?: number[][][]
  }
}

type SketchHandle = {
  create: (tool: string) => void
  cancel: () => void
  update: (
    targets: GraphicHandle | GraphicHandle[],
    options?: Record<string, unknown>,
  ) => void
  on: (
    eventName: string,
    callback: (event: Record<string, unknown>) => void,
  ) => {
    remove: () => void
  }
}

type WatchHandle = {
  remove: () => void
}

type ViewHandle = {
  destroy: () => void
  zoom: number
  extent?: Record<string, unknown>
  stationary: boolean
  interacting: boolean
  updating: boolean
  goTo: (target: unknown, options?: { animate?: boolean }) => Promise<unknown>
  toScreen: (target: unknown) => { x: number; y: number } | null
  watch: (propertyName: string, callback: () => void) => WatchHandle
}

type ArcGISPointCtor = new (properties: {
  longitude: number
  latitude: number
}) => unknown

type GraphicCtor = new (properties?: Record<string, unknown>) => GraphicHandle

type MapCtor = new (properties?: Record<string, unknown>) => MapHandle
type MapViewCtor = new (properties?: Record<string, unknown>) => ViewHandle
type GeoJSONLayerCtor = new (properties?: Record<string, unknown>) => LayerHandle
type FeatureLayerCtor = new (properties?: Record<string, unknown>) => LayerHandle
type TileLayerCtor = new (properties?: Record<string, unknown>) => LayerHandle
type GraphicsLayerCtor = new (
  properties?: Record<string, unknown>,
) => GraphicsLayerHandle
type SketchViewModelCtor = new (
  properties?: Record<string, unknown>,
) => SketchHandle

type WebMercatorUtilsHandle = {
  webMercatorToGeographic: (geometry: Record<string, unknown>) => Record<string, unknown>
}

type CoreModules = {
  MapCtor: MapCtor
  MapViewCtor: MapViewCtor
  GeoJSONLayerCtor: GeoJSONLayerCtor
  FeatureLayerCtor: FeatureLayerCtor
  TileLayerCtor: TileLayerCtor
  GraphicsLayerCtor: GraphicsLayerCtor
  SketchViewModelCtor: SketchViewModelCtor
  PointCtor: ArcGISPointCtor
  GraphicCtor: GraphicCtor
  webMercatorUtils: WebMercatorUtilsHandle
}

type LocatorModule = {
  addressToLocations: (
    url: string,
    params: { address: { SingleLine: string } },
  ) => Promise<
    Array<{
      location?: {
        longitude: number
        latitude: number
      }
      extent?: NumericBbox
    }>
  >
}

type ScalarGrid = {
  values: Float32Array
  cols: number
  rows: number
  xmin: number
  ymin: number
  xmax: number
  ymax: number
  dx: number
  dy: number
  min: number
  max: number
}

type NumericBbox = {
  xmin: number
  ymin: number
  xmax: number
  ymax: number
}

type GeoJsonFeatureCollection = {
  features: GeoJsonFeature[]
}

type GeoJsonFeature = {
  properties: Record<string, number>
  geometry: {
    type: string
    coordinates: number[][][]
  }
}

type Particle = {
  lon: number
  lat: number
  age: number
  ttl: number
  speedFactor: number
  turnBias: number
  shimmerPhase: number
}

type WindField = {
  extent: NumericBbox
  cols: number
  rows: number
  u: Float32Array
  v: Float32Array
  intensity: Float32Array
}

type ParticleStyle = {
  glowColor: string
  coreColor: string
  glowWidth: number
  coreWidth: number
}

declare global {
  interface Window {
    require?: (modules: string[], callback: (...loaded: unknown[]) => void) => void
    __arcgisPromise?: Promise<void>
  }
}

const SOLAR_DATA_URL = '/data/us_solar_surface.geojson'
const WIND_DATA_URL = '/data/us_wind_surface.geojson'
const SOLAR_FARMS_DATA_URL = '/data/solar_farm_sites.geojson'
const WIND_FARMS_DATA_URL = '/data/wind_farm_sites.geojson'
const POWER_LINES_DATA_URL =
  'https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0'
const HILLSHADE_DATA_URL =
  'https://services.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer'
const LOCATOR_URL =
  'https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer'
const MAX_WIND_CANVAS_DPR = 1.25
const WIND_RECOVERY_MS = 450
const scalarGridCache = new Map<string, Promise<ScalarGrid>>()
const LOWER_48_VIEW = {
  center: [-98.2, 38.5] as [number, number],
  zoom: 4.55,
}

let coreModulesPromise: Promise<CoreModules> | null = null
let locatorModulePromise: Promise<LocatorModule> | null = null
const SOLAR_FARM_ICON_URL = `data:image/svg+xml;utf8,${encodeURIComponent(
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
    <rect x="8" y="12" width="32" height="18" rx="3" fill="#17312a" stroke="#d8ff6d" stroke-width="2.4"/>
    <path d="M16 12v18M24 12v18M32 12v18M8 18h32M8 24h32" stroke="#8fe2b8" stroke-width="1.8" opacity="0.9"/>
    <path d="M24 30v8" stroke="#d8ff6d" stroke-width="2.4" stroke-linecap="round"/>
    <path d="M18 38h12" stroke="#d8ff6d" stroke-width="2.4" stroke-linecap="round"/>
  </svg>`,
)}`
const WIND_FARM_ICON_URL = `data:image/svg+xml;utf8,${encodeURIComponent(
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
    <path d="M24 17v20" stroke="#dff6ff" stroke-width="2.8" stroke-linecap="round"/>
    <circle cx="24" cy="14" r="2.7" fill="#dff6ff"/>
    <path d="M24 14L10 18" stroke="#8cd6ff" stroke-width="2.8" stroke-linecap="round"/>
    <path d="M24 14L34 7" stroke="#68f0cb" stroke-width="2.8" stroke-linecap="round"/>
    <path d="M24 14L30 28" stroke="#d8ff6d" stroke-width="2.8" stroke-linecap="round"/>
    <path d="M20 39h8" stroke="#dff6ff" stroke-width="2.8" stroke-linecap="round"/>
  </svg>`,
)}`

function createPinMarkerUrl(kind: 'solar' | 'wind') {
  const accent = kind === 'solar' ? '#d8ff6d' : '#9fe7ff'
  const fill = kind === 'solar' ? '#1d3d31' : '#193848'
  const inner =
    kind === 'solar'
      ? `
        <rect x="16" y="15" width="20" height="11" rx="2.2" fill="#17312a" stroke="${accent}" stroke-width="1.8"/>
        <path d="M21 15v11M26 15v11M31 15v11M16 20.5h20" stroke="#8fe2b8" stroke-width="1.35" opacity="0.95"/>
        <path d="M26 26v5.5" stroke="${accent}" stroke-width="1.8" stroke-linecap="round"/>
        <path d="M22.5 31.5h7" stroke="${accent}" stroke-width="1.8" stroke-linecap="round"/>
      `
      : `
        <path d="M26 17v13.5" stroke="#dff6ff" stroke-width="2.1" stroke-linecap="round"/>
        <circle cx="26" cy="15" r="2.1" fill="#dff6ff"/>
        <path d="M26 15L16 18.2" stroke="#8cd6ff" stroke-width="2.1" stroke-linecap="round"/>
        <path d="M26 15L33.2 10" stroke="#68f0cb" stroke-width="2.1" stroke-linecap="round"/>
        <path d="M26 15L30.4 25" stroke="#d8ff6d" stroke-width="2.1" stroke-linecap="round"/>
        <path d="M22.8 33h6.4" stroke="#dff6ff" stroke-width="2.1" stroke-linecap="round"/>
      `

  return `data:image/svg+xml;utf8,${encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 64">
      <defs>
        <filter id="shadow" x="-40%" y="-40%" width="180%" height="180%">
          <feDropShadow dx="0" dy="4" stdDeviation="3.4" flood-color="rgba(0,0,0,0.34)"/>
        </filter>
      </defs>
      <path d="M26 4C15.5 4 7 12.5 7 23c0 14.4 16.2 27.9 18.1 29.5a1.5 1.5 0 0 0 1.8 0C28.8 50.9 45 37.4 45 23 45 12.5 36.5 4 26 4Z" fill="${fill}" stroke="${accent}" stroke-width="2.4" filter="url(#shadow)"/>
      <circle cx="26" cy="23" r="13.5" fill="rgba(10,16,18,0.22)" stroke="rgba(255,255,255,0.14)" stroke-width="1"/>
      ${inner}
    </svg>`,
  )}`
}

const SOLAR_OPTIMIZATION_PIN_URL = createPinMarkerUrl('solar')
const WIND_OPTIMIZATION_PIN_URL = createPinMarkerUrl('wind')

const PARTICLE_STYLE_PALETTE: ParticleStyle[] = Array.from(
  { length: 10 },
  (_, index) => {
    const intensity = index / 9
    const hue = 202 + intensity * 88
    const coreAlpha = 0.34 + intensity * 0.16
    const glowAlpha = 0.045 + intensity * 0.08

    return {
      glowColor: `hsla(${hue.toFixed(0)}, 100%, 74%, ${glowAlpha.toFixed(2)})`,
      coreColor: `hsla(${(hue + 8).toFixed(0)}, 95%, 90%, ${coreAlpha.toFixed(2)})`,
      glowWidth: 2.8 + intensity * 2.2,
      coreWidth: 1.5 + intensity * 1.4,
    }
  },
)

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function normalize(value: number, min: number, max: number) {
  if (max <= min) {
    return 0
  }

  return clamp((value - min) / (max - min), 0, 1)
}

function emphasizeWindIntensity(intensity: number) {
  return clamp(intensity, 0, 1) ** 1.7
}

function loadArcGISApi() {
  if (window.require) {
    return Promise.resolve()
  }

  if (window.__arcgisPromise) {
    return window.__arcgisPromise
  }

  window.__arcgisPromise = new Promise<void>((resolve, reject) => {
    const existingCss = document.querySelector(
      'link[data-arcgis-theme="dark"]',
    ) as HTMLLinkElement | null

    if (!existingCss) {
      const css = document.createElement('link')
      css.rel = 'stylesheet'
      css.href = 'https://js.arcgis.com/4.30/esri/themes/dark/main.css'
      css.dataset.arcgisTheme = 'dark'
      document.head.appendChild(css)
    }

    const existingScript = document.querySelector(
      'script[data-arcgis-api="4.30"]',
    ) as HTMLScriptElement | null

    if (existingScript) {
      existingScript.addEventListener('load', () => resolve(), { once: true })
      existingScript.addEventListener(
        'error',
        () => reject(new Error('Unable to load ArcGIS API')),
        { once: true },
      )
      return
    }

    const script = document.createElement('script')
    script.src = 'https://js.arcgis.com/4.30/'
    script.async = true
    script.dataset.arcgisApi = '4.30'
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('Unable to load ArcGIS API'))
    document.body.appendChild(script)
  })

  return window.__arcgisPromise
}

function requireArcGISModules<T>(moduleNames: string[]) {
  return loadArcGISApi().then(
    () =>
      new Promise<T>((resolve, reject) => {
        if (!window.require) {
          reject(new Error('ArcGIS module loader is unavailable'))
          return
        }

        try {
          window.require(moduleNames, (...loaded: unknown[]) => {
            resolve(loaded as T)
          })
        } catch (error) {
          reject(error)
        }
      }),
  )
}

function loadCoreArcGISModules() {
  if (!coreModulesPromise) {
    coreModulesPromise = requireArcGISModules<
      [
        MapCtor,
        MapViewCtor,
        GeoJSONLayerCtor,
        FeatureLayerCtor,
        TileLayerCtor,
        GraphicsLayerCtor,
        SketchViewModelCtor,
        ArcGISPointCtor,
        GraphicCtor,
        WebMercatorUtilsHandle,
      ]
    >([
      'esri/Map',
      'esri/views/MapView',
      'esri/layers/GeoJSONLayer',
      'esri/layers/FeatureLayer',
      'esri/layers/TileLayer',
      'esri/layers/GraphicsLayer',
      'esri/widgets/Sketch/SketchViewModel',
      'esri/geometry/Point',
      'esri/Graphic',
      'esri/geometry/support/webMercatorUtils',
    ]).then(
      ([
        MapCtor,
        MapViewCtor,
        GeoJSONLayerCtor,
        FeatureLayerCtor,
        TileLayerCtor,
        GraphicsLayerCtor,
        SketchViewModelCtor,
        PointCtor,
        GraphicCtor,
        webMercatorUtils,
      ]) => ({
        MapCtor,
        MapViewCtor,
        GeoJSONLayerCtor,
        FeatureLayerCtor,
        TileLayerCtor,
        GraphicsLayerCtor,
        SketchViewModelCtor,
        PointCtor,
        GraphicCtor,
        webMercatorUtils,
      }),
    )
  }

  return coreModulesPromise
}

function loadLocatorModule() {
  if (!locatorModulePromise) {
    locatorModulePromise = requireArcGISModules<[LocatorModule]>([
      'esri/rest/locator',
    ]).then(([locator]) => locator)
  }

  return locatorModulePromise
}

function gridKey(col: number, row: number, cols: number) {
  return row * cols + col
}

function buildScalarGrid(
  featureCollection: GeoJsonFeatureCollection,
  propertyKey: string,
) {
  const features = featureCollection.features
  const firstFeature = features[0]

  if (!firstFeature) {
    throw new Error(`No features found for ${propertyKey}`)
  }

  const firstRing = firstFeature.geometry.coordinates[0]
  const dx = Math.abs(firstRing[1][0] - firstRing[0][0])
  const dy = Math.abs(firstRing[2][1] - firstRing[1][1])

  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  let xmin = Number.POSITIVE_INFINITY
  let ymin = Number.POSITIVE_INFINITY
  let xmax = Number.NEGATIVE_INFINITY
  let ymax = Number.NEGATIVE_INFINITY

  const centers: Array<{ x: number; y: number; value: number }> = []

  for (const feature of features) {
    const ring = feature.geometry.coordinates[0]
    const centerX = (ring[0][0] + ring[2][0]) / 2
    const centerY = (ring[0][1] + ring[2][1]) / 2
    const value = feature.properties[propertyKey]

    if (Number.isNaN(value)) {
      continue
    }

    min = Math.min(min, value)
    max = Math.max(max, value)
    xmin = Math.min(xmin, centerX)
    ymin = Math.min(ymin, centerY)
    xmax = Math.max(xmax, centerX)
    ymax = Math.max(ymax, centerY)

    centers.push({
      x: centerX,
      y: centerY,
      value,
    })
  }

  const cols = Math.round((xmax - xmin) / dx) + 1
  const rows = Math.round((ymax - ymin) / dy) + 1
  const values = new Float32Array(cols * rows)
  values.fill(Number.NaN)

  for (const center of centers) {
    const col = Math.round((center.x - xmin) / dx)
    const row = Math.round((center.y - ymin) / dy)
    values[gridKey(col, row, cols)] = center.value
  }

  return {
    values,
    cols,
    rows,
    xmin,
    ymin,
    xmax,
    ymax,
    dx,
    dy,
    min,
    max,
  } satisfies ScalarGrid
}

function loadScalarGrid(url: string, propertyKey: string) {
  const cacheKey = `${url}:${propertyKey}`
  const cached = scalarGridCache.get(cacheKey)

  if (cached) {
    return cached
  }

  const promise = fetch(url).then(async (response) => {
    if (!response.ok) {
      throw new Error(`Unable to load ${url}`)
    }

    const json = (await response.json()) as GeoJsonFeatureCollection
    return buildScalarGrid(json, propertyKey)
  })

  scalarGridCache.set(cacheKey, promise)
  return promise
}

function sampleGrid(grid: ScalarGrid, lon: number, lat: number) {
  const fx = (lon - grid.xmin) / grid.dx
  const fy = (lat - grid.ymin) / grid.dy

  if (fx < 0 || fy < 0 || fx > grid.cols - 1 || fy > grid.rows - 1) {
    return null
  }

  const x0 = Math.floor(fx)
  const y0 = Math.floor(fy)
  const x1 = Math.min(x0 + 1, grid.cols - 1)
  const y1 = Math.min(y0 + 1, grid.rows - 1)
  const v00 = grid.values[gridKey(x0, y0, grid.cols)]
  const v10 = grid.values[gridKey(x1, y0, grid.cols)]
  const v01 = grid.values[gridKey(x0, y1, grid.cols)]
  const v11 = grid.values[gridKey(x1, y1, grid.cols)]
  const tx = fx - x0
  const ty = fy - y0

  if (![v00, v10, v01, v11].some((value) => !Number.isNaN(value))) {
    return null
  }

  const fallbackCol = Math.round(fx)
  const fallbackRow = Math.round(fy)
  const fallback = grid.values[gridKey(fallbackCol, fallbackRow, grid.cols)]

  if ([v00, v10, v01, v11].some((value) => Number.isNaN(value))) {
    return Number.isNaN(fallback) ? null : fallback
  }

  const top = v00 + (v10 - v00) * tx
  const bottom = v01 + (v11 - v01) * tx
  return top + (bottom - top) * ty
}

function geographicExtentFromView(
  view: ViewHandle,
  webMercatorUtils: WebMercatorUtilsHandle,
) {
  if (!view.extent) {
    return null
  }

  const geographicExtent = webMercatorUtils.webMercatorToGeographic(view.extent)
  const xmin = geographicExtent.xmin
  const ymin = geographicExtent.ymin
  const xmax = geographicExtent.xmax
  const ymax = geographicExtent.ymax

  if (
    typeof xmin !== 'number' ||
    typeof ymin !== 'number' ||
    typeof xmax !== 'number' ||
    typeof ymax !== 'number'
  ) {
    return null
  }

  return {
    xmin,
    ymin,
    xmax,
    ymax,
  } satisfies NumericBbox
}

function createWindVector(lon: number, lat: number, windGrid: ScalarGrid) {
  const speed = sampleGrid(windGrid, lon, lat)
  if (speed === null) {
    return null
  }

  const intensity = normalize(speed, windGrid.min, windGrid.max)
  const sampleX1 = sampleGrid(windGrid, lon + windGrid.dx, lat) ?? speed
  const sampleX0 = sampleGrid(windGrid, lon - windGrid.dx, lat) ?? speed
  const sampleY1 = sampleGrid(windGrid, lon, lat + windGrid.dy) ?? speed
  const sampleY0 = sampleGrid(windGrid, lon, lat - windGrid.dy) ?? speed
  const gradientX = sampleX1 - sampleX0
  const gradientY = sampleY1 - sampleY0

  let u = 1.25 + intensity * 1.9
  let v = Math.sin((lat - 34) * 0.22) * 0.65

  const prairieJet =
    Math.exp(-((lon + 101.5) ** 2) / 54 - ((lat - 42) ** 2) / 18) * 1.8
  const gulfArc =
    Math.exp(-((lon + 90.5) ** 2) / 42 - ((lat - 29.5) ** 2) / 12) * 1.45
  const atlanticLift =
    Math.exp(-((lon + 74.5) ** 2) / 34 - ((lat - 36.5) ** 2) / 18) * 1.2
  const lakeCurl =
    Math.exp(-((lon + 85.5) ** 2) / 36 - ((lat - 45) ** 2) / 10) * 0.8
  const pacificSweep =
    Math.exp(-((lon + 122.5) ** 2) / 28 - ((lat - 44.5) ** 2) / 22) * 1.15

  u += prairieJet + gulfArc * 0.9 + atlanticLift * 0.55 + pacificSweep
  v += prairieJet * 0.2 + gulfArc * 0.9 + atlanticLift * 1.15 - pacificSweep * 0.5

  u += -gradientY * 0.65
  v += gradientX * 0.65

  const waveAngle =
    Math.sin((lon + 96) * 0.16) * 0.22 +
    Math.cos((lat - 37) * 0.31) * 0.2 +
    Math.sin((lon + lat) * 0.11) * 0.12

  const sinAngle = Math.sin(waveAngle)
  const cosAngle = Math.cos(waveAngle)
  const rotatedU = u * cosAngle - v * sinAngle
  const rotatedV = u * sinAngle + v * cosAngle + lakeCurl
  const magnitude =
    (0.42 + intensity * 0.95) /
    Math.max(0.0001, Math.hypot(rotatedU, rotatedV))

  return {
    u: rotatedU * magnitude,
    v: rotatedV * magnitude,
    intensity,
  }
}

function rotateVector(u: number, v: number, angle: number) {
  const sinAngle = Math.sin(angle)
  const cosAngle = Math.cos(angle)

  return {
    u: u * cosAngle - v * sinAngle,
    v: u * sinAngle + v * cosAngle,
  }
}

function createParticle(extent: NumericBbox, windGrid: ScalarGrid): Particle {
  for (let attempts = 0; attempts < 24; attempts += 1) {
    const lon = extent.xmin + Math.random() * (extent.xmax - extent.xmin)
    const lat = extent.ymin + Math.random() * (extent.ymax - extent.ymin)
    const value = sampleGrid(windGrid, lon, lat)

    if (value !== null) {
      return {
        lon,
        lat,
        age: Math.random() * 40,
        ttl: 85 + Math.random() * 110,
        speedFactor: 0.58 + Math.random() * 0.38,
        turnBias: (Math.random() - 0.5) * 0.2,
        shimmerPhase: Math.random() * Math.PI * 2,
      }
    }
  }

  return {
    lon: windGrid.xmin + Math.random() * (windGrid.xmax - windGrid.xmin),
    lat: windGrid.ymin + Math.random() * (windGrid.ymax - windGrid.ymin),
    age: 0,
    ttl: 72 + Math.random() * 96,
    speedFactor: 0.58 + Math.random() * 0.38,
    turnBias: (Math.random() - 0.5) * 0.2,
    shimmerPhase: Math.random() * Math.PI * 2,
  }
}

function buildWindField(
  extent: NumericBbox,
  windGrid: ScalarGrid,
  cols: number,
  rows: number,
) {
  const u = new Float32Array(cols * rows)
  const v = new Float32Array(cols * rows)
  const intensity = new Float32Array(cols * rows)
  const lonStep = cols > 1 ? (extent.xmax - extent.xmin) / (cols - 1) : 0
  const latStep = rows > 1 ? (extent.ymax - extent.ymin) / (rows - 1) : 0

  for (let row = 0; row < rows; row += 1) {
    const lat = extent.ymin + row * latStep

    for (let col = 0; col < cols; col += 1) {
      const lon = extent.xmin + col * lonStep
      const vector = createWindVector(lon, lat, windGrid)
      const index = gridKey(col, row, cols)

      if (!vector) {
        u[index] = Number.NaN
        v[index] = Number.NaN
        intensity[index] = Number.NaN
        continue
      }

      u[index] = vector.u
      v[index] = vector.v
      intensity[index] = vector.intensity
    }
  }

  return {
    extent: { ...extent },
    cols,
    rows,
    u,
    v,
    intensity,
  } satisfies WindField
}

function sampleWindField(field: WindField, lon: number, lat: number) {
  const spanX = field.extent.xmax - field.extent.xmin
  const spanY = field.extent.ymax - field.extent.ymin

  if (spanX <= 0 || spanY <= 0) {
    return null
  }

  const fx = ((lon - field.extent.xmin) / spanX) * (field.cols - 1)
  const fy = ((lat - field.extent.ymin) / spanY) * (field.rows - 1)

  if (fx < 0 || fy < 0 || fx > field.cols - 1 || fy > field.rows - 1) {
    return null
  }

  const x0 = Math.floor(fx)
  const y0 = Math.floor(fy)
  const x1 = Math.min(x0 + 1, field.cols - 1)
  const y1 = Math.min(y0 + 1, field.rows - 1)
  const tx = fx - x0
  const ty = fy - y0

  const indices = [
    gridKey(x0, y0, field.cols),
    gridKey(x1, y0, field.cols),
    gridKey(x0, y1, field.cols),
    gridKey(x1, y1, field.cols),
  ]
  const uValues = indices.map((index) => field.u[index])
  const vValues = indices.map((index) => field.v[index])
  const intensityValues = indices.map((index) => field.intensity[index])

  if (
    !uValues.some((value) => !Number.isNaN(value)) ||
    !vValues.some((value) => !Number.isNaN(value))
  ) {
    return null
  }

  if (
    [...uValues, ...vValues, ...intensityValues].some((value) => Number.isNaN(value))
  ) {
    const fallbackCol = Math.round(fx)
    const fallbackRow = Math.round(fy)
    const fallbackIndex = gridKey(fallbackCol, fallbackRow, field.cols)
    const fallbackU = field.u[fallbackIndex]
    const fallbackV = field.v[fallbackIndex]
    const fallbackIntensity = field.intensity[fallbackIndex]

    if (
      Number.isNaN(fallbackU) ||
      Number.isNaN(fallbackV) ||
      Number.isNaN(fallbackIntensity)
    ) {
      return null
    }

    return {
      u: fallbackU,
      v: fallbackV,
      intensity: fallbackIntensity,
    }
  }

  const topU = uValues[0] + (uValues[1] - uValues[0]) * tx
  const bottomU = uValues[2] + (uValues[3] - uValues[2]) * tx
  const topV = vValues[0] + (vValues[1] - vValues[0]) * tx
  const bottomV = vValues[2] + (vValues[3] - vValues[2]) * tx
  const topIntensity =
    intensityValues[0] + (intensityValues[1] - intensityValues[0]) * tx
  const bottomIntensity =
    intensityValues[2] + (intensityValues[3] - intensityValues[2]) * tx

  return {
    u: topU + (bottomU - topU) * ty,
    v: topV + (bottomV - topV) * ty,
    intensity: topIntensity + (bottomIntensity - topIntensity) * ty,
  }
}

function shouldRebuildWindField(
  field: WindField | null,
  extent: NumericBbox,
  cols: number,
  rows: number,
) {
  if (!field || field.cols !== cols || field.rows !== rows) {
    return true
  }

  const currentWidth = field.extent.xmax - field.extent.xmin
  const currentHeight = field.extent.ymax - field.extent.ymin
  const nextWidth = extent.xmax - extent.xmin
  const nextHeight = extent.ymax - extent.ymin
  const widthBaseline = Math.max(nextWidth, currentWidth, 0.0001)
  const heightBaseline = Math.max(nextHeight, currentHeight, 0.0001)

  return (
    Math.abs(field.extent.xmin - extent.xmin) > widthBaseline * 0.12 ||
    Math.abs(field.extent.ymin - extent.ymin) > heightBaseline * 0.12 ||
    Math.abs(currentWidth - nextWidth) > widthBaseline * 0.18 ||
    Math.abs(currentHeight - nextHeight) > heightBaseline * 0.18
  )
}

function windViewportCoverage(extent: NumericBbox, windGrid: ScalarGrid) {
  const extentArea = Math.max(
    1,
    (extent.xmax - extent.xmin) * (extent.ymax - extent.ymin),
  )

  return clamp(
    extentArea / ((windGrid.xmax - windGrid.xmin) * (windGrid.ymax - windGrid.ymin)),
    0.02,
    1,
  )
}

function projectToScreen(
  view: ViewHandle,
  PointCtor: ArcGISPointCtor,
  lon: number,
  lat: number,
) {
  return view.toScreen(
    new PointCtor({
      longitude: lon,
      latitude: lat,
    }),
  )
}

function pickParticleStyle(intensity: number) {
  const emphasizedIntensity = emphasizeWindIntensity(intensity)
  const index = Math.round(
    emphasizedIntensity * (PARTICLE_STYLE_PALETTE.length - 1),
  )

  return PARTICLE_STYLE_PALETTE[index]
}

function clearCanvas(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
) {
  context.save()
  context.setTransform(1, 0, 0, 1, 0, 0)
  context.clearRect(0, 0, canvas.width, canvas.height)
  context.restore()
}

function createThemeLayer(
  kind: ThemeLayerKind,
  GeoJSONLayerCtor: GeoJSONLayerCtor,
  FeatureLayerCtor: FeatureLayerCtor,
  TileLayerCtor: TileLayerCtor,
  visible: boolean,
) {
  if (kind === 'topography') {
    return new TileLayerCtor({
      url: HILLSHADE_DATA_URL,
      title: 'Terrain / elevation',
      visible,
      opacity: 0.3,
      blendMode: 'normal',
    })
  }

  if (kind === 'solar') {
    return new GeoJSONLayerCtor({
      url: SOLAR_DATA_URL,
      title: 'Solar irradiation',
      visible,
      popupEnabled: false,
      effect: 'blur(0.7px)',
      renderer: {
        type: 'class-breaks',
        field: 'solar_value',
        defaultSymbol: {
          type: 'simple-fill',
          color: 'rgba(61, 116, 100, 0.55)',
          outline: {
            color: 'rgba(255, 255, 255, 0)',
            width: 0,
          },
        },
        classBreakInfos: [
          {
            minValue: 3770000,
            maxValue: 4100000,
            symbol: {
              type: 'simple-fill',
              color: 'rgba(61, 116, 100, 0.50)',
              outline: { color: 'rgba(255,255,255,0)', width: 0 },
            },
            label: 'Lower solar resource',
          },
          {
            minValue: 4100000,
            maxValue: 4400000,
            symbol: {
              type: 'simple-fill',
              color: 'rgba(92, 149, 97, 0.56)',
              outline: { color: 'rgba(255,255,255,0)', width: 0 },
            },
            label: 'Moderate solar resource',
          },
          {
            minValue: 4400000,
            maxValue: 4700000,
            symbol: {
              type: 'simple-fill',
              color: 'rgba(126, 192, 106, 0.62)',
              outline: { color: 'rgba(255,255,255,0)', width: 0 },
            },
            label: 'Balanced solar resource',
          },
          {
            minValue: 4700000,
            maxValue: 5000000,
            symbol: {
              type: 'simple-fill',
              color: 'rgba(210, 205, 82, 0.68)',
              outline: { color: 'rgba(255,255,255,0)', width: 0 },
            },
            label: 'Strong solar resource',
          },
          {
            minValue: 5000000,
            maxValue: 5250000,
            symbol: {
              type: 'simple-fill',
              color: 'rgba(255, 189, 89, 0.74)',
              outline: { color: 'rgba(255,255,255,0)', width: 0 },
            },
            label: 'Very strong solar resource',
          },
          {
            minValue: 5250000,
            maxValue: 5600000,
            symbol: {
              type: 'simple-fill',
              color: 'rgba(255, 112, 76, 0.82)',
              outline: { color: 'rgba(255,255,255,0)', width: 0 },
            },
            label: 'Peak solar resource',
          },
        ],
      },
    opacity: 0.56,
  })
  }

  if (kind === 'solarFarms') {
    return new GeoJSONLayerCtor({
      url: SOLAR_FARMS_DATA_URL,
      title: 'Solar farm sites',
      visible,
      popupEnabled: false,
      renderer: {
        type: 'simple',
        symbol: {
          type: 'picture-marker',
          url: SOLAR_FARM_ICON_URL,
          width: '13px',
          height: '13px',
        },
      },
      opacity: 0.95,
    })
  }

  if (kind === 'windFarms') {
    return new GeoJSONLayerCtor({
      url: WIND_FARMS_DATA_URL,
      title: 'Wind farm sites',
      visible,
      popupEnabled: false,
      renderer: {
        type: 'simple',
        symbol: {
          type: 'picture-marker',
          url: WIND_FARM_ICON_URL,
          width: '12px',
          height: '12px',
        },
      },
      opacity: 0.9,
    })
  }

  if (kind === 'powerLines') {
    return new FeatureLayerCtor({
      url: POWER_LINES_DATA_URL,
      title: 'Major power lines',
      visible,
      popupEnabled: false,
      renderer: {
        type: 'simple',
        symbol: {
          type: 'simple-line',
          color: [210, 220, 226, 0.42],
          width: 1.2,
          cap: 'round',
          join: 'round',
        },
      },
      opacity: 0.72,
      minScale: 0,
      maxScale: 0,
    })
  }

  return new GeoJSONLayerCtor({
    url: WIND_DATA_URL,
    title: 'Wind speed',
    visible,
    popupEnabled: false,
    effect: 'blur(0.7px)',
    renderer: {
      type: 'class-breaks',
      field: 'wind_value',
      defaultSymbol: {
        type: 'simple-fill',
        color: 'rgba(52, 92, 168, 0.48)',
        outline: {
          color: 'rgba(255, 255, 255, 0)',
          width: 0,
        },
      },
      classBreakInfos: [
        {
          minValue: 1.6,
          maxValue: 2.1,
          symbol: {
            type: 'simple-fill',
            color: 'rgba(52, 92, 168, 0.48)',
            outline: { color: 'rgba(255,255,255,0)', width: 0 },
          },
          label: 'Lower wind resource',
        },
        {
          minValue: 2.1,
          maxValue: 2.7,
          symbol: {
            type: 'simple-fill',
            color: 'rgba(42, 158, 205, 0.54)',
            outline: { color: 'rgba(255,255,255,0)', width: 0 },
          },
          label: 'Moderate wind resource',
        },
        {
          minValue: 2.7,
          maxValue: 3.3,
          symbol: {
            type: 'simple-fill',
            color: 'rgba(0, 178, 198, 0.58)',
            outline: { color: 'rgba(255,255,255,0)', width: 0 },
          },
          label: 'Balanced wind resource',
        },
        {
          minValue: 3.3,
          maxValue: 4.0,
          symbol: {
            type: 'simple-fill',
            color: 'rgba(30, 188, 130, 0.64)',
            outline: { color: 'rgba(255,255,255,0)', width: 0 },
          },
          label: 'Strong wind resource',
        },
        {
          minValue: 4.0,
          maxValue: 4.6,
          symbol: {
            type: 'simple-fill',
            color: 'rgba(168, 212, 48, 0.70)',
            outline: { color: 'rgba(255,255,255,0)', width: 0 },
          },
          label: 'Very strong wind resource',
        },
        {
          minValue: 4.6,
          maxValue: 5.0,
          symbol: {
            type: 'simple-fill',
            color: 'rgba(255, 152, 48, 0.78)',
            outline: { color: 'rgba(255,255,255,0)', width: 0 },
          },
          label: 'Peak wind resource',
        },
      ],
    },
    opacity: 0.54,
  })
}

export const ArcGISMap = memo(function ArcGISMap({
  topographyVisible,
  solarVisible,
  windVisible,
  windParticlesVisible,
  solarFarmsVisible,
  windFarmsVisible,
  powerLinesVisible,
  optimizationSites,
  optimizationFocusRequest,
  boundingBox,
  boundingBoxSelectionActive,
  editSelectionRequest,
  locationSearchRequest,
  onBoundingBoxSelectionChange,
  onBoundingBoxSelect,
  onSelectionPolygonSelect,
}: ArcGISMapProps) {
  const mapElementRef = useRef<HTMLDivElement | null>(null)
  const windCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const mapRef = useRef<MapHandle | null>(null)
  const viewRef = useRef<ViewHandle | null>(null)
  const graphicsLayerRef = useRef<GraphicsLayerHandle | null>(null)
  const optimizationGraphicsLayerRef = useRef<GraphicsLayerHandle | null>(null)
  const sketchViewModelRef = useRef<SketchHandle | null>(null)
  const selectionGraphicRef = useRef<GraphicHandle | null>(null)
  const coreModulesRef = useRef<CoreModules | null>(null)
  const topographyLayerRef = useRef<LayerHandle | null>(null)
  const solarLayerRef = useRef<LayerHandle | null>(null)
  const windLayerRef = useRef<LayerHandle | null>(null)
  const solarFarmsLayerRef = useRef<LayerHandle | null>(null)
  const windFarmsLayerRef = useRef<LayerHandle | null>(null)
  const powerLinesLayerRef = useRef<LayerHandle | null>(null)
  const boundingBoxSelectionActiveRef = useRef(boundingBoxSelectionActive)
  const initialTopographyVisibleRef = useRef(topographyVisible)
  const initialSolarVisibleRef = useRef(solarVisible)
  const initialWindVisibleRef = useRef(windVisible)
  const initialSolarFarmsVisibleRef = useRef(solarFarmsVisible)
  const initialWindFarmsVisibleRef = useRef(windFarmsVisible)
  const initialPowerLinesVisibleRef = useRef(powerLinesVisible)
  const latestSearchIdRef = useRef(0)
  const interactionStateRef = useRef(true)
  const lastMotionAtRef = useRef(0)
  const [mapReadyToken, setMapReadyToken] = useState(0)

  const ensureThemeLayer = (kind: ThemeLayerKind) => {
    const map = mapRef.current
    const coreModules = coreModulesRef.current

    if (!map || !coreModules) {
      return null
    }

    if (kind === 'topography' && topographyLayerRef.current) {
      return topographyLayerRef.current
    }

    if (kind === 'solar' && solarLayerRef.current) {
      return solarLayerRef.current
    }

    if (kind === 'wind' && windLayerRef.current) {
      return windLayerRef.current
    }

    if (kind === 'solarFarms' && solarFarmsLayerRef.current) {
      return solarFarmsLayerRef.current
    }

    if (kind === 'windFarms' && windFarmsLayerRef.current) {
      return windFarmsLayerRef.current
    }

    if (kind === 'powerLines' && powerLinesLayerRef.current) {
      return powerLinesLayerRef.current
    }

    const layer = createThemeLayer(
      kind,
      coreModules.GeoJSONLayerCtor,
      coreModules.FeatureLayerCtor,
      coreModules.TileLayerCtor,
      true,
    )
    const indexByKind: Record<ThemeLayerKind, number> = {
      topography: 0,
      solar: 1,
      wind: 2,
      solarFarms: 3,
      windFarms: 4,
      powerLines: 5,
    }

    map.add(layer, indexByKind[kind])

    if (kind === 'topography') {
      topographyLayerRef.current = layer
    } else if (kind === 'solar') {
      solarLayerRef.current = layer
    } else if (kind === 'wind') {
      windLayerRef.current = layer
    } else if (kind === 'solarFarms') {
      solarFarmsLayerRef.current = layer
    } else if (kind === 'windFarms') {
      windFarmsLayerRef.current = layer
    } else {
      powerLinesLayerRef.current = layer
    }

    return layer
  }

  const updateZoom = (delta: number) => {
    if (!viewRef.current) {
      return
    }

    const nextZoom = Math.max(3, Math.min(11, viewRef.current.zoom + delta))
    viewRef.current.goTo({ zoom: nextZoom }, { animate: true }).catch(() => {})
  }

  const syncBoundingBoxFromGraphic = useCallback((graphic: GraphicHandle | null) => {
    const extent = graphic?.geometry?.extent
    const coreModules = coreModulesRef.current

    if (!extent) {
      onBoundingBoxSelect(null)
      onSelectionPolygonSelect(null)
      return
    }

    const geographicGeometry = coreModules?.webMercatorUtils
      ? (coreModules.webMercatorUtils.webMercatorToGeographic(
          graphic?.geometry as unknown as Record<string, unknown>,
        ) as NumericBbox & { rings?: number[][][] })
      : (graphic?.geometry as NumericBbox & { rings?: number[][][] } | undefined)

    const geographicExtent = (
      geographicGeometry &&
      typeof geographicGeometry === 'object' &&
      'extent' in geographicGeometry &&
      geographicGeometry.extent
        ? geographicGeometry.extent
        : geographicGeometry
    ) as NumericBbox | undefined

    if (
      !geographicExtent ||
      typeof geographicExtent.xmin !== 'number' ||
      typeof geographicExtent.ymin !== 'number' ||
      typeof geographicExtent.xmax !== 'number' ||
      typeof geographicExtent.ymax !== 'number'
    ) {
      onBoundingBoxSelect(null)
      onSelectionPolygonSelect(null)
      return
    }

    onBoundingBoxSelect({
      xmin: geographicExtent.xmin,
      ymin: geographicExtent.ymin,
      xmax: geographicExtent.xmax,
      ymax: geographicExtent.ymax,
    })

    const rings = geographicGeometry?.rings
    if (
      Array.isArray(rings) &&
      rings.every(
        (ring) =>
          Array.isArray(ring) &&
          ring.every(
            (coordinate) =>
              Array.isArray(coordinate) &&
              coordinate.length >= 2 &&
              typeof coordinate[0] === 'number' &&
              typeof coordinate[1] === 'number',
          ),
      )
    ) {
      onSelectionPolygonSelect({
        rings: rings.map((ring) =>
          ring.map((coordinate) => [coordinate[0], coordinate[1]]),
        ),
      })
    } else {
      onSelectionPolygonSelect(null)
    }
  }, [onBoundingBoxSelect, onSelectionPolygonSelect])

  const beginSelectionEdit = useCallback(() => {
    if (!selectionGraphicRef.current || !sketchViewModelRef.current) {
      return
    }

    sketchViewModelRef.current.cancel()
    sketchViewModelRef.current.update(selectionGraphicRef.current, {
      tool: 'transform',
      enableRotation: false,
      enableScaling: true,
      preserveAspectRatio: false,
      multipleSelectionEnabled: false,
      toggleToolOnClick: true,
    })
  }, [])

  useEffect(() => {
    boundingBoxSelectionActiveRef.current = boundingBoxSelectionActive
  }, [boundingBoxSelectionActive])

  useEffect(() => {
    lastMotionAtRef.current = performance.now()
  }, [])

  useEffect(() => {
    let cancelled = false
    let view: ViewHandle | null = null
    let createHandle: WatchHandle | null = null
    let updateHandle: WatchHandle | null = null
    let stationaryWatchHandle: WatchHandle | null = null
    let interactingWatchHandle: WatchHandle | null = null
    let updatingWatchHandle: WatchHandle | null = null

    loadCoreArcGISModules()
      .then((coreModules) => {
        if (cancelled || !mapElementRef.current) {
          return
        }

        coreModulesRef.current = coreModules

        const map = new coreModules.MapCtor({
          basemap: 'dark-gray-vector',
        })

        mapRef.current = map

        if (initialTopographyVisibleRef.current) {
          topographyLayerRef.current = createThemeLayer(
            'topography',
            coreModules.GeoJSONLayerCtor,
            coreModules.FeatureLayerCtor,
            coreModules.TileLayerCtor,
            true,
          )
          map.add(topographyLayerRef.current, 0)
        }

        if (initialSolarVisibleRef.current) {
          solarLayerRef.current = createThemeLayer(
            'solar',
            coreModules.GeoJSONLayerCtor,
            coreModules.FeatureLayerCtor,
            coreModules.TileLayerCtor,
            true,
          )
          map.add(solarLayerRef.current, 1)
        }

        if (initialWindVisibleRef.current) {
          windLayerRef.current = createThemeLayer(
            'wind',
            coreModules.GeoJSONLayerCtor,
            coreModules.FeatureLayerCtor,
            coreModules.TileLayerCtor,
            true,
          )
          map.add(windLayerRef.current, 2)
        }

        if (initialSolarFarmsVisibleRef.current) {
          solarFarmsLayerRef.current = createThemeLayer(
            'solarFarms',
            coreModules.GeoJSONLayerCtor,
            coreModules.FeatureLayerCtor,
            coreModules.TileLayerCtor,
            true,
          )
          map.add(solarFarmsLayerRef.current, 3)
        }

        if (initialWindFarmsVisibleRef.current) {
          windFarmsLayerRef.current = createThemeLayer(
            'windFarms',
            coreModules.GeoJSONLayerCtor,
            coreModules.FeatureLayerCtor,
            coreModules.TileLayerCtor,
            true,
          )
          map.add(windFarmsLayerRef.current, 4)
        }

        if (initialPowerLinesVisibleRef.current) {
          powerLinesLayerRef.current = createThemeLayer(
            'powerLines',
            coreModules.GeoJSONLayerCtor,
            coreModules.FeatureLayerCtor,
            coreModules.TileLayerCtor,
            true,
          )
          map.add(powerLinesLayerRef.current, 5)
        }

        const graphicsLayer = new coreModules.GraphicsLayerCtor()
        map.add(graphicsLayer)
        graphicsLayerRef.current = graphicsLayer

        const optimizationGraphicsLayer = new coreModules.GraphicsLayerCtor()
        map.add(optimizationGraphicsLayer)
        optimizationGraphicsLayerRef.current = optimizationGraphicsLayer

        view = new coreModules.MapViewCtor({
          container: mapElementRef.current,
          map,
          center: LOWER_48_VIEW.center,
          zoom: LOWER_48_VIEW.zoom,
          popupEnabled: false,
          constraints: {
            minZoom: 3,
            maxZoom: 11,
          },
          ui: {
            components: ['attribution'],
          },
          highlightOptions: {
            color: '#b4b6ba',
            haloOpacity: 0.85,
            fillOpacity: 0.12,
          },
        })

        viewRef.current = view
        setMapReadyToken((current) => current + 1)

        const syncMotionState = () => {
          if (!view) {
            return
          }

          const moving = !view.stationary || view.interacting || view.updating
          interactionStateRef.current = moving

          if (moving) {
            lastMotionAtRef.current = performance.now()
          }
        }

        syncMotionState()
        stationaryWatchHandle = view.watch('stationary', syncMotionState)
        interactingWatchHandle = view.watch('interacting', syncMotionState)
        updatingWatchHandle = view.watch('updating', syncMotionState)

        const sketchViewModel = new coreModules.SketchViewModelCtor({
          view,
          layer: graphicsLayer,
          defaultCreateOptions: {
            mode: 'click',
          },
          polygonSymbol: {
            type: 'simple-fill',
            color: [180, 182, 186, 0.12],
            outline: {
              color: [220, 222, 226, 0.92],
              width: 2,
            },
          },
        })

        createHandle = sketchViewModel.on('create', (event) => {
          const state = event.state

          if (state === 'start') {
            graphicsLayer.removeAll()
            selectionGraphicRef.current = null
          }

          if (state !== 'complete') {
            return
          }

          const graphic = event.graphic as GraphicHandle | undefined

          if (!graphic?.geometry?.extent) {
            onBoundingBoxSelectionChange(false)
            return
          }

          selectionGraphicRef.current = graphic
          syncBoundingBoxFromGraphic(graphic)
          onBoundingBoxSelectionChange(false)
          beginSelectionEdit()
        })

        updateHandle = sketchViewModel.on('update', (event) => {
          const graphics = event.graphics as GraphicHandle[] | undefined
          const graphic = graphics?.[0] ?? selectionGraphicRef.current

          if (!graphic) {
            return
          }

          selectionGraphicRef.current = graphic

          if (event.state === 'active' || event.state === 'complete') {
            syncBoundingBoxFromGraphic(graphic)
          }
        })

        sketchViewModelRef.current = sketchViewModel
      })
      .catch((error) => {
        console.error(error)
      })

    return () => {
      cancelled = true
      createHandle?.remove()
      updateHandle?.remove()
      stationaryWatchHandle?.remove()
      interactingWatchHandle?.remove()
      updatingWatchHandle?.remove()
      selectionGraphicRef.current = null
      topographyLayerRef.current = null
      solarLayerRef.current = null
      windLayerRef.current = null
      solarFarmsLayerRef.current = null
      windFarmsLayerRef.current = null
      powerLinesLayerRef.current = null
      graphicsLayerRef.current = null
      optimizationGraphicsLayerRef.current = null
      sketchViewModelRef.current = null
      coreModulesRef.current = null
      mapRef.current = null
      viewRef.current = null
      view?.destroy()
    }
  }, [beginSelectionEdit, onBoundingBoxSelectionChange, syncBoundingBoxFromGraphic])

  useEffect(() => {
    if (topographyVisible) {
      ensureThemeLayer('topography')
    }

    if (topographyLayerRef.current) {
      topographyLayerRef.current.visible = topographyVisible
    }
  }, [topographyVisible])

  useEffect(() => {
    if (solarVisible) {
      ensureThemeLayer('solar')
    }

    if (solarLayerRef.current) {
      solarLayerRef.current.visible = solarVisible
    }
  }, [solarVisible])

  useEffect(() => {
    if (windVisible) {
      ensureThemeLayer('wind')
    }

    if (windLayerRef.current) {
      windLayerRef.current.visible = windVisible
    }
  }, [windVisible])

  useEffect(() => {
    if (solarFarmsVisible) {
      ensureThemeLayer('solarFarms')
    }

    if (solarFarmsLayerRef.current) {
      solarFarmsLayerRef.current.visible = solarFarmsVisible
    }
  }, [solarFarmsVisible])

  useEffect(() => {
    if (windFarmsVisible) {
      ensureThemeLayer('windFarms')
    }

    if (windFarmsLayerRef.current) {
      windFarmsLayerRef.current.visible = windFarmsVisible
    }
  }, [windFarmsVisible])

  useEffect(() => {
    if (powerLinesVisible) {
      ensureThemeLayer('powerLines')
    }

    if (powerLinesLayerRef.current) {
      powerLinesLayerRef.current.visible = powerLinesVisible
    }
  }, [powerLinesVisible])

  useEffect(() => {
    const optimizationLayer = optimizationGraphicsLayerRef.current
    const coreModules = coreModulesRef.current

    if (!optimizationLayer || !coreModules) {
      return
    }

    optimizationLayer.removeAll()

    for (const site of optimizationSites) {
      const isSolarSite = site.device_type === 'solar'
      const selectedOutput =
        isSolarSite ? site.solar_power_kwh : site.wind_power_kwh

      const graphic = new coreModules.GraphicCtor({
        geometry: new coreModules.PointCtor({
          longitude: site.lon,
          latitude: site.lat,
        }),
        attributes: {
          deviceType: isSolarSite ? 'Solar' : 'Wind',
          selectedOutputKwh: Number(selectedOutput.toFixed(2)),
          expectedOutputKwh:
            site.expected_power_kwh !== undefined
              ? Number(site.expected_power_kwh.toFixed(2))
              : null,
          deviceCostUsd:
            site.device_cost_usd !== undefined
              ? Number(site.device_cost_usd.toFixed(2))
              : null,
          effectiveCostUsd:
            site.effective_cost_usd !== undefined
              ? Number(site.effective_cost_usd.toFixed(2))
              : null,
        },
        symbol: {
          type: 'picture-marker',
          url:
            isSolarSite
              ? SOLAR_OPTIMIZATION_PIN_URL
              : WIND_OPTIMIZATION_PIN_URL,
          width: '32px',
          height: '40px',
        },
      })

      optimizationLayer.add(graphic)
    }
  }, [optimizationSites, mapReadyToken])

  useEffect(() => {
    if (!sketchViewModelRef.current || !graphicsLayerRef.current) {
      return
    }

    if (boundingBoxSelectionActive) {
      graphicsLayerRef.current.removeAll()
      sketchViewModelRef.current.cancel()
      sketchViewModelRef.current.create('polygon')
      return
    }

    sketchViewModelRef.current.cancel()
  }, [boundingBoxSelectionActive])

  useEffect(() => {
    if (!editSelectionRequest || !selectionGraphicRef.current) {
      return
    }

    beginSelectionEdit()
  }, [beginSelectionEdit, editSelectionRequest])

  useEffect(() => {
    if (boundingBox !== null) {
      return
    }

    sketchViewModelRef.current?.cancel()
    graphicsLayerRef.current?.removeAll()
    selectionGraphicRef.current = null
  }, [boundingBox])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!selectionGraphicRef.current) {
        return
      }

      if (event.key !== 'Delete' && event.key !== 'Backspace') {
        return
      }

      const target = event.target
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        (target instanceof HTMLElement && target.isContentEditable)
      ) {
        return
      }

      event.preventDefault()
      sketchViewModelRef.current?.cancel()
      graphicsLayerRef.current?.removeAll()
      selectionGraphicRef.current = null
      onBoundingBoxSelectionChange(false)
      onBoundingBoxSelect(null)
      onSelectionPolygonSelect(null)
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [onBoundingBoxSelect, onBoundingBoxSelectionChange, onSelectionPolygonSelect])

  useEffect(() => {
    if (!locationSearchRequest || !viewRef.current) {
      return
    }

    latestSearchIdRef.current = locationSearchRequest.id

    loadLocatorModule()
      .then((locator) =>
        locator.addressToLocations(LOCATOR_URL, {
          address: {
            SingleLine: locationSearchRequest.query,
          },
        }),
      )
      .then((results) => {
        if (
          latestSearchIdRef.current !== locationSearchRequest.id ||
          !viewRef.current
        ) {
          return
        }

        const firstResult = results[0]
        if (!firstResult) {
          return
        }

        if (firstResult.extent) {
          viewRef.current.goTo(firstResult.extent, { animate: true }).catch(() => {})
          return
        }

        const point = firstResult.location
        if (!point) {
          return
        }

        viewRef.current
          .goTo(
            {
              center: [point.longitude, point.latitude],
              zoom: 9,
            },
            { animate: true },
          )
          .catch(() => {})
      })
      .catch(() => {})
  }, [locationSearchRequest])

  useEffect(() => {
    const view = viewRef.current
    const coreModules = coreModulesRef.current

    if (!optimizationFocusRequest || !view || !coreModules) {
      return
    }

    const southWest = new coreModules.PointCtor({
      longitude: optimizationFocusRequest.boundingBox.xmin,
      latitude: optimizationFocusRequest.boundingBox.ymin,
    })
    const northEast = new coreModules.PointCtor({
      longitude: optimizationFocusRequest.boundingBox.xmax,
      latitude: optimizationFocusRequest.boundingBox.ymax,
    })

    view
      .goTo(
        [southWest, northEast],
        { animate: true },
      )
      .catch(() => {})
  }, [optimizationFocusRequest])

  useEffect(() => {
    let cancelled = false
    let frameId = 0
    let timeoutId = 0
    let windField: WindField | null = null
    let lastTime = performance.now()
    let canvasWasCleared = false

    const canvas = windCanvasRef.current
    const view = viewRef.current
    const coreModules = coreModulesRef.current

    if (!canvas) {
      return
    }

    const context = canvas.getContext('2d')
    if (!context) {
      return
    }

    if (!windVisible || !windParticlesVisible || !view || !coreModules) {
      clearCanvas(context, canvas)
      return
    }

    loadScalarGrid(WIND_DATA_URL, 'wind_value')
      .then((windGrid) => {
        if (cancelled) {
          return
        }

        const particles: Particle[] = []

        const scheduleNextFrame = (delayMs = 0) => {
          if (cancelled) {
            return
          }

          if (delayMs > 0) {
            timeoutId = window.setTimeout(() => {
              frameId = window.requestAnimationFrame(tick)
            }, delayMs)
            return
          }

          frameId = window.requestAnimationFrame(tick)
        }

        const resizeCanvas = () => {
          const devicePixelRatio = Math.min(
            window.devicePixelRatio || 1,
            MAX_WIND_CANVAS_DPR,
          )
          const nextWidth = Math.max(
            1,
            Math.floor(canvas.clientWidth * devicePixelRatio),
          )
          const nextHeight = Math.max(
            1,
            Math.floor(canvas.clientHeight * devicePixelRatio),
          )

          if (canvas.width === nextWidth && canvas.height === nextHeight) {
            return
          }

          canvas.width = nextWidth
          canvas.height = nextHeight
          context.setTransform(1, 0, 0, 1, 0, 0)
          context.scale(devicePixelRatio, devicePixelRatio)
          context.lineCap = 'round'
          context.lineJoin = 'round'
        }

        const syncParticleCount = (
          extent: NumericBbox,
          recoveryMode: boolean,
        ) => {
          const coverage = windViewportCoverage(extent, windGrid)
          const targetCount = Math.round(
            (recoveryMode ? 82 : 112) + coverage * (recoveryMode ? 156 : 220),
          )

          while (particles.length < targetCount) {
            particles.push(createParticle(extent, windGrid))
          }

          if (particles.length > targetCount) {
            particles.length = targetCount
          }
        }

        const seedParticles = (extent: NumericBbox, recoveryMode: boolean) => {
          syncParticleCount(extent, recoveryMode)

          for (let index = 0; index < particles.length; index += 1) {
            const particle = particles[index]
            if (
              particle.lon < extent.xmin ||
              particle.lon > extent.xmax ||
              particle.lat < extent.ymin ||
              particle.lat > extent.ymax
            ) {
              particles[index] = createParticle(extent, windGrid)
            }
          }
        }

        const tick = (timestamp: number) => {
          if (cancelled || !viewRef.current || !coreModulesRef.current) {
            return
          }

          const hidden = document.visibilityState === 'hidden'
          const moving = interactionStateRef.current

          if (hidden) {
            if (!canvasWasCleared) {
              clearCanvas(context, canvas)
              canvasWasCleared = true
            }

            lastTime = timestamp
            scheduleNextFrame(240)
            return
          }

          canvasWasCleared = false
          resizeCanvas()

          const visibleExtent = geographicExtentFromView(
            viewRef.current,
            coreModulesRef.current.webMercatorUtils,
          )

          if (!visibleExtent) {
            scheduleNextFrame(120)
            return
          }

          const recoveryMode =
            moving || timestamp - lastMotionAtRef.current < WIND_RECOVERY_MS
          const viewportCoverage = windViewportCoverage(visibleExtent, windGrid)
          const fieldCols = recoveryMode ? 36 : 48
          const fieldRows = recoveryMode ? 22 : 30

          if (
            shouldRebuildWindField(
              windField,
              visibleExtent,
              fieldCols,
              fieldRows,
            )
          ) {
            windField = buildWindField(
              visibleExtent,
              windGrid,
              fieldCols,
              fieldRows,
            )
          }

          if (!windField) {
            scheduleNextFrame(120)
            return
          }

          seedParticles(visibleExtent, recoveryMode)

          const width = canvas.clientWidth
          const height = canvas.clientHeight
          const deltaSeconds = Math.min((timestamp - lastTime) / 1000, 0.04)
          const segmentCount =
            viewportCoverage < 0.1 ? 3 : recoveryMode ? 3 : 4
          const viewportSpeedFactor = 0.42 + viewportCoverage * 0.5
          lastTime = timestamp

          if (moving) {
            clearCanvas(context, canvas)
          } else {
            context.save()
            context.setTransform(1, 0, 0, 1, 0, 0)
            context.globalCompositeOperation = 'destination-in'
            context.fillStyle = 'rgba(255, 255, 255, 0.92)'
            context.fillRect(0, 0, canvas.width, canvas.height)
            context.restore()
          }

          context.lineCap = 'round'
          context.lineJoin = 'round'

          for (let index = 0; index < particles.length; index += 1) {
            let particle = particles[index]
            const vector = sampleWindField(windField, particle.lon, particle.lat)

            if (!vector) {
              particles[index] = createParticle(visibleExtent, windGrid)
              continue
            }

            const currentScreen = projectToScreen(
              viewRef.current,
              coreModulesRef.current.PointCtor,
              particle.lon,
              particle.lat,
            )

            if (
              !currentScreen ||
              currentScreen.x < -60 ||
              currentScreen.x > width + 60 ||
              currentScreen.y < -60 ||
              currentScreen.y > height + 60
            ) {
              particles[index] = createParticle(visibleExtent, windGrid)
              continue
            }

            const pathPoints = [currentScreen]
            let nextLon = particle.lon
            let nextLat = particle.lat
            let pathIntensity = vector.intensity
            let escaped = false
            const emphasizedIntensity = emphasizeWindIntensity(vector.intensity)
            const segmentStep =
              deltaSeconds *
              (0.1 + emphasizedIntensity * 0.18) *
              viewportSpeedFactor *
              particle.speedFactor

            for (let stepIndex = 0; stepIndex < segmentCount; stepIndex += 1) {
              const segmentVector = sampleWindField(windField, nextLon, nextLat)

              if (!segmentVector) {
                escaped = true
                break
              }

              pathIntensity = Math.max(pathIntensity, segmentVector.intensity)
              const bendAngle =
                particle.turnBias +
                (recoveryMode
                  ? 0
                  : Math.sin(
                      particle.shimmerPhase + particle.age * 0.025 + stepIndex * 0.55,
                    ) * 0.03)
              const curvedVector = rotateVector(
                segmentVector.u,
                segmentVector.v,
                bendAngle,
              )

              nextLon += curvedVector.u * segmentStep
              nextLat += curvedVector.v * segmentStep

              if (
                nextLon < visibleExtent.xmin - 1 ||
                nextLon > visibleExtent.xmax + 1 ||
                nextLat < visibleExtent.ymin - 1 ||
                nextLat > visibleExtent.ymax + 1
              ) {
                escaped = true
                break
              }

              const nextScreen = projectToScreen(
                viewRef.current,
                coreModulesRef.current.PointCtor,
                nextLon,
                nextLat,
              )

              if (
                !nextScreen ||
                nextScreen.x < -60 ||
                nextScreen.x > width + 60 ||
                nextScreen.y < -60 ||
                nextScreen.y > height + 60
              ) {
                escaped = true
                break
              }

              pathPoints.push(nextScreen)
            }

            particle.age += recoveryMode ? 0.45 : 0.7

            if (escaped || pathPoints.length < 2 || particle.age > particle.ttl) {
              particles[index] = createParticle(visibleExtent, windGrid)
              continue
            }

            const particleStyle = pickParticleStyle(pathIntensity)

            context.strokeStyle = particleStyle.glowColor
            context.lineWidth = particleStyle.glowWidth
            context.beginPath()
            context.moveTo(pathPoints[0].x, pathPoints[0].y)
            for (
              let pointIndex = 1;
              pointIndex < pathPoints.length;
              pointIndex += 1
            ) {
              context.lineTo(pathPoints[pointIndex].x, pathPoints[pointIndex].y)
            }
            context.stroke()

            context.strokeStyle = particleStyle.coreColor
            context.lineWidth = particleStyle.coreWidth
            context.beginPath()
            context.moveTo(pathPoints[0].x, pathPoints[0].y)
            for (
              let pointIndex = 1;
              pointIndex < pathPoints.length;
              pointIndex += 1
            ) {
              context.lineTo(pathPoints[pointIndex].x, pathPoints[pointIndex].y)
            }
            context.stroke()

            particle = {
              ...particle,
              lon: nextLon,
              lat: nextLat,
            }
            particles[index] = particle
          }

          scheduleNextFrame()
        }

        scheduleNextFrame()
      })
      .catch((error) => {
        console.error(error)
      })

    return () => {
      cancelled = true
      window.cancelAnimationFrame(frameId)
      window.clearTimeout(timeoutId)
      clearCanvas(context, canvas)
    }
  }, [mapReadyToken, windVisible, windParticlesVisible])

  return (
    <>
      <div ref={mapElementRef} className="arcgis-map" aria-label="United States map" />
      <canvas
        ref={windCanvasRef}
        className={
          windVisible && windParticlesVisible
            ? 'wind-overlay-canvas wind-overlay-canvas-active'
            : 'wind-overlay-canvas'
        }
        aria-hidden="true"
      />
      <div className="map-zoom-controls" aria-label="Map zoom controls">
        <button type="button" className="zoom-button" onClick={() => updateZoom(1)}>
          +
        </button>
        <button type="button" className="zoom-button" onClick={() => updateZoom(-1)}>
          -
        </button>
      </div>
    </>
  )
})
