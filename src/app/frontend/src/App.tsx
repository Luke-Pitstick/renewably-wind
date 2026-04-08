import { startTransition, useDeferredValue, useEffect, useState } from 'react'
import './App.css'
import { ArcGISMap } from './ArcGISMap'

type OptimizationMode = 'cash' | 'power'

type BoundingBox = {
  xmin: number
  ymin: number
  xmax: number
  ymax: number
}

type SelectionPolygon = {
  rings: number[][][]
}

type LocationSearchRequest = {
  id: number
  query: string
}

type LocationSuggestion = {
  text: string
}

type EditSelectionRequest = {
  id: number
}

type OptimizationFocusRequest = {
  id: number
  boundingBox: BoundingBox
}

type OptimizationPoint = {
  lat: number
  lon: number
  device_type: 'solar' | 'wind'
  solar_power_kwh: number
  wind_power_kwh: number
  solar_probability?: number
  wind_probability?: number
  expected_power_kwh?: number
  selected_power_kwh?: number
  device_cost_usd?: number
  effective_cost_usd?: number
}

type OptimizationResponse = {
  selected_count: number
  mode: 'cash' | 'power'
  sample_count: number
  total_cost_usd?: number
  total_actual_cost_usd?: number
  total_expected_power_kwh?: number
  total_power_kwh?: number
  total_raw_power_kwh?: number
  total_effective_cost_usd?: number
  power_basis?: string
  points?: OptimizationPoint[]
}

type GeoJsonPointFeature = {
  type: 'Feature'
  geometry: {
    type: 'Point'
    coordinates: [number, number]
  }
  properties: Record<string, number | string>
}

type GeoJsonFeatureCollection = {
  type: 'FeatureCollection'
  features: GeoJsonPointFeature[]
}

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ??
  'http://127.0.0.1:8000'
const LOCATION_SUGGEST_URL =
  'https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/suggest'
const REPOSITORY_URL = 'https://github.com/Luke-Pitstick/renewably'

function isFiniteCoordinate(value: number) {
  return Number.isFinite(value)
}

function buildOptimizationGeoJson(points: OptimizationPoint[]): GeoJsonFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: points.flatMap((point, index) => {
      if (!isFiniteCoordinate(point.lon) || !isFiniteCoordinate(point.lat)) {
        return []
      }

      return [
        {
          type: 'Feature' as const,
          geometry: {
            type: 'Point' as const,
            coordinates: [point.lon, point.lat] as [number, number],
          },
          properties: {
            site_id: index + 1,
            device_type: point.device_type,
            solar_power_kwh: point.solar_power_kwh,
            wind_power_kwh: point.wind_power_kwh,
            ...(point.solar_probability !== undefined
              ? { solar_probability: point.solar_probability }
              : {}),
            ...(point.wind_probability !== undefined
              ? { wind_probability: point.wind_probability }
              : {}),
            ...(point.expected_power_kwh !== undefined
              ? { expected_power_kwh: point.expected_power_kwh }
              : {}),
            ...(point.selected_power_kwh !== undefined
              ? { selected_power_kwh: point.selected_power_kwh }
              : {}),
            ...(point.device_cost_usd !== undefined
              ? { device_cost_usd: point.device_cost_usd }
              : {}),
            ...(point.effective_cost_usd !== undefined
              ? { effective_cost_usd: point.effective_cost_usd }
              : {}),
          },
        },
      ]
    }),
  }
}

function App() {
  const [topographyVisible, setTopographyVisible] = useState(true)
  const [solarVisible, setSolarVisible] = useState(false)
  const [windVisible, setWindVisible] = useState(true)
  const [windParticlesVisible, setWindParticlesVisible] = useState(true)
  const [solarFarmsVisible, setSolarFarmsVisible] = useState(false)
  const [windFarmsVisible, setWindFarmsVisible] = useState(false)
  const [powerLinesVisible, setPowerLinesVisible] = useState(false)
  const [optimizationPanelOpen, setOptimizationPanelOpen] = useState(false)
  const [layerMenuOpen, setLayerMenuOpen] = useState(false)
  const [helpModalOpen, setHelpModalOpen] = useState(false)
  const [optimizationMode, setOptimizationMode] =
    useState<OptimizationMode>('cash')
  const [optimizationValue, setOptimizationValue] = useState('')
  const [boundingBoxSelectionActive, setBoundingBoxSelectionActive] =
    useState(false)
  const [boundingBox, setBoundingBox] = useState<BoundingBox | null>(null)
  const [selectionPolygon, setSelectionPolygon] =
    useState<SelectionPolygon | null>(null)
  const [locationQuery, setLocationQuery] = useState('')
  const [locationSearchRequest, setLocationSearchRequest] =
    useState<LocationSearchRequest | null>(null)
  const [locationSuggestions, setLocationSuggestions] = useState<LocationSuggestion[]>([])
  const [locationSuggestionsOpen, setLocationSuggestionsOpen] = useState(false)
  const [locationSuggestionsLoading, setLocationSuggestionsLoading] = useState(false)
  const [activeLocationSuggestionIndex, setActiveLocationSuggestionIndex] =
    useState(-1)
  const [editSelectionRequest, setEditSelectionRequest] =
    useState<EditSelectionRequest | null>(null)
  const [optimizationSubmitting, setOptimizationSubmitting] = useState(false)
  const [optimizationStatusMessage, setOptimizationStatusMessage] = useState('')
  const [optimizationResult, setOptimizationResult] =
    useState<OptimizationResponse | null>(null)
  const [optimizationFocusRequest, setOptimizationFocusRequest] =
    useState<OptimizationFocusRequest | null>(null)

  const activeLayerCount =
    Number(topographyVisible) +
    Number(solarVisible) +
    Number(windVisible) +
    Number(solarFarmsVisible) +
    Number(windFarmsVisible) +
    Number(powerLinesVisible)

  const inputLabel =
    optimizationMode === 'cash' ? 'Available budget' : 'Target power need'
  const inputPlaceholder =
    optimizationMode === 'cash' ? 'Enter max budget' : 'Enter required power'
  const inputUnit = optimizationMode === 'cash' ? 'USD' : 'kWh'
  const deferredLocationQuery = useDeferredValue(locationQuery)
  const trimmedDeferredLocationQuery = deferredLocationQuery.trim()
  const boundingBoxSummary = boundingBox
    ? 'Polygon selected on map'
    : 'No area selected'
  const hasBoundingBox = boundingBox !== null
  const optimizationTargetValue = Number(optimizationValue)
  const canSubmitOptimization =
    hasBoundingBox &&
    Number.isFinite(optimizationTargetValue) &&
    optimizationTargetValue > 0 &&
    !optimizationSubmitting
  const optimizationPoints = optimizationResult?.points ?? []
  const downloadableOptimizationPointCount = optimizationPoints.filter(
    (point) => isFiniteCoordinate(point.lon) && isFiniteCoordinate(point.lat),
  ).length

  useEffect(() => {
    if (trimmedDeferredLocationQuery.length < 2) {
      setLocationSuggestions([])
      setLocationSuggestionsLoading(false)
      setActiveLocationSuggestionIndex(-1)
      return
    }

    const controller = new AbortController()
    setLocationSuggestionsLoading(true)

    fetch(
      `${LOCATION_SUGGEST_URL}?f=json&text=${encodeURIComponent(trimmedDeferredLocationQuery)}&maxSuggestions=6`,
      { signal: controller.signal },
    )
      .then(async (response) => {
        if (!response.ok) {
          throw new Error('Suggestion request failed')
        }

        return (await response.json()) as {
          suggestions?: Array<{
            text?: string
          }>
        }
      })
      .then((payload) => {
        const nextSuggestions = (payload.suggestions ?? [])
          .map((suggestion) => suggestion.text?.trim() ?? '')
          .filter((text, index, all) => text.length > 0 && all.indexOf(text) === index)
          .map((text) => ({ text }))

        setLocationSuggestions(nextSuggestions)
        setActiveLocationSuggestionIndex((current) =>
          nextSuggestions.length === 0
            ? -1
            : current >= 0
              ? Math.min(current, nextSuggestions.length - 1)
              : -1,
        )
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return
        }

        setLocationSuggestions([])
        setActiveLocationSuggestionIndex(-1)
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLocationSuggestionsLoading(false)
        }
      })

    return () => {
      controller.abort()
    }
  }, [trimmedDeferredLocationQuery])

  useEffect(() => {
    if (!helpModalOpen) {
      return
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setHelpModalOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [helpModalOpen])

  const submitLocationSearch = (queryOverride?: string) => {
    const query = (queryOverride ?? locationQuery).trim()
    if (!query) {
      return
    }

    setLocationSuggestions([])
    setLocationSuggestionsOpen(false)
    setActiveLocationSuggestionIndex(-1)

    startTransition(() => {
      setLocationSearchRequest({
        id: Date.now(),
        query,
      })
    })
  }

  const selectLocationSuggestion = (suggestion: LocationSuggestion) => {
    setLocationQuery(suggestion.text)
    submitLocationSearch(suggestion.text)
  }

  const resetOptimizationWorkflow = () => {
    setOptimizationResult(null)
    setOptimizationFocusRequest(null)
    setOptimizationStatusMessage('')
    setOptimizationMode('cash')
    setOptimizationValue('')
    setBoundingBoxSelectionActive(false)
    setBoundingBox(null)
    setSelectionPolygon(null)
    setOptimizationPanelOpen(false)
  }

  const submitOptimizationRequest = async () => {
    if (!boundingBox || !canSubmitOptimization) {
      return
    }

    setOptimizationSubmitting(true)
    setOptimizationStatusMessage('Submitting Optimization Request...')

    try {
      const response = await fetch(`${API_BASE_URL}/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          mode: optimizationMode,
          target_value: optimizationTargetValue,
          bounding_box: boundingBox,
          polygon: selectionPolygon,
        }),
      })

      if (!response.ok) {
        const errorBody = await response.json().catch(() => null)
        throw new Error(errorBody?.detail ?? 'Optimization Request Failed')
      }

      const result = (await response.json()) as OptimizationResponse
      setOptimizationResult(result)
      setOptimizationFocusRequest({
        id: Date.now(),
        boundingBox,
      })
      setOptimizationPanelOpen(false)
      setOptimizationStatusMessage(
        optimizationMode === 'cash'
          ? `${result.selected_count} sites selected · expected output ${Math.round(result.total_expected_power_kwh ?? 0).toLocaleString()} kWh`
          : `${result.selected_count} sites selected · total output ${Math.round(result.total_power_kwh ?? 0).toLocaleString()} kWh`,
      )
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Optimization Request Failed'
      setOptimizationStatusMessage(message)
    } finally {
      setOptimizationSubmitting(false)
    }
  }

  const downloadOptimizationGeoJson = () => {
    if (!optimizationResult || downloadableOptimizationPointCount === 0) {
      return
    }

    const geoJson = buildOptimizationGeoJson(optimizationPoints)
    const blob = new Blob([JSON.stringify(geoJson, null, 2)], {
      type: 'application/geo+json;charset=utf-8',
    })
    const downloadUrl = URL.createObjectURL(blob)
    const link = document.createElement('a')
    const dateStamp = new Date().toISOString().slice(0, 10)

    link.href = downloadUrl
    link.download = `renewably-${optimizationResult.mode}-optimization-sites-${dateStamp}.geojson`
    document.body.append(link)
    link.click()
    link.remove()
    window.setTimeout(() => {
      URL.revokeObjectURL(downloadUrl)
    }, 0)
  }

  return (
    <main className="app-shell">
      <section className="map-stage">
        <div className="map-frame">
          <div className="map-top-left">
            <div className="map-top-left-header">
              <div className="brand-banner">
                <div className="brand-copy">
                  <h1>Renewably</h1>
                  <p>Renewable Optimization Studio</p>
                </div>
              </div>

              <a
                className="map-project-link map-project-link-icon"
                href={REPOSITORY_URL}
                target="_blank"
                rel="noreferrer"
                aria-label="Open the Renewably GitHub repository"
                title="GitHub repository"
              >
                <svg viewBox="0 0 24 24" role="presentation" aria-hidden="true">
                  <path
                    d="M12 2C6.48 2 2 6.58 2 12.22c0 4.52 2.87 8.35 6.84 9.7.5.1.68-.22.68-.49 0-.24-.01-1.05-.02-1.9-2.78.62-3.37-1.2-3.37-1.2-.46-1.19-1.11-1.51-1.11-1.51-.91-.64.07-.63.07-.63 1 .07 1.53 1.05 1.53 1.05.89 1.56 2.34 1.11 2.91.85.09-.66.35-1.11.63-1.36-2.22-.26-4.56-1.14-4.56-5.05 0-1.11.39-2.01 1.03-2.72-.1-.26-.45-1.32.1-2.75 0 0 .84-.28 2.75 1.04A9.36 9.36 0 0 1 12 6.82c.85 0 1.71.12 2.51.35 1.9-1.32 2.74-1.04 2.74-1.04.55 1.43.21 2.49.1 2.75.64.71 1.03 1.61 1.03 2.72 0 3.92-2.34 4.79-4.57 5.05.36.32.68.95.68 1.92 0 1.39-.01 2.5-.01 2.84 0 .27.18.59.69.49A10.24 10.24 0 0 0 22 12.22C22 6.58 17.52 2 12 2Z"
                    fill="currentColor"
                  />
                </svg>
              </a>

              <button
                type="button"
                className="map-help-button"
                onClick={() => setHelpModalOpen(true)}
                aria-haspopup="dialog"
                aria-expanded={helpModalOpen}
                aria-controls="app-help-modal"
                aria-label="How to use Renewably"
                title="How to use Renewably"
              >
                <span aria-hidden="true">?</span>
              </button>
            </div>

            <form
              className="map-search"
              onSubmit={(event) => {
                event.preventDefault()
                if (
                  locationSuggestionsOpen &&
                  activeLocationSuggestionIndex >= 0 &&
                  activeLocationSuggestionIndex < locationSuggestions.length
                ) {
                  selectLocationSuggestion(
                    locationSuggestions[activeLocationSuggestionIndex],
                  )
                  return
                }

                submitLocationSearch()
              }}
            >
              <div className="map-search-input-group">
                <label className="map-search-field">
                  <span className="map-search-icon" aria-hidden="true">
                    ⌕
                  </span>
                  <input
                    type="text"
                    value={locationQuery}
                    onChange={(event) => {
                      setLocationQuery(event.target.value)
                      setLocationSuggestionsOpen(true)
                      setActiveLocationSuggestionIndex(-1)
                    }}
                    onFocus={() => {
                      if (trimmedDeferredLocationQuery.length >= 2) {
                        setLocationSuggestionsOpen(true)
                      }
                    }}
                    onBlur={() => {
                      setLocationSuggestionsOpen(false)
                      setActiveLocationSuggestionIndex(-1)
                    }}
                    onKeyDown={(event) => {
                      if (
                        !locationSuggestionsOpen ||
                        locationSuggestions.length === 0
                      ) {
                        if (event.key === 'Escape') {
                          setLocationSuggestionsOpen(false)
                        }

                        return
                      }

                      if (event.key === 'ArrowDown') {
                        event.preventDefault()
                        setActiveLocationSuggestionIndex((current) =>
                          current >= locationSuggestions.length - 1 ? 0 : current + 1,
                        )
                        return
                      }

                      if (event.key === 'ArrowUp') {
                        event.preventDefault()
                        setActiveLocationSuggestionIndex((current) =>
                          current <= 0 ? locationSuggestions.length - 1 : current - 1,
                        )
                        return
                      }

                      if (event.key === 'Escape') {
                        setLocationSuggestionsOpen(false)
                        setActiveLocationSuggestionIndex(-1)
                      }
                    }}
                    placeholder="Search city, state, or address"
                    aria-label="Search city, state, or address"
                    aria-autocomplete="list"
                    aria-expanded={locationSuggestionsOpen}
                    aria-controls="map-search-suggestions"
                    aria-activedescendant={
                      activeLocationSuggestionIndex >= 0
                        ? `map-search-suggestion-${activeLocationSuggestionIndex}`
                        : undefined
                    }
                  />
                </label>

                {locationSuggestionsOpen && trimmedDeferredLocationQuery.length >= 2 ? (
                  <div
                    id="map-search-suggestions"
                    className="map-search-suggestions"
                    role="listbox"
                    aria-label="Search suggestions"
                  >
                    {locationSuggestions.length > 0 ? (
                      locationSuggestions.map((suggestion, index) => (
                        <button
                          key={`${suggestion.text}-${index}`}
                          id={`map-search-suggestion-${index}`}
                          type="button"
                          role="option"
                          className={
                            index === activeLocationSuggestionIndex
                              ? 'map-search-suggestion map-search-suggestion-active'
                              : 'map-search-suggestion'
                          }
                          aria-selected={index === activeLocationSuggestionIndex}
                          onMouseDown={(event) => {
                            event.preventDefault()
                          }}
                          onClick={() => {
                            selectLocationSuggestion(suggestion)
                          }}
                        >
                          <span className="map-search-suggestion-icon" aria-hidden="true">
                            ↗
                          </span>
                          <span>{suggestion.text}</span>
                        </button>
                      ))
                    ) : (
                      <div className="map-search-empty" aria-live="polite">
                        {locationSuggestionsLoading ? 'Searching places...' : 'No matches found.'}
                      </div>
                    )}
                  </div>
                ) : null}
              </div>
              <button type="submit" className="map-search-button">
                Search
              </button>
            </form>
          </div>

          {helpModalOpen ? (
            <div
              className="app-help-modal-backdrop"
              onClick={() => setHelpModalOpen(false)}
            >
              <div
                id="app-help-modal"
                className="map-overlay app-help-modal"
                role="dialog"
                aria-modal="true"
                aria-labelledby="app-help-modal-title"
                onClick={(event) => {
                  event.stopPropagation()
                }}
              >
                <div className="legend-header app-help-modal-header">
                  <div>
                    <p className="panel-label">Quick start</p>
                    <h2 id="app-help-modal-title" className="app-help-modal-title">
                      How to use Renewably
                    </h2>
                  </div>
                  <button
                    type="button"
                    className="overlay-close-button"
                    onClick={() => setHelpModalOpen(false)}
                    aria-label="Close help dialog"
                  >
                    ×
                  </button>
                </div>

                <div className="app-help-modal-body">
                  <div className="app-help-step">
                    <span className="app-help-step-number">1</span>
                    <div>
                      <strong>Find a location</strong>
                      <p>
                        Search for a city, state, or address to move the map where
                        you want to plan.
                      </p>
                    </div>
                  </div>

                  <div className="app-help-step">
                    <span className="app-help-step-number">2</span>
                    <div>
                      <strong>Choose your layers</strong>
                      <p>
                        Use the layers button in the lower left to show solar,
                        wind, farms, terrain, and transmission context.
                      </p>
                    </div>
                  </div>

                  <div className="app-help-step">
                    <span className="app-help-step-number">3</span>
                    <div>
                      <strong>Select an area to optimize</strong>
                      <p>
                        Open Optimize in the lower right, draw a polygon on the
                        map, and set either your budget or required power target.
                      </p>
                    </div>
                  </div>

                  <div className="app-help-step">
                    <span className="app-help-step-number">4</span>
                    <div>
                      <strong>Review and export results</strong>
                      <p>
                        Inspect the recommended sites, compare totals, and export
                        the selected locations as GeoJSON when you are ready.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          <ArcGISMap
            topographyVisible={topographyVisible}
            solarVisible={solarVisible}
            windVisible={windVisible}
            windParticlesVisible={windParticlesVisible}
            solarFarmsVisible={solarFarmsVisible}
            windFarmsVisible={windFarmsVisible}
            powerLinesVisible={powerLinesVisible}
            optimizationSites={optimizationResult?.points ?? []}
            optimizationFocusRequest={optimizationFocusRequest}
            boundingBox={boundingBox}
            boundingBoxSelectionActive={boundingBoxSelectionActive}
            editSelectionRequest={editSelectionRequest}
            onBoundingBoxSelectionChange={setBoundingBoxSelectionActive}
            onBoundingBoxSelect={setBoundingBox}
            onSelectionPolygonSelect={setSelectionPolygon}
            locationSearchRequest={locationSearchRequest}
          />

          <div className="map-bottom-left">
            <div className="map-bottom-left-row">
              <div className="legend-stack">
                {solarVisible ? (
                  <div className="map-overlay overlay-legend overlay-legend-inline">
                    <div className="legend-header">
                      <p className="panel-label">Solar irradiation</p>
                      <span className="legend-unit">kWh/m²/day</span>
                    </div>
                    <div className="legend-bar" aria-hidden="true" />
                    <div className="legend-values">
                      <span>Lower resource</span>
                      <span>Strong resource</span>
                      <span>Peak resource</span>
                    </div>
                  </div>
                ) : null}

                {windVisible ? (
                  <div className="map-overlay overlay-legend overlay-legend-inline">
                    <div className="legend-header">
                      <p className="panel-label">Wind speed</p>
                      <span className="legend-unit">m/s</span>
                    </div>
                    <div className="legend-bar wind-legend-bar" aria-hidden="true" />
                    <div className="legend-values">
                      <span>Lower wind</span>
                      <span>Strong wind</span>
                      <span>Peak wind</span>
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="layer-dock-anchor">
                <button
                  type="button"
                  className="layer-menu-toggle"
                  onClick={() => setLayerMenuOpen((current) => !current)}
                  aria-expanded={layerMenuOpen}
                  aria-label="Toggle layer menu"
                >
                  <span className="layer-menu-toggle-icon" aria-hidden="true">
                    <svg viewBox="0 0 24 24" role="presentation">
                      <path
                        d="M12 3 4 7.4 12 11.8 20 7.4 12 3Z"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinejoin="round"
                      />
                      <path
                        d="M4 11.1 12 15.5 20 11.1"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d="M4 14.9 12 19.3 20 14.9"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </span>
                  <span className="layer-menu-toggle-count">{activeLayerCount}</span>
                </button>

                {layerMenuOpen ? (
                  <div className="map-overlay layer-dock">
                    <div className="legend-header">
                      <div>
                        <p className="panel-label">Map layers</p>
                        <span className="legend-unit">{activeLayerCount} active</span>
                      </div>
                      <button
                        type="button"
                        className="overlay-close-button"
                        onClick={() => setLayerMenuOpen(false)}
                        aria-label="Close layer menu"
                      >
                        ×
                      </button>
                    </div>

                    <div className="floating-layer-list">
                      <label className="floating-layer-card">
                        <div className="floating-layer-copy">
                          <strong>Topography</strong>
                          <p>Terrain hillshade for elevation context.</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={topographyVisible}
                          onChange={() => {
                            startTransition(() => {
                              setTopographyVisible((current) => !current)
                            })
                          }}
                        />
                      </label>

                      <label className="floating-layer-card">
                        <div className="floating-layer-copy">
                          <strong>Solar irradiation</strong>
                          <p>Estimated solar resource intensity.</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={solarVisible}
                          onChange={() => {
                            startTransition(() => {
                              setSolarVisible((current) => !current)
                            })
                          }}
                        />
                      </label>

                      <div className="floating-layer-group">
                        <label className="floating-layer-card">
                          <div className="floating-layer-copy">
                            <strong>Wind speed</strong>
                            <p>Estimated wind resource intensity.</p>
                          </div>
                          <input
                            type="checkbox"
                            checked={windVisible}
                            onChange={() => {
                              startTransition(() => {
                                setWindVisible((current) => !current)
                              })
                            }}
                          />
                        </label>

                        {windVisible ? (
                          <label className="floating-layer-subcard">
                            <div className="floating-layer-subcopy">
                              <strong>Wind particles</strong>
                              <p>Animated wind flow streaks.</p>
                            </div>
                            <input
                              type="checkbox"
                              checked={windParticlesVisible}
                              onChange={() => {
                                startTransition(() => {
                                  setWindParticlesVisible((current) => !current)
                                })
                              }}
                            />
                          </label>
                        ) : null}
                      </div>

                      <label className="floating-layer-card">
                        <div className="floating-layer-copy">
                          <strong>Solar farm sites</strong>
                          <p>Known solar farm locations.</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={solarFarmsVisible}
                          onChange={() => {
                            startTransition(() => {
                              setSolarFarmsVisible((current) => !current)
                            })
                          }}
                        />
                      </label>

                      <label className="floating-layer-card">
                        <div className="floating-layer-copy">
                          <strong>Wind farm sites</strong>
                          <p>Known wind farm locations.</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={windFarmsVisible}
                          onChange={() => {
                            startTransition(() => {
                              setWindFarmsVisible((current) => !current)
                            })
                          }}
                        />
                      </label>

                      <label className="floating-layer-card">
                        <div className="floating-layer-copy">
                          <strong>Major power lines</strong>
                          <p>Transmission line corridors.</p>
                        </div>
                        <input
                          type="checkbox"
                          checked={powerLinesVisible}
                          onChange={() => {
                            startTransition(() => {
                              setPowerLinesVisible((current) => !current)
                            })
                          }}
                        />
                      </label>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          {optimizationPanelOpen ? (
            <div className="map-overlay optimization-sheet">
              <div className="legend-header overlay-results-header">
                <div>
                  <h2 className="sheet-title">Optimization Request</h2>
                </div>
                <button
                  type="button"
                  className="overlay-close-button"
                  onClick={() => setOptimizationPanelOpen(false)}
                  aria-label="Close optimization Request"
                >
                  ×
                </button>
              </div>

              <div className="sheet-body">
                <div className="sheet-section-head">
                  <span className="panel-label">Mode selection</span>
                  <span className="sheet-step">Step 1</span>
                </div>
                <div className="mode-switch" role="tablist" aria-label="Optimization mode">
                  <button
                    type="button"
                    className={
                      optimizationMode === 'cash'
                        ? 'mode-button active'
                        : 'mode-button'
                    }
                    onClick={() => setOptimizationMode('cash')}
                  >
                    Cash optimization
                  </button>
                  <button
                    type="button"
                    className={
                      optimizationMode === 'power'
                        ? 'mode-button active'
                        : 'mode-button'
                    }
                    onClick={() => setOptimizationMode('power')}
                  >
                    Power optimization
                  </button>
                </div>

                <div className="sheet-section-head">
                  <span className="panel-label">Parameters</span>
                  <span className="sheet-step">Step 2</span>
                </div>
                <label className="input-block">
                  <span>{inputLabel}</span>
                  <div className="input-row">
                    <input
                      type="text"
                      value={optimizationValue}
                      onChange={(event) => setOptimizationValue(event.target.value)}
                      placeholder={inputPlaceholder}
                    />
                    <span className="input-unit">{inputUnit}</span>
                  </div>
                </label>

                <div className="sheet-section-head">
                  <span className="panel-label">Area selection</span>
                  <span className="sheet-step">
                    {hasBoundingBox ? 'Ready' : 'Step 3'}
                  </span>
                </div>
                <div className="selection-summary">
                  <strong>{boundingBoxSummary}</strong>
                  <p className="selection-hint">
                    Draw or edit a polygon on the map for the optimization area.
                  </p>
                </div>

                <div className="action-stack">
                  <button
                    type="button"
                    className={
                      boundingBoxSelectionActive
                        ? 'sidebar-action active'
                        : 'sidebar-action'
                    }
                    onClick={() =>
                      setBoundingBoxSelectionActive((current) => !current)
                    }
                  >
                    {boundingBoxSelectionActive
                      ? 'Cancel polygon'
                      : hasBoundingBox
                        ? 'Redraw polygon'
                        : 'Draw polygon'}
                  </button>
                  <button
                    type="button"
                    className="sidebar-action"
                    disabled={!hasBoundingBox}
                    onClick={() => {
                      if (!hasBoundingBox) {
                        return
                      }

                      startTransition(() => {
                        setBoundingBoxSelectionActive(false)
                        setEditSelectionRequest({
                          id: Date.now(),
                        })
                      })
                    }}
                  >
                    Edit selection
                  </button>
                  <button
                    type="button"
                    className="sidebar-action sidebar-action-danger"
                    disabled={!hasBoundingBox}
                    onClick={() => {
                      startTransition(() => {
                        setBoundingBoxSelectionActive(false)
                        setBoundingBox(null)
                      })
                    }}
                  >
                    Delete selection
                  </button>
                  <button
                    type="button"
                    className="sidebar-submit"
                    disabled={!canSubmitOptimization}
                    onClick={() => {
                      void submitOptimizationRequest()
                    }}
                  >
                    {optimizationSubmitting
                      ? 'Submitting...'
                      : 'Submit request'}
                  </button>
                </div>

                {optimizationStatusMessage ? (
                  <p className="optimization-status">{optimizationStatusMessage}</p>
                ) : null}
              </div>
            </div>
          ) : null}

          {optimizationResult ? (
            <div className="map-overlay overlay-optimization-results">
              <div className="legend-header overlay-results-header">
                <div>
                  <p className="panel-label">Optimization results</p>
                  <span className="legend-unit">Live</span>
                </div>
                <button
                  type="button"
                  className="overlay-close-button"
                  onClick={resetOptimizationWorkflow}
                  aria-label="Close optimization results"
                >
                  ×
                </button>
              </div>
              <div className="sidebar-results overlay-results-grid">
                <div className="result-row">
                  <span>Mode</span>
                  <strong>
                    {optimizationResult.mode === 'cash'
                      ? 'Budget target'
                      : 'Power target'}
                  </strong>
                </div>
                <div className="result-row">
                  <span>Sampled points</span>
                  <strong>{optimizationResult.sample_count.toLocaleString()}</strong>
                </div>
                <div className="result-row">
                  <span>Selected sites</span>
                  <strong>{optimizationResult.selected_count.toLocaleString()}</strong>
                </div>
                {optimizationResult.power_basis === 'average_hourly_kwh' ? (
                  <div className="result-row">
                    <span>Output basis</span>
                    <strong>Average hourly kWh</strong>
                  </div>
                ) : null}
                {optimizationResult.total_expected_power_kwh !== undefined ? (
                  <div className="result-row">
                    <span>Expected output</span>
                    <strong>
                      {Math.round(
                        optimizationResult.total_expected_power_kwh,
                      ).toLocaleString()}{' '}
                      kWh
                    </strong>
                  </div>
                ) : null}
                {optimizationResult.total_raw_power_kwh !== undefined ? (
                  <div className="result-row">
                    <span>Raw output</span>
                    <strong>
                      {Math.round(
                        optimizationResult.total_raw_power_kwh,
                      ).toLocaleString()}{' '}
                      kWh
                    </strong>
                  </div>
                ) : null}
                {optimizationResult.total_cost_usd !== undefined ? (
                  <div className="result-row">
                    <span>Total cost</span>
                    <strong>
                      {Math.round(
                        optimizationResult.total_cost_usd,
                      ).toLocaleString()}{' '}
                      USD
                    </strong>
                  </div>
                ) : null}
                {optimizationResult.total_power_kwh !== undefined ? (
                  <div className="result-row">
                    <span>Total output</span>
                    <strong>
                      {Math.round(
                        optimizationResult.total_power_kwh,
                      ).toLocaleString()}{' '}
                      kWh
                    </strong>
                  </div>
                ) : null}
                {optimizationResult.total_actual_cost_usd !== undefined ? (
                  <div className="result-row">
                    <span>Actual cost</span>
                    <strong>
                      {Math.round(
                        optimizationResult.total_actual_cost_usd,
                      ).toLocaleString()}{' '}
                      USD
                    </strong>
                  </div>
                ) : null}
                {optimizationResult.total_effective_cost_usd !== undefined ? (
                  <div className="result-row">
                    <span>Effective cost</span>
                    <strong>
                      {Math.round(
                        optimizationResult.total_effective_cost_usd,
                      ).toLocaleString()}{' '}
                      USD
                    </strong>
                  </div>
                ) : null}
              </div>
              <div className="overlay-results-actions">
                <button
                  type="button"
                  className="sidebar-submit"
                  disabled={downloadableOptimizationPointCount === 0}
                  onClick={downloadOptimizationGeoJson}
                >
                  Download sites as GeoJSON
                </button>
              </div>
            </div>
          ) : null}

          <div className="map-bottom-right">
            {!optimizationResult ? (
              <button
                type="button"
                className="optimization-fab"
                onClick={() => setOptimizationPanelOpen(true)}
              >
                Optimization Request
              </button>
            ) : null}

            {optimizationStatusMessage && !optimizationResult ? (
              <p className="map-status-pill">{optimizationStatusMessage}</p>
            ) : null}
          </div>
        </div>
      </section>
    </main>
  )
}

export default App
