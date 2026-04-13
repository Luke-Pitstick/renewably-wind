# GCS Migration Spec — Consolidate data & layer files to Google Cloud Storage

## Motivation
Today large data files (wind/transmission geojson, TIGER shapefiles, res-7 terrain cache, model pickle, map layer files) live in three places:

1. **Committed to git** — `src/app/frontend/public/data/*`, `src/python/backend/data/*`, `src/python/ml/data/*`. Repo is ~110MB+ and growing; GitHub warns/blocks >100MB single files.
2. **Bundled into the Modal image** — backend `image.add_local_file(...)` copies 250MB+ into every image build. Cold-start size and deploy time both scale with this.
3. **Informal one-offs** — GEE staging bucket (`gs://renewably-ee-staging`) already hosts the res-7 cells CSV.

Consolidating on a single GCS bucket gives us: (a) thinner repo, (b) thinner Modal image, (c) one source of truth, (d) ability to refresh data without redeploying code, (e) free CDN for static frontend layers via the bucket's public URL + optional Cloud CDN.

## Target layout

Single bucket: **`gs://renewably-data`** (multi-region `US`, uniform bucket-level access).

```
gs://renewably-data/
├── layers/                         # public, served to the browser
│   ├── us_wind_surface.geojson
│   ├── wind_farm_sites.geojson
│   ├── us_wind_speed_lower48.geojson
│   └── us_lower_48.geo.json
├── backend/                        # private-ish, Modal-readable
│   ├── terrain_res7_cache.csv
│   ├── us_wind_speed_dataset_2022.csv
│   ├── us_transmission_lines.geojson
│   ├── airports.geojson
│   └── roads/
│       ├── tl_2023_us_primaryroads.shp
│       ├── tl_2023_us_primaryroads.shx
│       ├── tl_2023_us_primaryroads.dbf
│       ├── tl_2023_us_primaryroads.prj
│       └── tl_2023_us_primaryroads.cpg
├── models/                         # private
│   ├── wind_xgboost_v2.pkl
│   └── wind_v2_feature_columns.json
└── ee-staging/                     # GEE exports land here (was gs://renewably-ee-staging)
    ├── res7_cells.csv
    └── exports/res7_terrain_*_shard*.csv
```

Access model:
- `layers/` — object-level `allUsers: Storage Object Viewer` (public read). Used directly by the browser.
- `backend/`, `models/`, `ee-staging/` — accessed via a service account that Modal and any ops scripts authenticate as.

## Phase 1 — Provision

```bash
# One-time setup
gcloud storage buckets create gs://renewably-data \
  --project=renewably --location=US --uniform-bucket-level-access

# Public read on layers/ only (object-level grant keeps rest private)
gcloud storage buckets add-iam-policy-binding gs://renewably-data \
  --member=allUsers --role=roles/storage.objectViewer \
  --condition='expression=resource.name.startsWith("projects/_/buckets/renewably-data/objects/layers/"),title=public-layers'

# Service account for Modal + scripts
gcloud iam service-accounts create renewably-runtime \
  --display-name="Renewably runtime (Modal + CI)"
gcloud storage buckets add-iam-policy-binding gs://renewably-data \
  --member=serviceAccount:renewably-runtime@renewably.iam.gserviceaccount.com \
  --role=roles/storage.objectViewer

# Key for Modal secret
gcloud iam service-accounts keys create ~/renewably-runtime.json \
  --iam-account=renewably-runtime@renewably.iam.gserviceaccount.com
modal secret create gcp-renewably GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat ~/renewably-runtime.json)"
rm ~/renewably-runtime.json
```

Optional CORS for frontend fetches:
```bash
cat > /tmp/cors.json <<EOF
[{"origin":["https://renewably-wind.onrender.com","http://localhost:5173"],
  "method":["GET","HEAD"],"responseHeader":["Content-Type"],"maxAgeSeconds":3600}]
EOF
gcloud storage buckets update gs://renewably-data --cors-file=/tmp/cors.json
```

## Phase 2 — Upload + remove from git

```bash
# Frontend layers
gcloud storage cp src/app/frontend/public/data/us_wind_surface.geojson gs://renewably-data/layers/
gcloud storage cp src/app/frontend/public/data/wind_farm_sites.geojson gs://renewably-data/layers/
gcloud storage cp src/app/frontend/public/data/us_wind_speed_lower48.geojson gs://renewably-data/layers/
gcloud storage cp src/app/frontend/public/data/us_lower_48.geo.json gs://renewably-data/layers/

# Backend seed data
gcloud storage cp src/python/backend/data/us_transmission_lines.geojson gs://renewably-data/backend/
gcloud storage cp src/python/backend/data/airports.geojson gs://renewably-data/backend/
gcloud storage cp "src/python/backend/data/tl_2023_us_primaryroads.*" gs://renewably-data/backend/roads/
gcloud storage cp src/python/ml/data/us_wind_speed_dataset_2022.csv gs://renewably-data/backend/
gcloud storage cp src/python/ml/data/renewably_exports/terrain_res7_cache.csv gs://renewably-data/backend/

# Models
gcloud storage cp src/python/ml/models/wind_xgboost_v2.pkl gs://renewably-data/models/
gcloud storage cp src/python/ml/models/wind_v2_feature_columns.json gs://renewably-data/models/
```

Then re-add to `.gitignore` (everything currently checked-in that now lives in GCS) and `git rm --cached` those files. Commit the removal in a separate PR so the history is clean.

## Phase 3 — Frontend

**Vars** — `src/app/frontend/.env`:
```
VITE_LAYER_BASE_URL=https://storage.googleapis.com/renewably-data/layers
```

**ArcGISMap.tsx** — replace the relative paths:
```ts
const LAYER_BASE = import.meta.env.VITE_LAYER_BASE_URL ?? '/data'
const WIND_DATA_URL        = `${LAYER_BASE}/us_wind_surface.geojson`
const WIND_FARMS_DATA_URL  = `${LAYER_BASE}/wind_farm_sites.geojson`
```
Fallback to `/data/*` preserves local dev-without-network.

Delete `src/app/frontend/public/data/us_wind_surface.geojson` and `wind_farm_sites.geojson` from the repo after the env var is wired up.

## Phase 4 — Backend (Modal)

Drop `image.add_local_file(...)` calls for the bulk data + model files. Replace with a startup-time GCS download into the Modal `Volume` (persistent across container starts). Rough shape in `App.py`:

```python
image = (
    modal.Image.debian_slim()
    .pip_install("google-cloud-storage", ...existing deps...)
    # keep small assets local; drop everything >5MB
)

GCS_BUCKET = "renewably-data"
SEED_DIR = Path("/seed")  # mounted via volume

def _ensure_asset(relpath: str) -> Path:
    local = SEED_DIR / relpath
    if local.exists():
        return local
    from google.cloud import storage
    client = storage.Client()
    blob = client.bucket(GCS_BUCKET).blob(relpath)
    local.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local)
    return local

@app.cls(image=image, volumes={"/seed": volume}, secrets=[modal.Secret.from_name("gcp-renewably")])
class ModelService:
    @modal.enter()
    def load(self):
        _ensure_asset("backend/terrain_res7_cache.csv")
        _ensure_asset("backend/us_wind_speed_dataset_2022.csv")
        _ensure_asset("backend/us_transmission_lines.geojson")
        _ensure_asset("backend/airports.geojson")
        for ext in ("shp", "shx", "dbf", "prj", "cpg"):
            _ensure_asset(f"backend/roads/tl_2023_us_primaryroads.{ext}")
        _ensure_asset("models/wind_xgboost_v2.pkl")
        _ensure_asset("models/wind_v2_feature_columns.json")
        # ... then rebind feature_engine path candidates to SEED_DIR
```

Update `feature_engine.py` `*_CANDIDATES` tuples to put `SEED_DIR` paths first, falling back to the current repo-relative paths for local runs without GCS (so tests keep working with synthetic data).

Modal secret `gcp-renewably` surfaces `GOOGLE_APPLICATION_CREDENTIALS_JSON`; `google.cloud.storage.Client()` picks it up automatically when `GOOGLE_APPLICATION_CREDENTIALS` points at a file. In a Modal container we write it to a tempfile in a tiny bootstrapper:

```python
import os, tempfile
if creds := os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
    fp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    fp.write(creds); fp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = fp.name
```

## Phase 5 — GEE pipeline

Rename bucket references in `ml/scripts/export_res7_terrain.py`: staging bucket becomes `gs://renewably-data/ee-staging/`. Keep the existing `gs://renewably-ee-staging` bucket for a grace period, then delete it once confirmed unused.

`download_res7_cache.py` reads shards directly from GCS instead of local Drive sync:
```bash
gcloud storage cp "gs://renewably-data/ee-staging/exports/res7_terrain_*.csv" /tmp/shards/
```

## Phase 6 — Cleanup

- `git rm --cached` the migrated files in one PR; bump `.gitignore` to prevent re-add.
- Verify `du -sh .git` shrinks; run `git gc --aggressive` on a throwaway clone to see the upper bound. If repo history is the concern, `git filter-repo` can retroactively strip the large blobs — but only do this if you're OK rewriting history.
- Delete `gs://renewably-ee-staging` once Phase 5 is in prod.
- Document the bucket layout in `docs/infra.md` (or wherever ops lives).

## Verification checklist
- [ ] `curl -I https://storage.googleapis.com/renewably-data/layers/us_wind_surface.geojson` → 200 from a browser origin (CORS included).
- [ ] `modal run backend/App.py::ModelService.predict_wind --lat 41.5 --lon -101.3` succeeds with GCS-sourced assets (no `add_local_file` for bulk data).
- [ ] Frontend deployed at Render shows wind speed + wind farm layers pulled from `storage.googleapis.com`.
- [ ] New Modal image size < 200MB (`modal image inspect`).
- [ ] Repo clone size < 30MB.
- [ ] `feature_engine` tests still pass locally against the synthetic cache (no GCS creds needed for tests).

## Cost (ballpark)
- Storage: ~0.5GB @ $0.020/GB·mo → **$0.01/mo**.
- Egress for public layers: ~50MB per uncached page view; at 1k views/mo ≈ 50GB → **$6/mo**. Cloud CDN drops this meaningfully if it becomes load-bearing.
- Modal runtime reads from `backend/` / `models/`: single cold-start download per container (~250MB) → **negligible** since containers are persistent and we cache to the Modal Volume.
