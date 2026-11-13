# Amazon Sentinel — Setup, Authentication & Scientific Methodology Documentation

## 1. System Overview & Architecture

Amazon Sentinel is a cloud-oriented geospatial data acquisition, remote-sensing processing, and FastAPI delivery system for monitoring the Amazon Basin across four satellite datasets:

1. **Sentinel-1 C-Band SAR (`COPERNICUS/S1_GRD`)** — Frequent 12-day all-weather flood and flooded-forest monitoring via Google Earth Engine.
2. **Global Forest Watch (`wur_radd_alerts`, `umd_glad_sentinel2_alerts`, `umd_glad_landsat_alerts`)** — Radar and optical canopy disturbance alerts via the WRI GFW Data API v3.
3. **NASA–ISRO NISAR L-Band SAR (`L2 GCOV` / `L2 GUNW`)** — 24 cm wavelength polarimetric covariance and sub-canopy double-bounce analysis via NASA Alaska Satellite Facility (ASF DAAC) Earthdata Cloud (`us-west-2`).
4. **NASA GPM IMERG V07 (`NASA/GPM_L3/IMERG_V07`)** — Half-hourly calibrated precipitation rate integration (`mm/hr * 0.5 hr`) via Google Earth Engine.

---

## 2. Environment Setup & Authentication

### 2.1 Install Dependencies

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r backend/requirements.txt
```

### 2.2 Configure `.env`

Copy `.env.example` to `.env` and populate your credentials:

```bash
cp .env.example .env
```

- **Google Earth Engine (`GEE_PROJECT_ID`, `GEE_SERVICE_ACCOUNT_EMAIL`, `GEE_PRIVATE_KEY_PATH`)**:
  Authenticate interactively via `earthengine authenticate` or provide a GCP service account key registered for Earth Engine.
- **Global Forest Watch (`GFW_API_KEY`)**:
  Obtain an API key from `https://data-api.globalforestwatch.org` and pass it via `GFW_API_KEY`.
- **NASA Earthdata / ASF DAAC (`EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` or `EARTHDATA_BEARER_TOKEN`)**:
  Used by `app/services/nisar_access.py` to exchange credentials at `https://nisar.asf.earthdatacloud.nasa.gov/s3credentials` for 1-hour temporary AWS STS credentials in `us-west-2`.
- **Transparent Demo Mode**:
  When external credentials are not configured in `.env`, the system never fabricates live status—it runs the exact same NumPy/Shapely/PyProj processing algorithms on calibrated reference observations and explicitly marks outputs with `status="demo"` and `result_nature="illustrative"`.

### 2.3 Run the FastAPI Server

```bash
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is served at `http://localhost:8000/docs`.

### 2.4 Run Unit & Integration Tests

```bash
PYTHONPATH=backend .venv/bin/pytest backend/tests/ -v
```

---

## 3. Scientific Methodology & Limitations

### 3.1 Sentinel-1 C-Band SAR Flood Detection (`COPERNICUS/S1_GRD`)
- **What GEE Already Performs**: Thermal noise removal, radiometric calibration to $\sigma^0$, Range-Doppler terrain correction using SRTM 30m DEM, and conversion to decibels ($10 \log_{10} \sigma^0$). Duplicate calibration or terrain correction is never re-applied.
- **Linear-Power Speckle Filtering**: Decibel grids are converted back to linear power ($\sigma^0 = 10^{\text{dB}/10}$) before applying a $5 \times 5$ Lee adaptive minimum-mean-square-error speckle filter ($\text{ENL} = 4.4$), avoiding the $\sim -2.5\text{ dB}$ logarithmic averaging bias.
- **5-Class Classification**:
  1. `open_water_inundation`: Specular reflection drop in both VH ($\le -17.5\text{ dB}$, $\Delta\text{VH} \le -2.8\text{ dB}$) and VV ($\le -12.0\text{ dB}$, $\Delta\text{VV} \le -2.5\text{ dB}$) on flat terrain ($\text{slope} \le 5^\circ$).
  2. `possible_flooded_vegetation`: Trunk-water dihedral double-bounce under forest canopy ($\Delta\text{VV} \ge +1.4\text{ dB}$, $\text{VV} \ge -7.2\text{ dB}$, $\text{VV} - \text{VH} \le 9.5\text{ dB}$).
  3. `permanent_water`: JRC Global Surface Water (`JRC/GSW1_4/GlobalSurfaceWater`) occurrence $\ge 80\%$.
  4. `uncertain`: Steep terrain shadow ($\text{slope} > 5^\circ$ with low backscatter), incidence angle outside $[30^\circ, 45^\circ]$, or borderline backscatter changes.
  5. `not_detected`: Remaining upland pixels.

### 3.2 Global Forest Watch Disturbance Alerts (`RADD` & `GLAD`)
- Queries `wur_radd_alerts` (Sentinel-1 C-band radar), `umd_glad_sentinel2_alerts` (Sentinel-2 optical), and `umd_glad_landsat_alerts` (Landsat 8/9 optical) while preserving each dataset's separate identity and original `confidence` (`high` vs `nominal`).
- **Limitation**: Canopy disturbance alerts indicate structural canopy loss (logging, fire, windthrow, clearing, or severe flood dieback) and do not automatically prove permanent deforestation.

### 3.3 NASA–ISRO NISAR L-Band SAR (`ASF DAAC`)
- Processed separately from Sentinel-1 via ASF Search API and windowed HDF5/NetCDF4 reads in AWS `us-west-2` (`/science/LSAR/GCOV/grids/frequencyA`).
- Explicitly separates **Direct Measurements** ($\gamma^0_{\text{HH}}$, $\gamma^0_{\text{HV}}$ in dB), **Derived Indicators** ($\text{HH}/\text{HV}$ ratio and repeat-pass $\Delta\text{HH}$ when $\ge 2$ passes exist), **Interpretations** (candidate sub-canopy double-bounce compared against Sentinel-1 C-band), and **Missing Observations**.

### 3.4 NASA GPM IMERG V07 Rainfall (`NASA/GPM_L3/IMERG_V07`)
- Each 30-minute granule reports `precipitation` as a **rate** in $\text{mm/hr}$.
- Every accumulation explicitly integrates over the $0.5\text{ hr}$ temporal interval:
  $$\text{Accumulated Rainfall (mm)} = \sum_{i} \text{Rate}_i\,(\text{mm/hr}) \times 0.5\text{ hr}$$

### 3.5 Combined Multi-Sensor Spatial & Temporal Analysis
- All polygon intersections (`Area(FloodGeometry ∩ ForestDisturbanceGeometry)`) and proximity buffers (configurable `distance_km`, default $5.0\text{ km}$) are computed in the **South America Albers Equal Area Conic** projection (`ESRI:102033`) and verified against WGS84 ellipsoid geodesic area integration.
- Direct intersection ($0\text{ km}$) is strictly separated from buffer-only proximity ($0 < d \le 5\text{ km}$).
- Rainfall context before, during, and after the flood window is provided as hydrological context, never as standalone causal proof.

---

## 4. Complete FastAPI Endpoint Reference

| Category | Method & Path | Description |
| :--- | :--- | :--- |
| **Health & Metadata** | `GET /api/health` | Service health and version check |
| | `GET /api/areas` | Predefined Amazon study areas with geodesic areas |
| | `GET /api/datasets/status` | Live credential and availability status of all 4 datasets |
| **Sentinel-1 Flood** | `GET /api/flood/coverage` | Scene discovery, temporal gaps, and orbit consistency |
| | `GET /api/flood/layer` | CesiumJS hybrid XYZ tile + GeoJSON flood layer & legend |
| | `GET /api/flood/statistics` | Inundated area in $\text{km}^2$ by class & period comparison |
| | `GET /api/flood/timeline` | Pass-by-pass flood observation timeline |
| **Forest Disturbance** | `GET /api/forest/alerts` | Paginated RADD & GLAD GeoJSON FeatureCollection |
| | `GET /api/forest/layer` | CesiumJS forest alert layer & legend |
| | `GET /api/forest/statistics` | Alert counts & $\text{km}^2$ breakdown by dataset & confidence |
| | `GET /api/forest/timeline` | Daily forest disturbance alert timeline |
| **NISAR L-Band** | `GET /api/nisar/coverage` | ASF DAAC granule discovery & repeat-pass check |
| | `GET /api/nisar/observations` | NISAR L-band observation timeline |
| | `GET /api/nisar/layer` | CesiumJS L-band polarimetric indicator layer |
| | `GET /api/nisar/statistics` | Direct measurements, derived indicators & S1 comparison |
| **GPM Rainfall** | `GET /api/rainfall/layer` | CesiumJS accumulated rainfall raster tile layer |
| | `GET /api/rainfall/statistics` | Accumulated rainfall ($\text{mm}$) & monthly summaries |
| | `GET /api/rainfall/timeline` | Daily integrated rainfall time series ($\text{mm/hr} \times 0.5\text{h}$) |
| **Combined Analysis** | `GET /api/analysis/summary` | Multi-sensor regional synthesis & rainfall context |
| | `GET /api/analysis/overlap` | `Area(Flood ∩ ForestDisturbance)` in $\text{km}^2$ & GeoJSON |
| | `GET /api/analysis/proximity` | Buffer proximity ($5\text{ km}$) vs direct overlap |
| | `GET /api/analysis/timeline` | Unified chronological multi-sensor timeline |
| **Jobs, Cache & Tiles** | `POST /api/processing/jobs` | Submit asynchronous processing job |
| | `GET /api/processing/jobs/{job_id}` | Poll job progress and status |
| | `GET /api/processing/jobs/{job_id}/results` | Retrieve completed job results |
| | `DELETE /api/cache` | Invalidate cached outputs by key, dataset, or region |
| | `GET /api/tiles/{dataset}/{region_id}/{z}/{x}/{y}.png` | Dynamic 256x256 RGBA PNG XYZ tiles for CesiumJS |
