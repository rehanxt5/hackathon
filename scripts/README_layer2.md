# Layer 2 — Context Seeder (SIH26162 / AGNI-Net)

`seed_layer2_colab.py` pulls the five context datasets that let us classify a FIRMS
hotspot instead of just plotting it. All free, no API keys, no logins.

| # | Dataset | Source | Output | Why it matters |
|---|---------|--------|--------|----------------|
| D1 | Industrial infrastructure | OpenStreetMap / Overpass API | `industrial_sites.csv` | distance-to-refinery / tank farm / kiln / mine = the strongest single feature |
| D2 | Power plants | WRI Global Power Plant DB | `power_plants_india.csv` | 1,589 Indian plants + fuel type; coal yards & gas flares live here |
| D3 | Persistent thermal sources | derived from **your** FIRMS archive | `persistent_sources.csv`, `hotspot_cells.csv` | free ground-truth labels for flare / coal-seam class (Jharia, Talcher, Hazira…) |
| D4 | Land cover | ESA WorldCover 10 m (AWS COGs) | `landcover_at_cells.csv` | cropland → stubble burn, tree cover → wildfire, built-up → industrial |
| D5 | Population | WorldPop 1 km India | `population_ind_1km.tif` | alert severity ("~2,300 people within 5 km") |

## Run in Colab

```python
from google.colab import drive; drive.mount('/content/drive')
!pip -q install rasterio
!python seed_layer2_colab.py --firms-glob "/content/drive/MyDrive/firms_history/*.csv" --out-dir /content/drive/MyDrive/layer2_seeds
```

D1 is the slow/fragile one (Overpass rate-limits). It is chunked into 2° tiles, rotated
across 6 mirrors, backed off exponentially, and **cached per tile** — if it dies, just
re-run the same command and it resumes from the cache.

Useful flags:

* `--skip D1,D5` — run only what you need
* `--tile-deg 1.5` — smaller Overpass tiles (slower, but survives rate limits better)
* `--workers 2` — be gentler on the mirrors

## Design decisions worth defending to judges

* **Overpass is pulled once, never at request time.** Static world = static table.
* **D3 replaces the World Bank flare database.** Point-level WB flare data thins out after
  2018; deriving flares from our own 19-month NOAA-21 archive (≥150 nights active in the
  same ~1 km cell) is fresher, India-specific, and needs zero manual labelling.
* **WorldCover is sampled, not downloaded.** Reading the COG overview level over
  `/vsicurl` gives ~80 m land cover per hotspot cell with a few MB of traffic per tile,
  instead of ~6 GB of raster.

## Verified (2026-09-04, on this machine)

* **D2** — GPPD downloaded, 1,589 Indian plants: Solar 851, Coal 253, Hydro 233, Wind 108, Gas 68, Oil 17, Nuclear 9.
* **D3** — persistence aggregation correct on synthetic archive: a 320-night cell and a
  333-night cell were both tagged `flare_or_coalseam`, an 8-night stubble cell and a
  2-night high-FRP cell were correctly left out.
* **D4** — live `/vsicurl` reads against ESA WorldCover succeeded; Jharia → `bare_sparse`,
  Punjab cell → `cropland`. ~5–10 s per tile, a few MB of traffic.
* **D1** — one 1°×1° tile over the Surat industrial belt returned **965 sites**
  (403 storage_tank, 319 industrial_zone, 117 factory, 78 chimney, 42 power_plant),
  correctly naming Kawas Thermal, SUGEN Mega, Utran Gas, Shell Energy.
  The first mirror answered HTTP 504 and the backoff rotated to overpass-api.de — as designed.
