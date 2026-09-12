#!/usr/bin/env python3
"""
seed_layer2_colab.py  --  LAYER 2 SEEDER for SIH26162 (AGNI-Net)

Pulls the five CONTEXT datasets that turn a FIRMS hotspot into a classifiable event.

  D1  industrial_sites      OpenStreetMap via Overpass API   (refineries, factories,
                            tank farms, kilns, mines, quarries, industrial zones)
  D2  power_plants          WRI Global Power Plant Database  (lat/lon + fuel + MW)
  D3  persistent_sources    derived from YOUR FIRMS archive  (flare / coal-seam ground truth)
  D4  landcover             ESA WorldCover 10 m (sampled at hotspot cells, via overviews)
  D5  population            WorldPop 1 km India grid         (for alert severity, optional)

Design notes
  * Overpass is the only fragile dependency -> it is CHUNKED into a lat/lon grid,
    rotated across 6 public mirrors, backed off exponentially, and CACHED per tile,
    so a re-run resumes instead of restarting.
  * D2..D5 run in parallel threads alongside the Overpass worker pool.
  * Everything is free / no API key / no login.

Colab usage
  !python seed_layer2_colab.py --firms-glob "/content/drive/MyDrive/firms_history/*.csv"
"""

import argparse
import glob as globmod
import itertools
import json
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
INDIA_BBOX = (68.0, 6.0, 97.5, 37.5)          # west, south, east, north
OUT_DIR = "layer2_seeds"
FIRMS_GLOB = ""                                # e.g. "/content/drive/MyDrive/firms_history/*.csv"

OVERPASS_TILE_DEG = 4.0                        # fallback tile size when a whole-India shard fails
OVERPASS_WORKERS = 4                           # keep low: mirrors ban aggressive clients
OVERPASS_TIMEOUT = 90
OVERPASS_MAX_ROUNDS = 4                        # full re-sweeps over still-failing tiles

PERSIST_CELL_DEG = 0.01                        # ~1.1 km cell for persistence aggregation
PERSIST_MIN_NIGHTS = 30                        # >=30 distinct nights -> "persistent source"
PERSIST_FLARE_NIGHTS = 150                     # >=150 nights -> high-confidence flare/coal seam

LANDCOVER_DOWNSAMPLE = 8                       # 10 m * 8 = ~80 m sampling grid
LANDCOVER_MAX_TILES = 150   # India spans 96 WorldCover tiles; never silently drop cells

OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
# NOTE: overpass.osm.ch and overpass.osm.jp are deliberately EXCLUDED -- they serve
# regional extracts (Switzerland / Japan) and answer HTTP 200 with zero elements for
# an India bbox, which would silently poison the dataset rather than raise an error.

WRI_URLS = [
    "https://raw.githubusercontent.com/wri/global-power-plant-database/master/output_database/global_power_plant_database.csv",
    "https://wri-dataportal-prod.s3.amazonaws.com/manual/global_power_plant_database_v_1_3.zip",
]

WORLDPOP_URL = ("https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IND/"
                "ind_ppp_2020_1km_Aggregated.tif")

WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"

WORLDCOVER_CLASSES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built_up", 60: "bare_sparse", 70: "snow_ice", 80: "water",
    90: "herbaceous_wetland", 95: "mangroves", 100: "moss_lichen", 0: "unknown",
}

# --------------------------------------------------------------------------------------
# OSM tag -> category mapping (what counts as "industrial" for us)
# --------------------------------------------------------------------------------------
# Overpass uses a tag INDEX for exact `key=value` (and bare `key`) lookups, but a
# regex value match forces a full scan -- measured 6.8 s vs 187 s for the same tile.
# So: exact matches only, sharded BY TAG (each shard is cheap and independent) and
# only tiled geographically when a whole-India shard is too big for the server.
OVERPASS_SHARDS = {
    "works":        ['nwr["man_made"="works"]'],
    "storage_tank": ['nwr["man_made"="storage_tank"]'],
    "kiln":         ['nwr["man_made"="kiln"]'],
    "chimney":      ['nwr["man_made"="chimney"]'],
    "mining":       ['nwr["man_made"="mineshaft"]', 'nwr["man_made"="adit"]'],
    "quarry":       ['nwr["landuse"="quarry"]'],
    "oil_gas":      ['nwr["man_made"="petroleum_well"]', 'nwr["man_made"="gasometer"]',
                     'nwr["man_made"="gasworks"]'],
    "industrial":   ['nwr["industrial"]'],          # bare key -> still index-backed
    "power_plant":  ['nwr["power"="plant"]'],
    "landuse_ind":  ['nwr["landuse"="industrial"]'],
}


def categorise(tags):
    """Collapse messy OSM tags into the 7 categories the classifier cares about."""
    t = tags or {}
    ind = (t.get("industrial") or "").lower()
    mm = (t.get("man_made") or "").lower()
    lu = (t.get("landuse") or "").lower()
    name = (t.get("name") or "").lower()

    if ind == "refinery" or "refinery" in name or "refineries" in name:
        return "refinery"
    if ind in ("oil", "gas") or mm in ("petroleum_well", "gasometer", "gasworks"):
        return "oil_gas"
    if mm == "storage_tank":
        return "storage_tank"
    if mm in ("mineshaft", "adit") or lu == "quarry" or "colliery" in name or "coalfield" in name:
        return "mining"
    if t.get("power") == "plant":
        # 91% of OSM power=plant in this bbox is solar/hydro/wind -- those emit NO thermal
        # signature, so "near a power plant" would be a meaningless feature unless split.
        src = (t.get("plant:source") or t.get("generator:source") or "").lower()
        if src in ("solar", "wind", "hydro", "battery", "tidal", "geothermal"):
            return "power_plant_nonthermal"
        return "power_plant_thermal"
    if mm == "kiln" or "brick" in name:
        return "kiln"
    if mm == "works" or ind == "factory":
        return "factory"
    if mm == "chimney":
        return "chimney"
    if lu == "industrial":
        return "industrial_zone"
    return "other_industrial"


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
_print_lock = threading.Lock()


def log(msg, tag="*"):
    with _print_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}][{tag}] {msg}", flush=True)


def ensure(pkg, pipname=None):
    try:
        __import__(pkg)
        return True
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q {pipname or pkg}")
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False


def download(url, dest, tag="dl", retries=3, timeout=300):
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        log(f"cached  {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.1f} MB)", tag)
        return True
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code != 200:
                    log(f"HTTP {r.status_code} for {url[:70]}", tag)
                    time.sleep(2 ** attempt)
                    continue
                tmp = dest + ".part"
                n = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                        n += len(chunk)
                os.replace(tmp, dest)
                log(f"saved   {os.path.basename(dest)} ({n/1e6:.1f} MB)", tag)
                return True
        except Exception as e:
            log(f"retry {attempt+1}/{retries}: {type(e).__name__} {e}", tag)
            time.sleep(2 ** attempt * 2)
    return False


# --------------------------------------------------------------------------------------
# D1 -- OpenStreetMap industrial infrastructure (tag-sharded Overpass)
# --------------------------------------------------------------------------------------
def tile_grid(bbox, step):
    w, s, e, n = bbox
    lat = s
    while lat < n:
        lon = w
        while lon < e:
            yield (round(lat, 3), round(lon, 3),
                   round(min(lat + step, n), 3), round(min(lon + step, e), 3))
            lon += step
        lat += step


def overpass_query(selectors, tile):
    s, w, n, e = tile
    box = f"({s},{w},{n},{e})"
    body = "\n  ".join(f"{sel}{box};" for sel in selectors)
    # `out tags bb` returns each element's BOUNDING BOX, not just a centroid. A refinery
    # complex can be km across, so centroid-distance badly understates proximity for a
    # hotspot on its perimeter. The bbox lets enrichment measure distance to the FOOTPRINT.
    return f"[out:json][timeout:{OVERPASS_TIMEOUT}];\n(\n  {body}\n);\nout tags bb;"


class MirrorPool:
    """Round-robins mirrors and parks ones that just rate-limited us."""

    def __init__(self, mirrors):
        self._cycle = itertools.cycle(mirrors)
        self._cooldown = {m: 0.0 for m in mirrors}
        self._lock = threading.Lock()
        self._n = len(mirrors)

    def take(self):
        with self._lock:
            for _ in range(self._n):
                m = next(self._cycle)
                if time.time() >= self._cooldown[m]:
                    return m
            m = min(self._cooldown, key=self._cooldown.get)
        wait = max(0.0, self._cooldown[m] - time.time())
        if wait:
            time.sleep(min(wait, 30))
        return m

    def penalise(self, mirror, seconds):
        with self._lock:
            self._cooldown[mirror] = time.time() + seconds


def fetch_job(job, pool, cache_dir, tag="osm"):
    """job = (shard_name, selectors, tile). Returns (elements|None, source_label)."""
    shard, selectors, tile = job
    s, w, n, e = tile
    cache = os.path.join(cache_dir, f"{shard}__{s}_{w}_{n}_{e}.json")
    if os.path.exists(cache):
        try:
            with open(cache) as fh:
                return json.load(fh), "cached"
        except Exception:
            os.remove(cache)

    q = overpass_query(selectors, tile)
    for attempt in range(3):
        mirror = pool.take()
        host = mirror.split("/")[2]
        try:
            r = requests.post(mirror, data={"data": q}, timeout=OVERPASS_TIMEOUT + 30,
                              headers={"User-Agent": "AGNI-Net/SIH26162 (research)"})
        except Exception as ex:
            log(f"{shard} {s},{w} {type(ex).__name__} on {host}", tag)
            pool.penalise(mirror, 20)
            continue

        if r.status_code == 200:
            try:
                els = r.json().get("elements", [])
            except Exception:
                pool.penalise(mirror, 30)
                continue
            with open(cache, "w") as fh:
                json.dump(els, fh)
            time.sleep(0.4 + random.random() * 0.4)
            return els, host

        retry_after = float(r.headers.get("Retry-After", 0) or 0)
        back = max(retry_after, min(60, 5 * (2 ** attempt))) + random.random() * 3
        pool.penalise(mirror, back)
        log(f"{shard} {s},{w} HTTP {r.status_code} on {host} -> cooling {back:.0f}s", tag)
    return None, "failed"


def elements_to_rows(elements):
    rows = []
    for el in elements:
        b = el.get("bounds")
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
            min_lat = max_lat = lat
            min_lon = max_lon = lon
        elif b:
            min_lat, max_lat = b["minlat"], b["maxlat"]
            min_lon, max_lon = b["minlon"], b["maxlon"]
            lat, lon = (min_lat + max_lat) / 2, (min_lon + max_lon) / 2
        else:
            c = el.get("center") or {}
            lat, lon = c.get("lat"), c.get("lon")
            min_lat = max_lat = lat
            min_lon = max_lon = lon
        if lat is None or lon is None:
            continue
        # half-diagonal of the footprint in metres -- 0 for a point feature
        dlat_m = (max_lat - min_lat) * 111_320.0
        dlon_m = (max_lon - min_lon) * 111_320.0 * math.cos(math.radians(lat))
        tags = el.get("tags", {}) or {}
        rows.append({
            "osm_id": f"{el.get('type')}/{el.get('id')}",
            "latitude": lat,
            "longitude": lon,
            "min_lat": min_lat, "min_lon": min_lon,
            "max_lat": max_lat, "max_lon": max_lon,
            "extent_m": round(math.hypot(dlat_m, dlon_m) / 2, 1),
            "category": categorise(tags),
            "name": tags.get("name", ""),
            "operator": tags.get("operator", ""),
            "raw_tags": json.dumps({k: v for k, v in tags.items()
                                    if k in ("landuse", "man_made", "industrial", "power",
                                             "product", "resource", "substance",
                                             "plant:source", "generator:source",
                                             "plant:output:electricity")}),
        })
    return rows


def _whole_india_shard(job, pool, cache_dir, tag, attempts=3):
    """Try a whole-India shard a few times before condemning it to 240 tiled queries.
    A single 504 or a truncated 0-element answer is usually transient / one bad mirror."""
    for i in range(attempts):
        els, src = fetch_job(job, pool, cache_dir, tag)
        if els:
            return els, src
        stale = os.path.join(cache_dir,
                             f"{job[0]}__{job[2][0]}_{job[2][1]}_{job[2][2]}_{job[2][3]}.json")
        if os.path.exists(stale):
            os.remove(stale)          # never keep a 0-element whole-India answer
        if i < attempts - 1:
            why = "failed" if els is None else f"suspicious 0 elements from {src}"
            log(f"shard {job[0]} {why}; retry {i + 2}/{attempts}", tag)
            time.sleep(5 + 5 * i)
    return None, "failed"


def pull_industrial_sites(out_dir, bbox=INDIA_BBOX):
    """Pass 1: one whole-India query per tag shard (cheap, index-backed).
       Pass 2: any shard that failed is re-tried tiled at OVERPASS_TILE_DEG."""
    tag = "D1-osm"
    cache_dir = os.path.join(out_dir, "overpass_cache_bb")   # bumped: bbox payload format
    os.makedirs(cache_dir, exist_ok=True)
    pool = MirrorPool(OVERPASS_MIRRORS)
    w, s, e, n = bbox
    india = (s, w, n, e)

    all_rows, failed_shards = [], []
    jobs = [(name, sel, india) for name, sel in OVERPASS_SHARDS.items()]
    log(f"pass 1: {len(jobs)} whole-India tag shards", tag)
    with ThreadPoolExecutor(max_workers=OVERPASS_WORKERS) as ex:
        futs = {ex.submit(_whole_india_shard, j, pool, cache_dir, tag): j for j in jobs}
        for fut in as_completed(futs):
            shard = futs[fut][0]
            els, src = fut.result()
            if not els:
                log(f"shard {shard} -> exhausted retries; will tile", tag)
                failed_shards.append(futs[fut])
                continue
            all_rows.extend(elements_to_rows(els))
            log(f"shard {shard:13s} -> {len(els):6,} elements  [{src}]", tag)

    if failed_shards:
        tiles = list(tile_grid(bbox, OVERPASS_TILE_DEG))
        jobs2 = [(name, sel, t) for name, sel, _ in failed_shards for t in tiles]
        log(f"pass 2: {len(failed_shards)} shards x {len(tiles)} tiles = {len(jobs2)} queries", tag)
        for rnd in range(1, OVERPASS_MAX_ROUNDS + 1):
            if not jobs2:
                break
            still = []
            with ThreadPoolExecutor(max_workers=OVERPASS_WORKERS) as ex:
                futs = {ex.submit(fetch_job, j, pool, cache_dir, tag): j for j in jobs2}
                for fut in as_completed(futs):
                    els, src = fut.result()
                    if els is None:
                        still.append(futs[fut])
                    else:
                        all_rows.extend(elements_to_rows(els))
            log(f"pass 2 round {rnd}: {len(jobs2) - len(still)} ok, {len(still)} left", tag)
            jobs2 = still
            if jobs2:
                time.sleep(10)
        if jobs2:
            log(f"WARNING: {len(jobs2)} queries never succeeded (re-run to resume)", tag)

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["osm_id"])
    path = os.path.join(out_dir, "industrial_sites.csv")
    df.to_csv(path, index=False)
    log(f"industrial_sites.csv -> {len(df):,} unique sites", tag)
    if len(df):
        log("categories: " + ", ".join(f"{k}={v}" for k, v in
                                       df.category.value_counts().head(12).items()), tag)
    return {"rows": len(df), "path": path}


# --------------------------------------------------------------------------------------
# D2 -- WRI Global Power Plant Database
# --------------------------------------------------------------------------------------
def pull_power_plants(out_dir):
    tag = "D2-wri"
    raw = os.path.join(out_dir, "_gppd_raw")
    os.makedirs(raw, exist_ok=True)
    csv_path = None

    for url in WRI_URLS:
        dest = os.path.join(raw, url.rsplit("/", 1)[-1])
        if not download(url, dest, tag):
            continue
        if dest.endswith(".zip"):
            import zipfile
            try:
                with zipfile.ZipFile(dest) as z:
                    member = next(n for n in z.namelist() if n.endswith(".csv") and "database" in n)
                    z.extract(member, raw)
                    csv_path = os.path.join(raw, member)
            except Exception as e:
                log(f"zip failed: {e}", tag)
                continue
        else:
            csv_path = dest
        break

    if not csv_path:
        log("ERROR: could not obtain GPPD", tag)
        return {"rows": 0, "path": None}

    df = pd.read_csv(csv_path, low_memory=False)
    ind = df[df.country == "IND"].copy() if "country" in df.columns else df
    keep = [c for c in ["name", "gppd_idnr", "capacity_mw", "latitude", "longitude",
                        "primary_fuel", "other_fuel1", "commissioning_year", "owner"]
            if c in ind.columns]
    ind = ind[keep].dropna(subset=["latitude", "longitude"])
    path = os.path.join(out_dir, "power_plants_india.csv")
    ind.to_csv(path, index=False)
    log(f"power_plants_india.csv -> {len(ind):,} plants", tag)
    if "primary_fuel" in ind.columns:
        log("fuels: " + ", ".join(f"{k}={v}" for k, v in
                                  ind.primary_fuel.value_counts().head(8).items()), tag)
    return {"rows": len(ind), "path": path}


# --------------------------------------------------------------------------------------
# D3 -- persistent thermal sources derived from YOUR FIRMS archive
# --------------------------------------------------------------------------------------
def load_firms(firms_glob, tag="D3"):
    files = sorted(globmod.glob(firms_glob)) if firms_glob else []
    if not files:
        return None
    frames = []
    for f in files:
        try:
            head = pd.read_csv(f, nrows=0)
        except Exception as e:
            log(f"skip {os.path.basename(f)}: {e}", tag)
            continue
        cols = set(head.columns)
        want = [c for c in ["latitude", "longitude", "acq_date", "acq_time", "frp",
                            "daynight", "confidence", "satellite", "bright_ti4",
                            "bright_ti5", "brightness", "bright_t31", "type"] if c in cols]
        try:
            df = pd.read_csv(f, usecols=want, low_memory=False)
        except Exception as e:
            log(f"skip {os.path.basename(f)}: {e}", tag)
            continue
        # normalise MODIS column names onto the VIIRS schema
        if "brightness" in df.columns and "bright_ti4" not in df.columns:
            df = df.rename(columns={"brightness": "bright_ti4", "bright_t31": "bright_ti5"})
        df["src_file"] = os.path.basename(f)
        frames.append(df)
        log(f"read {os.path.basename(f)}: {len(df):,} rows", tag)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def derive_persistent_sources(df, out_dir, tag="D3-persist"):
    d = df.copy()
    d["latitude"] = pd.to_numeric(d.latitude, errors="coerce")
    d["longitude"] = pd.to_numeric(d.longitude, errors="coerce")
    d["frp"] = pd.to_numeric(d.get("frp"), errors="coerce")
    d = d.dropna(subset=["latitude", "longitude", "acq_date"])

    q = PERSIST_CELL_DEG
    d["cell_lat"] = (d.latitude / q).round().astype(int)
    d["cell_lon"] = (d.longitude / q).round().astype(int)

    # vectorise the day/night share BEFORE grouping: a per-group Python apply over
    # ~600k groups takes minutes, a boolean mean takes seconds.
    d["is_night"] = (d.get("daynight") == "N") if "daynight" in d.columns else False

    g = d.groupby(["cell_lat", "cell_lon"], sort=False)
    agg = g.agg(
        nights=("acq_date", "nunique"),
        detections=("acq_date", "size"),
        first_seen=("acq_date", "min"),
        last_seen=("acq_date", "max"),
        frp_mean=("frp", "mean"),
        frp_max=("frp", "max"),
        frp_std=("frp", "std"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        night_ratio=("is_night", "mean"),
    ).reset_index()

    # flat FRP + always-on  ->  flare / coal seam.  spiky FRP -> episodic burning.
    agg["frp_cv"] = agg.frp_std / agg.frp_mean.replace(0, float("nan"))
    persistent = agg[agg.nights >= PERSIST_MIN_NIGHTS].sort_values("nights", ascending=False)
    persistent["label_hint"] = "persistent_source"
    persistent.loc[persistent.nights >= PERSIST_FLARE_NIGHTS, "label_hint"] = "flare_or_coalseam"

    path = os.path.join(out_dir, "persistent_sources.csv")
    persistent.drop(columns=["cell_lat", "cell_lon"]).to_csv(path, index=False)
    log(f"persistent_sources.csv -> {len(persistent):,} cells "
        f"(>= {PERSIST_MIN_NIGHTS} nights); {int((persistent.nights >= PERSIST_FLARE_NIGHTS).sum())} "
        f"look like flares/coal seams", tag)
    for _, r in persistent.head(5).iterrows():
        log(f"  top: {r.latitude:.3f},{r.longitude:.3f}  {int(r.nights)} nights  "
            f"frp_mean={r.frp_mean:.1f}", tag)

    cells = agg[["latitude", "longitude", "nights", "detections"]].copy()
    cells_path = os.path.join(out_dir, "hotspot_cells.csv")
    cells.to_csv(cells_path, index=False)
    log(f"hotspot_cells.csv -> {len(cells):,} unique ~1 km cells (input for land cover)", tag)
    return {"persistent_rows": len(persistent), "cells": len(cells),
            "path": path, "cells_path": cells_path}


# --------------------------------------------------------------------------------------
# D4 -- ESA WorldCover 10 m, sampled at hotspot cells (overview reads, no bulk download)
# --------------------------------------------------------------------------------------
def worldcover_tile_name(lat, lon):
    tlat = int(math.floor(lat / 3.0) * 3)
    tlon = int(math.floor(lon / 3.0) * 3)
    ns = "N" if tlat >= 0 else "S"
    ew = "E" if tlon >= 0 else "W"
    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(tlat):02d}{ew}{abs(tlon):03d}_Map.tif", tlat, tlon


def sample_landcover(cells_csv, out_dir, tag="D4-lc"):
    if not ensure("rasterio"):
        log("rasterio unavailable -> land cover deferred to enrichment step", tag)
        return {"rows": 0, "path": None, "skipped": "no rasterio"}
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

    cells = pd.read_csv(cells_csv)
    cells["tile"] = [worldcover_tile_name(la, lo)[0]
                     for la, lo in zip(cells.latitude, cells.longitude)]
    groups = list(cells.groupby("tile"))
    groups.sort(key=lambda kv: -len(kv[1]))
    groups = groups[:LANDCOVER_MAX_TILES]
    log(f"{len(cells):,} cells across {len(groups)} WorldCover tiles", tag)

    out = []
    for i, (tile, sub) in enumerate(groups, 1):
        url = f"/vsicurl/{WORLDCOVER_BASE}/{tile}"
        try:
            with rasterio.open(url) as src:
                h = src.height // LANDCOVER_DOWNSAMPLE
                w = src.width // LANDCOVER_DOWNSAMPLE
                arr = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
                b = src.bounds
                xres = (b.right - b.left) / w
                yres = (b.top - b.bottom) / h
                col = ((sub.longitude.values - b.left) / xres).astype(int)
                row = ((b.top - sub.latitude.values) / yres).astype(int)
                ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
                vals = np.zeros(len(sub), dtype="uint8")
                vals[ok] = arr[row[ok], col[ok]]
                s = sub.copy()
                s["landcover_code"] = vals
                s["landcover"] = [WORLDCOVER_CLASSES.get(int(v), "unknown") for v in vals]
                out.append(s)
                log(f"[{i}/{len(groups)}] {tile} -> {len(sub):,} cells sampled", tag)
        except Exception as e:
            log(f"[{i}/{len(groups)}] {tile} failed ({type(e).__name__}) - skipped", tag)

    if not out:
        return {"rows": 0, "path": None, "skipped": "all tiles failed"}
    df = pd.concat(out, ignore_index=True).drop(columns=["tile"])
    path = os.path.join(out_dir, "landcover_at_cells.csv")
    df.to_csv(path, index=False)
    log(f"landcover_at_cells.csv -> {len(df):,} rows", tag)
    log("classes: " + ", ".join(f"{k}={v}" for k, v in
                                df.landcover.value_counts().head(8).items()), tag)
    return {"rows": len(df), "path": path}


# --------------------------------------------------------------------------------------
# D5 -- WorldPop 1 km population (for alert severity)
# --------------------------------------------------------------------------------------
def pull_population(out_dir, tag="D5-pop"):
    dest = os.path.join(out_dir, "population_ind_1km.tif")
    ok = download(WORLDPOP_URL, dest, tag, retries=2)
    return {"path": dest if ok else None, "ok": ok}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    global OVERPASS_TILE_DEG, OVERPASS_WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--firms-glob", default=FIRMS_GLOB,
                    help='e.g. "/content/drive/MyDrive/firms_history/*.csv"')
    ap.add_argument("--skip", default="", help="comma list of D1,D2,D3,D4,D5 to skip")
    ap.add_argument("--tile-deg", type=float, default=OVERPASS_TILE_DEG)
    ap.add_argument("--workers", type=int, default=OVERPASS_WORKERS)
    args = ap.parse_args()

    OVERPASS_TILE_DEG = args.tile_deg
    OVERPASS_WORKERS = args.workers

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    t0 = time.time()
    manifest = {"run_utc": datetime.now(timezone.utc).isoformat(), "bbox": INDIA_BBOX,
                "datasets": {}}

    log("LAYER 2 SEEDER  ---  five context datasets, all free, no API keys")
    log(f"output -> {os.path.abspath(out_dir)}")

    # D3 must precede D4 (it produces the cell list that D4 samples).
    def d3_then_d4():
        res = {}
        if "D3" in skip or not args.firms_glob:
            log("no --firms-glob given -> D3/D4 skipped "
                "(mount Drive and pass the FIRMS csv glob)", "D3")
            return {"D3": {"skipped": True}, "D4": {"skipped": True}}
        df = load_firms(args.firms_glob)
        if df is None or df.empty:
            log("no FIRMS rows matched the glob -> D3/D4 skipped", "D3")
            return {"D3": {"skipped": "no data"}, "D4": {"skipped": "no data"}}
        log(f"FIRMS archive loaded: {len(df):,} rows", "D3")
        res["D3"] = derive_persistent_sources(df, out_dir)
        if "D4" in skip:
            res["D4"] = {"skipped": True}
        else:
            res["D4"] = sample_landcover(res["D3"]["cells_path"], out_dir)
        return res

    jobs = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        if "D1" not in skip:
            jobs["D1"] = ex.submit(pull_industrial_sites, out_dir)
        if "D2" not in skip:
            jobs["D2"] = ex.submit(pull_power_plants, out_dir)
        jobs["D34"] = ex.submit(d3_then_d4)
        if "D5" not in skip:
            jobs["D5"] = ex.submit(pull_population, out_dir)

        for key, fut in jobs.items():
            try:
                r = fut.result()
            except Exception as e:
                log(f"{key} CRASHED: {type(e).__name__}: {e}", "!")
                r = {"error": f"{type(e).__name__}: {e}"}
            if key == "D34" and isinstance(r, dict) and "D3" in r:
                manifest["datasets"].update(r)
            else:
                manifest["datasets"][key] = r

    manifest["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    log("=" * 70)
    log(f"DONE in {manifest['seconds']/60:.1f} min -> {os.path.abspath(out_dir)}")
    for k, v in manifest["datasets"].items():
        log(f"  {k}: {v}")
    log("If any Overpass tiles failed, just re-run: cached tiles are skipped.")


if __name__ == "__main__":
    main()
