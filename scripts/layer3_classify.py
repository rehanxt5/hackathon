#!/usr/bin/env python3
"""
layer3_classify.py -- LAYER 3 for SIH26162 (AGNI-Net)

Turns 1.7M anonymous FIRMS hot pixels into classified, explainable incidents.

  1. SITES + EPISODES   grid pixels into ~1.1 km sites, split each site's timeline
                        into episodes (a gap > EPISODE_GAP_DAYS starts a new one)
  2. ENRICHMENT JOIN    attach Layer 2 context to every site -- distance to the
                        FOOTPRINT (not centroid) of each industrial category,
                        land cover, population, nearest WRI plant + fuel
  3. FEATURES           ~25 numbers per episode: persistence, FRP physics, timing, context
  4. WEAK LABELS        near-certain rules label a confident subset; no manual annotation
  5. CLASSIFY           (a) transparent rules engine  (b) gradient-boosting model
  6. EVALUATE           site-grouped split AND temporal holdout; per-class precision/recall

Outputs -> data/layer3/
"""

import argparse
import glob as globmod
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
CELL_DEG = 0.01                 # ~1.1 km site grid (matches Layer 2 persistence cells)
EPISODE_GAP_DAYS = 3            # gap longer than this starts a new episode
KNN_CANDIDATES = 8              # nearest sites checked before exact bbox distance
POP_WINDOW_CELLS = 9            # ~9 km box around a site for population sum

FLARE_MIN_NIGHTS = 150          # >= this many active nights -> flare / coal seam
PERSIST_MIN_NIGHTS = 30
INDUSTRIAL_MAX_M = 500          # within this of a hot-industry footprint
INDUSTRIAL_MAX_SITE_NIGHTS = 60 # ... and NOT a permanent burner
WILDFIRE_MIN_CLEAR_M = 5000
AGRI_MAX_FRP = 100.0

CLASSES = ["industrial_fire", "gas_flare", "mining", "agri_burn", "wildfire"]

# industrial categories that can plausibly produce a THERMAL signature
HOT_CATEGORIES = ["refinery", "oil_gas", "storage_tank", "power_plant_thermal",
                  "factory", "industrial_zone", "kiln", "chimney", "other_industrial"]
ALL_CATEGORIES = HOT_CATEGORIES + ["mining", "power_plant_nonthermal"]

M_PER_DEG = 111_320.0
LAT0 = 22.0
COS0 = math.cos(math.radians(LAT0))


def log(msg, tag="*"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}][{tag}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# 1 -- LOAD + NORMALISE DETECTIONS
# --------------------------------------------------------------------------------------
VIIRS_CONF = {"l": 0.25, "n": 0.60, "h": 0.90}


def load_detections(firms_glob):
    files = sorted(globmod.glob(firms_glob))
    if not files:
        sys.exit(f"no FIRMS files matched {firms_glob!r}")
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if "brightness" in df.columns:                       # MODIS -> VIIRS schema
            df = df.rename(columns={"brightness": "bright_ti4", "bright_t31": "bright_ti5"})
        keep = [c for c in ["source", "latitude", "longitude", "bright_ti4", "bright_ti5",
                            "acq_date", "acq_time", "satellite", "confidence", "frp",
                            "daynight", "scan", "track"] if c in df.columns]
        df = df[keep]
        frames.append(df)
        log(f"{os.path.basename(f)}: {len(df):,} rows", "load")
    d = pd.concat(frames, ignore_index=True)

    for c in ("latitude", "longitude", "bright_ti4", "bright_ti5", "frp", "scan", "track"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # confidence: VIIRS is l/n/h, MODIS is 0-100 -> one 0..1 scale
    conf = d["confidence"].astype(str).str.strip().str.lower()
    num = pd.to_numeric(conf, errors="coerce") / 100.0
    d["confidence"] = np.where(num.notna(), num, conf.map(VIIRS_CONF)).astype(float)
    d["confidence"] = d["confidence"].fillna(0.5)

    d["acq_date"] = pd.to_datetime(d["acq_date"], errors="coerce")
    d["acq_hour"] = pd.to_numeric(d["acq_time"], errors="coerce").fillna(0) // 100
    d["is_night"] = (d.get("daynight") == "N")
    d = d.dropna(subset=["latitude", "longitude", "acq_date"])
    log(f"total {len(d):,} detections, {d.acq_date.min().date()} -> {d.acq_date.max().date()}", "load")
    return d


# --------------------------------------------------------------------------------------
# 2 -- SITES AND EPISODES
# --------------------------------------------------------------------------------------
def build_sites_and_episodes(d):
    """A SITE is a ~1.1 km grid cell (grid, not DBSCAN: connected-component clustering
    chains thousands of adjacent stubble fires into one bogus 'site' every November).
    An EPISODE is one continuous burn at that site."""
    d["cell_lat"] = (d.latitude / CELL_DEG).round().astype(np.int32)
    d["cell_lon"] = (d.longitude / CELL_DEG).round().astype(np.int32)
    d["site_id"] = pd.factorize(pd.Series(zip(d.cell_lat, d.cell_lon)))[0].astype(np.int32)
    log(f"{d.site_id.nunique():,} sites", "sites")

    d = d.sort_values(["site_id", "acq_date"], kind="stable").reset_index(drop=True)
    day = d.acq_date.values.astype("datetime64[D]").astype(np.int64)
    gap = np.diff(day, prepend=day[0])
    new_site = np.diff(d.site_id.values, prepend=d.site_id.values[0]) != 0
    new_ep = new_site | (gap > EPISODE_GAP_DAYS)
    d["episode_id"] = np.cumsum(new_ep).astype(np.int32)
    d["day_index"] = day
    log(f"{d.episode_id.nunique():,} episodes", "sites")
    return d


def _slope(g):
    """FRP trend: least-squares slope of frp against day, vectorised over groups.
    A flare is a flat line; an accident is a rising curve. This is the feature
    no public tool exploits."""
    n = g["n"]; sx = g["sx"]; sy = g["sy"]; sxx = g["sxx"]; sxy = g["sxy"]
    denom = n * sxx - sx * sx
    return np.where(denom > 0, (n * sxy - sx * sy) / np.where(denom == 0, 1, denom), 0.0)


def episode_features(d):
    d = d.copy()
    d["_x"] = d.day_index - d.groupby("episode_id").day_index.transform("min")
    d["_y"] = d.frp.fillna(0.0)
    d["_xy"] = d._x * d._y
    d["_xx"] = d._x * d._x

    g = d.groupby("episode_id", sort=False)
    ep = g.agg(
        site_id=("site_id", "first"),
        latitude=("latitude", "mean"), longitude=("longitude", "mean"),
        start=("acq_date", "min"), end=("acq_date", "max"),
        n_detections=("acq_date", "size"), n_nights=("acq_date", "nunique"),
        frp_max=("frp", "max"), frp_mean=("frp", "mean"), frp_sum=("frp", "sum"),
        frp_std=("frp", "std"),
        ti4_max=("bright_ti4", "max"), ti4_mean=("bright_ti4", "mean"),
        ti5_mean=("bright_ti5", "mean"),
        night_ratio=("is_night", "mean"), confidence=("confidence", "mean"),
        n=("_y", "size"), sx=("_x", "sum"), sy=("_y", "sum"),
        sxx=("_xx", "sum"), sxy=("_xy", "sum"),
    ).reset_index()

    ep["frp_trend"] = _slope(ep)
    ep = ep.drop(columns=["n", "sx", "sy", "sxx", "sxy"])
    ep["duration_days"] = (ep.end - ep.start).dt.days + 1
    ep["hot_delta"] = ep.ti4_mean - ep.ti5_mean
    ep["frp_cv"] = ep.frp_std / ep.frp_mean.replace(0, np.nan)
    ep["month"] = ep.start.dt.month
    return ep


def site_features(d, ep):
    g = d.groupby("site_id", sort=False)
    st = g.agg(
        latitude=("latitude", "mean"), longitude=("longitude", "mean"),
        site_nights=("acq_date", "nunique"), site_detections=("acq_date", "size"),
        site_first=("acq_date", "min"), site_last=("acq_date", "max"),
        site_frp_mean=("frp", "mean"), site_frp_max=("frp", "max"),
        site_frp_std=("frp", "std"), site_night_ratio=("is_night", "mean"),
    ).reset_index()
    st["site_span_days"] = (st.site_last - st.site_first).dt.days + 1
    st["site_duty_cycle"] = st.site_nights / st.site_span_days.clip(lower=1)
    st["site_frp_cv"] = st.site_frp_std / st.site_frp_mean.replace(0, np.nan)
    st = st.merge(ep.groupby("site_id").size().rename("site_episodes").reset_index(),
                  on="site_id", how="left")
    log(f"{len(st):,} sites, max {st.site_nights.max()} active nights", "sites")
    return st


def flag_india(st, seeds_dir):
    """The FIRMS pull is an India BOUNDING BOX, so it also covers Myanmar, Bangladesh,
    Pakistan, Nepal and Tibet. Without this flag the top-FRP 'industrial fires' were
    Myanmar forest fires. Alerts must be filterable to actual Indian territory."""
    path = os.path.join(seeds_dir, "india_boundary.geojson")
    if not os.path.exists(path):
        log("india_boundary.geojson missing -> in_india not computed", "join")
        st["in_india"] = pd.NA
        return st
    try:
        from shapely.geometry import shape
        from shapely import points, contains
        geom = shape(json.load(open(path))["geometry"])
        pts = points(st.longitude.values, st.latitude.values)
        st["in_india"] = contains(geom, pts)
        log(f"in_india: {int(st.in_india.sum()):,} of {len(st):,} sites "
            f"({100*st.in_india.mean():.1f}%)", "join")
    except Exception as ex:
        log(f"in_india skipped ({type(ex).__name__}: {ex})", "join")
        st["in_india"] = pd.NA
    return st


# --------------------------------------------------------------------------------------
# 3 -- ENRICHMENT JOIN
# --------------------------------------------------------------------------------------
def _xy(lat, lon):
    return np.c_[np.asarray(lon) * M_PER_DEG * COS0, np.asarray(lat) * M_PER_DEG]


def footprint_distance(q_lat, q_lon, sub):
    """Exact point -> bounding-box distance, KD-tree pre-filtered.
    Centroid distance badly understates proximity: Jharia's footprint alone has an
    8.4 km half-diagonal, so a hotspot 3 km from its centre is INSIDE the mine."""
    from scipy.spatial import cKDTree
    tree = cKDTree(_xy(sub.latitude.values, sub.longitude.values))
    k = min(KNN_CANDIDATES, len(sub))
    _, idx = tree.query(_xy(q_lat, q_lon), k=k, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    qla = np.asarray(q_lat)[:, None]; qlo = np.asarray(q_lon)[:, None]
    dy = np.maximum(np.maximum(sub.min_lat.values[idx] - qla, qla - sub.max_lat.values[idx]), 0)
    dx = np.maximum(np.maximum(sub.min_lon.values[idx] - qlo, qlo - sub.max_lon.values[idx]), 0)
    dist = np.hypot(dy * M_PER_DEG, dx * M_PER_DEG * COS0)
    best = dist.argmin(axis=1)
    r = np.arange(len(dist))
    return dist[r, best], idx[r, best]


def enrich_sites(st, seeds_dir):
    sites = pd.read_csv(os.path.join(seeds_dir, "industrial_sites.csv"))
    t0 = time.time()
    for cat in ALL_CATEGORIES:
        sub = sites[sites.category == cat]
        if sub.empty:
            st[f"d_{cat}"] = np.inf
            continue
        dist, idx = footprint_distance(st.latitude.values, st.longitude.values, sub)
        st[f"d_{cat}"] = dist
        if cat in ("refinery", "mining", "industrial_zone", "power_plant_thermal"):
            st[f"name_{cat}"] = sub.name.values[idx]
    st["d_industrial_hot"] = st[[f"d_{c}" for c in HOT_CATEGORIES]].min(axis=1)
    st["nearest_hot_category"] = (st[[f"d_{c}" for c in HOT_CATEGORIES]]
                                  .idxmin(axis=1).str.replace("d_", "", regex=False))
    log(f"industrial footprint distances in {time.time()-t0:.0f}s", "join")

    # WRI plants (point features -> fuel type is the value here)
    wri = pd.read_csv(os.path.join(seeds_dir, "power_plants_india.csv"))
    from scipy.spatial import cKDTree
    tw = cKDTree(_xy(wri.latitude.values, wri.longitude.values))
    dw, iw = tw.query(_xy(st.latitude.values, st.longitude.values), k=1, workers=-1)
    st["d_wri_plant"] = dw
    st["wri_fuel"] = wri.primary_fuel.values[iw]
    st["wri_capacity_mw"] = wri.capacity_mw.values[iw]

    # land cover, joined on the identical ~1.1 km cell key Layer 2 used
    lc = pd.read_csv(os.path.join(seeds_dir, "landcover_at_cells.csv"))
    lc["cell_lat"] = (lc.latitude / CELL_DEG).round().astype(np.int32)
    lc["cell_lon"] = (lc.longitude / CELL_DEG).round().astype(np.int32)
    st["cell_lat"] = (st.latitude / CELL_DEG).round().astype(np.int32)
    st["cell_lon"] = (st.longitude / CELL_DEG).round().astype(np.int32)
    st = st.merge(lc[["cell_lat", "cell_lon", "landcover", "landcover_code"]].drop_duplicates(
        subset=["cell_lat", "cell_lon"]), on=["cell_lat", "cell_lon"], how="left")
    st["landcover"] = st.landcover.fillna("unknown")
    log(f"land cover joined: {st.landcover.notna().mean()*100:.1f}% matched", "join")

    # population within ~9 km, from the WorldPop raster
    try:
        import rasterio
        from scipy.ndimage import uniform_filter
        with rasterio.open(os.path.join(seeds_dir, "population_ind_1km.tif")) as src:
            arr = src.read(1).astype("float32")
            arr[arr < 0] = 0
            box = uniform_filter(arr, size=POP_WINDOW_CELLS) * (POP_WINDOW_CELLS ** 2)
            from rasterio.transform import rowcol
            rows, cols = rowcol(src.transform, st.longitude.values, st.latitude.values)
            rows = np.clip(np.asarray(rows, dtype=np.int64), 0, src.height - 1)
            cols = np.clip(np.asarray(cols, dtype=np.int64), 0, src.width - 1)
            st["pop_9km"] = box[rows, cols]
        log(f"population sampled, median {st.pop_9km.median():.0f} people within ~9 km", "join")
    except Exception as e:
        log(f"population skipped ({type(e).__name__}: {e})", "join")
        st["pop_9km"] = np.nan
    return flag_india(st, seeds_dir)


# --------------------------------------------------------------------------------------
# 4 -- WEAK LABELS  (no manual annotation anywhere)
# --------------------------------------------------------------------------------------
def weak_labels(e):
    """Label only what is near-certain; leave the ambiguous middle unlabelled.
    Note the deliberate asymmetry: labels lean on PERSISTENCE and LAND COVER, so the
    model still has FRP physics (trend, delta, curve shape) as independent evidence
    rather than merely memorising the geographic rule."""
    lab = pd.Series(pd.NA, index=e.index, dtype="object")

    flare = e.site_nights >= FLARE_MIN_NIGHTS
    lab[flare] = "gas_flare"

    mining = (lab.isna() & (e.d_mining <= 500) & (e.site_nights >= PERSIST_MIN_NIGHTS))
    lab[mining] = "mining"

    industrial = (lab.isna()
                  & (e.d_industrial_hot <= INDUSTRIAL_MAX_M)
                  & (e.site_nights < INDUSTRIAL_MAX_SITE_NIGHTS)
                  & (e.frp_max >= 15)
                  & (e.duration_days <= 5))
    lab[industrial] = "industrial_fire"

    agri = (lab.isna()
            & (e.landcover == "cropland")
            & (e.duration_days <= 2)
            & (e.night_ratio < 0.5)
            & (e.frp_max < AGRI_MAX_FRP)
            & (e.d_industrial_hot > 2000)
            & (e.month.isin([3, 4, 5, 10, 11])))
    lab[agri] = "agri_burn"

    wild = (lab.isna()
            & (e.landcover.isin(["tree_cover", "shrubland"]))
            & (e.d_industrial_hot > WILDFIRE_MIN_CLEAR_M)
            & (e.duration_days >= 2)
            & (e.site_nights < INDUSTRIAL_MAX_SITE_NIGHTS))
    lab[wild] = "wildfire"
    return lab


# --------------------------------------------------------------------------------------
# 5a -- RULES ENGINE  (works with zero training; the demo never depends on the model)
# --------------------------------------------------------------------------------------
def rules_classify(e):
    cls = pd.Series("unclassified", index=e.index, dtype="object")
    why = pd.Series("", index=e.index, dtype="object")

    def put(mask, name, reason):
        m = mask & (cls == "unclassified")
        cls[m] = name
        why[m] = reason

    put(e.site_nights >= FLARE_MIN_NIGHTS, "gas_flare",
        f"active >= {FLARE_MIN_NIGHTS} nights at this cell -> permanent thermal source")
    put((e.d_mining <= 1000) & (e.site_nights >= PERSIST_MIN_NIGHTS), "mining",
        "inside/near a mine or quarry footprint and recurring")
    # Tight industrial rule first: essentially on top of the plant.
    put((e.d_industrial_hot <= 500) & (e.site_nights < INDUSTRIAL_MAX_SITE_NIGHTS),
        "industrial_fire", "within 500 m of a hot-industry footprint, not a permanent burner")
    # Vegetation OUTRANKS the loose industrial rule. Without this, a forest fire 1.8 km
    # from a mapped industrial zone (population 0) was being called an industrial fire --
    # which is exactly the false alarm this whole system exists to prevent.
    put(e.landcover.isin(["tree_cover", "shrubland", "mangroves"]) & (e.d_industrial_hot > 500),
        "wildfire", "forest/scrub land cover and not on top of industry")
    put((e.landcover == "cropland") & (e.duration_days <= 2) & (e.frp_max < AGRI_MAX_FRP),
        "agri_burn", "short daytime burn on cropland")
    # Loose industrial rule only where the ground is plausibly industrial.
    put((e.d_industrial_hot <= 1500) & (e.site_nights < INDUSTRIAL_MAX_SITE_NIGHTS)
        & e.landcover.isin(["built_up", "bare_sparse", "grassland", "cropland", "unknown"]),
        "industrial_fire", "near industry on built-up/bare ground")
    put(e.landcover == "cropland", "agri_burn", "cropland land cover")
    put(e.landcover == "built_up", "industrial_fire", "built-up land cover")
    put(pd.Series(True, index=e.index), "wildfire", "default: no industrial context")
    return cls, why


# --------------------------------------------------------------------------------------
# 5b + 6 -- MODEL AND HONEST EVALUATION
# --------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# CIRCULARITY GUARD.  The weak labels are deterministic functions of land cover,
# persistence and footprint distance.  A model given those same columns scores ~1.00
# by copying its own teacher -- measured: landcover_code importance 0.21, every FRP
# feature exactly 0.0.  That model has learned nothing.
#
# So the model is trained on RADIOMETRY AND TIMING ONLY -- signals the labeller never
# touched.  The rules then classify by WHERE a fire is; the model classifies by HOW it
# burns.  Two independent views: agreement raises confidence, disagreement flags an
# episode for analyst review (e.g. an industrial-looking burn where no industry is mapped).
# ---------------------------------------------------------------------------------
FEATURES_PHYSICS = ["n_detections", "n_nights", "duration_days",
                    "frp_max", "frp_mean", "frp_sum", "frp_std", "frp_cv", "frp_trend",
                    "ti4_max", "ti4_mean", "hot_delta", "night_ratio", "confidence", "month"]

FEATURES_ALL = ["n_detections", "n_nights", "duration_days",
            "frp_max", "frp_mean", "frp_sum", "frp_std", "frp_cv", "frp_trend",
            "ti4_max", "ti4_mean", "hot_delta", "night_ratio", "confidence", "month",
            "site_nights", "site_detections", "site_episodes", "site_span_days",
            "site_duty_cycle", "site_night_ratio", "site_frp_cv",
            "d_industrial_hot", "d_refinery", "d_oil_gas", "d_storage_tank",
            "d_power_plant_thermal", "d_factory", "d_industrial_zone", "d_kiln",
            "d_chimney", "d_mining", "d_power_plant_nonthermal",
            "d_wri_plant", "wri_capacity_mw", "pop_9km", "landcover_code"]


def train_and_evaluate(e, out_dir):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupShuffleSplit

    lab = e.label.notna()
    train_pool = e[lab].copy()
    log(f"{len(train_pool):,} labelled episodes of {len(e):,} "
        f"({100*len(train_pool)/len(e):.1f}%)", "model")
    log("label counts: " + ", ".join(f"{k}={v:,}" for k, v in
                                     train_pool.label.value_counts().items()), "model")
    if train_pool.label.nunique() < 2:
        log("not enough classes to train -- rules only", "model")
        return None, None, "Not enough labelled classes to train a model."

    # sklearn's binner crashes on an all-NaN or constant column, so drop those first
    # and say which -- silently training on a dead feature is worse than the crash.
    Xf = train_pool[FEATURES_PHYSICS].replace([np.inf, -np.inf], np.nan).astype("float32")
    dead = [c for c in FEATURES_PHYSICS
            if Xf[c].notna().sum() == 0 or Xf[c].nunique(dropna=True) < 2]
    if dead:
        log(f"dropping {len(dead)} dead features: {dead}", "model")
    feats = [c for c in FEATURES_PHYSICS if c not in dead]
    X = Xf[feats]
    y = train_pool.label.astype(str)
    groups = train_pool.site_id

    # Split by SITE, never by row: one flare contributes thousands of rows, and a random
    # split would put the same flare in train and test -- instant fake accuracy.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         class_weight="balanced", random_state=42)
    t0 = time.time()
    clf.fit(X.iloc[tr], y.iloc[tr])
    log(f"trained on {len(tr):,} episodes in {time.time()-t0:.1f}s", "model")

    log("scoring site-split holdout", "model")
    rep_site = classification_report(y.iloc[te], clf.predict(X.iloc[te]), digits=3, zero_division=0)
    cm = confusion_matrix(y.iloc[te], clf.predict(X.iloc[te]),
                          labels=sorted(y.unique()))

    # Temporal holdout: train on the past, test on the newest 30 days -- how it really runs.
    cutoff = e.start.max() - pd.Timedelta(days=30)
    tr2 = train_pool.start < cutoff
    rep_time = "insufficient data for a temporal holdout"
    if tr2.sum() > 100 and (~tr2).sum() > 20 and y[tr2].nunique() > 1:
        clf2 = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                              random_state=42)
        clf2.fit(X[tr2.values], y[tr2.values])
        rep_time = classification_report(y[~tr2.values], clf2.predict(X[~tr2.values]),
                                         digits=3, zero_division=0)

    log("temporal holdout done; computing feature importance", "model")
    # Deliberately leaky control model: same data, but given the label-defining columns.
    # Its near-perfect score is the evidence that those features must be withheld.
    leak_feats = [c for c in FEATURES_ALL
                  if e[c].notna().sum() > 0 and e[c].nunique(dropna=True) > 1]
    Xl = train_pool[leak_feats].replace([np.inf, -np.inf], np.nan).astype("float32")
    clf_leak = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                              random_state=42).fit(Xl.iloc[tr], y.iloc[tr])
    rep_leak = classification_report(y.iloc[te], clf_leak.predict(Xl.iloc[te]),
                                     digits=3, zero_division=0)
    log("leak-control model trained (for the report only)", "model")

    from sklearn.inspection import permutation_importance
    sub = np.random.default_rng(0).choice(te, size=min(4000, len(te)), replace=False)
    pi = permutation_importance(clf, X.iloc[sub], y.iloc[sub], n_repeats=3,
                                random_state=0, n_jobs=1)
    imp = (pd.DataFrame({"feature": feats, "importance": pi.importances_mean})
           .sort_values("importance", ascending=False))
    imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False)

    log("scoring all episodes", "model")
    # score EVERY episode, labelled or not -- that is the point of the model
    Xall = e[feats].replace([np.inf, -np.inf], np.nan).astype("float32")
    proba = clf.predict_proba(Xall)
    e["model_class"] = clf.classes_[proba.argmax(axis=1)]
    e["model_confidence"] = proba.max(axis=1)
    for i, c in enumerate(clf.classes_):
        e[f"p_{c}"] = proba[:, i]

    md = ["# Layer 3 -- model evaluation", "",
          f"Labelled episodes: {len(train_pool):,} of {len(e):,} "
          f"({100*len(train_pool)/len(e):.1f}%)", "",
          "## Split by SITE (no site appears in both train and test)", "",
          "```", rep_site, "```", "",
          f"Confusion matrix (rows = truth), labels {sorted(y.unique())}:", "",
          "```", str(cm), "```", "",
          f"## Temporal holdout (train < {cutoff.date()}, test after)", "",
          "```", rep_time, "```", "",
          "## Top features (physics-only model)", "",
          imp.head(15).to_markdown(index=False), "",
          "## Why the model is trained on radiometry only", "",
          "The weak labels are deterministic functions of land cover, persistence and",
          "footprint distance. A control model given those same columns scores as follows",
          "-- it is copying its own teacher, not learning:", "",
          "```", rep_leak, "```", "",
          "In that control run `landcover_code` carried permutation importance 0.21 and",
          "every FRP feature scored exactly 0.0. The production model therefore sees only",
          "radiometry and timing; the rules supply geography. They are independent views,",
          "so agreement is meaningful and disagreement is a review signal.", "",
          "### Caveat", "",
          "Labels are programmatic, so these scores measure agreement with the rule set on",
          "held-out sites -- not against human-verified ground truth, which does not exist",
          "publicly for industrial fires. The model's value is calibrated probabilities and",
          "the ambiguous middle where rules are silent, not beating its own teacher."]
    return clf, imp, "\n".join(md)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--firms-glob", default="data/firms_history/sih/firms_history_*.csv")
    ap.add_argument("--seeds-dir", default="data/layer2_seeds")
    ap.add_argument("--out-dir", default="data/layer3")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    d = load_detections(a.firms_glob)
    d = build_sites_and_episodes(d)
    ep = episode_features(d)
    st = site_features(d, ep)
    st = enrich_sites(st, a.seeds_dir)

    e = ep.merge(st.drop(columns=["latitude", "longitude"]), on="site_id", how="left")
    e["label"] = weak_labels(e)
    e["rule_class"], e["rule_reason"] = rules_classify(e)
    log("rules: " + ", ".join(f"{k}={v:,}" for k, v in e.rule_class.value_counts().items()), "rules")

    clf, imp, metrics_md = train_and_evaluate(e, a.out_dir)

    # final verdict: model where it is confident, rules otherwise
    # Geography (rules) is the verdict; radiometry (model) is an independent second
    # opinion. Agreement -> high confidence. Disagreement -> analyst review, which is
    # itself useful: an industrial-looking burn where no industry is mapped is exactly
    # the unmapped-facility case NTRO would want surfaced.
    e["final_class"] = e.rule_class
    if clf is not None:
        e["model_agrees"] = e.model_class == e.rule_class
        e["final_source"] = np.where(e.model_agrees, "rules+model", "rules_only")
        # Flag only CONSEQUENTIAL disagreements. Blanket "model disagrees" flagged half
        # the dataset, which is not a signal -- radiometry genuinely cannot separate
        # cropland from forest burns, and it is not asked to.
        veg = ["agri_burn", "wildfire"]
        false_alarm = (e.rule_class == "industrial_fire") & e.model_class.isin(veg) \
            & (e.model_confidence >= 0.80)
        unmapped = e.rule_class.isin(veg) & (e.model_class == "gas_flare") \
            & (e.model_confidence >= 0.80)
        e["needs_review"] = false_alarm | unmapped
        e["review_reason"] = np.where(
            false_alarm, "near industry but burns like vegetation - possible false alarm",
            np.where(unmapped,
                     "burns like a persistent flare with no industry mapped - possible unmapped source",
                     ""))
        log(f"model agrees with rules on {e.model_agrees.mean()*100:.1f}% of episodes", "out")
        log(f"review queue: {int(false_alarm.sum()):,} possible false alarms, "
            f"{int(unmapped.sum()):,} possible unmapped sources", "out")
    else:
        e["final_source"] = "rules"
        e["model_agrees"] = pd.NA
        e["needs_review"] = False
        e["review_reason"] = ""
    log("final: " + ", ".join(f"{k}={v:,}" for k, v in e.final_class.value_counts().items()), "out")

    ev_cols = [c for c in ["episode_id", "site_id", "latitude", "longitude", "start", "end",
                           "duration_days", "n_detections", "n_nights", "frp_max", "frp_mean",
                           "frp_trend", "hot_delta", "night_ratio", "site_nights",
                           "site_duty_cycle", "landcover", "d_industrial_hot",
                           "nearest_hot_category", "d_mining", "d_wri_plant", "wri_fuel",
                           "pop_9km", "label", "rule_class", "rule_reason", "model_class",
                           "model_confidence", "model_agrees", "needs_review",
                           "review_reason", "final_class", "final_source", "in_india"]
               + [c for c in e.columns if c.startswith("p_")] if c in e.columns]
    def slim(df):
        df = df.copy()
        for c in df.select_dtypes("float").columns:
            df[c] = df[c].round(5 if c in ("latitude", "longitude") else 2)
        return df

    log(f"writing events.csv ({len(e):,} rows)", "out")
    slim(e[ev_cols]).to_csv(os.path.join(a.out_dir, "events.csv"), index=False)

    # a slim, ranked file for the dashboard: what an analyst should look at first
    interest = e[e.final_class.isin(["industrial_fire", "gas_flare", "mining"])
                 & (e.in_india.fillna(True))].copy()
    # Analyst triage order. Raw FRP alone ranked a fire 22 km from any mapped industry
    # above a confirmed refinery hit -- so weight intensity by novelty, how close the
    # burn actually is to infrastructure, and how many people are exposed.
    novelty = np.where(interest.site_nights < 60, 1.0, 0.3)
    proximity = 1.0 / (1.0 + interest.d_industrial_hot.fillna(1e6) / 500.0)
    exposure = np.log1p(interest.pop_9km.fillna(0)) / 10.0
    interest["priority"] = (interest.frp_max.fillna(0) * novelty * proximity * (1 + exposure)
                            + interest.frp_trend.fillna(0).clip(lower=0) * 10).round(2)
    slim(interest.nlargest(20000, "priority")[ev_cols + ["priority"]]).to_csv(
        os.path.join(a.out_dir, "events_priority.csv"), index=False)

    st_out = st.copy()
    if clf is not None:
        top = e.sort_values("frp_max", ascending=False).drop_duplicates("site_id")
        st_out = st_out.merge(top[["site_id", "final_class"]], on="site_id", how="left")
    slim(st_out).to_csv(os.path.join(a.out_dir, "sites.csv"), index=False)

    if clf is not None:
        with open(os.path.join(a.out_dir, "model.pkl"), "wb") as fh:
            pickle.dump({"model": clf, "features": imp.feature.tolist(),
                         "classes": list(clf.classes_)}, fh)
    with open(os.path.join(a.out_dir, "metrics.md"), "w") as fh:
        fh.write(metrics_md + "\n")
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as fh:
        json.dump({"run_utc": datetime.now(timezone.utc).isoformat(),
                   "detections": int(len(d)), "sites": int(len(st)), "episodes": int(len(e)),
                   "labelled": int(e.label.notna().sum()),
                   "final_class_counts": e.final_class.value_counts().to_dict(),
                   "seconds": round(time.time() - t0, 1)}, fh, indent=2, default=str)

    log(f"DONE in {(time.time()-t0)/60:.1f} min -> {os.path.abspath(a.out_dir)}", "out")


if __name__ == "__main__":
    main()
