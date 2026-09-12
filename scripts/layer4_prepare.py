#!/usr/bin/env python3
"""
layer4_prepare.py -- build the compact data bundle the dashboard embeds.

1.09M episodes will not fit in a browser page, and they should not: an analyst needs
the incidents that matter plus enough background to see the pattern. So we take every
high-value event, a representative sample of the vegetation classes for context, and
pre-compute the aggregates the charts need.

Values are packed as positional arrays rather than objects -- same data, ~4x smaller.
"""
import json, os
import numpy as np
import pandas as pd

SRC = "data/layer3"
OUT = "dashboard/data.js"
CLASS_CAPS = {"industrial_fire": 3500, "gas_flare": 2500, "mining": 1500,
              "agri_burn": 2000, "wildfire": 2000}

os.makedirs("dashboard", exist_ok=True)
ev = pd.read_csv(f"{SRC}/events.csv", low_memory=False)
print(f"loaded {len(ev):,} episodes")

ev = ev[ev.in_india.fillna(False).astype(bool)]
print(f"{len(ev):,} inside India")

# priority score for ranking (same shape as Layer 3's triage score)
novelty = np.where(ev.site_nights < 60, 1.0, 0.3)
prox = 1.0 / (1.0 + ev.d_industrial_hot.fillna(1e6) / 500.0)
expo = np.log1p(ev.pop_9km.fillna(0)) / 10.0
ev["priority"] = (ev.frp_max.fillna(0) * novelty * prox * (1 + expo)).round(2)

parts = []
for cls, cap in CLASS_CAPS.items():
    sub = ev[ev.final_class == cls]
    if cls in ("agri_burn", "wildfire"):        # context: sample, don't rank
        sub = sub.sample(min(cap, len(sub)), random_state=7)
    else:                                        # incidents: keep the important ones
        sub = sub.nlargest(min(cap, len(sub)), "priority")
    parts.append(sub)
    print(f"  {cls:16s} {len(sub):>5,} of {int((ev.final_class == cls).sum()):>7,}")
sel = pd.concat(parts, ignore_index=True)

CLASSES = list(CLASS_CAPS)
LC = sorted(ev.landcover.dropna().unique().tolist())
CATS = sorted(ev.nearest_hot_category.dropna().unique().tolist())

def code(s, vocab):
    return s.map({v: i for i, v in enumerate(vocab)}).fillna(-1).astype(int).tolist()

sel["start"] = pd.to_datetime(sel.start)
epoch = pd.Timestamp("2025-01-01")
rows = [
    sel.latitude.round(4).tolist(),
    sel.longitude.round(4).tolist(),
    ((sel.start - epoch).dt.days).astype(int).tolist(),      # days since 2025-01-01
    code(sel.final_class, CLASSES),
    sel.frp_max.fillna(0).round(1).tolist(),
    sel.site_nights.fillna(0).astype(int).tolist(),
    sel.duration_days.fillna(1).astype(int).tolist(),
    sel.d_industrial_hot.fillna(-1).round(0).astype(int).tolist(),
    code(sel.landcover, LC),
    code(sel.nearest_hot_category, CATS),
    sel.pop_9km.fillna(0).round(0).astype(int).tolist(),
    sel.night_ratio.fillna(0).round(2).tolist(),
    sel.needs_review.fillna(False).astype(int).tolist(),
    sel.priority.round(1).tolist(),
    ((sel.start.max() - sel.start).dt.days).astype(int).tolist(),   # age: days behind the feed
    None,   # placeholder, filled below with the state index
]


# ---- aggregates for the charts (computed on the FULL India set, not the sample) ----
ev["month"] = pd.to_datetime(ev.start).dt.to_period("M").astype(str)
monthly = (ev.groupby(["month", "final_class"]).size().unstack(fill_value=0)
           .reindex(columns=CLASSES, fill_value=0).sort_index())
lc_mix = (ev.groupby(["final_class", "landcover"]).size().unstack(fill_value=0)
          .reindex(index=CLASSES, fill_value=0))

top = ev[ev.final_class.isin(["industrial_fire", "gas_flare", "mining"])] \
        .nlargest(25, "priority")
alerts = [{
    "lat": round(float(r.latitude), 4), "lon": round(float(r.longitude), 4),
    "cls": r.final_class, "start": str(r.start)[:10],
    "frp": round(float(r.frp_max or 0), 1),
    "dur": int(r.duration_days or 1), "nights": int(r.site_nights or 0),
    "cat": (r.nearest_hot_category if isinstance(r.nearest_hot_category, str) else "-"),
    "dist": int(r.d_industrial_hot) if pd.notna(r.d_industrial_hot) else -1,
    "pop": int(r.pop_9km or 0), "lc": r.landcover,
    "reason": r.rule_reason, "review": bool(r.needs_review),
    "priority": round(float(r.priority), 1),
} for r in top.itertuples()]

# The newest thing the feed has seen. This is what "real time" actually means here:
# NASA publishes VIIRS near-real-time roughly 3 hours after the overpass.
ev["ts"] = pd.to_datetime(ev.start)
newest = ev.ts.max()
recent = ev[(ev.ts >= newest - pd.Timedelta(days=7))
            & ev.final_class.isin(["industrial_fire", "gas_flare", "mining"])]
arrivals = [{
    "lat": round(float(r.latitude), 4), "lon": round(float(r.longitude), 4),
    "cls": r.final_class, "start": str(r.ts)[:10],
    "age": int((newest - r.ts).days),
    "dur": int(r.duration_days or 1),
    "frp": round(float(r.frp_max or 0), 1),
    "cat": (r.nearest_hot_category if isinstance(r.nearest_hot_category, str) else "-"),
    "dist": int(r.d_industrial_hot) if pd.notna(r.d_industrial_hot) else -1,
    "pop": int(r.pop_9km or 0), "nights": int(r.site_nights or 0),
} for r in recent.nlargest(14, "priority").itertuples()]
print(f"arrivals in the last 7 days of feed: {len(recent):,} incidents")

persist = ev.nlargest(15, "site_nights").drop_duplicates("site_id")
sources = [{
    "lat": round(float(r.latitude), 4), "lon": round(float(r.longitude), 4),
    "nights": int(r.site_nights), "duty": round(float(r.site_duty_cycle or 0), 2),
    "night_ratio": round(float(r.night_ratio or 0), 2), "lc": r.landcover,
    "cls": r.final_class,
} for r in persist.itertuples()]

# ---------------------------------------------------------------------------------
# BASEMAP.  Map TILES cannot be used (the artifact CSP blocks external images, and a
# tile server is one more thing that can fail live), so the map is drawn from vector
# boundaries on a canvas.
#
# Source: district boundaries for India (760 districts, 36 states/UTs), dissolved to
# state level here.  This dataset is used rather than GADM because GADM is pre-2014:
# it has no Telangana, no Ladakh, spells Odisha "Orissa" and Uttarakhand "Uttaranchal",
# and it truncates Jammu & Kashmir at 35.5 N instead of showing India's full claimed
# territory to 37.1 N.  For a submission to an Indian agency the boundary depiction has
# to be the Indian one.
# ---------------------------------------------------------------------------------
DISTRICTS = "data/layer2_seeds/india_districts.geojson"


def india_states(path=DISTRICTS, tol=0.02, min_area=0.0008):
    from shapely.geometry import shape
    from shapely.ops import unary_union
    import collections
    gj = json.load(open(path))
    by_state = collections.defaultdict(list)
    for f in gj["features"]:
        by_state[f["properties"]["st_nm"]].append(shape(f["geometry"]).buffer(0))

    out, shapes = [], []
    for name, parts in sorted(by_state.items()):
        g = unary_union(parts)
        shapes.append((name, g))
        gs = g.simplify(tol, preserve_topology=True)
        polys = list(gs.geoms) if gs.geom_type == "MultiPolygon" else [gs]
        rings = [[[round(x, 4), round(y, 4)] for x, y in pl.exterior.coords]
                 for pl in polys if pl.area >= min_area]
        if not rings:
            # never drop a state or UT for being small -- Lakshadweep's islands are
            # 1-4 km2 each, well under the threshold, but it is still on the map
            big = max(polys, key=lambda x: x.area)
            rings = [[[round(x, 4), round(y, 4)] for x, y in big.exterior.coords]]
        c = max(polys, key=lambda x: x.area).representative_point()
        out.append({"n": name, "r": rings, "c": [round(c.x, 3), round(c.y, 3)]})
    # the national outline is the union of the states, so the two can never disagree
    nat = unary_union([g for _, g in shapes]).simplify(tol, preserve_topology=True)
    npolys = list(nat.geoms) if nat.geom_type == "MultiPolygon" else [nat]
    outline = [[[round(x, 3), round(y, 3)] for x, y in pl.exterior.coords]
               for pl in sorted(npolys, key=lambda x: -x.area) if pl.area >= 0.05]
    print(f"states: {len(out)}, {sum(len(r) for st in out for r in st['r'])} points")
    print(f"outline: {len(outline)} rings, {sum(len(r) for r in outline)} points, "
          f"north to {max(p[1] for r in outline for p in r):.2f}N")
    return out, shapes, outline


def label_states(lats, lons):
    """Which state is each detection in? STRtree keeps 11.5k lookups instant."""
    from shapely.strtree import STRtree
    from shapely import points as mk_points
    geoms = [g for _, g in state_shapes]
    names = [n for n, _ in state_shapes]
    tree = STRtree(geoms)
    idx = tree.query(mk_points(lons, lats), predicate="within")
    res = [-1] * len(lats)
    for pi, gi in zip(idx[0], idx[1]):
        if res[pi] == -1:
            res[pi] = gi
    return names, res


states, state_shapes, outline = india_states()

# which state is each plotted detection in?  (STRtree keeps 11.5k lookups instant)
def label_states(lats, lons):
    from shapely.strtree import STRtree
    from shapely import points as mk_points
    geoms = [g for _, g in state_shapes]
    names = [n for n, _ in state_shapes]
    tree = STRtree(geoms)
    pts = mk_points(lons, lats)
    idx = tree.query(pts, predicate="within")      # (2, n) pairs: point, geom
    res = [-1] * len(lats)
    for pi, gi in zip(idx[0], idx[1]):
        if res[pi] == -1:
            res[pi] = gi
    return names, res

STATE_NAMES, sel_state = label_states(sel.latitude.values, sel.longitude.values)
rows[-1] = [int(v) for v in sel_state]
_an, _ai = label_states([a["lat"] for a in arrivals], [a["lon"] for a in arrivals])
for a, gi in zip(arrivals, _ai):
    a["state"] = _an[gi] if gi >= 0 else ""
print(f"state resolved for {sum(1 for v in sel_state if v>=0):,} of {len(sel):,} plotted points")

# reference points so the eye can place a detection without reading coordinates
CITIES = [("New Delhi",28.61,77.21),("Mumbai",19.08,72.88),("Kolkata",22.57,88.36),
          ("Chennai",13.08,80.27),("Bengaluru",12.97,77.59),("Hyderabad",17.39,78.49),
          ("Ahmedabad",23.03,72.58),("Jaipur",26.91,75.79),("Lucknow",26.85,80.95),
          ("Bhopal",23.26,77.41),("Patna",25.59,85.14),("Nagpur",21.15,79.09),
          ("Guwahati",26.14,91.74),("Bhubaneswar",20.30,85.82),("Surat",21.17,72.83),
          ("Visakhapatnam",17.69,83.22),("Kochi",9.93,76.27),("Chandigarh",30.73,76.78)]

bundle = {
    "outline": outline,
    "states": states, "state_names": STATE_NAMES,
    "cities": [[n, la, lo] for n, la, lo in CITIES],
    "classes": CLASSES, "landcover": LC, "categories": CATS,
    "epoch": "2025-01-01",
    "fields": ["lat", "lon", "day", "cls", "frp", "nights", "dur", "dist",
               "lc", "cat", "pop", "night_ratio", "review", "priority", "age", "st"],
    "rows": rows, "n": len(sel),
    "totals": ev.final_class.value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
    "grand_total": int(len(ev)),
    "detections": 1706805, "sites_all": 744538, "episodes_all": 1091136,
    "sites_india": int(ev.site_id.nunique()),
    "monthly": {"months": monthly.index.tolist(),
                "series": {c: monthly[c].tolist() for c in CLASSES}},
    "lc_mix": {"landcover": lc_mix.columns.tolist(),
               "series": {c: lc_mix.loc[c].tolist() for c in CLASSES}},
    "alerts": alerts, "sources": sources,
    "latest": str(pd.to_datetime(ev.start).max())[:10],
    "arrivals": arrivals,
    "review_counts": {"total": int(ev.needs_review.fillna(False).sum())},
}

with open(OUT, "w") as fh:
    fh.write("window.AGNI = ")
    json.dump(bundle, fh, separators=(",", ":"))
    fh.write(";\n")
print(f"\n{OUT}: {os.path.getsize(OUT)/1e6:.2f} MB, {len(sel):,} mapped events")
print("totals:", bundle["totals"])
