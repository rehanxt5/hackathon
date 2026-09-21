# Agni-Net — the interface

One HTML file, ~0.84 MB, no server and no network calls at runtime (the Google
Fonts stylesheet is the one external fetch). `open dashboard/index.html`.

## The design brief, answered

**The sheet is the page.** The dashboard reads as tonight's plate of the
national thematic atlas — a Survey of India / NATMO compilation rather than a
tech console: warm rag-paper ground, prussian-ink hairline cartography, boxed
legend furniture with double-rule frames, graticule margin ticks, and a
segmented atlas scale bar. Map tiles are external images: blocked by the
artifact CSP, and one more thing that can fail during a live demo. So the
basemap is drawn from real boundary data — **32 states and union territories**
(GADM, simplified to 7,094 points), the national coastline over that, a
graticule, and 18 reference cities that fade in as you zoom.

Every plotted detection is resolved to the state it falls in (11,490 of
11,500), because "23.76, 86.40" tells an analyst nothing and "Jharkhand" tells
them where to send someone.

**Built on the satellite's rhythm, not a fake ticker.** Sun-synchronous polar
orbits cross a given latitude at near-fixed local solar times, so the feed does
not stream — it arrives in passes. The status rail names which satellite
crossed last, which crosses next, and when, computed live from real overpass
windows (Terra 10:30 / 22:30, Suomi-NPP and Aqua 13:30 / 01:30, NOAA-20 12:40 /
00:40, NOAA-21 14:20 / 02:20 IST). The ~3-hour NASA publication delay is stated
as the floor it is, not hidden.

**One orchestrated motion.** On load, the press runs west to east and the
detections print as it passes — a vermilion press edge crosses the sheet once.
1.5 s, once, and it honours `prefers-reduced-motion`. Nothing else animates
except the feed's live lamp.

## Design tokens

| | |
|---|---|
| Paper | `#F2ECDD` warm sheet ground; boxes on `#F7F3E8`, wells on `#EAE2CF`. |
| Ink | `#23374E` prussian — linework, text, coast; `#4A5D72` / `#5C6F85` for secondary and tertiary text. Sea tint `#D8E2DF`, land `#F4EEDD`. |
| Type | **Halant** (masthead, headings, prose) and **Mukta** (labels, keys, tabular figures) — both Indian Type Foundry faces, the print-atlas pairing. |
| Classes | Colour encodes the **fuel** as print-plate inks: deep teal `#2E6B5E` wildfire, goldenrod `#AD8512` crop residue, vermilion `#BF4136` industrial alarm, burnt orange `#C06A1E` flare, atlas violet `#6B5AA0` mining. |

Detections are drawn as pre-rendered radial ink sprites composited with
`multiply`, so density reads as print staining — overlapping marks darken the
paper the way litho ink builds up — and 11,500 points still paint in one frame.

## The chart that carries the argument

The plate strip along the bottom stacks every class by month as inked bars,
with industrial, flare and mining overlaid as a vermilion line at **20× scale**
— labelled as such. At true scale that line is flat against the axis. That is
the whole problem statement in one graphic: the thing NTRO needs to see is 3.5%
of what the satellite reports.

## Build

```bash
python scripts/layer4_prepare.py   # data/layer3 -> dashboard/data.js
python scripts/layer4_build.py     # template + data -> dashboard/index.html
```

Chart aggregates are computed over the **full** 714,953 India episodes; only
the plotted points are sampled. The numbers on screen are real even though the
map is a sample.
