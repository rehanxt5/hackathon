# Agni-Net — the interface

One HTML file, 0.73 MB, no server and no network calls at runtime. `open dashboard/index.html`.

## The design brief, answered

**The map is the page.** Not a widget inside dashboard chrome — full-bleed canvas, with the
readouts floating over it in the corners India's own shape leaves empty. Map tiles are
external images: blocked by the artifact CSP, and one more thing that can fail during a live
demo. So the basemap is drawn from real boundary data — **32 states and union territories**
(GADM, simplified to 7,094 points), the national coastline over that, a graticule, and 18
reference cities that fade in as you zoom.

Every plotted detection is resolved to the state it falls in (11,490 of 11,500), because
"23.76, 86.40" tells an analyst nothing and "Jharkhand" tells them where to send someone.

**Built on the satellite's rhythm, not a fake ticker.** Sun-synchronous polar orbits cross a
given latitude at near-fixed local solar times, so the feed does not stream — it arrives in
passes. The status panel names which satellite crossed last, which crosses next, and when,
computed live from real overpass windows (Terra 10:30 / 22:30, Suomi-NPP and Aqua 13:30 /
01:30, NOAA-20 12:40 / 00:40, NOAA-21 14:20 / 02:20 IST). The ~3-hour NASA publication delay
is stated as the floor it is, not hidden.

**One orchestrated motion.** On load, a swath sweeps west to east and the detections appear
as it passes — which is how the sensor actually sees. 1.5 s, once, and it honours
`prefers-reduced-motion`. Nothing else animates except the feed's live dot.

## Design tokens

| | |
|---|---|
| Ground | `#0D1230` deep indigo — the dye India traded and the sky the sensor looks through. Deliberately not a tinted near-black. |
| Land / coast | `#1C2246` / `#3B4480` |
| Type | **Anek Latin**, the variable superfamily by **Ek Type, Mumbai**, designed for Indian scripts. One family throughout; hierarchy comes from its width axis (78→100), not a second typeface. No all-caps labels, no monospace — Anek's tabular figures handle the numbers. |
| Classes | Colour encodes the **fuel**: wheat `#C9B458` crop residue, forest teal `#3FA98A` wildfire, vermilion `#FF5335` industrial alarm, flame amber `#FFA62B` flare. Mining `#8B7BE8` sits deliberately outside that logic because it is neither fire nor fuel. |

Detections are drawn as pre-rendered radial sprites composited with `lighter`, so density
reads as heat the way a night-lights image does — and 11,500 points still paint in one frame.

## The chart that carries the argument

The ribbon along the bottom stacks every class by month, with industrial, flare and mining
overlaid as a line at **20× scale** — labelled as such. At true scale that line is flat
against the axis. That is the whole problem statement in one graphic: the thing NTRO needs
to see is 3.5% of what the satellite reports.

## Build

```bash
python scripts/layer4_prepare.py   # data/layer3 -> dashboard/data.js
python scripts/layer4_build.py     # template + data -> dashboard/index.html
```

Chart aggregates are computed over the **full** 714,953 India episodes; only the plotted
points are sampled. The numbers on screen are real even though the map is a sample.
