"""Re-aggregate the persisted series-depth rows, with acyclic split out.

WHY NOT JUST READ THE APP'S TABLES
----------------------------------
The app buckets panel_depth starting at 0, but depth 0 is not the bottom of a
series-depth scale: all 361 of those entries are acyclic molecules whose Murcko
scaffold is the empty string, so they were never assigned a scaffold at all.
Reading them as "shallowest series" would put a category error at the exact end
of the curve the hypothesis is about. They are reported separately here.

Depth 1 is the real singleton bucket: one panel source, ring scaffold, no other
panel member sharing it.
"""
import gzip
import json
import statistics
import collections
import sys
from pathlib import Path

rows = json.load(gzip.open(sys.argv[1], "rt"))
acyclic = {i.strip() for i in Path(sys.argv[2]).read_text().split()} if len(sys.argv) > 2 else set()

DEPTH = ((1, 1), (2, 4), (5, 9), (10, 24), (25, 10**9))
SUPPORT = ((0, 0), (1, 4), (5, 24), (25, 10**9))
MINIMUM_CELL = 40


def label(v, edges):
    for lo, hi in edges:
        if lo <= v <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
    return f"?{v}"


def agg(cell):
    return (len(cell),
            statistics.mean(r["identity_nll"] for r in cell),
            statistics.mean(r["family_nll"] for r in cell),
            statistics.mean(r["teacher_successor_nll"] for r in cell))


ring = [r for r in rows if r["entry_id"] not in acyclic] if acyclic else \
       [r for r in rows if r["panel_depth"] > 0]
skipped = len(rows) - len(ring)
print(f"scored rows {len(rows):,}   ring-scaffold rows {len(ring):,}   "
      f"acyclic held out {skipped:,}")


def table(key, edges, title, source):
    g = collections.defaultdict(list)
    for r in source:
        g[label(key(r), edges)].append(r)
    print(f"\n=== {title} ===")
    print(f"{'bucket':>10} {'n':>7} {'identity':>9} {'family':>8} {'joint':>8}")
    prev, mono = None, True
    for lab in sorted(g, key=lambda s: int(s.split('-')[0].rstrip('+'))):
        cell = g[lab]
        if len(cell) < MINIMUM_CELL:
            print(f"{lab:>10} {len(cell):7,}   (below MINIMUM_CELL, not reported)")
            continue
        n, ident, fam, joint = agg(cell)
        arrow = "" if prev is None else ("  up" if ident > prev else "  DOWN")
        if prev is not None and ident <= prev:
            mono = False
        prev = ident
        print(f"{lab:>10} {n:7,} {ident:9.3f} {fam:8.3f} {joint:8.3f}{arrow}")
    print(f"  monotonic increasing: {mono}")
    return mono


if skipped:
    ac = [r for r in rows if r not in ring] if acyclic else \
         [r for r in rows if r["panel_depth"] == 0]
    if len(ac) >= MINIMUM_CELL:
        n, ident, fam, joint = agg(ac)
        print(f"\nacyclic (no Murcko scaffold, reported separately): "
              f"n={n:,} identity={ident:.3f} family={fam:.3f} joint={joint:.3f}")

table(lambda r: r["panel_depth"], DEPTH, "SERIES DEPTH (ring scaffolds only)", ring)
table(lambda r: r["train_support"], SUPPORT, "TRAIN SCAFFOLD SUPPORT (ring only)", ring)

print("\n=== DEPTH within fixed provenance (ring only) ===")
for kind in ("real", "synthetic"):
    sub = [r for r in ring if r["provenance"] == kind]
    if len(sub) >= MINIMUM_CELL:
        table(lambda r: r["panel_depth"], DEPTH, f"depth | provenance={kind}", sub)

print("\n=== DEPTH at fixed SUPPORT band (the collinearity check) ===")
for lo, hi in SUPPORT:
    band = [r for r in ring if lo <= r["train_support"] <= hi]
    if len(band) < MINIMUM_CELL * 2:
        continue
    g = collections.defaultdict(list)
    for r in band:
        g[label(r["panel_depth"], DEPTH)].append(r)
    usable = {k: v for k, v in g.items() if len(v) >= MINIMUM_CELL}
    tag = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+")
    if len(usable) < 2:
        print(f"  support {tag:>6}: only {len(usable)} usable depth bucket(s) "
              f"of {len(g)} -- axes too collinear to separate here")
        continue
    print(f"  support {tag:>6}:")
    for lab in sorted(usable, key=lambda s: int(s.split('-')[0].rstrip('+'))):
        n, ident, _f, _j = agg(usable[lab])
        print(f"      depth {lab:>6} n={n:6,} identity={ident:.3f}")
