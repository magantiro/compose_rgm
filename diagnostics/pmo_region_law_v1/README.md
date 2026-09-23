# Uncapping the region draw does not move PMO's changed-region size

Zero oracle calls. 12 matched sources, 300 draws/arm, matched RNG seeds, PMO kernel
rdkit 2023.09.6.

## End to end

| metric | baseline | law + uncapped replacement | paired delta |
|---|---:|---:|---:|
| largest changed region (median) | 4.000 | 4.167 | **+0.167 +- 0.216** (ns) |
| largest changed region (p90) | 10.58 | 10.93 | +0.350 +- 0.371 (ns) |
| MCS retention | 0.905 | 0.858 | -0.048 +- 0.016 |
| primitives | 4.08 | 4.83 | +0.750 +- 0.250 |
| executable yield | 1.000 | 1.000 | 0.000 |

Inertness of the patch on `law=None`: **CONFIRMED across all 12 sources**, so the contrast
is causal rather than an RNG perturbation.

## Conditional on `segment_replace` firing (family dilution removed)

| | median | p90 | max | >=15 atoms | retention |
|---|---:|---:|---:|---:|---:|
| capped | 4.0 | 7.0 | **8** | **0.0%** | 0.923 |
| uncapped | 5.0 | 8.0 | **15** | **0.2%** | 0.840 |

The ceiling lifts from 8 to 15 and the distribution barely moves.

## Why: uniform over regions is not large regions

`BridgeRegionLaw(maximum=None, margin=None)` draws UNIFORMLY over bridge-separated regions.
Most such regions are small, so uncapping only ADDS large regions to the menu without giving
any reason to prefer one. The isolated delete module reaches 25 atoms and 18.3% >= 15 because
it is measured over its own draws; inside a program the uniform draw returns to small regions.

**T4 measured this same thing and it is in `learnings.md`:** raising the cap ALONE degrades a
working control, and conditioning ALONE leaves braf_0 at 109/319 -- *"it is the joint change
or nothing"*. This probe ran the cap-only half. The conditioned half is not expressible here
because `free_gate_margin_v1` consumes a similarity reference and delta that PMO does not have.

## What this establishes

The cap is NOT the binding constraint on PMO's changed-region size, and neither is executor
refusal. What is missing is a **task-independent conditioning signal on region choice** -- a
reason to prefer a large coherent region, not merely permission to draw one.

That is the evidence-based justification for building the explicit
`(R, anchors, replacement)` proposer: it must supply the intent, not just the capability.

## The patch is a PREREQUISITE, not a fix

`segment_replace` re-capped the regrowth at `MAX_SEGMENT_LENGTH` regardless of how much was
excised, so a large excision was refilled with <=8 atoms and the endpoint carried only 8 new
ones. That ceiling is removed when a law is supplied (`a replacement may be as large as what
it replaced`), and `region_law=None` stays byte-identical. Any future proposer that picks a
large region deliberately still needs this, but on its own it changes nothing.

---

# Scale-balancing the region draw is ALSO null — the region law is not the layer

`ScaleBalancedRegionLaw` gives every OCCUPIED size class equal total mass
(`q(R|G) = q(s|G) q(R|s,G)`, classes small `<5` / medium `5-11` / large `>=12`), so a state
with one 20-atom region and thirty small ones draws the large one as often as the small ones
collectively. That is the property a uniform law cannot have at any cap.

| metric | baseline | scale-balanced | paired delta |
|---|---:|---:|---:|
| largest changed region (median) | 4.000 | 3.792 | **-0.208 +- 0.208** |
| largest changed region (max) | 20.83 | 19.83 | -1.000 +- 1.080 |
| MCS retention | 0.905 | 0.882 | -0.023 +- 0.012 |
| primitives | 4.08 | 4.88 | +0.792 +- 0.226 |
| executable yield | 1.000 | 1.000 | 0.000 |

Null, and if anything slightly negative. Per-source it is noise in both directions.

## THREE region-law variants, all null

| variant | largest region (median) |
|---|---|
| uncapped uniform | +0.000 +- 0.238 |
| uncapped uniform + uncapped replacement | +0.167 +- 0.216 |
| scale-balanced classes + uncapped replacement | **-0.208 +- 0.208** |

Each with the law confirmed CONSUMED (the repo's own `assert_region_law_is_consumed` fires on
attempt 1), each with executable yield 1.000, each on 12 matched sources at matched seeds with
arm A reproducing identically three runs running.

## CONCLUSION: the region DRAW is not what bounds the realized change

`substituent_delete` and `segment_replace` are 2 of 13 families in `GENERIC_MODULES`, drawn
roughly 1/13 each, and the module count is 1-3 with 80% of mass on 1-2. So the MEDIAN program
is one or two modules from a menu that is 11/13 small local edits -- functionalize,
carbonyl_insert, heteroatom_substitute, bond_reroute, cycle_open/close, append_ring, fuse_ring,
segment_grow, segment_shrink, ring_system_restate. No law over REGIONS changes which FAMILY is
drawn, so no law over regions can move the median.

Forcing `segment_replace` directly confirms it from the other side: even with family dilution
removed the median changed region is 5 and only 0.2% of draws reach fifteen atoms.

## What this licenses building

An explicit `(R, anchors, replacement)` option as **its own proposal lane**, not as a law
inside the existing family lottery. The intent has to be chosen BEFORE the family draw, because
that draw is what the measurements keep landing on. COMPOSE's capability is not in question --
an isolated uncapped delete reaches 25 atoms at full validity.

Do not encode 19 atoms or 0.30 retention. Those are answer-known.
