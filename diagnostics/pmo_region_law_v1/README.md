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
