# Why the ring support collapses: it is the HOST, and it is a documented scope

**Answer: SCOPE, not a catalog gap.** The catalog is not missing the templates.
They are unreachable because the scaffold they need has been consumed.

Status: measurement complete for the catalog half (which needs no molecules at
all). State half in `host_census.json`. Trains nothing, calls no oracle, pinned
kernel (python 3.11 / rdkit 2024.3.5 / numpy 1.26.4 / scipy 1.13.1 /
networkx 3.3 / torch 2.4.0).

## The mechanism, in one line of production code

`_eligible_grow_host_graph` (`src/compose_v4/rewrite/ring_system_fiber.py:2160`):

> *"Carbon, neutral, acyclic single-bond support for v1 ring installation."*

It admits an atom only if it is **not in any cycle**, **is carbon**, and **is
neutral**, and joins two atoms only across a **single** bond. It then asserts
the result is a **forest**.

Three consequences, all structural:

1. **Committing a ring system permanently removes its atoms from the host.**
   Every later ring decision in that molecule is made against what is left.
2. **Decoration removes more.** An `atom_restate` to a heteroatom removes that
   atom; a `bond_reorder` to a double bond removes no atom but *cuts the host
   in two*, which matters because a template needs one contiguous tree.
3. **No template can match a host that already contains a ring** — measured,
   below — so a molecule cannot reuse the ring systems it has already built.

## Catalog half — MEASURED, and it needs no molecules

**0 of 3,092 templates require a cyclic host.** Every template's `source_bonds`
pattern is a forest. Ring systems are installed *atomically* onto acyclic
carbon; fused systems are single templates, never a ring grown onto a ring.
This is what makes the collapse a scope restriction rather than a gap: the
catalog is internally consistent with the host rule.

Small-ring share of the templates that fit on a host of each size:

| host atoms available | templates that fit | 3/4-ring share |
| --- | --- | --- |
| 3 | 2 | **100.0%** |
| 4 | 5 | **100.0%** |
| 5 | 13 | 46.2% |
| 6 | 34 | 47.1% |
| 7 | 77 | **57.1%** |
| 8 | 154 | 42.9% |
| 9 | 324 | 25.0% |
| 10 | 564 | 17.0% |
| 12 | 955 | 15.3% |
| 14 | 1,883 | 10.5% |
| 18 | 2,965 | 9.4% |
| 30+ | 3,092 (all) | **9.2%** |

**This table is the whole mechanism.** A large ring needs a large contiguous
carbon tree to sit on; a three-ring needs three atoms. So a support measured on
a small host is small-ring-enriched *by construction*, with no defect in the
policy, the reward, or the catalog. The knee is at host ≈ 7-9 atoms.

It also explains the correlation that looked backwards in the acceptance run:
`r(mass, free_slots) = +0.13`, i.e. *smaller* molecules showed *more*
small-ring mass. Smaller molecule → smaller host → the surviving templates are
the small rings.

## State half — MEASURED on real compiled traces

90 ring decisions over 34 real training molecules, `ring_dependency_block`
schedule (which commits rings *before* decoration, so the host loss here is
almost purely from prior rings):

| ordinal | n | small mass | legal | host atoms | largest tree | components | lost to cycles | lost to heteroatoms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 34 | 0.1292 | 1040.8 | 26.9 | 26.5 | 1.21 | 0.0 | 0.4 |
| 1 | 30 | 0.1827 | 657.7 | 20.1 | 17.6 | 1.83 | 6.6 | 0.4 |
| 2 | 20 | 0.2976 | 295.6 | 14.2 | 11.1 | 2.35 | 12.6 | 1.2 |
| 3 | 6 | 0.2726 | 272.5 | 13.3 | 10.8 | 2.00 | 17.2 | 0.0 |

`r(small mass, largest_host_tree) = -0.673`,
`r(log legal, largest_host_tree) = +0.772`. Host loss is carried almost entirely
by **committed cycles** (0.0 → 17.2) and not by heteroatoms (0.4 → 1.2) under
this schedule, which is exactly what it should be: this arm commits rings
first.

### The catalog table is a LOWER BOUND, not the whole story

Observed small-ring mass runs consistently **1.4x to 2x above** what
"templates that fit on a host of this size" predicts:

| ordinal | largest host tree | catalog table predicts | observed |
| --- | --- | --- | --- |
| 0 | 26.5 | ~9.2% | 12.9% |
| 1 | 17.6 | ~9.4% | 18.3% |
| 2 | 11.1 | ~15.3% | 29.8% |
| 3 | 10.8 | ~15.3% | 27.3% |

Two reasons, and both are second mechanisms the atom count alone misses:

1. **Shape.** Fitting needs the right branching pattern, not just enough atoms.
   "Host requirement <= k" is necessary, not sufficient.
2. **Fragmentation.** Host components rise **1.21 → 1.83 → 2.35** with ordinal.
   A committed ring does not only remove atoms, it can SPLIT the remaining
   forest, and a template needs one contiguous piece. A census tracking atoms
   alone would miss this entirely, which is why `largest_host_tree` and
   `host_components` are reported and separately tested.

So the host explains the direction and most of the magnitude; shape and
fragmentation account for the rest. Nothing here points back at the catalog.

## Rollout half — the same mechanism produces the established 44.3%

The committed reference audit stores both the support statistic and the ROLLOUT
state it was measured on, so this is directly attributable rather than
analogous. `scripts/denovo_rollout_host_attribution.py` reproduces the audit's
own stored mean (0.4431) before reporting anything, so a misread of the
artifact cannot be mistaken for a finding.

| row | legal | small | mass | real atoms | host | largest tree | components | lost to cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 808 | 120 | 0.1485 | 36 | 36 | 18 | 3 | 0 |
| 6 | 1098 | 127 | 0.1157 | 31 | 27 | 22 | 2 | 0 |
| 10 | 802 | 114 | 0.1421 | 21 | 18 | 18 | 1 | 0 |
| 1 | 653 | 102 | 0.1562 | 36 | 26 | 17 | 6 | 10 |
| 7 | 561 | 88 | 0.1569 | 31 | 21 | 15 | 3 | 6 |
| 4 | 469 | 90 | 0.1919 | 21 | 18 | 15 | 2 | 0 |
| 5 | 113 | 54 | 0.4779 | 21 | 13 | 9 | 3 | 5 |
| 8 | 12 | 5 | 0.4167 | 31 | 12 | 5 | 4 | 15 |
| 11 | 45 | 23 | 0.5111 | 21 | 8 | 7 | 2 | 10 |
| **2** | **2** | **2** | **1.0000** | 36 | 16 | **3** | 10 | **20** |
| **3** | **2** | **2** | **1.0000** | 36 | 13 | **3** | 9 | **23** |
| **9** | **5** | **5** | **1.0000** | 31 | 7 | **5** | 3 | **20** |

`r(mass, largest_host_tree) = -0.890`, `r(log legal, largest_host_tree) =
**+0.962**`. Mean largest host tree 11.4.

**The headline the 0.4431 mean hides: on 3 of 12 audited rollout states EVERY
legal template is a small ring.** Those states have 2-5 legal templates and a
host reduced to a 3-5 atom fragment by 20-23 atoms already locked into
committed rings. The model has no non-small option to choose. That is not
miscalibration -- it cannot be fixed by any amount of training, reward shaping,
or better sampling, because the alternative does not exist in the support.

The distribution is bimodal, not centrally enriched: states with an intact host
score 0.116-0.192 (they would pass a 0.15-ish gate), states whose host has been
eaten score 0.417-1.000. Quoting the 0.4431 mean alone describes neither
population.

This closes the loop on the generated-molecule defect. ~50% of generated
molecules carry a 3- or 4-ring because at their later ring decisions the
support contained nothing else.

## What this rules in and out

- **Ruled out: a catalog gap.** The large templates exist (2,965 of 3,092 fit on
  an 18-atom host). Adding templates changes nothing while the host is small.
- **Ruled out: a reward or policy fix — now conclusively.** The model already
  sits *below* uniform on small rings, and on 3 of 12 audited rollout states
  the support contains *no* non-small template at all. A policy cannot select
  an option that does not exist.
- **Ruled out: a better schedule.** Scheduling moves *when* a ring is committed;
  it cannot stop a committed ring from consuming its atoms. This is why the
  acceptance run's ordinal wall was unavoidable.
- **Ruled IN, and these are the only two doors:**
  1. **Extend the scope** — permit ring installation on a host containing a
     ring (fused growth onto an existing system) and/or on heteroatom and
     multiply-bonded scaffolds. That is a genuine capability change to
     `_eligible_grow_host_graph` plus templates with cyclic source patterns,
     and it invalidates the forest assertion the current DP relies on.
  2. **Scope the claim** — state that v1 ring generation installs ring systems
     atomically onto acyclic carbon, and report the small-ring behaviour as a
     property of that scope rather than as a defect being fixed.

## Second, independent item: `exact_early_ring` is unwired

Separate from the catalog question and actionable on its own. Lineage B trained
on `sequential`, while `exact_early_ring` — which delivers **0.3049 → 0.1926**
on the acceptance statistic — has existed since 2026-07-20 and reaches no
training path.

**MEASURED plumbing surface** (`event_schedule` appears nowhere in the training
chain; grep over `src/`, `scripts/`, `modal_apps/`, `recipes/` returns only
`tree_transport.py`, `commuting_schedule.py`, tests, and this session's probes):

| hop | file | call sites |
| --- | --- | --- |
| 1 | recipe JSON argument | 1 |
| 2 | `experiments/recipe.py: build_tracelet_recipe_argv` | 1 |
| 3 | `scripts/train_tracelet_cnof_gate.py` CLI arg → builder | **4** |
| 4 | `experiments/tracelet_conditional.py: build_tree_transport_path_records` | 1 |
| 5 | its `compile_carbon_tree_to_target` sinks | **2** |

**This CORRECTS the recorded estimate of "five additive hops":** it is five
hops but **nine call sites**, and hops 3 and 5 are multi-site. The two sinks in
hop 5 share one parameter, so a consultation test exercising only one stays
green — the exact near-miss the region-law wiring hit. Mutation-test at the
call sites, not at the definition.

**Do not build it yet.** Whether it is worth a retrain depends on the scope
decision above: if the claim is scoped, the gain may not need to be realised in
the corpus at all.
