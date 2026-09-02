# COMPOSE — Rewrite Generator Matching for pathwise-constrained molecular generation

COMPOSE learns a **validity-closed, flexible-size molecular generator**: a
continuous-time Markov chain (CTMC) over molecular graphs whose committed states
are complete, chemically valid, connected molecules and whose transitions are
**executable chemical rewrites**. Generator Matching learns the contextual firing
rates of those rewrites from endpoint-conditioned programs; at inference the
process is target-free, ancestral, and **non-monotone** (it grows *and* shrinks) —
no endpoint, beam search, candidate bank, or terminal repair.

## Paper 1 thesis — *pathwise*-constrained molecular generation

Every guided generator constrains the **endpoint**: the final molecule must
satisfy a predicate `P`. COMPOSE can constrain the **path** — `P` holds at *every
committed state*, not just the last one. This is a problem class expressible
**only because our intermediates are molecules**, which is exactly the structural
gap of diffusion/flow models (their intermediates are latents / partial graphs,
so `P` can't even be evaluated mid-generation). Formalizing pathwise-constrained
generation, and showing COMPOSE solving instances of it, is the paper's spine.

> **New to the project?** Read (1) this file, (2)
> [`docs/CONDITIONAL_RESULTS_AND_SCOPING.md`](docs/CONDITIONAL_RESULTS_AND_SCOPING.md)
> — the current, honest results + scoping, then (3) the paper in
> [`paper_iclr_stochastic_rewriting/`](paper_iclr_stochastic_rewriting/)
> (`main.tex`, builds to `main.pdf` with `latexmk -pdf`). Everything else under
> `docs/` is historical — see the index at the bottom.

## The contribution, graded by claim hardness

The paper grades its own claims so a reviewer can trust the self-assessment.

| Layer | What it is | Status | Number |
|---|---|---|---|
| **Spotlight** — pathwise **safety** (Tier 1) | never pass through a reactive/toxicophore state; exact invariant in the legal fiber | committed | endpoint-only generation traverses a reactive intermediate in **66.9%** of even its *clean-ending* trajectories; COMPOSE **0%** audited (3,067 states), at QED cost **−0.016** (free) |
| **Structural control** | exact fixed-scaffold + required/forbidden-substructure satisfaction, where generate-then-filter collapses and no learned constrained generator can even *represent* the constraint | committed | scaffold **100%** vs baseline **96%→12%**; substructure **100%**; multi-property physchem box under exact scaffold **17.8% vs 3.5% per sample (5×)**, 100% scaffold |
| **Oracle efficiency / anytime** | oracle spent only on in-fiber candidates; every intermediate usable | committed | **100%** usable-oracle vs **39.8%**; median lead reaches 90% of its gain by **~13** oracle calls |
| **Foundation (rigor)** | exact conditioning verified; guidance measured against ground truth | committed | Doob h-transform reproduces the exact conditional to **1.1e-16**; finite-particle guidance provably converges to it |
| **De novo sufficiency** | the base is a credible generator (not a distribution-learning leaderboard entry) | in progress | validity/uniqueness/novelty ≈ 100% (smoke); V/U/N + descriptor Wassersteins, **no FCD arms race** |

**What we do *not* claim:** exact *property* conditioning (property targeting is
*steered*, only rule-closed structural constraints are exact); unconditional
distribution SOTA (FCD is deferred, not a headline); a head-to-head win on
oracle-*light* property optimization (our SMC is oracle-hungry — GrIDDD's home
turf, which we do not contest). See `docs/CONDITIONAL_RESULTS_AND_SCOPING.md`.

## Repository map

```
paper_iclr_stochastic_rewriting/   the ICLR paper (main.tex, numbers.tex, references.bib, figures/)
src/compose_v4/                    the model: chem state, rewrite kernel/fibers, Generator Matching, experiments
scripts/                           experiment drivers (see "Results" below)
diagnostics/conditional_smc/       committed conditional result JSONs
diagnostics/exactness/             committed E0 / guidance-ground-truth JSONs
configs/benchmarks/                frozen inputs (cnof_leads.json = 271 CNOF optimization leads, Jin QED set)
modal_apps/                        distributed training / rollout evaluation on Modal
tests/                             pytest (E0 exactness, reward-FT, value-twist, ring-hazard)
docs/                              documentation (see index at bottom; canonical vs historical)
```

## Results at a glance (script → figure → data)

Each conditional result is a self-contained script; figures land in
`paper_iclr_stochastic_rewriting/figures/`, summaries in `diagnostics/`.

| Result | Script | Figure | Data |
|---|---|---|---|
| Pathwise safety (spotlight) | `scripts/tier1_pathwise_safety.py` | `pathwise_safety.pdf` | `diagnostics/conditional_smc/tier1_pathwise_safety.json` |
| Scaffold collapse curve | `scripts/make_paper_figures.py` | `scaffold_collapse.pdf` | `diagnostics/conditional_smc/smc_scaffold_control48.json` |
| Physchem box under exact scaffold | `scripts/physchem_box.py` | `physchem_box.pdf` | `diagnostics/conditional_smc/physchem_box.json` |
| Anytime | `scripts/make_paper_figures.py` | `anytime.pdf` | `diagnostics/conditional_smc/scaffold_opt_panel12.json` |
| Property dial (warm-up) | `scripts/property_dial.py` | `property_dial.pdf` | `diagnostics/conditional_smc/property_dial.json` |
| E0 Doob exactness | `scripts/e0_toy_h_exactness.py` | — | `diagnostics/exactness/e0_toy_h_*.json` |
| Guidance ground truth | `scripts/doob_guidance_ground_truth.py` | `doob_guidance_ground_truth.pdf` | `diagnostics/exactness/doob_guidance_ground_truth.json` |
| Usable-oracle efficiency | (from scaffold data) | — | `diagnostics/conditional_smc/usable_oracle_efficiency.json` |
| De novo sufficiency (E1) | `scripts/e1_unconditional_metrics.py` | — | (run pending) |

The conditional-control machinery is the value-guided SMC controller
`scripts/griddd_value_guided_smc_controller.py` (hard scaffold / similarity /
required-SMARTS / forbidden-SMARTS(tuple) fiber constraints, arbitrary
state→float objective, population dump, best-molecule + anytime traces).

## Reproduce

```bash
# environment: python 3.12+, rdkit, torch, numpy, scipy, matplotlib (see pyproject.toml)
export PYTHONPATH=src:scripts
export KMP_DUPLICATE_LIB_OK=TRUE           # macOS OpenMP guard
# optional: where scratch outputs go (default: ./scratch)
export COMPOSE_SCRATCH=./scratch

# base checkpoint (Lineage B) — pull once from the Modal artifact volume
#   compose-v4-artifacts : compose-v4-stage3-flexible-graft-3k-1ac6f19-v1/checkpoint.best_so_far.pt
# scripts default to /private/tmp/lineage_b_checkpoint/checkpoint.best_so_far.pt (override CKPT in-script)

python scripts/tier1_pathwise_safety.py      # the spotlight result + figure
python scripts/physchem_box.py               # multi-property box under exact scaffold
python scripts/e0_toy_h_exactness.py         # exactness (self-contained, no checkpoint)
python -m pytest tests/test_e0_toy_h_exactness.py -q
```

Notes: leads are read from `configs/benchmarks/cnof_leads.json` (tracked). The
E1 de novo run additionally needs a one-SMILES-per-line ZINC/GuacaMol file
(extract the `smiles` column from the source CSV). SMC runs are single-process
(no fork) — the multiprocessing rollout path deadlocks on some macOS setups, so
prefer serial for local runs and Modal (`modal_apps/`) for scale.

## Current state & honest caveats

- **Base = Lineage B**, GuacaMol-trained (C/N/O/F), a **step-1,000 preview** of a
  3,000-step schedule, chosen for clean edit dynamics. It **overproduces small
  rings** — an open defect. Read
  [`docs/HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md`](docs/HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md)
  before committing to a final backbone or another large unconditional run.
- Conditional results are **mechanism-driven and base-independent** (fiber +
  SMC), so they carry to whatever backbone is finalized; only **E1 (unconditional
  quality)** and the pathwise-safety *magnitude* depend on the base.
- The pathwise-safety 67% is partly inflated by the small-ring defect (aziridine/
  epoxide dominate the per-alert incidence); the **phenomenon, the 0% guarantee,
  and the free cost are base-robust**, and the traversed alerts are not only
  strained rings.
- **Corpus:** base trained on **GuacaMol**; constrained-design **leads** are
  ZINC-derived (Jin QED set). "Matches GrIDDD" holds at the *element* level
  (CNOF), not the corpus.

## Documentation index

**Read these (current):**
- `docs/CONDITIONAL_RESULTS_AND_SCOPING.md` — canonical conditional results + scoping (option A + pathwise).
- `docs/HANDOFF_GENERATOR_RUN_LINEAGE_CORRECTION_V2.md` — the unconditional-backbone decision (read before large runs).
- `paper_iclr_stochastic_rewriting/main.tex` — the paper.

**Historical / superseded** (kept for provenance, not the current framing):
`docs/HANDOFF*.md`, `docs/PROJECT_STATUS.md`, `docs/ACTIVE_LANES.md`,
`docs/48_HOUR_RESULTS_TRACKER.md`, `docs/PAPER_POSITIONING_EXACT_CONTROL.md`
(superseded by the pathwise framing), the gate docs (`*_GATE.md`,
`SCALE16_GATE.md`), and the `docs/research_plans/` HTML plans (the original
distribution-learning framing, now reframed to structural + pathwise control).
```
