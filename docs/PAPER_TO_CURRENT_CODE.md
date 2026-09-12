# COMPOSE paper-to-current code map

This page separates the implementation represented in the submitted workshop
paper from controller development performed after that submission. It is an
onboarding and lineage map, not a new scientific result. For current outcomes
and authorization, read `START_HERE_ICLR.md` first.

## Exact revision boundary

- `34da388` is the repository commit that first records the submitted NeurIPS
  2026 workshop packages. The COMPOSE package is
  `paper_gem_neurips2026/`; its full-paper source at that point is
  `paper_iclr2027/`.
- The post-submission local-to-global controller campaign begins at `f0a49ab`
  and is integrated on branch `compose-iclr`.
- The paper results were produced by several frozen experiment revisions, not
  by one monolithic controller commit. Their exact producer revisions and
  input identities live in the result artifacts and traceability documents
  named below.

To inspect the boundary without relying on prose:

```bash
git show 34da388:paper_gem_neurips2026/main_gem.tex
git diff --stat 34da388..compose-iclr
git diff 34da388..compose-iclr -- src/compose_v4/control
```

Do not use `paper_gem_neurips2026/RESULTS_STATUS.md` as current campaign
status. It is an August 26 build note retained inside the submitted-paper
package. Current status is in `docs/START_HERE_ICLR.md`.

## What is shared by the paper and current work

The present controller is not a separate molecule generator. It retains the
paper's production substrate and changes how that substrate is controlled.

| Scientific object | Production implementation |
| --- | --- |
| Complete molecular graph state | `src/compose_v4/chem/molecular_graph.py` |
| Legal executable rewrites | `src/compose_v4/rewrite/` and `src/compose_v4/rewrite/kernel.py` |
| Frozen factorized reference model | `src/compose_v4/model/factorized_tracelet_rate_model.py` |
| Canonical molecular-successor law | `src/compose_v4/experiments/production_successor_kernel.py` |
| Successor objective and training semantics | `factorized_successor_objective.py`, `factorized_successor_training.py`, `editing_successor_trainer.py` |
| Molecular-scale rollout and twisted SMC machinery | `src/compose_v4/experiments/hphi_rollout.py`, `hphi_smc.py` |

The scientific definitions are in the paper and
`paper_arxiv/SCIENTIFIC_TRACEABILITY.md`. Checkpoint identities and locations
are catalogued in `docs/GENERATOR_LINEAGE_MAP.md`. Bulk checkpoints and run
records live on the `compose-v4-artifacts` Modal volume, not in Git.

## Paper-era experiment map

### Submitted manuscript

- Workshop source: `paper_gem_neurips2026/main_gem.tex` and its `sections/`.
- Full source used for that abridgment: `paper_iclr2027/main.tex` and its
  `sections/`.
- Claim-to-code and claim-to-artifact boundaries:
  `paper_arxiv/SCIENTIFIC_TRACEABILITY.md`.

The directories `paper_iclr_control_substrate/` and
`paper_iclr_stochastic_rewriting/` are historical manuscript variants. Their
own `_STATUS.md` files say not to use their experimental sections as current.

### QED editing and future-aware control

- Controller contract: `docs/LEARNED_REACHABILITY_CONTROLLER.md` and
  `docs/AMENDMENT_VALIDATION_128.md`.
- Core implementation: `src/compose_v4/experiments/hphi_rollout.py` and
  `src/compose_v4/experiments/hphi_smc.py`.
- Frozen-panel launch and readers: `scripts/hphi_official800_launch.py`,
  `scripts/hphi_official800_status.py`, and
  `scripts/hphi_official800_curve.py`.
- Paper-facing summaries: `docs/VALID128_K8_RESULT.json`,
  `docs/VALID128_CURVE.json`, and the official-panel records named in the
  manuscript appendix.

### Exactness and trajectory capabilities

- Exact finite-state checks: `scripts/e0_toy_h_exactness.py` and
  `scripts/doob_guidance_ground_truth.py`.
- Future-reachability evidence: the artifacts under `diagnostics/reachability/`
  and the experiment described in the manuscript appendix.
- Retargeting: `modal_apps/retarget_intervention_app.py` and the
  `scripts/retarget_*` analysis tools.
- Pathwise constraints: `scripts/tier1_pathwise_safety.py` and
  `scripts/pathwise_precheck.py`.

### Workshop T4 table

- Protocol and original controller decisions: `docs/AMENDMENT_GENMOL_T4.md`.
- Production app: `modal_apps/genmol_t4_opt_app.py`.
- Table reducer: `scripts/t4_combined_table.py`.
- Historical result summaries: `diagnostics/genmol_t4_official_s1.json`,
  `diagnostics/t4_all_runs.json`, and
  `diagnostics/t4_combined_table.json`.
- Reconciled lineage: `diagnostics/controller_baselines/t4_lineage.json`.

Important limitation: the workshop T4 table selects between two historical
controller generations. It is not one frozen 500-call controller or a matched
mean-of-three result. Preserve that distinction in all new comparisons.

## Post-submission controller extensions

The newer work preserves the executor and frozen reference process while
adding a hierarchical search controller:

```text
Q(M | x,z)  ->  Q(o | x,M,z)  ->  q(w | x,M,o)
   WHERE              WHAT              HOW
```

| Extension | Primary implementation | Evidence or contract |
| --- | --- | --- |
| Local-to-global connected regions | `src/compose_v4/control/region.py`, `region_selector.py` | `experiments/region_resampling/` |
| Applicability-aware options | `option_selector.py`, `macro_engine.py` | `docs/MACRO_INVENTORY.md` |
| Primitive execution of complete options | `option_continuation.py` and option-specific modules | `docs/OPTION_CONTROLLER_V1.md` |
| Persistent semi-Markov controller | `option_controller.py`, `option_controller_runtime.py` | `docs/OPTION_CONTROLLER_RUNTIME_V1.md` |
| Resumable frontier search | `frontier_search.py`, `src/compose_v4/experiments/t4_frontier_search.py` | `docs/T4_FRONTIER_COMPARISON.md` |
| Paired T4 comparison | `t4_frontier_compare.py`, `t4_frontier_audit.py` | `diagnostics/t4_frontier_compare/` |
| PMO option continuation bank | `pmo_option_controller_bank.py` | `docs/PMO_OPTION_CONTROLLER_BANK.md` |

Macros and programs are proposal channels. They do not teleport to an
endpoint, replace the executor, or reduce the broad primitive support to a
finite fragment vocabulary. Every realized state still comes from the same
primitive executor.

## First hour on a fresh laptop

After cloning the repository:

```bash
git switch compose-iclr
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q \
  tests/test_option_controller.py \
  tests/test_option_controller_runtime.py \
  tests/test_t4_frontier_compare.py \
  tests/test_t4_frontier_search.py
```

Then read, in order:

1. `docs/START_HERE_ICLR.md` for the current result and next decision.
2. `AGENTS.md` for scientific and launch authority.
3. This map for paper versus post-paper lineage.
4. `diagnostics/t4_frontier_compare/README.md` for the current primary
   controller evidence.
5. `experiments/INDEX.md` to locate a specific runnable entrypoint.

Run `python3 tools/preflight.py` before any scientific launch. A fresh Git clone
is sufficient for reading the paper, developing code, and running unit tests.
It is not sufficient to reproduce remote scientific results: the required
checkpoint, lock, raw receipt, receptor, and docking assets must be obtained
from their recorded durable locations and verified by hash.

## Branches collaborators should know

- `compose-iclr`: reviewed integration branch and only recommended starting
  point.
- `legacy-main-recovery-20260912` at `f20f7bb`: exact, unvalidated quarantine
  of the prior dirty checkout. Do not merge it wholesale.
- Topic branches and detached historical worktrees are evidence lineage, not
  alternative starting points.

Both `compose-iclr` and `legacy-main-recovery-20260912` are published on
`origin`. New collaborators should check out `compose-iclr`; the recovery
branch exists only to make the preserved legacy state durable.
