# STAGE6_SMOKE_REPORT.md — local end-to-end wiring + smoke-training gate

**Status: `GO_FOR_PRODUCTION_PREFLIGHT`** (not `GO_FOR_FULL_A100`). Machine-readable:
`diagnostics/composition/stage6_smoke_summary.json`.

> `GO_FOR_FULL_A100` is intentionally **not** returnable from this local stage. It requires (a) the at-scale
> GuacaMol trace pool, (b) the final mixture locked from measured statistics, and (c) a short
> production-environment preflight that loads the **real** B checkpoint. **No Modal was touched.**

## 1. Clean commit hash
`c7fe5ff` (branch `claude/control-closed-pareto-editing`). Audit + Stage-5/6 series is committed; the
pre-existing paper/script working-tree changes were deliberately left uncommitted.

## 2. Environment
Python 3.14.2; torch 2.11.0, numpy 2.4.2, rdkit 2025.09.6, scipy 1.17.1, networkx 3.6.1.

## 3. Exact commands
- Gate: `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src pytest tests/`
- Coverage/overfit + contract: `PYTHONPATH=src python scripts/stage6a_coverage_smoke.py`
- Characterization / manifest: `PYTHONPATH=src:scripts python scripts/characterize_edit_traces.py`;
  `PYTHONPATH=src python scripts/build_edit_data_manifest.py`
- Pool (deterministic): `PYTHONPATH=src:scripts python scripts/build_analogue_trace_pool.py`

## 4. §0b pipeline reproducibility
Pipeline source committed (`analogue_prior.py` + the `--analogue-trace-pool` gate wiring). The generated
pool is recorded by provenance (SHA-256 `58ace3f4…`, generation command, corpus id+hash, schema) in the
data manifest, not committed. **Reproduced in a pristine worktree: the pool SHA-256 matched exactly**
(mining is rng-free/deterministic), and the clean-worktree gate passed (463 at `e88cf81`, plus the new
stage-6 tests since).

## 5. Ring-catalog repair + tests
The pilot "`ring_system_delete` zero support" was a **characterization-config artifact** (the char ran
`make_edit_pair` without a catalog). The trainer already threads `catalog=ring_catalog`; with a catalog,
**92/200** corruption trims contain a ring-open. Verified in `tests/test_ring_opening_supervision.py` (5):
legal-set membership across ring types + a not-openable acyclic negative; replay validity; heteroatom
preservation; inverse round-trip; and a `ring_system_delete` **positive teacher is learned** (overfit
lowers loss and raises its probability). The shared versioned catalog's `ring_catalog_fingerprint` is
recorded in checkpoint metadata + manifest + SYSTEM_CONTRACT.

## 6. Positive target counts by family
`atom_insert/delete/restate`, `bond_reorder`, `bond_reroute`, `ring_system_restate`, `ring_system_delete`
= 3 each; `ring_system_grow` = 2 (from de-novo carbon-tree traces). `cycle_insert/cycle_attach` = 0
(`DISABLED_EVERYWHERE`). Full table: `diagnostics/composition/stage6a_family_contract.json`.

## 7. Gradient by family
Every production-enabled family's **family head receives a finite nonzero gradient from its positive
teacher** (contract `gradient_observed = true` for all 8) — positive-target supervision, distinguished
from a suppressive denominator gradient by the overfit raising each family's selected-mark probability.

## 8. Tiny-overfit results
23 examples over **23 unique states** (no conflicting targets). Loss **5.235 → 1.138** (converged by
~step 250), **gap 0.138** above the per-example Poisson-Bregman lower bound `r(1−log r) = 1`, **every
family's selected-mark probability rises**, no family missing → the model can overfit the verified
corpus. `tests/test_stage6a_coverage_overfit.py`. (Wiring verification only — **not** a quality claim.)

## 9. Requested vs observed mixture
`FactorizedMarkDataset` indexes the record tuple, so uniform/shuffled sampling makes the realized layer
mixture equal the **count ratio** (requested 0.400 vs observed 0.4005, |z|=0.08 within the multinomial
band). Reverse pairs are separate equal-count records (balanced). Caveat: a long trace yields more
state-action rows (one `PathRecord` → many rows); the path-length **curriculum bins** are the balancing
lever, and the 53-pair pilot pool is too small/concentrated (scaffold Gini 0.39) to validate the final
recipe — usable for wiring only.

## 10. Save/reload equivalence
Exact. A saved B-edit checkpoint reloads via `load_factorized_rollout_checkpoint` with organic vocab +
editing capabilities reconstructed from metadata, reproducing the original's sampled mark, hazard, and a
fixed-seed editing trajectory. A de-novo B reloads CNOF + editing-off byte-identical; a checkpoint missing
required rollout metadata **fails loudly**. `tests/test_checkpoint_roundtrip_and_init.py` (4).

## 11. Held-out rollout diagnostics
Over held-out drug-like leads and budgets 1/2/4/8: rollouts **start exactly at the lead** (no carbon-tree
seed), recompute the legal set per state, commit only executor-accepted actions, and keep **100%
all-intermediate validity** (valid + connected). The reversal/revisit/return-to-source diagnostic is
measured for plumbing; its **quality** reading requires the trained model. `tests/test_heldout_rollout_wiring.py` (2).

## 12. Remaining unverified production dependencies
- The **real trained B / B-edit checkpoint** (none on disk locally — only a B-compatible fixture loaded).
- The **at-scale GuacaMol-TRAIN analogue pool** (pilot = 53 pairs, scaffold Gini 0.39).
- **A2.2/A2.3** fused-ring + sparse-NN scaffold-pair compilation.
- The **final locked mixture percentages** (proposed, not locked).
- The **goal/region-conditioned controller** `h_φ(b,x,z)`, docking oracle, and external baselines (all post-training).

## 13. Precise status
**`GO_FOR_PRODUCTION_PREFLIGHT`.** All local data / training / sampling plumbing pass (→ `GO_FOR_AT_SCALE_MINING`),
AND the B-compatible init / head-widening / metadata / save-reload / short-rollout paths are verified with
the local fixture (→ `GO_FOR_PRODUCTION_PREFLIGHT`). The next step — the **production-environment preflight**
(requires separate user authorization) — loads the actual B checkpoint + scaled data manifest, runs a very
short train/save/reload/sample job, and only then decides `GO_FOR_FULL_A100`.

### Decision-logic check (all clear)
No `NO_GO_LOCAL_WIRING` condition holds: the ring-catalog gap is resolved; every enabled family has positive
targets; the tiny corpus overfits; save/reload is exact; no invalid intermediate or illegal sampled action
occurred; and the mixture loader implements its declared (count-ratio) recipe.
