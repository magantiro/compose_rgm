# AUDIT_REPORT.md — universal-edit-prior pre-training correctness audit

**Status: adversarial pass complete** (three Fable-5 agents: cyclic-graft quotient derivation, GM-loss
measure normalization, master-plan readiness — all returned). All fixable items surfaced are fixed +
regression-tested (**full suite 446 passed**); one pre-existing issue (A7) is surfaced for an owner
decision rather than blindly changed, per the deriving agent's own recommendation.

- **Base commit audited:** `94a6009` (branch `claude/control-closed-pareto-editing`)
- **Env:** Python 3.14.2; torch 2.11.0, numpy 2.4.2, rdkit 2025.09.6, scipy 1.17.1, networkx 3.6.1.
- **Scope:** the factorized tracelet model (`FactorizedTraceletRateModel`) — the de-novo base B and its
  editing fine-tune B-edit — its training loss, the CTMC/embedded-chain samplers, the executor, and the
  editing controllers' pancake sampler. Companion: `docs/SYSTEM_CONTRACT.md`.

### Committed audit state (frozen — do not verify against a dirty tree)
- **A1** (capability propagation, ring-restate + ring-opening) → commit **`d26817c`**
- **A2** (cyclic-graft successor quotient, `survival=1`) → commit **`207c76f`**
- Docs + stale-results manifest → commit **`2f887bb`** (this record itself is a docs-only follow-up).
- **Clean-worktree gate** — a *fresh* `git worktree` at `2f887bb`, pristine (empty `git status`), run with
  the main venv interpreter and `PYTHONPATH=<worktree>/src`:
  `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src pytest tests/` →
  **`446 passed, 6 warnings in 41.78s`**. No untracked source/test file is required for the pass; the
  committed state verifies independently of the working tree's pre-existing uncommitted files.
- **Only** these files are part of the audit commits (the pre-existing uncommitted paper/script/diagnostic
  changes in the working tree are NOT included): `src/compose_v4/experiments/canonical_successor_distillation.py`,
  `tests/test_editing_sampler_capabilities.py` (6 tests), `docs/SYSTEM_CONTRACT.md`, `docs/AUDIT_REPORT.md`,
  `diagnostics/STALE_RESULTS_MANIFEST.json`.

---

## Executive summary

The core mathematics is sound: the training loss is the correct Poisson-Bregman generator-matching
objective (`SYSTEM_CONTRACT §8`), training-trace construction and sampling share **one executor**, the
vocabulary has a single source of truth, no carbon-tree seed is reachable in editing mode, and the
stopping rule degrades to `Λ=0` (no NaN). Stereo/isotope/radical are deliberately unencoded, not bugs.

The audit found **one silent, high-severity inference defect** — the editing sampler enumerated the
**de-novo** vocabulary for a B-edit checkpoint — and, while fixing it, a **second high-severity defect**
my own first-pass fix exposed: cyclic graft NaN-crashes the pancake sampler on cyclic leads. Both would
have invalidated the headline editing experiments. The first is fixed and regression-tested for
de-aromatization + clean ring-opening; cyclic graft is gated behind a fail-fast guard pending the
quotient derivation now in flight. **No A100 training or headline experiment should run until the
cyclic-graft quotient is derived + verified (or a consistent no-cyclic-graft checkpoint is trained), and
the remaining correctness gates below pass.**

## Issue table

| ID | Severity | Subsystem | Symptom | Root cause | Evidence | Fix | Regression test | Status |
|----|----------|-----------|---------|-----------|----------|-----|-----------------|--------|
| A1 | HIGH | editing sampler | B-edit checkpoint silently samples the DE-NOVO vocabulary (de-aromatization / clean ring-opening / cyclic graft unreachable) despite loading wide organic heads | `AnalyticPancakeQuotientSampler` built its mark batch without the editing-capability flags → `compute_ring_restates/cyclic_graft/ring_opening` defaulted `False`; unlike `model.sample_rewrite_mark` which passes them | `canonical_successor_distillation.py:322,:434` vs `factorized_tracelet_rate_model.py:3105`; reproduced (rate 0 for ring_system_restate before fix, 0.15 after) | Propagate `model.enable_*` into both batch builds | `tests/test_editing_sampler_capabilities.py::test_pancake_sampler_enumerates_de_aromatization`, `::..._clean_ring_opening_preserves_heteroatom` | **FIXED** (ring-restate + ring-opening); cyclic graft → A2 |
| A2 | HIGH | graft quotient | Enabling cyclic graft in the pancake sampler → `productive_total_hazard = NaN` → sampler crashes on every cyclic lead | Pancake graft "survival" = productive/raw graft mass; the **raw** mask `_legacy_prequotient_graft_tables` is TREE-GATED (empty on cyclic graphs), so `0·exp(mass−(−inf)) = NaN` in the group-rate loop | `factorized_tracelet_rate_model.py:714`; `canonical_successor_distillation.py:527`; reproduced on `c1ccccc1CCC` | **Derived (Fable agent) + wired:** cyclic graft is already quotiented by the general canonical successor key (self-grafts dropped, aliases grouped) → raw partition falls back to the quotient partition, `survival[graft]=1`; enable the two cyclic-graft gates; remove the guard. Verified finite on 4 cyclic leads, `rate == Σ successor-group rates`, **inert on trees** | `::test_pancake_sampler_cyclic_graft_finite_and_matches_quotient`, `::..._fix_inert_on_trees` | **FIXED** |
| A7 | HIGH (pre-existing) → FIXED | graft quotient (de-novo tree) | The pancake sampler's **de-novo tree** graft used `Zr + survival`, which did NOT equal training's / the raw sampler's `Zq` normalization — graft family rate 0.097 vs 0.136 on octane | Family softmax used the raw partition `Zr` (self-grafts included) then survival; training normalizes graft over the quotient mask `Zq`. `softmax(·+Zr)·exp(Zq−Zr) ≠ softmax(·+Zq)` (family denominators differ) | cyclic-graft agent's symmetric-tree comparison; measure agent confirms the raw sampler is `Zq` | **Full-Zq (owner decision):** the whole pancake lane (analytic sampler + calibrated target) normalizes graft over `Zq` for tree AND cyclic; `survival[graft]=1`. Legacy `Zr`+survival retained behind `PancakeQuotientCalibration.legacy_raw_graft_survival` (named ablation). Verified pancake family kernel == raw sampler `<1e-5` on 6 state classes; graft mass = Σ group rates; MC null-band; Zr diverges | `tests/test_pancake_graft_kernel_equality.py` (7 tests) | **FIXED (full-Zq)** |
| A6 | CRITICAL→resolved | recipe ↔ sampler coupling | The recipe couples all four `enable_*` flags to `corrupted_prior_mix`, so the B-edit checkpoint it produces has `enable_cyclic_graft=True` — under the interim guard that was unsampleable | flag coupling (`train_tracelet_cnof_gate.py:2625-2628`, `evaluate_tracelet_rollouts.py:191-194`) | flag sites | **Resolved by the A2 fix:** cyclic graft is now correctly sampleable, so an all-flags-on B-edit checkpoint samples correctly — no flag decoupling needed. (Decoupling remains available as the spec-#9 fallback if a cyclic-graft-free ablation is ever wanted.) | covered by A2 tests | **RESOLVED (by A2)** |
| A3 | MEDIUM | framing / process | Editing sampling described/assumed as a timed CTMC, but it is a fixed-step **embedded jump chain** — the hazard Λ is discarded; only the mark distribution is used | Controllers loop `for step in range(max_steps)`, advance a fixed `time_step`, read only `action is None` | `griddd_value_guided_smc_controller.py:206`, `pareto_editing_hero.py:139` | Documented in `SYSTEM_CONTRACT §10`; exact Doob steering for editing = the finite-horizon transform (Appendix A), not the generator transform | Doob enumerable check (task #35) | **DOCUMENTED**; enumerable check PENDING |
| A4 | LOW | sampler cache | `model._sampling_state_cache` and the pancake `_context_cache` are keyed by state only (no capability component) | No `OperatorCapabilities` in the cache key; safe today because capabilities are fixed per model instance, but a latent flag/vocabulary collision if two capability sets ever share a process | `factorized_tracelet_rate_model.py:247`, `canonical_successor_distillation.py:588` | Deferred: route raw + pancake batch builds through one immutable `OperatorCapabilities` + a fail-fast startup capability assert (spec #1/#2) — after A2 | — | **OPEN** (deferred, post-A2) |
| A5 | resolved (not a bug) | measure / normalization | Is `p_θ` a proper distribution (loss algebraic validity); do non-graft families over-count symmetric successors? | — | **measure agent:** `Σ exp(selected_mark_log_probability) = 1` to ~1e-7 on every molecule; raw sampler consistent with the loss (bootstrap-TV PASS incl. neopentane's 12-fold graft group). Non-graft duplicates **split** mass correctly (no over-count); graft over-count is handled **by design** via group-logsumexp. `unseen=0` (sampler never leaves analytic support) | — | — | **VERIFIED CORRECT** |
| A8 | LOW→resolved | corruption teacher (graft) | A graft teacher enumerating one molecular successor via k coordinate-marks would inject k× reward (each carries the full group mass) | mark-level vs successor-level graft supervision | measure agent | Guarded: the DRY shared enumerator `pendant_graft_candidates` + grouped logsumexp means a corruption graft always lands as one group; non-graft families are immune | existing enumerator sharing | **NO ACTION (guarded)** |

## Verified correct (direct source evidence)

- **GM Bregman loss** is `Λ_θ − r·(logΛ_θ + log p_θ(selected))` — exact Poisson/KL generator matching
  (`factorized_tracelet_rate_model.py:4136`). Illegal teacher → `-inf` → non-finite loss guard (`:3789`).
- **Shared executor** train↔sample: `de_novo_rewrite_system().apply` (`kernel.py:51`; `trace.py`,
  `analogue_prior.py`, controllers).
- **No carbon-tree seed reachable in editing** (seeds from the lead; `evaluate_tracelet_rollouts` is
  de-novo-only, hard-raises otherwise).
- **Vocabulary** single source of truth; **charge neutral-only heads** → edit-protection is the correct
  scope boundary; **no stereo/isotope/radical** by design.
- **Termination** degrades to `Λ=0` (`<TERMINAL>`), no NaN; self-loops excluded from the fiber.
- **De-aromatization + clean ring-opening reachable through the (fixed) pancake sampler**, valid+connected
  successors, inverse round-trips, decoration preserved (clean keeps ring N; structured carbon-izes) —
  `tests/test_editing_sampler_capabilities.py` (4 tests), full suite 444 passed.

## Correctness gates before the A100 headline run (tracking)

1. Vocabulary round-trips exact — *pending explicit test (task #26 machinery).*
2. Checkpoint vocab/operator mismatch fails loudly — *partial: capability fail-fast added (A2 guard); OperatorCapabilities assert deferred (A4).*
3. Every registered operator: applicability/execution/validity/inverse — *partial (§9.1 inverse-consistency prior; fuzz task #30).*
4. Random legal-op fuzz: zero unexplained validity failures — *pending (task #30).*
5. Compiler replay exact + endpoint iso + inverse round-trip — *prior work; re-confirm.*
6. Training target always in the enumerated legal set — *pending fuzz (task #29/F).*
7. Padding/illegal → zero prob/rate after masking — *pending (task #26).*
8. Every intended head gets finite gradients — *pending (task #34).*
9. Model overfits a tiny verified dataset to its minimum — *pending (task #28).*
10. Save/reload preserves predictions + sampling — *pending (task #30/G).*
11. Production sampler matches an exact tiny-state-space kernel within MC error — *pending (task #27/#35).*
12. Held-out-source trajectories start at the lead + stay valid — *pending (task #29).*
13. No carbon-tree init in editing — **PASS** (verified).
14. No val/test leakage in trace construction — *pending (analogue pool).*
15. Metric implementations pass hand-computed unit tests — *pending (downstream).*
16. **Cyclic-graft quotient derived + verified, kernel sums to 1, all controllers on identical successor
    support** — **BLOCKING**, in flight (task #33).

## Verdict

**The editing sampler is now correct for the full B-edit vocabulary** (de-aromatization, clean
ring-opening, ring-atom bioisosterism, cyclic graft), and the GM loss/measure is verified proper. The
two correctness blockers this audit found (A1, A2) are fixed + regression-tested; A5/A6/A8 are resolved.

- **Small local editing smoke (any/all editing families):** the sampler now enumerates the full B-edit
  vocabulary with correct measure — **GO** for scoped local smoke *once a checkpoint exists* (none is on
  disk locally today, not even de-novo B).
- **Full headline editing experiments (H1/H2/H3):** **NO-GO**, but now gated on *program* work, not
  editing-sampler correctness:
  1. the universal edit prior is **not trained** (A100, user-gated);
  2. the master-plan §0b recipe (MMP-at-scale + ε full-support mixture + budget curriculum) is not
     finalized (analogue pool is a 53-pair pilot → identity-bias risk);
  3. the **goal/region-conditioned controller `h_φ(t,x,z)` is NOT-STARTED** (only a fingerprint-only,
     single-objective value net exists);
  4. multi-objective metric infra (normalized/HV-AUC/IGD+/feasible-HV) and a docking oracle are missing;
     external baselines are 0/10 runnable.
- **A7 (RESOLVED — full-Zq):** the pancake lane now normalizes graft over the quotient partition `Zq` for
  tree and cyclic grafts, inducing exactly the raw-sampler / GM-trained canonical-successor kernel
  (verified `<1e-5`); the legacy `Zr`+survival convention is a named ablation only.
- **Stale results:** any editing rollout produced through the pre-fix pancake sampler (e.g.
  `diagnostics/composition/composition_learned_vs_uniform*.json`, Jul 22–23) sampled the de-novo
  vocabulary and must be treated as **stale/structurally-uninformative** and rerun — never cited.
