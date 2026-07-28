# Program Coherence Report — COMPOSE/RGM (audit @ HEAD 4204c9d)

## Verdict: **COHESIVE_WITH_QUARANTINED_LEGACY**

One canonical scientific program exists; every production path implements it; training/validation/testing/
sampling enumerate identical legal support; one executor + one canonical successor; legacy paths are
explicitly quarantined and **not reachable-and-silently-wrong** from the canonical recipe; the papers describe
the active program (2 MEDIUM overstatements, correctly hedged). **0 CRITICAL deviations.** The 3 HIGH items are
structural (no canonical model factory), experimental-lane (reward-FT silent default), and one device bug
(**already fixed**) — none is a live divergence in the RingCore production path.

## The 12 questions
1. **One canonical program?** Yes — RingCore-V1, contract fingerprint `3917a0bf4a3de9830722663f`
   (`PROGRAM_CONTRACT.md`).
2. **Which entry points implement it?** `train_tracelet_cnof_gate.py` (PRODUCTION_CANONICAL trainer) via
   `train_tracelet_gm.py` (wrapper); `evaluate_tracelet_rollouts.py` + `ring_core_rollout_panel.py` (eval).
   Full inventory + classification in `ENTRYPOINT_MATRIX.md` — nothing UNKNOWN.
3. **Train/validation/test/sampling/control support identical?** Yes — the data-gen fibers, the dense-mask
   enumerator, and the sampler share **one** `prepare_factorized_mark_batch` + **one** `RewriteSystem.apply`.
   Proven by the golden-path test (`tests/test_golden_path_coherence.py`) + the teacher-in-candidate invariant
   (zero mismatches across neutral/cation/anion/zwitterion/S/Cl/fused strata).
4. **Artifacts contract-compatible?** Yes — one contract fingerprint; loaders reject incompatible checkpoints
   (`test_ring_core_identity_gate.py`, 10 negative tests); eval-cache v2 + capability-keyed.
5. **Parallel/stale model definitions?** No live duplicate in the RingCore path — trainer + loader are
   flag-consistent (verified symmetric persist/read). Structural risk H1 (no canonical factory) + DEV-4
   (trainer multiplexes legacy backends, only factorized is production).
6. **RingCore-V1 consistent everywhere?** Yes, with one nuance (R1): ring GENERATION is compositional
   (catalog-independent), but ring DELETION (`ring_delete`) is catalog-bounded to the 5 seed topologies — so
   "macros disabled / catalog acceleration-only" is only partially realized (`ring_delete` is a retained,
   catalog-bounded macro by design).
7. **Legacy branches reachable?** No — pancake/CNOF/lipid/strict-warmstart/v1-cache/superposed are all
   flag-gated off, dead, migration loaders, or explicit ablations (`LEGACY_QUARANTINE.md`). One guard GAP (G1):
   no assertion `cycle_op_mix ⇒ disable_ring_grow_macro` (canonical passes the flag; a forgetful launch would
   run grow+cycle).
8. **Paper matches code?** Yes — no HIGH-risk archetype present (no planned-controller-as-implemented, no
   disabled-macro-as-active, no "trained on 500k", no arbitrary charged editing, no synthesizability-from-
   validity, no exactness-at-scale). 2 MEDIUM overstatements (P1 charged-base-corpus wording; P2 catalog-
   independence claimed for the released base rather than the unreleased core) + LOW terminology/context drift.
9. **Deviations found?** 25 total: 0 CRITICAL · 3 HIGH · 8 MEDIUM · 14 LOW (`DEVIATION_REGISTER.md`).
10. **Which fixed?** DEV-CUDA (HIGH device bug in the warm-start checksum) fixed this session.
11. **Which need an owner decision?** The fix queue (below) — factory consolidation (H1), reward-FT
    capabilities (H2), the `cycle_op_mix⇒disable_grow` guard (G1), paper wording (P1/P2), doc precision (R1).
12. **Active run interpretable under its launch commit?** Yes — pinned to `4204c9d`, no drift
    (`active_run_lineage.json`). It failed on DEV-CUDA (device bug, not a science issue); relaunching from the
    fix commit.

## Top 5 architectural risks
1. **No canonical model factory (H1)** — trainer + loader hand-duplicate a ~15-arg constructor; consistent
   today, silent mis-load if a flag is added to one side.
2. **`OperatorCapabilities` not the single source in production (M1) + omits cycle_ops (L1)** — three parallel
   manual capability reconstructions; the fingerprint under-identifies the defining RingCore feature.
3. **Guard gap G1** — `--cycle-op-mix` without `--disable-ring-grow-macro` silently yields a two-ring-addition-
   mechanism model.
4. **Device-portability blind spot (DEV-CUDA)** — a CPU-only dry-launch cannot validate the CUDA path; needs a
   minimal GPU smoke.
5. **Corruption in-memory serial build** — a full-run bottleneck (~14h) requiring precompile+parallelize.

## Owner decisions required
- Approve the fix queue for the HIGH/MEDIUM items (H1 factory, H2 reward-FT, G1 guard) as failing-test-first
  patches (do not touch model/mixture/optimizer/schedule).
- Approve the paper-wording corrections (P1, P2) — text-only.
- Approve the corruption-pool precompile+parallelize as the full-run scaling prerequisite.

## Required return
- **Verdict:** COHESIVE_WITH_QUARANTINED_LEGACY
- **Contract fingerprint:** `3917a0bf4a3de9830722663f`
- **Active launch commit:** `4204c9d` · **current HEAD:** advanced by audit deliverables + the DEV-CUDA fix
- **Production entry points:** ~10 CANONICAL + 4 WRAPPER (see matrix)
- **Deviations:** 0 CRITICAL · 3 HIGH (1 fixed) · 8 MEDIUM · 14 LOW
- **Smallest next action:** relaunch the bounded run from the DEV-CUDA fix commit (blocker cleared); then the
  fix queue (H1/H2/G1) failing-test-first.
