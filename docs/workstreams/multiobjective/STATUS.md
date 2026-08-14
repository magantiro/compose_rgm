# STATUS — multiobjective workstream (Lane 5)

**Status:** redirected 2026-08-13. Stage 0/1 accepted. The published-protocol
alignment audit is complete on the dimensions that decide the verdict.

**Branch:** `codex/compose-multiobjective-package`, based on
`codex/editing-v2-successor-fiber-fastpath` @ **`f6146d7`**.

**The base has since advanced to `667c5ec`** (five commits: Lane 4's top-up
work, the main-lane baseline philosophy, and the DRD2 batch-invariance defect).
This branch is **behind, not divergent** — it modifies **11 files, all its own**,
and touches nothing those commits touched. Verified with
`git diff --name-only f6146d7..HEAD`. Rebasing is a trivial fast-forward merge
whenever main wants it.

**Governing document:** `docs/BASELINE_PHILOSOPHY_AND_MAIN_LANE_DECISIONS.md`
(`7039cd1`), which binds this lane. This audit executes its Decision 2, and the
outcome lands on that decision's **"Not aligned → label published values
contextual, non-head-to-head"** branch.

**What is running:** nothing. **Compute spent: zero.** No Modal, no GPU, no
baseline execution, nothing installed into any project environment.

**Held-out data opened:** **no.** This lane opened no data at all.

**Last completed gate:** GSK3β/JNK3 alignment audit — **freeze declined**.

**Next action:** the lead chooses between fallback option 1 (contextual citation
only, recommended) and option 2 (resource frontier, which needs an oracle
extraction that does not exist).

---

## The headline

> **GSK3β/JNK3 cannot be frozen as a head-to-head Panel B.** Two dimensions fail
> independently. The published values are **contextual, non-head-to-head**.

| # | dimension | verdict |
|---|---|---|
| 1 | oracle files / score definitions | **FAIL** |
| 2 | allowed query / oracle budget | **FAIL** |
| 3 | hypervolume reference point | **FAIL** |
| 4 | top-K and preference weighting | **FAIL** |
| 5 | validity and canonicalization | **FAIL** |
| 6 | seeds and reporting convention | **FAIL** |

**All six fail.** The audit expected one or two to fail and the rest to pass.

**The single most damaging fact.** InversionGNN's Table 3 attributes
**50K+20K = 70,000** oracle calls to HN-GFN (`inv.txt:395-399`), while HN-GFN's
own paper states **1,000** (`hngfn.txt:489-491`, matching `main_mobo.py:50-52`).
A **70× disagreement between two published papers about the same method**. Their
hypervolumes disagree accordingly — HN-GFN reports its own as 0.669 ± 0.061,
InversionGNN reports HN-GFN's as 0.592 ± 0.042. That is a **protocol mismatch,
not a reproduction failure**, and it closes the last route to a valid
head-to-head table: citing each paper is blocked by dimension 1, and citing
InversionGNN's single self-consistent table would misrepresent HN-GFN by 70× in
budget.

**Dimension 1.** HN-GFN scores GSK3β/JNK3 with a **1024-bit** ECFP4 against its
own RandomForest pickles (`utils/chem.py:63`, `kinase_scorer.py:18`). TDC — which
InversionGNN calls (`molecular/denovo.py:9,25-26`) — uses a **2048-bit** ECFP4
(`tdc/.../oracle.py:682,710`). A model trained on 1,024 features cannot consume
2,048. **The two papers are already not head-to-head with each other**, before
COMPOSE enters. TDC additionally swaps model file by installed sklearn version,
so "the TDC GSK3β oracle" is itself two objects.

**Dimension 2.** Published budget: **1,000** true-oracle evaluations
(`main_mobo.py:50-52`). COMPOSE per single preference trajectory, computed from
`diagnostics/pareto_semantic_oracle_accounting.json`: `greedy_pref` **2,187**,
`verified_pref` **53,977**. COMPOSE cannot finish one trajectory inside the
entire published budget; a full five-preference front is **270×** it.

**This generalizes.** Oracle-budget benchmarks live at 10³–10⁴; PMO uses 10,000.
COMPOSE's primary arm needs ~5.4 × 10⁴ *per trajectory*. **Changing benchmark
does not fix this** — the mismatch is with the benchmark class.

## Cost of running COMPOSE alone under it

**Compute is not the obstacle:** ~58 min per source on 8 CPUs for all five arms,
so a 12-source panel is roughly **12 core-hours, CPU-only**.

**The blocker is an artifact that does not exist.** COMPOSE cannot call
`tdc.Oracle` — installing PyTDC is barred, and TDC's sklearn-dependent model
selection is exactly the drift `artifacts/oracles/drd2_svm_v1` was built to
prevent. It would need a version-independent extraction of two large
RandomForests (97 MB, 38 MB), hash-bound with a parity panel: **1–2 days of
engineering, zero GPU**. Since dimensions 1 and 2 fail regardless, **that
extraction should not be built to serve this comparison.**

## Panel B status: `BLOCKED_ON_QUERY_EFFICIENT_COMPOSE`

**Not `BASELINE_UNAVAILABLE` — the blocker is ours.** Relabelled per
`docs/AMENDMENT_PUBLISHED_NUMBER_FIRST.md` §3.

> **Stop looking for a benchmark that aligns.** No choice of benchmark fixes a
> 5–50× budget-class mismatch.

Paused, not impossible. External baseline execution and dependency archaeology
stop here.

**Named unblock condition:** `R_θ` plausibility shortlisting, upgraded to
`CONDITIONAL_LOAD_BEARING_FOR_ORACLE_BENCHMARKS`. One preregistered operating
point at ~10× query reduction, **no `K` sweep**, primary result **HV retained per
oracle query retained**. Ordering unchanged: repaired P3/P4 → full-fiber Pareto
capability → then this. **Not designed here, and this lane does not scope it.**

It is not needed for the core capability claim. It is needed only for the
separate claim that COMPOSE competes in ordinary oracle-budget MOO.

## Execution tiers — what could ever be cited vs what needs running

| tier | methods here |
|---|---|
| **1** exact alignment → run COMPOSE only, cite as "reported" | **none** — tier 1 needs exact protocol alignment and the audit found none |
| **2** partial → contextual only, never head-to-head | HN-GFN; InversionGNN (to cite); CDD (Lane 6) |
| **3** rerun smallest necessary set on our custom task | **DDSBM, GraphXForm** — the only two this lane would ever expect executed |
| **4** cite and discuss; do not reconstruct | OP-GFN, Edit Flows, Expanding Flow Maps, MOG-DFM, PepTune, pCoMole, AReUReDi, InversionGNN (to run), GrIDDD's task row |

## Where the argument now rests

With Panel B blocked, the multiobjective argument rests on **Panel A and the
`MATCHED_CAUSAL_CONTROL` arms**. That is where the scientific content was anyway.

**And that evidence just got stronger, from another lane.** P3/P4 source 000 has
**MATCHED** — the committed 368-kernel target reached at 369 after 1,025
generate-and-rank trajectories, 16.8× more search than the broken matcher gave
it. The fair closed-loop-versus-open-loop comparison now exists. The
preregistered `MATCHING_UNREACHABLE` fallback did not fire for that source, which
is why writing it down before the outcome was worth doing.

**The one gap that remains open** is E1's framework counterfactual: no
`FRAMEWORK_NEIGHBOR` has yet been qualified. DDSBM is the principal candidate and
GrIDDD's qualification is in progress. That is the largest hole in the package,
and it is a hole in *novelty* evidence, not competence evidence.

## Standing findings from Stage 0/1 (unchanged, now project record)

- HN-GFN is GPU-only with no released checkpoint — **now moot**, since we cite
  rather than rerun.
- InversionGNN is not runnable as shipped and has **no LICENSE**.
- OP-GFN is excluded on CC BY-NC-**ND** NoDerivatives.
- `main.py:145` opens every HN-GFN rollout with an empty
  `BlockMoleculeDataExtended()`; no qualified external method is
  source-conditioned, so none may enter Panel A.
- `surrogate_calls` axis and `INCOMPARABLE_SURROGATE_ASYMMETRY` verdict adopted.

## Escalation — unchanged, and still the highest-priority human task

**pCoMole has not been read.** OpenReview is not machine-reachable; not on arXiv.
All ten schema fields remain `UNVERIFIED`, deliberately not filled from search
summaries. The lead has an OpenReview login.

## Deliverables in this directory

`STATUS.md` · `PROTOCOL.md` · `BASELINE_TASK_MATRIX.md` ·
`SOURCE_CONDITIONING_AUDIT.md` · `BENCHMARK_ALIGNMENT_AUDIT.md` ·
`SAME_LAB_LINEAGE.md` · `DECISION_LOG.md` · `HANDOFF.md` · `handoff.json`

Code: `src/compose_v4/experiments/multiobjective_qualification.py`,
`tests/test_multiobjective_qualification.py` (23 tests, all passing).

Every artifact carries `DESIGN_ONLY`.
