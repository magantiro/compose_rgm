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

**The one gap that remains open** is E1's framework counterfactual: **no
`FRAMEWORK_NEIGHBOR` has been qualified.** GrIDDD was examined and declined;
**DDSBM is the only remaining candidate.** That is the largest hole in the
package, and it is a hole in *novelty* evidence, not competence evidence.

Two candidates have now fallen to the same rule — the adapter would have had to
supply the mechanism under test. A pattern, not a coincidence.

## GrIDDD verdict — `CONCEPTUAL_LINEAGE_ONLY`, tier 4 to run / tier 2 to cite

Official code at `cc3dc31` (2025-09-28, one commit). Full evidence in
`FRAMEWORK_NEIGHBOR_GRIDDD.md`.

| axis | verdict |
|---|---|
| (a) supplied source | **YES** — `freegress.py:311,334-337`. The **only** external method audited here that clears this. |
| (b) edit budget | **NO** — similarity is a post-hoc rejection filter, `freegress.py:393-397` |
| (c) preference-conditioned | **PARTIAL** — target values at inference; property set baked in by CFG at training |

**The deciding fact is the authors' own ablation:** disabling insertions and
deletions leaves DRD2 and LogP success *"relatively unchanged"* (only QED drops,
to 33.8%, where the objective is a function of molecular weight). The mechanism
that motivated the `FRAMEWORK_NEIGHBOR` nomination is inert on the task nearest
ours. Plus: **no licence** anywhere in the tree (not vendorable), no checkpoints,
and an adapter that would have to supply an edit budget, a successor-fiber
notion, a separable frozen reference law and a checkpoint.

**But it is the closest external method to our task semantics found so far**, and
a strong contextual citation: same DRD2 oracle lineage — it repackaged the same
Python-3.6 pickle into an npz with the same `gamma=0.015625` and 2048-fold count
fingerprint COMPOSE extracted — and its 20-candidate / Tanimoto-0.40 protocol is
the one already frozen in our internal `GridDDProtocol`. Reported: **5.0% DRD2**,
**45.1% QED**.

**Also: GrIDDD is already in this repository, and not as GrIDDD.**
`griddd_conditional.py` (93 KB) imports no GrIDDD code — it is COMPOSE arms on a
GrIDDD-shaped task. A naming hazard, flagged. And `GridDDBenchmarkFairnessContract`
already records `exact_griddd_leads_available = False`, so a task-competence row
was already blocked before this lane looked.

## Editing-competence lane (E4) — opened 2026-08-14

**Question:** ignoring Pareto, is COMPOSE a credible source-conditioned editor?
Reviewer insurance, **not** a scientific pillar. Full audit in
`EDITING_COMPETENCE_AUDIT.md`.

### Can COMPOSE enter a published protocol directly? Yes physically, no as tier 1

The classic Jin et al. similarity-constrained DRD2 task is the closest anything
in this workstream has come to tier 1. **COMPOSE already holds every frozen
component**, verified in `artifacts/oracles/drd2_svm_v1/`:

- the **same** DRD2 oracle, parity to **2.19e-14** against the original estimator;
- the **exact** success threshold — `drd2_oracle.py:274-282` documents
  `margin ≥ 0 ⟺ P = 0.5`, no conversion ambiguity;
- the **exact** ECFP4 Tanimoto similarity;
- **200 of the 800 source molecules**, probabilities 0.000037–0.048158, all under
  the benchmark's ≤ 0.05 criterion.

**And it is still tier 2.** The protocol budgets *candidates*, not oracle calls —
because its native methods never query the oracle at inference. GrIDDD's DRD2
calls are dataset labelling, post-hoc metrics and success evaluation, **none
inside the sampling loop**; its `algorithmic_oracle_requests` are **zero**, as are
JT-VAE's, CG-VAE's and GCPN's. COMPOSE's would be thousands per source.

> The absence of an oracle budget is an unstated assumption, not permission.

This is the HN-GFN surrogate finding pointed at us. There the baseline had the
hidden advantage; here COMPOSE would. The same refusal applies in the direction
that costs us.

**Any COMPOSE row on this benchmark carries its `algorithmic_oracle_requests`
column or is not reported.**

### Costed, if ever authorized — CPU only, no GPU

| arm | 200 sources | 800 sources |
|---|---:|---:|
| greedy control | **~190 core-h** | ~750 core-h |
| verified control | ~3,200 core-h | ~12,800 core-h |

Cheapest useful first move: the 200 manifest sources, greedy arm, **~190
core-hours**, no dataset acquisition.

### Per-method tier verdicts

| method | role | tier |
|---|---|---|
| **DDSBM** | `FRAMEWORK_NEIGHBOR` | **1 on its own ZINC logP-shift task; 4 on anything DRD2** |
| **InVirtuoGen** — *selected practical editor* | `TASK_COMPETENCE` | **2** |
| GraphXForm | `TASK_COMPETENCE` | **4 — capability-disqualified** |
| GrIDDD, JT-VAE, CG-VAE, GCPN | contextual competence numbers | **2** |

### The practical editor — InVirtuoGen, on criteria fixed in advance

**GraphXForm is capability-disqualified: it cannot delete.** `grep` for
`RemoveAtom|RemoveBond|delete_atom|delete_bond` returns **zero hits**; the action
space is terminate / create atom / pick atom / bond order. The authors name atom
and bond removal as future work **twice**. It also budgets **8 h of H100
wall-clock** and explicitly rejects the oracle-call convention, so its numbers are
not citable across labs. Otherwise the best-licensed thing here (MIT, live
checkpoint, peer-reviewed *Digital Discovery*) — none of which repairs an
append-only action space.

**InVirtuoGen wins the three pre-registered criteria**: source-conditioned lead
optimization under a Tanimoto floor; real `delete_atom` / `delete_cyclic_bond`
operators; budget in **oracle calls (10,000)**, our currency. It is also unusually
honest about budget, flagging that GenMol and f-RAG prescreen ZINC250k so their
effective budget is ~260k, not 10k.

**Selected, but tier 2, and the caveats travel:**

- its citable `drd2` number (0.985 / 0.995) is **de novo PMO**, not our task;
- its one structurally matching experiment **does not run** — the shipped
  `ppo_docking.py` is a results-aggregation script with zero `torch`/`vina`/`ckpt`
  references, and 3 of 5 receptors are missing;
- its delete operators are lifted verbatim from **jensengroup/GB_GA**, so the
  honest attribution for its editing power is Graph GA, not the flow model;
- **licence, actionable:** `LICENSE` contradicts itself (CC BY-NC-SA 4.0 vs
  Attribution-NonCommercial 4.0), GitHub reports `NOASSERTION`, and the weights
  terms forbid using its output to train molecular-generation models.
  **No InVirtuoGen output may ever become `R_θ` training data.** For main to
  decide deliberately, not assume.

> **Neither practical editor's native benchmark aligns** with source-conditioned
> DRD2-style editing. InVirtuoGen is chosen because it wins criteria fixed before
> the evidence, not because it fits.

### DDSBM — the first tier-1 path this workstream has found

**Source-conditioned: YES**, verified in code — `predict_step` uses `target="0"`,
*"using original data to generate new data"*, then `sample_forward_bridge_batch(X_0,
E_0)`. (Trap: `on_test_epoch_end` discards the source for prior noise, but only
for the unconditional datasets. Reading one function would have called it de novo.)

**Tier 1 on its own task.** The ZINC logP 2→4 paired CSV is shipped in its repo,
so the 23,936 / 5,984 split is exactly reproducible and its metrics are
model-agnostic. **COMPOSE runs alone; DDSBM's Table 1 is cited as reported** on
validity, uniqueness, novelty, NSPDK, logP $W_1$, QED MAD, SA MAD, FCD.
**NLL excluded** — model-relative, defined against DDSBM's own reference process,
and it is the column carrying its headline win.

**This fills the framework-counterfactual slot** that has been empty since the
role amendment — on a distribution-shift task, not on DRD2.

**Tier 4 for anything DRD2:** `DRD2` occurs zero times in the paper; no budget, no
success criterion, no similarity gate. Requires a new paired dataset and full
retraining. **No checkpoints** (README TODO unchecked; Zenodo returns 0 records —
a search summary claiming otherwise is false), **no licence** (GitHub API `null`,
`/license` 404), 4× RTX A4000 × 3 seeds, wall-clock `UNVERIFIED`.

### A conflict surfaced, not resolved

The predecessor lane's frozen **T3** imposes a **1,000-evaluation budget** on a
similar-sounding similarity-constrained task; the published benchmark imposes
none. Two different tasks with similar descriptions — captioning one as the other
is the failure the audit exists to prevent. And a single budget-6 trajectory
touches ~3,500 candidates, so **T3's 1,000 does not cover two edit steps**.
Whether T3 is runnable as frozen belongs to whoever owns that suite.

## Defect found in a committed artifact — reported, not fixed

`artifacts/oracles/drd2_svm_v1/drd2_oracle_manifest.json` describes the pickle as
the oracle behind *"the VJTNN/GrIDDD success numbers"*. **GrIDDD reports no VJTNN
number** — VJTNN appears exactly once in that paper, in a related-work paragraph
setting it aside as requiring paired data. GrIDDD's baselines are JT-VAE, CG-VAE
and GCPN.

Not fixed: the manifest is hash-bound and consumed by claim-bearing code, this is
prose provenance, and no measurement depends on it. **For main to correct in a
future artifact version.**

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
