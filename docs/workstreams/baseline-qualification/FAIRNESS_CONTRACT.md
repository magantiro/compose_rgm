# Fairness contracts for the MUST_RUN comparators

**Artifact status: `DESIGN_ONLY`.** No comparison has been run. This document
says what "matched budget" would mean, and where each comparison stops being
fair, so that neither can be decided after seeing results.

The four `MUST_RUN` methods are **MARS**, **GraphXForm**, **GraphGA** and
**REINVENT**. `HN-GFN` is `CONDITIONAL` (no CPU path, broken upstream import) and
`DDSBM` is `CONTEXT_ONLY` (no license, no checkpoints, no oracle in the loop);
their contracts are in the registry but they are not expected to enter a matched
table.

---

## 0a. What each comparison is actually testing

**"Future-aware control helps" is not a universal COMPOSE claim.** It is
established on hard exact-target recovery and *measured absent* on an easy
target-free property goal:

| | established | measured absent |
|---|---|---|
| task | exact-target recovery, finite edit budget, sealed 65 pairs | bounded developability goal (cLogP box + QED floor) after a same-prefix switch, 3+3 |
| result | greedy 26/65 (40%) → verified rollout 40/65 (62%), +21.5 pp [12.3, 33.5] | greedy 28/30 = verified 28/30, binary headroom **0**; gate CLOSED |
| artifact | `diagnostics/editing_v2_sealed67_result.json` | `diagnostics/retarget_calibration_result_3plus3_fixed.json` |

Three consequences that change how the tables below must be read:

1. **The static-optimization table (C4a) is a competitiveness sanity check, not
   a claim-bearing test.** PMO's DRD2/GSK3β/JNK3 tasks and the developability box
   are exactly the *easy target-free property goals* where COMPOSE has now
   measured that future-aware control adds nothing. A baseline winning there
   refutes nothing COMPOSE claims. A COMPOSE win there supports the weaker
   sentence "the substrate plus ordinary control is competitive" — which is worth
   having, and is not the novelty argument.
2. **The regime where a baseline would actually have to beat COMPOSE is C4b**,
   and on C4b none of the six methods is applicable at all: GraphXForm cannot
   delete an atom, MARS has no remaining-budget concept, GraphGA does not
   preserve the source, REINVENT and HN-GFN have no intermediate molecular
   states, DDSBM has no oracle in the loop. Report `N/A`. **Do not report the
   absence of an applicable baseline as a COMPOSE win.**
3. **C4c is about intervention responsiveness and prefix reuse, not post-switch
   planning quality** — because the post-switch planning subclaim was gated on a
   measurement and the gate closed. So the REINVENT-staged-learning and
   MARS-restart arms are judged on **post-switch oracle cost and whether a
   restart was required**, not on endpoint score. That is already the standard in
   `docs/EXPERIMENT_PLAN.md` Experiment 6.

**Binding anti-tuning rule.** Do not select a comparison task because it makes
greedy fail. A goal on which future-aware control shows no advantage is
*reported*, not replaced. This rule is inherited from
`docs/RETARGETING_SAME_PREFIX_DESIGN.md` and applies to baseline selection as
much as to goal selection.

---

## 0. Dual oracle accounting — FROZEN 2026-08-13

**Decided by the main workstream. Not open for reinterpretation at run time.**

Every run logs **both** counters. Neither may ever be substituted for the other,
and no table may report one without naming which it is.

| counter | definition | used for |
|---|---|---|
| **`benchmark_native`** | unique **valid canonical** molecules scored. A repeat of a canonical SMILES already scored in this run does not increment. An invalid or unparseable molecule does not increment. | **only** comparison against published PMO numbers |
| **`raw_compute`** | **every** oracle invocation: duplicates, rejected proposals, invalids, particles, prescreened candidates, reranked endpoints, and rescores. | **all** actual efficiency claims |

The two exist because they answer different questions, and each one alone is
misleading in a predictable direction:

- `benchmark_native` is the only quantity comparable to the literature, and it
  **understates** the cost of a method that re-evaluates. MARS has no cache
  anywhere and rescores its current molecule whenever a proposal is invalid;
  under `benchmark_native` alone that work is invisible.
- `raw_compute` is the honest efficiency number, and it **overstates** a method
  relative to published results. REINVENT caches per scoring component by design
  and its diversity filter zero-scores repeats; GraphGA natively charges for
  duplicate offspring. Reporting `raw_compute` beside a PMO leaderboard number
  would be comparing two different quantities.

**The rule this freeze exists to prevent:** a later run picking whichever
convention flatters us. Both numbers are always logged, so the choice is made
once, in the caption, and is visible.

### Binding consequences

1. A results table states its counter in the caption. `benchmark_native` and
   `raw_compute` never appear in the same column.
2. A PMO-comparable row uses `benchmark_native` **and** may not carry a
   REINVENT 4 capability claim (PMO wraps a REINVENT 2.0-era reimplementation).
3. An efficiency claim — "COMPOSE reaches X at fewer oracle calls" — uses
   `raw_compute`, for every method including COMPOSE.
4. The ratio `raw_compute / benchmark_native` is itself reported per method. It
   is the cache-and-duplicate rate, it differs by orders of magnitude across
   these methods, and hiding it is how an unfair comparison survives review.
5. Caching is allowed for every method with identical semantics (same canonical
   SMILES ⇒ same score). A cache hit increments `raw_compute` and does not
   increment `benchmark_native`.
6. Invalid or unscorable molecules increment `raw_compute` only, and are
   additionally reported as `failed_proposals`.

Implemented and enforced by `src/compose_v4/experiments/oracle_accounting.py`;
instrument check in `diagnostics/baselines/graph_ga_accounting_smoke.json`.

---

## 1. MARS

> **Naming rule, binding.** The MARS arm is **"restart at `x_τ` under a new
> objective"**. It must **never** be described as native same-prefix retargeting,
> mid-trajectory goal switching, or continuation. **It is a new optimization run
> launched from the current molecule.** The realized molecule is preserved as an
> initial state and nothing else is: the editor is randomly re-initialised
> because no shipped code path saves it, the imitation dataset starts empty, and
> the temperature counter resets. Use the phrase "restart at the switch molecule"
> in every table, caption and sentence, because "continue from" and "restart at"
> are the exact pair a reader will conflate.

**Matched quantity.** Property-oracle calls, counting every scored candidate
including rejected proposals and including the redundant re-scoring MARS
performs when a proposal is invalid. Secondary: edits — one attempted edit per
chain per step against one committed COMPOSE edit.

**Where it is fair.** MARS's state is a complete molecule, it can be started
from an exact supplied molecule with stock code, and it needs no offline
retraining for a new objective. On source-conditioned static optimization at a
declared budget, this is a genuine like-for-like comparison of two iterative
molecular editors.

**Where it is not fair.**

1. **Regime.** MARS's designed operating point is ~10⁶ molecule scorings per run
   (1000 chains × 1000 steps at code defaults; 5000 × ~550 at paper settings).
   Its adaptive proposal only becomes useful *after* being trained online on many
   accepted edits. A COMPOSE-matched budget of hundreds to thousands runs it
   three orders of magnitude below that. **Both rows are required** — matched
   budget and MARS's native regime — with the gap stated. Reporting only the
   matched row would be a misrepresentation dressed as rigour.
2. **Support.** Fragment-level edits from a ChEMBL-derived 1000-fragment
   vocabulary versus canonical successors of the frozen rewrite kernel. Never
   claim equal support.
3. **Caching.** MARS has none. Adding one (which the fairness rules require) is
   a change to the method's cost profile and must be reported with the hit rate.
4. **Pathwise metrics are `N/A`.** MARS has no constraint mechanism. Adding a
   motif filter to `Proposal.propose` would be *our* method, not MARS.

**In the retargeting experiment.** MARS is the **restart-at-the-switch-molecule**
arm and it works with stock code. It is not a continuation arm: the proposal is
randomly re-initialised (no checkpoint path exists), the imitation dataset starts
empty, and the temperature counter resets. Report it as a restart, and report
that MARS has no remaining-budget concept at all — it is an infinite-horizon
annealed sampler whose temperature is pinned at a 1e-2 floor from roughly step
450 onward.

---

## 2. GraphXForm

**Matched quantity.** Property-oracle calls, counting every beam leaf scored
**and** every call consumed by per-objective fine-tuning. **Two totals must be
reported** — with and without the fine-tuning phase. Edit budget is `N/A` and
must be printed as `N/A`: its actions are atom and bond additions, not
molecule-to-molecule edits.

**Where it is fair.** `start_from_smiles` gives exact source conditioning with
the model continuing to modify the supplied molecule, and the full action history
is exported. For source-conditioned static optimization from a designated
starting structure, GraphXForm is the closest external analogue of what COMPOSE
does.

**Where it is not fair.**

1. **Constructive-only action space.** No atom or bond removal exists; the paper
   defers removal to future work. On any task requiring a deletion GraphXForm
   cannot reach the answer at all. That is a **scope difference, not a quality
   result**, and must be stated wherever it binds rather than reported as a loss.
2. **No native oracle budget.** The paper explicitly rejects one and runs to
   convergence or an 8-hour wall clock, arguing its objectives are cheap.
   Imposing a budget runs it outside its designed protocol; say so.
3. **Per-objective fine-tuning has no COMPOSE counterpart** because `R_theta` is
   frozen. That asymmetry *is* the claim under test. Do not normalise it away,
   and do not hide it by reporting only the search-phase calls.
4. **Intermediate states are valid only at action level 0.** Between levels the
   graph carries a dangling unbonded atom. Pathwise metrics must be scored at
   level-0 boundaries or not at all.

**In the pathwise experiment.** GraphXForm's shipped substructure mechanism is a
**terminal −∞ filter**, so it instantiates the *endpoint-only filtering* arm, not
the pathwise arm. Its per-step masking is real but covers only valence, atom
type, atom count and bonding legality. Reporting it as a native pathwise-motif
baseline would be wrong — see the correction in `COMPARATOR_MATRIX.md`.

---

## 3. GraphGA

**Matched quantity.** Property-oracle calls under the single declared convention
(§0). Edits are `N/A`: a GA offspring is not one edit from a designated source.

**Where it is fair.** Zero training cost, zero checkpoint, zero warm-up. Under a
fixed oracle budget on a target-free scalar objective, this is the cleanest
comparison in the registry — nothing is hidden on either side. PMO ranks it
second of 25 methods, so it is a serious baseline.

**Where it is not fair.**

1. **The source is not preserved.** The paper states it outright: molecules found
   "bear little resemblance to the molecules used to construct the initial mating
   pool", with nearest-ZINC Tanimoto of 0.27 and 0.12. On a similarity-constrained
   or source-conditioned panel, GraphGA either fails the constraint or has the
   constraint folded into its scoring function — which changes its task. Whichever
   is chosen must be declared, and folding a constraint into the score is a
   deviation.
2. **No trajectory.** Lineage is a two-parent DAG that the code does not persist.
   Every pathwise and same-prefix metric is `N/A`. Synthesising a "trajectory"
   from a GA genealogy is a forbidden adaptation.
3. **Variance.** PMO's own note: "Graph GA suffers from a relatively large
   variance due to its random-walk-like exploration". Seed count matters more
   here than for the learned methods.

**Expectation management.** PMO's headline conclusion is that simple established
methods frequently outperform newer ones under controlled oracle budgets.
COMPOSE should expect to be *comparable* to GraphGA on C4a, not to beat it, and
the paper's argument does not depend on beating it.

---

## 4. REINVENT

**Matched quantity.** Property-oracle calls under the declared convention, plus
the step count. Edit budget `N/A`.

**Where it is fair.** It is PMO's top-ranked method (14.196 summed AUC top-10
across 23 tasks), it runs CPU-only, and via PMO's `BaseOptimizer` it shares an
oracle counter with GraphGA exactly. For an oracle-efficiency comparison this is
the strongest available reference point.

**Where it is not fair.**

1. **The entire budget is objective-specific RL fine-tuning.** COMPOSE pays none
   of that because `R_theta` is frozen. This asymmetry is the claim under test.
2. **Its raw oracle count is systematically lower for the same work**, because it
   caches per scoring component and its diversity filter zero-scores repeats.
   Putting REINVENT's and MARS's raw counts side by side without stating the
   convention would mislead.
3. **No intermediate molecular states.** Autoregressive SMILES decoding means
   intermediates are token prefixes. All pathwise and intermediate-state metrics
   are `N/A`. Same-prefix metrics are `N/A` at the molecular level but
   **measurable at the policy level**, and that distinction must be printed
   rather than elided.
4. **Hyperparameter transfer.** PMO warns that performance is highly dependent
   on σ and to re-tune whenever the environment changes. Using PMO's σ = 500 on
   a COMPOSE objective is a recorded deviation; re-tuning it is a *different*
   recorded deviation. Pick one and say which.
5. **Two systems.** PMO's `main/reinvent` is a REINVENT 2.0-era reimplementation.
   REINVENT 4's staged learning, Mol2Mol, LibInvent and LinkInvent are **not** in
   the PMO-benchmarked configuration. A row may cite PMO numbers *or* claim a
   REINVENT 4 capability, never both.

**In the retargeting experiment.** REINVENT 4 is the one external arm that
natively changes objective mid-run, and it must be run for the C4c claim to be
stated honestly. Its contract there is: report the post-switch **re-adaptation
cost** (further RL steps × batch size) alongside COMPOSE's post-switch oracle
cost, and state explicitly that REINVENT re-targets a policy while COMPOSE
continues a realized molecular history.

---

## 5. Rules that bind all four

- Identical source molecules, identical objective implementation
  (`src/compose_v4/drd2_oracle.py` and the frozen goal-language normalizers),
  identical feasibility predicates, identical final evaluator.
- Caching allowed for everyone, identical semantics (same canonical SMILES ⇒
  same score), hit rate reported separately.
- Invalid or unscorable molecules are failed oracle proposals, not free actions.
- Wall time and accelerator class reported, never matched.
- A method with no notion of an intermediate molecule gets `N/A`, printed.
- No baseline is excluded because its preliminary results are strong.
- The COMPOSE-side environment stays on the production pins (python 3.11, torch
  2.4.0, numpy 1.26.4, scipy 1.13.1, networkx 3.3, rdkit 2024.3.5). Baselines run
  in their own containers and molecules cross the boundary as SMILES only —
  except GraphGA, which can share the COMPOSE rdkit pin exactly and should.
