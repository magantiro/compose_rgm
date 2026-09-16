# Objective-driven Dynamic reset, revision 1

## Question and scope

Can minimal support repairs plus actual scored parent allocation improve
autonomous T4 search? This is a bounded development experiment, not an IVG win,
not a held-out benchmark and not a newly trained route-distilled policy.

The longer-term controller is route-informed Dynamic program evolution:
complete coordinated proposal priors, shared exact execution, an objective-aware
diverse archive and unchanged v0 mutation/recombination/refinement. The current
slice tests search/support defects before fitting another model.

Inputs are benchmark start graphs, generic operators/configuration and charged
observations from this run. There are zero initial task-specific stored routes.
Historical winners are diagnostic/training evidence, never runtime seeds.

## Empirical basis

Read `diagnostics/t4_strategy_reset/20260916/REPORT.md` and its verification
receipt. The audit recovered 35,895 scored observations, of which 14,085 have
an exact call index and 35,328 have parent provenance. Missing fields are not
invented. Counts are observations, not independent molecules or clean labels.

Key measured defects: historical Dynamic parent selection was score-blind;
empty bootstrap rounds repeated deterministic proposals; fixed v1 ring panels
excluded one known useful construction in its required context. V2.1 also
removed v0's archive-composition channel. These do not establish that any one
repair is sufficient. Historical winners sometimes descend from worse parents,
so greedy selection is not the recommendation.

## Modules and changes

| Module | Responsibility |
| --- | --- |
| `control/progressive_bootstrap.py` | Persistent proposal streams, dedup and resumable cold start |
| `control/progressive_structured_sampler.py` | Fresh bounded ring panels and generic protected v1 composition |
| `control/objective_program_search.py` | Unchanged v0 engine, separate structured stream, one archive, eligible-pool arbitration |
| `experiments/t4_objective_reset.py` | Candidate locks, charged-call ledger, interruption-safe unit runner |
| `experiments/t4_docking_adapter.py` | Historical OpenBabel/QuickVina command protocol without teacher artifacts |
| `modal_apps/t4_objective_reset_app.py` | Isolated CPU workers and source-only image |
| `tools/t4_objective_reset.py` | Source-only capsule and local zero-oracle probe |

No existing v0/v1/v2.1 controller, compiler, frozen contract or running experiment
is modified. The v0 branch keeps 25% program composition, its original
mutation/recombination implementation and 1–3-module synthesis. Its proposal RNG
is isolated from the structured RNG. The portfolio does not guarantee identical
v0 scored trajectories because candidate competition and measured parents differ.

Structured compilation remains bounded at 1–3 high-level modules, 32 primitives,
eight blocks and the existing exact executor. Intermediate executability is
required; task eligibility is checked only at completed endpoints. This does not
claim all 77 teachers are autonomously reachable through the structured sampler.
The separate corrected complete-region realizer remains unchanged and available
for the later learned prior. It is not silently represented as implemented here.

## Frozen comparison

`configs/t4_objective_reset_runtime_v1.json` binds the starting molecules and
historical receptor/binary hashes. Both arms use the same support repairs,
candidate budgets, seeds, scored endpoint rules and incumbent-gain channel
allocation. The only experimental difference is parent selection:

- `support_control`: score-blind historical parent allocation.
- `objective_search`: existing generic niche-score parent allocation, retaining
  exploration rather than selecting only the champion.

Three cells: BRAF seed 1, JAK2 seed 1, FA7 seed 0, delta=0.4. At most 20 charged
calls per arm/cell, six scored workers, total ceiling **120**. No automatic retry,
backfill, extra seed or hidden third arm. This is narrower than the strategy
report's proposed three-arm 180-call ladder because a learned prior is not fitted.
The two-arm pilot isolates scored search allocation on common repaired support.

Shared proposal budget: 64 v0 attempts / eight candidates / 20-second batch cap;
16 structured attempts / four candidates / 20-second cap; at most eight dockings
per selected pool. The bootstrap uses persistent shallow/structured streams and
a 32-round ceiling. Wall caps are disclosed stopping budgets, not scientific
eligibility relaxation. Actual attempts and elapsed time are recorded.

Both pools are exact-executed, endpoint-filtered and deduplicated before oracle
allocation. An eligible-channel exploration floor remains; UCB credit is bounded
incumbent improvement per charged call, not improvement over a weak parent.
Failed dockings consume budget. Unscored candidates have no invented values.

## Gates and decisions

1. **Support/engineering:** focused tests for unchanged v0 candidate law,
   independent RNGs, startup advancement, exact resume, real receipt validation,
   zero-call empty pools, failure accounting and historical docking-command parity.
   Failures block launch; repair only the relevant defect.
2. **Zero-oracle chemistry probe:** five cells, up to four bounded startup rounds,
   each on one pinned CPU container. This is a yield measurement, not a requirement
   that every expert solve every seed within an arbitrary attempt count.
3. **20-call pilot:** only the three declared cells/two arms, after focused checks
   and the one repository-wide verification at the launch boundary. Preserve a
   failed gate, rather than widening it until it passes.
4. **Promotion to 50:** separate prospective lock only if objective allocation
   improves the common control in at least two yielding cells without a >0.5
   docking-unit loss on BRAF, and JAK2 approaches historical v0 call-20 performance
   (−9.9). These small noisy samples justify development, not superiority claims.
   A zero-yield FA7 is reported separately, never as a scored success.
5. **Kill/revise:** unchanged zero-yield, persistent >1-unit deficits to historical
   matched-call Dynamic, or systematic worse allocation means no budget extension.
   Determine whether missing eligible support, parent allocation or finite-pool
   utility caused it. Do not reopen the compiler or inflate primitive beam search.
6. **Five-cell then full qualification:** only a subsequent versioned contract.
   Compare at 1/5/10/20/50/100/etc charged calls where available, never against a
   convenient final-only reference. IVG beating requires matched budget/protocol,
   declared training exposure, replicated evidence and all failures included.

## Operational safeguards

New isolated Modal volume: `compose-t4-objective-reset-20260916`.
Use `MODAL_PROFILE=nitya`. Workers persist started records, proposal locks,
per-query receipts, checkpoints and progress. Ambiguous started queries stop and
are not redocked automatically. A final result is idempotently recoverable.
The image contains code and a source-only capsule, not historic output folders.
Every docking input is hash checked against the old benchmark.

Runtime is Python 3.11, RDKit 2024.03.5, NumPy 1.26.4, Torch 2.4.0 CPU;
OpenBabel/QuickVina command flags match the historical adapter, including seed
1701, one CPU and exhaustiveness one. Local chemistry diagnostics use a newer
RDKit and are explicitly not substituted for pinned-image results.

## Revision 1a: split-clean proposal-prior corpus (zero oracle, no fit)

The learned prior stays separate until its split-clean data and its actual-sampler
support are verified. This revision closes the first of those two prerequisites and
fits nothing.

Contract `configs/t4_proposal_prior_dataset_v1.json` (payload
`d8e5feb4a15bf0aa12eaed883622a256b4f464f618b9dfc6758c27cc5886ebd7`) was frozen before
any extraction. It declares the atomic group (`cell`), the identities that must not
cross a group, leave-one-target-out folds, admission rules with reason codes, label
semantics, the target/cell/lineage/record weight hierarchy, and a coverage gate whose
required construction families come from sections 4 and 5 of the strategy report
rather than from a per-fold count. It also discloses that the pooled, fold-blind
block-label census had already been inspected while the contract was written.

| Module | Responsibility |
| --- | --- |
| `experiments/t4_proposal_prior_dataset.py` | Disjointness proof, admission, observation collapse, lineage closure, folds, weights, census, gate |
| `tools/t4_proposal_prior_dataset.py` | Contract-input verification and artifact publication |
| `configs/t4_proposal_prior_checkpoint_manifest_v1.json` | Content identity of the 71 historical checkpoints that carry program payloads |

```sh
PYTHONPATH=src:. .venv/bin/python tools/t4_proposal_prior_dataset.py
.venv/bin/pytest -q tests/test_t4_proposal_prior_dataset.py
```

Measured outcome (`diagnostics/t4_proposal_prior/dataset_v1/REPORT.md`, decision
**GO**): 35,895 recovered rows admit without exclusion and collapse into 34,073
labelled constructions. No endpoint, source state, entry id or protocol crosses a
cell, so no lineage component crosses a fold. Eighteen generic construction families
are dense in every fold, including all six required ones; `ring_path_remodel`,
`core_carbonyl_insertion` and `ring_carbonyl` are sparse, and `pendant_benzene` and
`remodel_linker` are confined to one target and absent from some fold's training
side. A prior over the sparse or absent families is not supported by this corpus.

Two limits matter more than the record count. First, the corpus holds only 133
lineage components, so a held-out fold contains 16 to 44 independent genealogies
behind its thousands of records, and its weighted effective sample size is 65 to 676.
Second, the frozen equal-mass-per-lineage rule over-corrects on this shape: the
heaviest single record carries 454 times uniform mass and the heaviest hundred carry
23% of the corpus. Both numbers were computed after the hierarchy was frozen and
neither changed it; a capped or size-damped variant is a decision for the separate
fitting contract. Program payloads are bound by checkpoint hash and entry id, not
copied, so the fitting step resolves them inside its own training fold.

## Revision 1b: actual-sampler support probe (zero oracle, no fit)

The second prerequisite the reset contract sets before any learned prior. Contract
`configs/t4_proposal_access_probe_v1.json`, artifact
`diagnostics/t4_proposal_prior/access_probe_v1/`. Two lanes over the five recovered
champion contexts: a teacher-free **bootstrap** lane from the frozen benchmark root,
which is the autonomous measurement, and an answer-known **archive** lane that
injects one historical champion for diagnostics only. Decision **PASS** on all three
frozen criteria.

```sh
PYTHONPATH=src:. .venv/bin/python tools/t4_proposal_access_probe.py
.venv/bin/pytest -q tests/test_t4_proposal_access_probe.py
```

**The JAK2 support loss is closed.** Strategy report section 5 measured that the
substituted-ring construction the productive JAK2 transformation needed was absent
from all four deterministically seeded v1 panels, so a compiler that reached the
endpoint when the target was supplied could not reach it through the runtime
sampler. Through the repaired progressive sampler, pooled over 1,280 teacher-free
bootstrap attempts and 1,600 archive attempts, every one of the fifteen declared
generic families is realized, and `construct_substituted_ring` is realized 359 times
against 374 attempts. `ring_path_remodel`, the other v1-only construction, is 263 of
263.

**The repetition defect is repaired at scale.** 198 to 255 distinct proposals per 256
attempts per cell, and every round after the first draws proposals never drawn
before. Exploration is judged on proposal identity, not endpoint identity, because a
small legal endpoint space would otherwise read as the defect.

**All five supplied champions still realize exactly** through unchanged archive
admission, which replays the program and asserts endpoint identity.

### The measured cause of low and zero yield is the endpoint gate, not the sampler

| Cell | Root QED | Root passes its own gate | Bootstrap eligible / 256 |
| --- | ---: | --- | ---: |
| parp1_0 | 0.888 | yes | 84 |
| jak2_1 | 0.712 | yes | 55 |
| 5ht1b_0 | 0.438 | no | 16 |
| braf_1 | 0.346 | no | 5 |
| fa7_0 | 0.284 | no | 0 |

FA7-0 produced 255 distinct proposals and 240 unique endpoints from 256 attempts,
realized all fifteen families, and had **zero** executor rejections: the chemistry
runs, and then 239 of 256 completed endpoints fail the endpoint filter. On the pinned
production image, all 34 recorded FA7 bootstrap attempts fail for one reason, QED not
strictly above 0.6, while similarity sits at 0.51 to 0.72, comfortably inside the
delta-0.4 gate. The cell's own root has QED 0.284.

So FA7's zero yield is not a proposal-support or repetition problem, and widening the
sampler will not address it: a proposal must clear QED 0.6 at the endpoint while
staying similar to a molecule at QED 0.284, and those two demands pull against each
other. This supersedes the reading that repairing the deterministic retry defect would
recover FA7 yield; that defect was real and is repaired, and the yield did not follow.
Eligible yield is ordered by root QED across all five cells. That is a measured
association over five cells, not a proof that no supported program clears the gate;
256 draws is not an impossibility result and the contract says so.

**Consequence for the 120-call pilot.** FA7-0 is one of the three declared cells. On
this evidence it will most likely consume its charged allowance producing no eligible
candidate, exactly as v2.1 FA7 did historically. That is a budget decision to take
with this number in hand, not a reason to relax the endpoint gate.

## Next learned-prior work

Do not train a large whole-patch model first. Use the recovered successful and
unsuccessful scored programs to fit a small **joint** context-conditioned proposal
prior over generic complete-program families, role-based WHERE/bindings and
parameters. Split source/lineage groups before reusable extraction. Preserve the
v0 lane and treat utility labels as noisy measured outcomes, not structural
similarity. Teacher-forced support checks and injected-known-answer rank probes
must precede autonomous proposals. Runtime must not retrieve teacher routes or
endpoints. This remains future work, not an accomplished part of revision 1.
