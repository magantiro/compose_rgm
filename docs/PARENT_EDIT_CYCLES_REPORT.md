# Parent/edit learning cycles: results and implementation

Status at 2026-09-13 00:30 UTC: T4 completed; PMO interrupted, not completed.
The deployed application has zero active tasks. No recovery campaign has launched.
This is a development result, not a matched IVG/GenMol benchmark win.

## Executive decision

Keep the coordinated-program engine and verified branch recombination. The
learned selector produced a repeat-supported JAK2 candidate, but did not improve
BRAF's incumbent. Do not promote this selector as generally superior yet.
Proposal throughput and candidate yield limited the comparison. PMO adds an
operational failure, not evidence that a completed learning campaign succeeded
or definitively failed. Repair restart isolation and throughput before scaling.

## T4: completed observations

Both arms used BRAF seed1 and JAK2 seed1 at similarity 0.4, two search seeds,
three rounds with up to four new queries per round. Identical historical warm
archives, broad proposal recipe and endpoint gates were available to both arms.
The learned selector, not parent-score ranking, was the intended intervention.

| Cell | Arm | New first calls | Best new first score | Fresh champion repeats |
| --- | --- | ---: | ---: | --- |
| BRAF seed1 | Learned | 21 | -10.6 | -10.5, -10.5 |
| BRAF seed1 | Score-blind | 16 | -10.4 | -10.2, -10.3 |
| JAK2 seed1 | Learned | 11 | -10.9 | -10.5, -10.7 |
| JAK2 seed1 | Score-blind | 12 | -10.4 | -9.4, -9.8 |

The predeclared BRAF incumbent repeated at -11.0/-10.9; the predeclared JAK2
incumbent repeated at -9.5/-10.0. These controls are the declared first-generation
incumbents, not retroactively selected historical winners. The prior JAK2 best
single score was already -10.9, so this run did not improve that best-ever number.

Total: 60 first calls plus 12 confirmations, 72 of the 108-call ceiling; no oracle
failures. The 36 unused calls were not filled by extending candidate generation.
Arm champions were selected across search randomizations before confirmation.
Repeat-only means exclude the favorable selection score. Two repeats and two
search seeds do not establish statistical superiority.

Both learned JAK2 randomizations generated the same best molecule via verified
branch recombination, scoring -10.9 and -10.3 in their respective first evaluations:

`NC(=O)C1CCN(C(=O)CC2Nc3ccccc3-c3cccc4[nH]cc2c34)C1`

For search seed 20260921, the informative second-round JAK2 proposal pools were
identical sets of nine endpoints: learned selection chose the eventual strong
candidate, while score-blind selection missed it. This is the clearest local
selector signal. The other randomization had different wall-limited pools, so
it does not isolate selection in the same way.

## Throughput and remaining search limitations

Across the eight T4 units, first oracle evaluation averaged 4.424 seconds.
Summed proposal time was 1,445.947 seconds, including 1,070.402 seconds of
reference-law computation. Summed search-worker time was 2,242.044 seconds.
These are summed parallel-work measurements, not elapsed campaign time, and
law time is a component of proposal work, not an extra quantity to add to it.

Only three of twelve learned-arm rounds had more candidates than their four-query
allowance. Three rounds across both arms had empty pools. Thus most rounds gave
the selector little or no opportunity to improve allocation. The nominal
45-second proposal limit is checked between attempts; an in-flight complete
broad proposal can overrun it. An empty round can reflect an expensive failed
attempt, not exhaustion of the full molecular neighborhood.

The recorded 533 attempts break down as follows. These are attempts, not
independent or unique molecules; statuses classify the actual sampled proposals.

| Channel | Eligible new candidates | Duplicate | Execution rejected | Endpoint ineligible |
| --- | ---: | ---: | ---: | ---: |
| Program mutation | 65 | 65 | 97 | 139 |
| Branch recombination | 19 | 49 | 40 | 13 |
| Broad reference continuation | 3 | 1 | 4 | 38 |

Do not remove broad support in response. The evidence motivates bounded,
checkpointable broad execution and independent scheduling of productive program
work, with exact caching and explicit resource accounting. It does not yet
establish the optimal allocation.

## PMO: partial observations and failures

The intended comparison was four tasks, three search seeds and two arms, with
1,000 queries per unit including sixteen task-independent initial molecules.
All 100 initialization-bank molecules passed the declared support check before
the sixteen-per-seed locks were selected. No historical PMO task scores were
imported. The shared structural program library still has its declared public
T4-winner development origin.

The receipt census contains 647 charged reservations, 646 completed evaluations,
and one unresolved reservation. It remains charged and has not been retried:
`pmo_perindopril_mpo_20260922_score_blind/oracle/query_000041`.

| Task | Search seed suffix | Completed learned / blind | Best partial score in both arms |
| --- | --- | ---: | ---: |
| Albuterol similarity | 921 | 83 / 59 | 0.362140 |
| Albuterol similarity | 922 | 89 / 80 | 0.403509 |
| Albuterol similarity | 923 | 46 / 46 | 0.393162 |
| Perindopril MPO | 921 | 48 / 48 | 0.465475 |
| Perindopril MPO | 922 | 48 / 41 | 0.352261 |
| Perindopril MPO | 923 | 29 / 29 | 0.411693 |

Full seeds are 20260921, 20260922 and 20260923. These partial endpoint maxima
are not PMO top-ten AUC, full-budget results, or a matched comparison to IVG.
They do not improve the separately reported historical Perindopril endpoint
0.694808, which used a different information/initialization regime.

Two distinct failures occurred:

1. The reused older PMO factory supported only its older probe's task subset.
   All six isomer units and all six scaffold-hop units failed before any oracle
   call. This was an integration error. Local dispatch now reads the frozen
   campaign's allowed task names, with focused tests, but is not redeployed.
2. A worker invocation encountered an existing unit start receipt. The explicit
   no-retry guard rejected it; the fail-fast map propagated the exception and
   cancelled the other active workers around 00:20 UTC. The precise reason the
   invocation re-entered has not been established. Do not label it as an OOM,
   platform preemption or user cancellation without additional evidence.

The completed PMO oracle callbacks together took approximately 0.752 seconds.
The observed delay was therefore outside these property calculations, in
proposal work and runtime/persistence overhead. That total does not separate
every overhead component. No completed PMO unit provides the intended full
multi-round comparison. Preserve all partial data and interrupted state.

## What was actually implemented and exercised

### Exact substrate and complete programs

The frozen reference model and executor were unchanged. T4 remains within the
40-heavy-atom support; the persistent-slot representation is not a 40-slot array.
Programs use typed references to input and created atoms. Dependency scheduling,
capacity checks and serial exact execution validate every intermediate. Mutable
sites can be disconnected; molecular states must remain connected and valid.
Endpoint-only task checks do not prune unfinished supported transformations.

The common library contained 146 verified program representations. Programs can
change attachments, created-atom identities and segment lengths, replace verified
branches, and recombine compatible constructions. The library itself was not
automatically relearned or expanded into new abstractions during this campaign.
Measured program/endpoints enter the search archive for later proposals.

### Frozen shared search recipe

- Top-level mutation/recombination/broad probabilities: 70% / 20% / 10%.
- Score-blind parent allocation in both arms, with duplicate/exploration handling.
- Fresh exact-current-state continuation for new edits; ancestry is retained.
- Verified program decomposition enabled. Inseparable branches are not forced apart.
- Existing-input-atom editing probability 0.25 within the mutation route.
- Double program mutation probability 0.25.
- Maximum 32 new primitives, eight blocks and 64 attachment bindings.
- Maximum 128 attempts and nominal 45 seconds per proposed batch.
- Candidate-pool targets: twelve for T4, 48 for PMO; query batches: four and sixteen.

Fresh-state continuation is a new per-proposal allowance, not erasure of prior
construction/oracle cost. Program-parameter mutations can still rebuild from
their constructor source. The selected-parent delta is then a graph comparison,
not an invented executed parent-to-child route.

### Learned component and its limitations

The v2 model is a five-member, endpoint-bootstrapped ridge ensemble with ridge
penalty 1.0 and fixed seed 20260913. Features include normalized 512-bit Morgan
fingerprints of selected parent, candidate and constructor; fingerprint changes;
128-bin hashed attachment/operation features; construction source/peak/final size;
selected-parent size delta; mutation events, branch changes, observed-parent-score
masks and endpoint similarity context where applicable.

Training uses genuine accessible completed outcomes within one task/oracle
domain. Distinct repeats are averaged by endpoint, with equal endpoint mass
and balanced representations. The model predicts completed oriented utility
(negative docking score for T4, reward for PMO). It is not a future-value model.
Uncalibrated bootstrap disagreement is not used as optimistic acquisition.

Learned selection reserves 25% of query slots, rounded up to at least one, for
random audit selection. Remaining slots use plug-in marginal improvement to
the best-one T4 or best-ten PMO archive, with predicted-score tie-breaking.
It is not posterior-integrated expected improvement. Both arms use the same
proposal architecture; the learned arm changes query allocation, not a trained
atom-pointer/program proposal law. No neural decoder, new Doob twist, twisted
SMC, search-to-proposal distillation or learned future head was deployed here.

PMO bootstraps program construction from the first four locked initial structures
in its first four rounds. All sixteen initial scores count, but not all sixteen
are immediately construction parents. This is an implementation limitation to
include in the next initialization review, not a claim that the best initial
structure necessarily received refinement.

### Task and runtime contracts

T4 uses original-seed Morgan similarity at least 0.4, QED at least 0.6 and SA at
most 4, plus the existing endpoint gate. PMO does not inherit these T4 filters.
The PMO metric helper implements logged top-ten trapezoids and optional flat tail;
no completed or extrapolated PMO AUC is promoted for the interrupted comparison.
The frozen image used RDKit 2024.03.5 and PyTDC 0.3.6 with the oracle source hash
checked. Docking binary/receptor hashes and the existing preparation were checked.
First docking seed: 1701; confirmation seeds: 1702 and 1703.

Every query is reserved durably before evaluation and has an identified receipt.
Failures remain charged; unresolved attempts fail closed. Completed rounds retain
exact archive, RNG and proposal state. Pending-round recovery is manual, not an
automatic restart. A zero-query dispatch-recovery helper exists locally, but
does not repair the interrupted driver/partial-round case and has not launched.

## Files, verification and costs

| Responsibility | Source |
| --- | --- |
| Typed programs and replay | `src/compose_v4/control/edit_program.py` |
| Dependencies and size profiles | `src/compose_v4/control/edit_program_graph.py` |
| Parameter/attachment/branch mutations | `src/compose_v4/control/program_mutation.py` |
| Persistent search and channel dispatch | `src/compose_v4/control/adaptive_program_optimizer.py` |
| Features, fitting and selection | `src/compose_v4/control/parent_edit_model.py`, `parent_edit_search.py` |
| T4/PMO semantics and query ledger | `src/compose_v4/control/program_task.py`, `program_campaign.py` |
| Campaign orchestration | `src/compose_v4/experiments/parent_edit_cycles.py` |
| Launch/retrieval/review CLI | `tools/parent_edit_cycles.py` |
| Remote resources | `modal_apps/parent_edit_cycles_app.py` |

The launched source was clean commit `5553ef0ca97c`. Forty focused tests passed
before launch. Ten focused tests passed after local dispatch/recovery/T4-review
repairs; touched Python lint/format checks passed. This is not a full repository
verification or milestone sign-off. The report, retrieved artifacts and review
repairs are preserved on the shared `compose-iclr` branch; unrelated work was
preserved.

The approved reservation remains $20 combined, not measured spend. The deployment
used twelve shared single-CPU workers, two drivers and at most two confirmation
workers, below thirty containers. No GPU was used. Exact billed dollars have not
been reconciled, so there is no claimed measured dollar cost. Query ceilings were
respected. No additional campaign was launched after interruption.

Authoritative local evidence: `diagnostics/parent_edit_cycles/t4_result.json`,
`t4_units/`, `t4_review.json`, `pmo_reconciliation.json`, `pmo_review.json`, and
`pmo_call_graph.json`. The review artifacts include input/source hashes.
Remote traces and pending state remain on the identified Modal volume.

Reproduce the read-only reviews with:

```sh
PYTHONPATH=src:. .venv/bin/python tools/parent_edit_cycles.py review-t4 --output /tmp/compose_t4_review.json
PYTHONPATH=src:. .venv/bin/python tools/parent_edit_cycles.py review-pmo --output /tmp/compose_pmo_review.json
```

## Recommended next decision, not another launched experiment

1. Preserve the JAK2 recombinant and existing BRAF incumbent. Do not scale the
   current selector unchanged on a broad benchmark claim.
2. Isolate worker failures instead of cancelling a whole mapped campaign. Implement
   receipt-aware pending-round recovery with identical locks, RNG and budgets.
   Resolve the one ambiguous query without automatically calling it again.
3. Profile and reduce persistence overhead; make long broad work checkpointable
   and prevent it consuming all proposal opportunities. Preserve broad support.
4. Resume the bounded paired comparison only after those operational repairs are
   qualified. If learning still lacks useful candidate choice, prioritize proposal
   yield/parameterized recombination over another large model or particle increase.

The current evidence supports a useful recombination-plus-selection mechanism on
one cell, a BRAF null, and a PMO runtime/throughput problem. It does not support
claiming that the overall competitive-controller goal has been achieved.
