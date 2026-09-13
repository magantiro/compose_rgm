# Frozen full T4 delta=0.4 program-controller benchmark

## Identity and claim

- **Scientific problem:** optimize each prescribed T4 lead under the official
  endpoint constraints and a fixed docking-call budget.
- **Primary output:** for every target, source cell and search replicate, the
  complete best-eligible-score curve and the final molecule with its exact
  executable construction program.
- **Claim under test:** one sealed, shared COMPOSE program controller is
  competitive with the reported InVirtuoGen results across the complete T4
  delta=0.4 suite.
- **Setting:** all 15 starting cells (five targets, three sources each), three
  predeclared search replicates and at most 1,000 distinct candidate dockings per
  cell/replicate. Results are aggregated over the three replicates.
- **Reported baselines:** the official IVG, GenMol, RetMol and GraphGA table.
  This run is not a causal ablation of controller components.
- **Support:** the existing 48-slot representation with at most 40 active heavy
  atoms, broad supported organic atom/bond vocabulary, connected legal molecular
  intermediates and the exact COMPOSE executor. Stereochemistry and formal-charge
  editing remain outside the declared editing support.

The shared controller library was developed using public T4 winner routes. It is
sealed before this run and is identical for every cell. No recipient-specific
route lookup, score, endpoint template, controller version or parameter change is
available after launch. This establishes prospective performance of the sealed
development-informed controller, not a held-out benchmark-development claim.

The first attempted lock reconstructed source tensors from SMILES and failed its
zero-oracle BRAF seed-1 yield gate. It was not launched. The v2 lock reuses the
exact saved 48-slot benchmark source states, including the same state and
controller seed used by the earlier successful shared-controller preparation.
The failed v1 receipt remains under
`diagnostics/t4_frozen_program_benchmark/failed_preflight_v1.json`.

## Verified external protocol

The official IVG repository revision
`b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb` provides a lead-optimization command
with `--max_oracle_calls 1000`. Its table builder groups best eligible scores by
three search seeds and applies strict filters: similarity greater than the
threshold, QED greater than 0.6 and SA less than 4. One inspected official
delta=0.4 CSV has 998, 987 and 996 recorded candidates for its three seeds and
contains no seed-molecule row, consistent with 1,000 being a per-run candidate
call ceiling rather than a requirement to charge the supplied seed.

The pinned upstream `in_virtuo_reinforce/ppo_docking.py` blob is incorrectly
packaged as an unrelated PMO table script. Therefore the README command and
released result files establish the reported ceiling and aggregation, but do not
independently expose complete call-by-call implementation accounting. This
limitation must accompany any matched-protocol claim.

## Frozen controller

The same recipe is used for every target, cell and replicate:

| Component | Frozen choice |
| --- | --- |
| Shared program library | existing 146-program all-task development library |
| Proposal mode | fast program-only |
| Mutation / branch recombination | 7/9 / 2/9 |
| Parent allocation | score-blind with 0.2 exploration floor |
| Continuation root | exact current state, full ancestry retained |
| Existing-state edit allocation | 0.25 |
| Verified decomposition | enabled |
| Mutation sampling | random |
| Protected composition | disabled after failed structural gate |
| Reference-law inference | disabled |
| Learned selector | disabled |
| Proposal cache | 128 entries |
| Per-batch work | 128 attempts, 16 candidates, 45 seconds |
| Per-proposal work | at most 32 primitives and eight blocks |

The cold-start batch uses the same shared program library and program-only
mutation/recombination mixture. Subsequent scored batches update only the archive
of measured constructions. The score-blind parent law remains fixed. A unit stops
at 1,000 distinct candidate evaluations, after three consecutive empty proposal
batches, or at a predeclared competitive plateau. The plateau rule applies only
after at least 500 calls, only after the run reaches or improves on its cell's
reported IVG mean, and only when the best score improves by less than 0.3 docking
units over the previous 250 calls. A weak run therefore retains its full 1,000-call
opportunity. Candidate shortages are reported, not filled by unbounded retries.

Endpoint admission uses the official strict T4 filters with the original source
as the ECFP4 radius-two, 2048-bit similarity reference. The executor supplies
valid connected molecules, so no additional medicinal-chemistry heuristic is
added to the official endpoint gate.

The cold-start binding seed is `20260913`, matching the earlier successful shared
controller preparation, and is identical across replicates. Search continuation
seeds are `20260913`, `20260914` and `20260915`; corresponding docking seeds are
`1701`, `1702` and `1703`. Thus every replicate starts from the same deterministic
structural population but has an independent continuation and docking stream.
Every score is retained in query order. Duplicate endpoints within a run are not
redocked. Runs do not share docking labels with one another.

## Compute and query contract

- 15 cells x three replicates x at most 1,000 distinct candidate calls gives a
  maximum of 45,000 search calls.
- After the search, select the lowest first-evaluation score across the three
  replicates for each cell. Obtain two fresh confirmation dockings for each of
  the 15 locked champions, at most 30 additional calls.
- The benchmark table uses first-evaluation search scores. Confirmation scores
  are a separate reliability summary and are never substituted by the most
  favorable repeat.
- At most one driver and 29 single-CPU workers may exist concurrently. GPU use,
  automatic oracle retry and ambiguous-call recovery are disabled. Reserved
  compute may not exceed $20.
- Candidate locks precede docking. A started call without a result remains
  charged and blocks automatic unit resume. Failures are retained.

Each result records input hashes, code revision, frozen configuration, RDKit and
OpenBabel versions, target receptor and docking-binary hashes, seeds, hardware,
query counts, proposal attempts/time, candidate and trace identities, eligibility
properties, every docking receipt, termination reason and best score versus
query count. Every unit publishes a durable heartbeat at least every 30 seconds
during active work and an end-of-round progress record. The aggregate reports the
first call that reaches the corresponding reported IVG mean for each replicate.

The score-versus-query curves and calls-to-IVG thresholds test a sample-efficiency
claim separately from final endpoint quality. Early termination does not establish
efficiency by itself; it only prevents spending the remaining ceiling on a run that
is already competitive and empirically flat under the frozen rule.

## Preflight and acceptance

Before paid execution:

1. Generate a first zero-oracle batch for every one of the 15 cells under the
   frozen controller and verify exact replay, official endpoint admission,
   deterministic serialization and nonempty yield.
2. Exercise a local synthetic multi-round resume test and a remote image/input
   preflight without invoking docking.
3. Run focused tests and formatting/lint for touched code, then the repository
   launch-boundary verification and full test suite.
4. Inspect the frozen lock and final diff, commit coherent changes and launch
   only from that clean pushed revision.

The benchmark is complete only when all nonfailed units terminate under their
declared budget, the confirmation lock is fixed before repeats, the aggregate
table and curves are self-hashed, and failures/shortfalls are included. No
intermediate cell result may change the sealed controller.
