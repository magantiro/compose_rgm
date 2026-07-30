# RingCore-V1 canonical-successor leaderboard readiness

## Outcome

The 32-snapshot leaderboard is now locally specified and regression-tested, but it is intentionally
**not authorized to execute yet**. No checkpoint has been scored, ranked, or selected.

The preparation path now guarantees:

- exactly the 32 frozen snapshots from steps 500 through 16,000;
- SHA-256, byte-count, filename, payload-step, history-step, and provenance checks before scoring;
- strict installation of each recovery snapshot's `current_state_dict`;
- validation-only scoring;
- the production canonical molecular pushforward rather than an independent alias enumerator;
- production-law importance weighting for the primary NLL;
- equal weighting of frozen joint semantic cells for the secondary NLL;
- family, candidate-count, alias, productive-mass, virtual-mass, family-choice, rank, MRR, and uniform-law
  diagnostics;
- hard failure if a nonterminal teacher successor is absent, duplicated after grouping, non-finite, or
  silently skipped.

## Corrections made before any leaderboard result

The previous leaderboard JSON named balanced-family NLL as the secondary criterion. That was inconsistent
with the V2 editing gate and the first-principles corpus audit: balancing operator names does not balance
capability regime, evidence origin, path scale, size/topology delta, chemistry stratum, or split unit. The
secondary metric is now:

```text
balanced_semantic_cell_canonical_successor_nll
```

Family-balanced NLL remains a diagnostic only.

The checked-in inventory had also been a compact projection carrying the remote inventory hash. A read-only
artifact-volume retrieval recovered the exact immutable freezer output. Its recomputed self-hash is:

```text
df815175f8b75a319cc87c01b923db431e53834e28d8b6c5f232ac34e66ddbc0
```

The exact JSON is now checked in, so provenance can be verified locally rather than trusted by assertion.

## Why the legacy selection script is not evidence

`scripts/ring_core_checkpoint_selection.py` predates the authoritative production evaluator. It:

- builds a small synthetic corruption/cycle set instead of drawing the frozen packed validation law;
- fixes model time at 0.5 instead of reproducing the frozen time law;
- omits the MMP validation layer and progress importance weighting;
- reconstructs aliases independently and catches broad exceptions;
- permits skipped examples;
- lacks the fixed family-forensics panel and structural gates;
- historically reported an automatic best checkpoint before capability and statistical gates.

Its old JSON outputs must not be used for checkpoint selection. The current legacy script has also been
made non-selecting: it emits `selection_performed=false`, emits no rank or best-checkpoint field, and
requires the explicit `:current` state source. The new preparation module does not import or call it.

## Exact remaining blockers

1. Implement and freeze the six semantic-axis labelers over packed validation records, then freeze the
   resulting nonempty cell census. The versioned joint-cell identity function is implemented; the
   chemistry/data-derived labels are not yet inferred from the corpus.
2. Build the deterministic 4,096-row production panel and the minimum-256-per-family forensic panel from
   the actual packed validation layers. Include terminal rows only for hazard reporting and exclude them
   from successor NLL.
3. Implement the all-32 runner around the shared checkpoint evaluator and production pushforward. It must
   score every snapshot on the same final panel size and write per-row records.
4. Implement paired bootstrap comparisons, the 4,096-to-16,384 expansion rule, and the per-family
   256-to-1,024 ambiguity expansion without emitting a winner.
5. On the artifact volume, resolve each snapshot and recheck its frozen SHA, payload, current-state source,
   and RingCore capability signature before the first model forward pass.

The expensive part should not begin until blockers 1–4 are complete and tested locally. Snapshot downloads
are unnecessary: the future runner should execute where the immutable artifacts already reside.

## Verification

The targeted leaderboard, production-kernel, and checkpoint-evaluator test suite passes locally. The
readiness artifact is:

```text
diagnostics/coherence/ringcore_v1_successor_leaderboard_readiness_2026-07-30.json
```
