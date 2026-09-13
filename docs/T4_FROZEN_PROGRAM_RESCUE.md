# Frozen T4 interrupted-unit rescue

## Identity and claim

- **Scientific problem:** finish the already frozen T4 delta=0.4 benchmark
  without losing completed work or double-charging interrupted oracle calls.
- **Primary output:** one reconciled 45-unit result with exact search-call
  chronology, final cell aggregates and separately reported champion repeats.
- **Central claim:** operational recovery leaves the sealed controller and its
  scientific protocol unchanged while completing every recoverable unit from
  its durable checkpoint.
- **Validation setting:** the original run
  `54c3cb6d4a708cecc45c5037d8873a3f051538c6bc787bf84eff161f30be90da`,
  deployed from commit `c272b881bd23ebf1f46bbd1989e6e2871ad20687`.
- **Baselines and ablations:** none. This is recovery of one frozen benchmark,
  not a controller comparison.
- **Support:** unchanged exact COMPOSE executor, shared 146-program library,
  48 persistent slots, at most 40 active heavy atoms, and the original strict
  similarity, QED and synthetic-accessibility gates.

The original top-level result is a stale partial aggregate. It sealed before a
manual JAK2 seed-2 resume completed and before the other interrupted units were
reconciled. It remains immutable evidence and is never overwritten.

## Recovery rule

An interrupted unit is recoverable only when all of the following hold:

1. Its exact unit identity occurs in the original 45-unit contract.
2. It has no final successful unit result.
3. Its query ledger is contiguous except for exactly one final `started.json`
   reservation with no corresponding `result.json`.
4. The ambiguous reservation belongs to the already locked round and candidate.
5. Its checkpoint is not ahead of the completed query ledger.

For that reservation, publish an exact tombstone containing the original query
identity, `score: null`, `status: failed`,
`failure: ambiguous_charged_query_unobserved`, and
`oracle_call_charged: true`. This does not assert that docking failed. It states
that the reserved external call has no recoverable observation. The candidate
must never be redocked as that query, and no task label may be inferred.

After the tombstone is committed, invoke the original deployed worker with the
original run task plus the original unit identifier. The existing checkpoint,
round lock and random state control continuation. The tombstone participates in
the outcome sequence as a failed, charged observation. All later calls retain
their original query indices.

## Frozen scope and ceilings

- Original search ceiling: 45,000 distinct candidate calls.
- Original total ceiling: 45,030 calls.
- Units complete after the late JAK2 result: 23 of 45.
- Units requiring reconciliation: 22.
- Charged calls in those units before rescue: 5,057, including 22 ambiguous
  reservations.
- Maximum new search calls: 16,943.
- Maximum new final-confirmation calls: 30.
- Maximum concurrent rescue workers: 22, each with one CPU and no GPU.
- Automatic retry: zero.

The 24 confirmations selected by the stale aggregate remain charged historical
observations. After search completion, create a distinct final confirmation
lock. Reuse a historical confirmation only if every field in its query identity
matches the final lock. Otherwise obtain a fresh confirmation without deleting
or replacing the earlier result. The realized cumulative total, including all
historical confirmations, must remain below 45,030 calls.

No controller parameter, program, source state, eligibility threshold, docking
setting, seed, oracle protocol or plateau rule may change. No PMO work, model
training, candidate replacement or result-guided repair is part of this rescue.

## Artifacts and procedure

1. Audit the live volume read-only and freeze a rescue lock containing the
   original run-start and stale-result hashes, every completed unit identity,
   every ambiguous query payload and hash, ledger counts, checkpoint and round
   lock hashes, and the exact maximum remaining work.
2. Generate all 22 tombstone envelopes locally. Verify their payload hashes and
   commit the lock and tombstones before any remote write.
3. Recheck the live volume against the lock. Refuse recovery if any ambiguous
   query acquired a result, if any file identity changed, or if a unit no longer
   has the exact locked state.
4. Upload each missing tombstone without overwrite and spawn only the 22 frozen
   unit tasks through the already deployed worker.
5. Persist call identifiers immediately. Monitor durable progress and retain
   every stopped, failed or completed outcome.
6. After all unit outcomes are durable, freeze the final confirmation lock,
   execute only unmatched confirmations, and produce a new rescue aggregate.

## Acceptance

The rescue is complete only when focused tests verify tombstone construction,
identity rejection and exact accounting; all 45 unit results are present or a
remaining failure is explicitly reported; the final aggregate uses all and only
the frozen unit identities; search and confirmation counts reconcile to their
receipts; the original partial aggregate is unchanged; and no call or check is
reported as completed unless its artifact exists.
