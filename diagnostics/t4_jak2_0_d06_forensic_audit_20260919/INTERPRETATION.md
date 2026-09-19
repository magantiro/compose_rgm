# JAK2 seed-0 delta=0.6 miss: bounded forensic audit

## Outcome

The 49-call scored run contains no candidate at or below the reported IVG mean of
−9.7. Its best was −9.6, produced by a shallow proposal selected through
exploration. The exact scored proposal pool cannot be reconstructed from the local
checkout because the per-cell result and six candidate locks remain only on the
bound run volume.

Every Dynamic, Full146 and route score below is a reported historical label, not a
current-run oracle observation. No historical label is transferred into the current
archive or treated as a score under the current docking seed.

The strongest supported diagnosis is checkpoint lineage, not a current binding or
joint-planning failure. The scored run used a leave-JAK2-out route checkpoint with
111 templates. A later shared, task-independent checkpoint fitted on all 77 locked
routes has 137 templates and autonomously emits two exact, one-region JAK2 routes
at ranks 3 and 6. Both pass the current `compose_valid` delta=0.6 support in 10 and
12 primitives. Their reported historical endpoint labels are −10.6 and −10.8.
Their exact templates are absent from the checkpoint used by the scored run and
present in the later checkpoint.

The historical Dynamic-v0 delta=0.6 run has 912 query-verified reported historical
labels and reached a reported −9.5, so it also did not meet −9.7. Its best endpoint
is absent from the later route shard. Six weaker Dynamic-v0 endpoints overlap that
shard, with a best reported historical label of −8.1. This is endpoint overlap only,
not an impossibility claim about Dynamic's stochastic law.

A third reported delta=0.6 route, labelled −10.5, is also exactly emitted by the
later route runtime. It passes the benchmark QED, synthetic accessibility and
similarity thresholds, but the current stricter `compose_valid` endpoint gate
rejects its four-membered N-N ring. This is an endpoint-support exclusion, not a
binding or compilation loss.

## Stage localization

- **Current scoring:** observed miss. The complete current archive best is −9.6,
  so none of its 49 charged observations meets −9.7.
- **Proposal families:** every measured parent was scheduled through shallow,
  anchored replacement and route-complete-region workers. The first two rounds had
  an expert floor; later selection mixed the value model with two exploration
  slots. The winner was shallow exploration in round 5.
- **Proposal scales:** the scored revision attached local, medium or large scale to
  route records but had no scale floor. Exact offered and selected scale counts are
  in the missing locks and are not inferred. The later zero-oracle support pool has
  96 complete programs, including 16 eligible at delta=0.6 across all three
  primitive bands.
- **Binding and compilation:** the later all-route shard exactly realizes the two
  eligible strong routes with precision 96/96 over all committed programs. Binding
  and compilation therefore do not explain a present support loss. Their exact
  templates were unavailable to the old scored checkpoint; the missing locks leave
  alternative-template behavior unresolved.
- **Old standalone validation:** the earlier forensic represented the routes but
  correctly abstained on a missing sealed compiler payload and found its older
  one-to-two-event generator out of support. The later complete-region runtime
  supersedes that particular support failure for the two routes above.
- **Joint composition:** the three historically reported delta=0.6 routes each
  contain one dependency region. Deferred multi-region validity and depth planning
  are therefore not causal for this miss.
- **Selection and continuation:** the exact old candidate locks are missing, so it
  is unknown whether an alternative template reached one of these endpoints and
  was dropped by selection. Archive continuation is not needed for the historical
  route endpoints themselves to beat −9.7, although stronger Full146 descendants
  show that continuation can add value. Historical labels are not transferred into
  the current run.

## Claim boundary and missing asset

This audit made zero oracle, docking, network, Modal or live-run calls. It does not
claim that the historical labels would repeat under the current docking seed or
that the later route pool would have been selected by the old controller.

Exact reconstruction requires the SHA-bound
`jak2_0/result.json` (`52df52490281aadda3f68db1ec0de880a85d332eb6ed884e9c3e050840cbd8f5`)
and its referenced `round_001_lock.json` through `round_006_lock.json` under
`compose-t4-integrated-route-fiber-v1/2f223bd1210f3da8e5adc99e93679b1303bfb2922c0509f570b9b213bb96d066/jak2_0/`.
They were not fetched because live-run and Modal access were outside scope.

Reproduce the local evidence ledger with:

```bash
PYTHONPATH=src python3 diagnostics/t4_jak2_0_d06_forensic_audit_20260919/audit.py
```

The machine-readable authority is `result.json`; every local material input and
teacher receipt is SHA-256 bound there.
