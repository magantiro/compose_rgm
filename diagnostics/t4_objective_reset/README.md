# Reset revision 1: actual outcomes

Runtime commit: `80dc1aa3`. No new oracle calls and no learned-prior fit.

## Pinned Modal zero-oracle probes

| Cell | Fresh attempts | Eligible endpoints | Seconds |
| --- | ---: | ---: | ---: |
| 5HT1B-0 | 54 | 5 | 23.73 |
| BRAF-1 | 54 | 1 | 21.15 |
| JAK2-1 | 14 | 3 | 20.99 |
| PARP1-0 | 17 | 8 | 12.49 |
| FA7-0 | 34 | 0 | 92.58 |

These are proposal-yield measurements, not docking outcomes. Wall caps and
chemical runtime affect the number of attempts; this is not a fixed-count
performance comparison. See hash-checked `remote_probe/unit_results/`,
`remote_probe/unit_rounds/`, `remote_probe/status_complete.json`, and the launch
receipt. The older `local_probe/` is a preliminary newer-RDKit diagnostic, not
the pinned benchmark environment or a passing gate.

## Conditional support

`conditional_ring_support.json`: after supplying the known JAK2 prefix and
module family, fresh generic ring sampling produced 64 distinct panels / 803
unique endpoints, with no exact teacher-ring hit. The old four-panel bottleneck
is removed, but this result does not demonstrate useful autonomous proposal
probability. It motivates a compact learned joint parameter/binding prior, not
blindly increasing random width. Source/code hashes and the supplied-answer
limitation are in the artifact.

## Verification and launch

- `verification/focused.xml`: 14 passed runtime tests.
- `verification/handoff.xml`: 5 passed cross-lane/resume/pack/reporting tests.
- `verification/touched_lint.json`: clean touched code.
- `verification/repository_lint.json`: 513 legacy violations; zero in changed
  runtime source files. No mass rewrite of frozen code was attempted.
- The full repository run contained failures/errors and was stopped after 54%
  reported progress. It is **not a complete or passing run**. Operational status
  and the initial collection-only environment error are preserved.
- `verification/final.json`: scored launch remains blocked; no false green gate.

The 120-call two-arm pilot is prepared but **not launched**. Read `HANDOFF.md`
and `docs/T4_AUTONOMOUS_RESET.md` before taking the next action.
