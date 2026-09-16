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
- `verification/milestone.xml`: 73 passed tests across all five branch test files.
- `verification/repository_lint.json`: 513 legacy violations in `src/`, unchanged
  from the baseline, zero in any file this branch adds. `repository_lint_all_summary.json`
  records the whole worktree at 4,274 so the narrower figure is not mistaken for it.
- `verification/full_suite.xml` and `full_suite_execution_completed.json`: the
  repository-wide run **completed** at 5,396 tests, 5,225 passed, 110 failed, 58
  errors, 3 skipped, 38m30s. Status `completed_and_failed`, `launch_authority: none`.
  The earlier terminated receipt (`full_suite_execution.json`, 54%) is preserved.
- `verification/full_suite_classification.json`: every failing node bucketed by its
  own error text, plus three independent provenance measurements showing none of
  them belongs to this branch. `full_suite_isolated_recheck.xml` reruns exactly the
  failing nodes alone on cleared bytecode; 48 of 168 pass there.
- `verification/final.json`: `passed: false`, `branch_tests_passed: true`. The
  scored launch remains blocked and no false green gate is claimed.

The 120-call two-arm pilot is prepared but **not launched**. Read `HANDOFF.md`
and `docs/T4_AUTONOMOUS_RESET.md` before taking the next action.

## Learned-prior prerequisites (zero oracle, nothing fitted)

Both gates `AGENTS.md` sets before proposal-prior training are closed; the artifacts
live under `diagnostics/t4_proposal_prior/`.

- `dataset_v1/`: split-clean corpus, decision **GO**. 34,073 labelled constructions,
  no identity crossing a leave-one-target-out fold, 18 dense construction families.
- `access_probe_v1/`: actual-sampler support, decision **PASS**. All fifteen declared
  families realized through the production sampler, exploration repaired at
  256-attempt scale, all five supplied champions realize exactly. Eligible yield is
  ordered by each cell's root QED, and FA7-0's zero yield is the endpoint gate rather
  than proposal support.
