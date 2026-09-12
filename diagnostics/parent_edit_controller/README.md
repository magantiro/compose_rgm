# Parent/edit implementation evidence

Local CPU development, 2026-09-12. **Zero new oracle calls. No prospective
controller improvement or benchmark win is established.** The existing measured
archives and docking receipts are unchanged. Implementation and limitations:
[`PARENT_EDIT_CONTROLLER.md`](../../docs/PARENT_EDIT_CONTROLLER.md).

## Authoritative source-bound artifacts

All three computations use code
`6bcc220fdc34e68d93ca545565a09e71cd87a2b8`, Python 3.12.9, RDKit 2024.03.5,
NumPy 1.26.4, arm64 CPU, one worker per command and float64 model computation.
The commands ran concurrently; reported elapsed times are not isolated throughput
benchmarks. Input SHA-256 hashes, static implementation dependencies, seeds,
recipes, timestamps and result hashes are in each result JSON.

| Artifact | Finding | Evidence boundary |
| --- | --- | --- |
| [`committed_audit/result.json`](committed_audit/result.json) | Eight-endpoint-per-target v1 fitting improved archive gain in 3 pools, harmed 2, tied 3; 5.08 seconds | Chronological retrospective diagnostic on already inspected, correlated second-generation pools; not qualification |
| [`committed_branches/result.json`](committed_branches/result.json) | 20/32 constructions decomposed; all 20 admitted cases replay-verified; 12 abstentions; 5.46 seconds | Bound to the observed attachments/schedules, not universal decomposition or commutativity |
| [`committed_next_fit/result.json`](committed_next_fit/result.json) | Four v2 models fitted on 81 unique endpoints and 101 observations including repeats; 1.00 second | Training-only snapshots for a new prospective decision, not an evaluation on these training pools |

The branch artifact includes a bounded real BRAF branch exchange that retains
another recipient branch and produces a different executable endpoint. It is
**unscored**. It also records a corrected BRAF comparison: attachment-only equals
the parent canonically, and extension-only equals the joint candidate. Only two
distinct endpoints exist in that four-arm panel. The descriptive interaction is
zero under the existing first observations. This supports the extension, not
attachment synergy; it is not a biological mechanism or a noise estimate.

The v2 models add selected-parent-to-candidate mutation features separately from
original-source construction features. Fitting counts are:

| Target/context | Unique endpoints | Genuine observations, including repeats |
| --- | ---: | ---: |
| JAK2 seed1 | 19 | 23 |
| FA7 seed0 | 23 | 29 |
| BRAF seed1 | 21 | 27 |
| 5HT1B seed0 | 18 | 22 |

Models are target/protocol-specific; repeated measurements are retained and
averaged rather than selecting the most favorable repeat. Duplicate program
representations do not multiply endpoint weight. A reconstructed candidate's
comparison with its measured parent is not labeled as an executed parent-to-child
trajectory. No future-value interpretation or calibrated uncertainty is claimed.

## Verification

- 44 focused controller, runner and dependency tests passed in 6.68 seconds.
- Lint and formatting passed for the touched Python files; `git diff --check`
  passed. No repository-wide suite or milestone release sign-off was performed.
- All three authoritative result self-hashes and eight model payload hashes were
  checked. All 104 distinct recorded input/source files matched their physical
  SHA-256 hashes. Bound implementation bytes also matched the code commit.
- The multi-round runner test used an executor-backed synthetic PMO reward.
  It validates query accounting, updates and completed-round resume, not PMO
  benchmark performance. Real task-oracle manifests and paid launch authority
  remain prerequisites for an experiment.

## Preserved development history

`attempt_1` retains the mixed v1 selector audit and initial operand-only splitter
failure (0/32). `branch_repair_1` records observed-write decomposition (20/32).
`branch_repair_2` adds the real donation and corrected BRAF panel. `next_fit_1`
retains the pre-commit v2 fit. These artifacts bind working-source hashes; use the
committed directories above as the final source-bound reproductions. Do not count
reruns as independent evidence or overwrite the original diagnostic with a fit
that includes its evaluation outcomes.

The next experiment must compare several learning rounds against score-blind
selection using the **same repaired proposal support**, with new query and
proposal-work limits. The current local unit does not authorize a new docking or
PMO campaign.
