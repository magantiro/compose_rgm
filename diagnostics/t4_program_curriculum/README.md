# Four-target shared-program docking result

The same expanded program library produced seed improvements on all four T4
development cells while keeping the 40-heavy-atom lane fixed. All 32 candidates
were locked before scoring and all were scored, without surrogate selection.

| Cell, delta=0.4 | Fresh seed | Best of eight | Candidates beating seed | Best endpoint atom change |
| --- | ---: | ---: | ---: | ---: |
| JAK2 seed1 | -7.9 | -10.3 | 8/8 | 22 to 28 |
| FA7 seed0 | -7.2 | -9.2 | 8/8 | 32 to 27 |
| BRAF seed1 | -9.4 | -11.1 | 5/8 | 39 to 32 |
| 5HT1B seed0 | -5.8 | -13.1 | 8/8 | 39 to 32 |

Every cell used eight candidate calls plus one charged seed control. Total:
36 calls, zero failures, 61.16 driver seconds, 139.82 summed worker seconds,
eight single-CPU workers plus one driver. Deployment took 89.04 seconds,
separate from scoring. These timings are not a provider-billed dollar amount.
No fresh confirmation repeats are included in this round.

## Interpretation and next decision

Shrink/remodeling programs improved more than endpoint yield: the best FA7,
BRAF and 5HT1B candidates also docked better than their original seeds. This
is development evidence, not proof that the architecture is superior under
matched compute, nor proof of a binding mechanism. The previous PARP1 -13.3
result remains intact.

For orientation, the pinned released IVG per-run bests are:

| Cell | Released IVG run bests |
| --- | --- |
| JAK2 seed1 | -12.6, -11.3, -10.8 |
| FA7 seed0 | -9.1, -8.5, -8.6 |
| BRAF seed1 | -10.5, -10.6, -9.9 |
| 5HT1B seed0 | -13.1, -14.0, -12.8 |

BRAF is promising relative to those reported scores. JAK2 remains weaker;
5HT1B does not beat the best released run; FA7's 0.1 difference from the best
released score is not resolved by this experiment. These values are external
single-run bests, not redocked controls or a matched aggregate comparison.
Historical COMPOSE scores and their budgets are not overwritten by this round.

The existing hash-verified census considered 87,448 released rows and retained
91 distinct winners, none above 40 atoms. For BRAF seed1 the selected winners
contain 26–28 atoms; for 5HT1B seed0 they contain 32–33. Current evidence favors
capacity-aware transformation selection, not increasing the T4 state-space cap.
The census does not describe the size distribution of every nonwinning candidate.

All 32 new measurements now reside in four separate target/protocol-bound
optimizer archives. Their transformation logic can be reused; their scores
cannot be relabeled as another target's observations. The next decision is to
test measured-score adaptation and confirm promising results under a separately
bounded budget. No new head, extra particles, cap expansion or automatic paid
round was launched. Cumulative follow-on use is 64/66 calls.

## Artifacts and verification

- `attempt_1/launch.json`: durable call and clean image identity.
- `attempt_1/remote_result.json`: unchanged sealed remote result.
- `attempt_1/review.json`: verified score ledger, curves, provenance and archive hashes.
- `attempt_1/*_archive.json`: exact programs and genuine observations, ready for local optimizer reuse.
- `configs/t4_program_curriculum_lock.json`: immutable 36-row allocation.

Scientific source `1a5c0ccc169e7910a70d0f8b9af75952ecfec69a`;
call `fc-01M2BSMX75V6SM1695M6SFNQ62`; volume `compose-v4-artifacts`;
prefix `t4_program_curriculum/abc7c4426db514975b658e52ae4ea31b60c7b3ab5ca7c33b03a454c50fa0257c`.
Per-row ligand and docking-pose hashes and files are retained there.

Before launch, all 32 exact traces and endpoint gates were checked locally;
27 focused checks passed plus the repaired driver fixture passed separately.
The fixture repair removed input-only trace tuples from fake worker outcomes,
matching the actual row schema. Eight additional reducer tests passed, protecting
oracle identity, missing outcomes, nonfinite scores and failure accounting.
Archive conversion replayed every newly measured program. Ruff checks passed
for touched Python files. No repository-wide suite was run and no full controller
milestone is declared complete.

Reproduce the cache-only review with the pinned RDKit 2024.03.5 environment:

```sh
PYTHONPATH=src:. python tools/review_t4_program_curriculum.py
```

The reducer hashes its exact implementation, all input locks and receipts, and
the existing census. It performs zero oracle calls.
