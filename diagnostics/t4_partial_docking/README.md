# Approved saved-prefix docking: completed

Source implementation: `993fbef8b720` (full revision in `attempt_1/spawn.json`).
Prospective contract: `configs/t4_partial_docking.json` and
`docs/T4_PARTIAL_DOCKING.md`. The user approved at most 13 saved eligible
candidates and container parallelism. No new molecular proposals, R_theta
initialization, executor work, or automatic optimizer continuation occurred.

Run: `fc-01M20XESZV3Y3HFF58BHMSV20W`, volume `compose-v4-artifacts`, prefix
`t4_partial_docking/6162234b802b661012c4773171623a0dd964cb1394dbeb65405160248abf4a6d`.
All 13 approved candidates passed the pinned-runtime eligibility recheck and
were locked before any worker started. Thirteen candidates from 13 distinct
bundles were docked once each; zero failed. The completed diagnostic consumed
13 new calls, giving 33 cumulative guided calls and 27 unused calls from the
previous allowance. Those unused calls are not automatic launch authority.

## Measured results

| Proposal | Parent docking score | New docking score | Change in graph cycle rank |
| --- | ---: | ---: | ---: |
| Best new candidate, generic | -9.3 | -9.2 | 0 |
| Fused C6 program, hydroxy-containing parent | -7.5 | -9.1 | +1 |
| Fused C6 program, isopropyl-containing parent | -8.2 | -8.9 | +1 |
| Cyclize | -7.6 | -7.9 | +1 |

The best-so-far score remains **-9.3**. Both verified aromatic C6 fused builds
produced lower docking scores than their respective parents, by 1.6 and 0.7
score units, but neither surpassed the existing best. These are observed
differences under an unseeded oracle, not causal affinity estimates or evidence
that guidance outperforms a matched reference.

Both fused products satisfy QED >= 0.6, SA <= 4 and original-seed similarity
>= 0.4. Their SA values are 2.601 and 2.955, and similarities 0.534 and 0.413,
respectively. Each adds four carbon atoms and one cycle without increasing the
ring-system count. The prior exact-slot audit verified aromatic C6 fusion and
no added bridgeheads/spiro atoms. Complex bridged source products and the
SA-failing second pendant addition were not docked.

Intended release versus realized coherent change remains distinct: the -9.1
fused product has 0.100 versus 0.300; the -8.9 fused product has 0.545 versus
0.273. The cyclize product has 0.850 versus 0.100. These are per-parent
structural measurements, not evidence that a large selected region necessarily
produces a proportionally large transformation.

## Runtime, provenance, and verification

The docking phase took 26.768 seconds, including worker startup/receipt handling.
Total driver time was 34.219 seconds. At most four docking containers were
configured, each with one CPU and 2 GiB requested memory. The driver requested
the same resources. Driver peak RSS was 3,250,368 Linux-native KiB; this is not
the combined peak of all containers, and exceeds the requested memory floor.
Actual billing has not been measured. The oracle used Open Babel 3.1.1,
RDKit 2024.03.5, and the verified frozen QuickVina/receptor hashes. Its inherited
unseeded protocol is unchanged.

`attempt_1/remote_inventory.json` records the physical hash and size of every
downloaded remote receipt. `candidate_lock.json` retains all 25 source candidates,
the 13 selected exact states, exclusions, source traces, and pinned-runtime
properties. `rows/00` through `rows/12` each retain started and result receipts.
`docking.json` is the sealed reduction; `result.json` adds the remote wrapper's
provenance, profiling, and later completion timestamp. No failed batch or row
was retried.

Focused implementation/launcher checks: 83 passed in 7.82 seconds (`focused.xml`).
The exact worker retry test additionally passed in 2.53 seconds (`worker_test.xml`).
Two saved-receipt audit checks passed in 2.22 seconds (`audit_tests.xml`). The
auditor checks all physical hashes, source/lock identity, eligibility, unique
canonical identities, every worker's index and chronology, oracle accounting,
and exact-state structural metadata. It distinguishes the earlier sealed
docking timestamp from the common wrapper's later completion timestamp while
requiring equality of all scientific fields. Formatting/lint passed for touched
standalone code; the legacy app retains the same 17 existing lint findings with
none introduced. No broad suite was repeated.

Reproduce the offline verification:

```sh
PYTHONPATH=src:. .venv/bin/python tools/t4_partial_audit.py \
  diagnostics/t4_partial_docking/attempt_1 \
  diagnostics/t4_warm_continuation/attempt_2 \
  --output /tmp/t4_partial_review.json
```

Operational note: an initial preflight attempt raced clean-worktree creation
and could not find its script. After worktree creation completed, strict
preflight passed; the cached deployment was finalized again, and the launcher
independently passed clean-source preflight before spawning the only scientific
call. No oracle was invoked during that setup race. The launch worktree was
`/private/tmp/compose-t4-partial.5hJCHU`; unrelated scaffold-construction work
and its tracelet-compiler edits were excluded and left untouched.

## Interpretation and stop

Fused-ring construction now delivers feasible, competitive docking candidates
in this inspected cell. This partial batch did not improve the best-so-far
score. It is not a completed eight-parent round, a multiround optimization
result, a matched causal ablation, or an IVG performance claim. The next
scientific question is accumulation through feedback-driven rounds; this
diagnostic does not answer it or authorize another run. All work is local and
unpushed. The separate warm attempt's compute-ceiling failure remains recorded.
