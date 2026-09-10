# Complete-edit preference-policy development result

The 2026-09-10 bounded run completed in 877.46 seconds with 28 new PMO calls,
under the frozen 32-call cap. It used 200 previously charged historical calls
(194 unique labels). This is a warm-start development experiment on
`perindopril_mpo`, not a from-scratch PMO result or an IVG comparison.

| Outcome | Balanced | Learned |
| --- | ---: | ---: |
| Fresh evaluation query requests | 8 | 8 |
| Mean selected endpoint score | 0.406653 | 0.454261 |
| Mean parent-to-child score change | -0.051976 | -0.001643 |
| Initial best | 0.491055 | 0.491055 |
| Final best | 0.494575 | 0.494575 |
| Initial top-ten score sum | 4.694462 | 4.694462 |
| Final top-ten score sum | 4.742637 | 4.742637 |

Interpretation: learned choice avoided more damaging edits in this small run,
but did not improve final best or top-ten archive quality relative to balanced
choice. Most learned selections were score-neutral or nearly neutral. The
learned arm selected no ring-system additions; balanced selected two, and both
lowered immediate score. These observations do not show that rings are generally
unhelpful, or establish learned long-horizon construction. The updated policy was
frozen across the two evaluation rounds. No future-value learning is claimed.

Calibration produced 15 completed candidates and consumed 14 new physical calls.
Evaluation consumed eight physical calls for balanced and six for learned, with
two independently selected cross-arm duplicates reusing their physical labels.
Both arms have eight logical evaluation requests. The initial and updated policy
fits took 1.95 and 2.10 seconds, respectively. The latter converged in seven
iterations over 472 preference pairs from 88 parent groups.

The selected-candidate records retain exact states, primitive witnesses through
their remote origin, intended region scope, realized structural displacement,
cycle-rank/ring-system deltas, canonical identity and allocation probabilities.
Intended scope is not counted as realized change. In the learned arm, a region
releasing 0.727 of the parent yielded largest changed fraction 0.091 and no ring
change. Both arms' new best arose from the carbonyl option.

## Measured speed bottleneck

The recorded slow worker took 293.75 seconds: 128.65 seconds in runtime
initialization and 160.51 seconds in 13 fresh full marked-law enumerations.
Enumeration accounts for 97.22% of its post-initialization wall time. The four
options took 86.48, 1.17, 0.60 and 76.56 seconds. The first and fourth were
six-membered pendant-ring construction options. There were 147 in-memory law
cache hits, zero historical/disk law hits and 1,341 metered executor calls.

These data implicate full primitive enumeration, not option-name selection, as
the dominant post-startup cost for this worker. The deployed code enumerates a
full molecular law before applying the region/option restriction. The earlier
serialization-cache experiment reported 1.76x and 1.94x speedups on two small-model
exact-state checks; that optimization is preserved on `pmo-enumeration-speed`
and is not active in this run. Its frozen-model deployment compatibility remains
unqualified. No full-run speedup is inferred from those two checks.

## Provenance and verification

- Run: `f162a5ff3560e62b502ae78925e626d96c4101f894237d7935746ae689685afe`.
- Producer: `3a0b6115af179617e06a39e43be433f22bc41500`.
- Contract: `5d29f594388f1ec34cdd763e6e4b788888c75242906d269fa42108c054565e7d`.
- Volume: `compose-v4-artifacts`, prefix `pmo_branch_policy/<run>`.
- `result_sealed.json` file SHA-256: `23854af517573f7bcee140778f8467c93964865b6bd4fdb209dc31b110389a2d`.
- `slow_worker_sealed.json` file SHA-256: `a47fae8134949740e2830a0a9207608b038289a2808dc721ed612872f8bbdb44`.
- Full locks and proposal receipts remain under `phase/0..2`, `policies`, `workers`
  and `oracle` in that remote namespace. The local sealed result records software,
  hardware, policy identities, input contract, query-ledger hashes and all selected
  outcomes. Preserve the separate zero-call failed launch and its correction.

Prelaunch checks: five focused tests passed in 3.29 seconds, covering process
identity drift, preference direction/KL/floor, canonical query allocation,
lock-before-score/frozen evaluation/resume, and exact primitive interruption
replay. Ruff check/format and strict clean-source preflight passed. Downloaded
payload seals and role counts were checked; the query count sums to 28 and stays
below 32. No repository-wide suite was run. This result is not milestone/release
qualification. No further queries, deployment, or reference training were launched.
