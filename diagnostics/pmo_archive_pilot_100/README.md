# Completed 100-query development pair

Observed on 2026-09-10. No new optimization run is authorized here.
Run: `029a27392680e2e57bd9aa9567b005057f9c68fb5450807a768cc80e8e05c892`.
Producer revision: `8ef5eb220435aa76f373b0099fbde858391e7292`.
Contract: `b49ffcd7eb228d70c5a9a48fe82dc2bec3593efce1a9350f5fe4a9dbcdf9b382`.
Source: Modal volume `compose-v4-artifacts`, prefix
`pmo_archive_pilot/<run>/perindopril_mpo__<arm>__0`.
These are exact copies of the finished function results. They bind model inputs,
configuration contract, software, hardware, seeds/case, query ledger, archive,
attempts, credit and complete curves. The recipe is in `docs/PMO_ARCHIVE_PILOT.md`.

| Producer-reported quantity | Balanced | Adaptive |
|---|---:|---:|
| Unique oracle queries, including initialization | 100 | 100 |
| Initial best | 0.360237411 | 0.360237411 |
| Final best | 0.491054734 | 0.460348270 |
| Final top-ten mean | 0.466391532 | 0.450011774 |
| Zero-padded top-ten AUC over 100 queries | 0.395920108 | 0.387317135 |
| Completed option attempts | 97 | 102 |
| Support dead ends | 1 | 0 |
| Maximum option ancestry | 9 | 11 |
| Maximum primitive ancestry | 33 | 30 |
| Mean pairwise fingerprint distance, all 100 queried molecules | 0.724474460 | 0.733349945 |
| Fresh marked laws | 176 | 164 |
| Marked-law seconds | 1307.274388 | 1762.210631 |
| Proposal-loop elapsed seconds | 1737.763650 | 2214.749179 |
| Initialization seconds, separate | 109.762965 | 122.715225 |

Interpretation: adaptive credit changed allocations (mean row total-variation
distance 0.3272 versus zero in balanced) but lost on the declared AUC objective
and final best in this single paired seed. No confidence interval or superiority
claim is supported by one pair. Keep the negative result; do not change the recipe
and relabel a rerun as this experiment. There was no prescreen or winner input.
This is not official PMO initialization and cannot be compared directly with IVG's
10,000-query runs.

The fixed-prefix structural audit is `adaptive_prefix_topology.json`, produced by
`tools/pmo_archive_snapshot.py`. It binds every source receipt by SHA-256 and
recomputes structural displacement from exact states. At 62 queries it covers
60/60 completed attempts, 14/14 completed ring constructors and 15 cycle-rank
increases, with chains reaching nine options and 26 primitives. These are prefix
counts, not totals for the final 100-query run. Primitive replay relies on the
producer receipt; the offline check recomputes topology, not replay. A full final
graph/credit diagnosis remains the next analysis, without further oracle calls.

File SHA-256 identities:

- `balanced.json`: `5d9ff064faa95e6204c2aa5ce69315474cbdec00f319118d12fa7cc77903c471`
- `adaptive.json`: `1f07d33f5716dd1efb9dca112f846f93ac0608ed20fe2d99faabc9cafe842685`
- `adaptive_prefix_topology.json`: `d7bf08c63ec813e95354bd9cd5d41933efa7555632ac57cd35cf9b9fa98da314`

The separately committed serialization repair was not deployed into either arm.
No result here depends on that repair. No full repository verification or broader
scientific milestone is claimed by this operational report.
