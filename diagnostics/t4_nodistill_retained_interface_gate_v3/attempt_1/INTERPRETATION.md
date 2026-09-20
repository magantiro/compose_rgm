# NoDistill retained-interface v3 support gate

Decision: **NO_PROMOTION**.

The prospectively frozen v3 allocator compiled 98 unique exact programs at replay
precision 1.0, all within the unchanged 40-heavy-atom, 32-primitive and 8-block
support. Generation used only the current graph, deterministic seed, retained
interface roles, current topology, capacity and work budgets. All five locks were
sealed before teacher descriptors were loaded. Free endpoint admission was applied
only after generation.

The strict all-three growth-cell gate failed:

| Cell | Raw compile successes | Exact unique | Fiber/archive-ready | Ready plan diversity | Result |
|---|---:|---:|---:|---:|---|
| `jak2_0_d06` | 614 | 34 | 0 | 0 | failed endpoint feasibility |
| `parp1_0_d04` | 775 | 36 | 4 | 3 | passed the per-cell minimum |
| `5ht1b_1_d04` | 851 | 28 | 2 | 2 | passed the preregistered v3 per-cell minimum |
| `braf_0_d06` | 0 | 0 | 0 | 0 | expected capacity abstention |
| `5ht1b_0_d04` | 0 | 0 | 0 | 0 | expected capacity abstention |

The six archive-ready endpoints span `Δheavy=+4..+8` and `Δcycle=+1`; none is
an exact teacher endpoint. The archive-ready PARP and 5HT1B1 subsets cover no
teacher coarse or semantic descriptor, so v3 demonstrates limited free-feasible
growth support, not closure of the audited large multi-cycle transformation gap.
Exact-teacher recovery was not a gate requirement.

The authoritative post-format result is `verified_result.json`, payload SHA-256
`0bd9a932d16d5c286e76f9e43ea77fa90d53e594c21a4f4bd535f05dea261c7f`.
The earlier immutable `result.json` records the same `NO_PROMOTION` scientific
projection before final formatting. Candidate locks and all 81 deterministic
particle locks are retained for audit and resume.

This is zero-oracle proposal-support evidence only. No docking score, objective
utility, scored selection, production integration or benchmark improvement was
measured.
