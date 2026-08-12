# Workstream C — Pathwise Constraints: Status

- **Status:** `DESIGN_ONLY` — implementation complete, awaiting run authorisation
- **Branch:** `codex/compose-pathwise-constraints`
- **Base commit:** `04f1c46` (Add parallel workstream plan and agent handoff template)
- **What is running:** **NOTHING.** Zero Modal invocations have been made.
- **Last completed gate:** local held-in census + 64 local tests, all passing
- **Held-out data opened:** **NO.** The only pool touched is `training_source_keys`.

## Next action (one, bounded)

Authorise **stage A** of the smoke:

```bash
modal run modal_apps/pathwise_constraints_app.py --stage A --sources 6
```

- **6 held-in sources**, panel `8b47a0e4…`, CPU only, `cpu=2.0`, `max_containers=10`
- **≈ 2.5 container-hours** (range 1.9 – 3.1), **≈ 30 min wall** with 6 parallel containers
- Resolves gates **G1 – G4**, including the vacuity gate that decides whether
  this workstream has a claim at all

Stage B (the two verified arms, ≈ 4.3 further container-hours) is **not**
requested and should not be paid for until G1 passes.

## What the local census settled

| Question | Answer | Denominator |
|---|---|---|
| Does a protected motif exist? | 98.3% have a ring system | 20 000 held-in |
| How large? | median 9 atoms, 31% of the molecule | 19 665 |
| How much room is left? | median 18 heavy atoms outside the motif | 19 665 |
| Are sources eligible? | 74.9% yield | 20 000 |
| Objective headroom? | only 0.9% already satisfy `B`; 2.3% satisfy `P` | 20 000 |
| SMARTS writer correctness | 0 failures | 4 000 fuzz + 1 500 in-suite |

## What the census could NOT settle

The two questions the vacuity gate turns on — **how often unconstrained
control violates the motif mid-path**, and **what fraction of legal support the
mask removes** — both require enumerating successors, which requires the frozen
`R_theta` on Modal. They are resolved by stage A, not locally. No local number
in this lane may be quoted as the gate.
