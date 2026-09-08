# Ring feasibility: construction alternatives and saved-endpoint refinement

Computed 2026-09-08, inspected PARP1 seed0 delta 0.4 development bundles.
**Neither a different zero-refinement path in the same bundle nor two existing
ring-scoped refinement steps recovered a feasible endpoint for the six failed
saved rings under the declared neutral reference.** The checks used 487 public
executor calls in 9.62 seconds, with zero docking, training or cloud calls.
The live T4 controller was not changed or redeployed. Best docking therefore
remains -9.7 at 51 cumulative calls, as recorded in the preceding audited round.

## What was computed

1. Reused the seven exact saved construction paths. Recomputed the unchanged
   T4 properties and reconciled endpoint values with the original run within
   1e-7. Verified path continuity, canonical identity and completed-cycle
   witnesses. Measured fingerprint overlap and union after every edit.
2. Held each parent, region and ring request fixed, and completed the existing
   finite-horizon backup over neutral broad-organic primitive candidates with
   the production executor. Terminal value was exactly the existing T4
   feasibility indicator, not docking or a new quality score. All seven backups
   completed: 26 bundle-wise canonical endpoint records, three feasible, all in
   the already-successful bundle. The other six bundles had zero terminal mass.
   Work: 325 public executor calls, 7.990731 seconds.
3. Resumed the six saved infeasible ring endpoints for the existing `refine=2`
   continuation, preserving the original parent, region, cycle, exact slots and
   atom-birth lineage. All six backups completed: 56 bundle-wise canonical
   two-step endpoint records, zero feasible. All 27 initial successor records
   also failed feasibility. Work: 162 executor calls, 1.627691 seconds.

No program was declared impossible because of budget exhaustion; none exhausted
the declared 1,024-call/30-second per-bundle allowance. No intermediate was
pruned merely for failing an endpoint constraint. The reference-versus-guided
initial-decision calculation used the existing kappa=1 tilt. It had no positive
feasibility contrast to exploit in the six failed bundles and correctly returned
the zero-terminal-mass baseline. The high-margin bundle had terminal mass one.

| Bundle suffix | Request | Parent similarity | Saved endpoint similarity | Best alternative construction similarity | Best two-step refinement similarity |
| --- | --- | ---: | ---: | ---: | ---: |
| `ea73f` | pendant 5, C4N1 aromatic | 0.5614 | 0.4478 | 0.5079 | excluded, already feasible |
| `0a8` | fused 6, C6 aromatic | 0.4091 | 0.3662 | 0.3662 | 0.3662 |
| `67a6` | pendant 5, C4N1 aromatic | 0.4375 | 0.3562 | 0.3889 | 0.3562 |
| `5cf98` | pendant 6, C6 aromatic | 0.4091 | 0.3611 | 0.3889 | 0.3611 |
| `00b2fc` | pendant 5, C4O1 aromatic | 0.4127 | 0.3562 | 0.3714 | 0.3600 |
| `cfbd` | pendant 6, C5O1 nonaromatic | 0.4179 | 0.3636 | 0.3733 | 0.3766 |
| `c7ab` | fused 5, C4N1 nonaromatic | 0.4179 | 0.2469 | 0.2469 | 0.2658 |

The required similarity is 0.4; QED >=0.6 and SA <=4 remain additional unchanged
requirements. Maxima in the last two columns are separately enumerated outcomes,
not the same molecule and not a sampled-controller performance score.

## Mechanism and decision

Tanimoto similarity is shared fingerprint bits divided by union bits. These
paths lose similarity through both effects, not only a bad attachment site.
For example, the saved six-member oxygen pendant ring preserves all 28 shared
bits, but increases the union from 67 to 77: 28/67=0.4179 becomes 28/77=0.3636.
The saved fused five-member nitrogen ring loses eight shared bits (28 to 20)
and increases the union (67 to 81), giving 20/81=0.2469. Exact stepwise bit lists
are saved in the result, so these are fingerprint measurements, not inferred
atom matches or chemical mechanisms.

**Do not integrate stronger feasibility weighting on these fixed, empty-feasible
construction supports.** Do not increase kappa, relax similarity, fit Q(o), or
spend docking to confirm already-known constraint failures. The next proposed
search change should address when/where material is added: retain construction
opportunities before similarity margin is spent, or test replacement/reuse of
existing substituent material instead of addition alone. Those strategies have
not been implemented or validated by this diagnostic. Q(M), the balanced Q(o),
generic support, frozen structural committor and R_theta remain unchanged.

## Evidence limits and verification

This is a **neutral-reference support/continuation diagnostic**, not a learned
R_theta run. In particular, the neutral primitive enumerator does not establish
coverage of the learned correlated `ring_system_restate` proposals. Its negative
refinement result is not a proof that every production refinement route fails.
The fixed regions, C/N/O request menu and two-step refinement bound also limit
the conclusion. No claim of global unreachability, medicinal quality, calibrated
task guidance, or IVG superiority follows. There is no new docking evidence.

- [result.json](result.json), scientific revision `e6fa7ce`, SHA-256
  `3d49817a43475743fa1a85a52334b5ffa34d9f0dea597320a8f8ad29ad884a3c`.
- [refinement.json](refinement.json), scientific revision `43be17d`, SHA-256
  `404247d938a12edbf9a5154f575d045f24df0507aeaff2412af3b0973dc6b4da`.
- Both ran from clean detached worktrees, outside unrelated user edits. Input
  hashes, tracked scientific-code hashes, exact states, all executor receipts,
  configuration and software are included. Local RDKit is 2026.03.6, whereas
  the original T4 run used 2024.03.5; saved endpoint properties reconciled.
- Focused dependencies: 43 passed in 2.93 s at `e6fa7ce`; the three affected
  diagnostic/lineage checks passed in 2.32 s at `43be17d`. Ruff and whitespace
  checks passed. Artifact inspection verified source hashes, complete unit
  counts, every executor ledger index, normalized probabilities, terminal-mass
  arithmetic and KL <=1. No repository-wide suite or full milestone claim.

Reproduce the first and second commands at their respective scientific revisions:

```sh
PYTHONPATH=src:. python tools/t4_ring_feasibility.py \
  diagnostics/t4_ring_program_round/attempt_1/run --output NEW_RESULT_PATH
PYTHONPATH=src:. python tools/t4_ring_feasibility.py \
  diagnostics/t4_ring_program_round/attempt_1/run --refine-saved --output NEW_REFINEMENT_PATH
```

The scripts refuse to overwrite an existing result. Intermediate unit receipts
are saved atomically; the outer `status` distinguishes a completed census from
an interrupted or budget-limited one. Do not rerun the completed units solely
because this interpretation file receives a new commit.
