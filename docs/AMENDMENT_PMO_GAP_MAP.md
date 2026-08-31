# Amendment: the PMO deficit is concentrated, and not where we were developing

Status: measured 2026-08-22. Source of truth for GenMol targets is
`docs/genmol_pmo_targets.json`, extracted programmatically from arXiv:2501.06158v3
and asserted to hold 23 tasks summing to 18.362. **No benchmark number may be
hand-copied into prose or code.**

## Result

22 verified tasks, generic controller (`L1`, oracle-prescreened init, no
surrogate), 100 counted calls, one seed. Reported as the official PMO
`top_auc(..., finish=True)` over a COUNTED-ONLY buffer, normalised by 10,000.

Full table in `diagnostics/pmo_gap_map.json`. Aggregate:

| | 22 tasks |
|---|---|
| COMPOSE @100 calls | **12.703** |
| GenMol, same 22 | **17.367** |
| deficit | **+4.664** |

The deficit is concentrated. The five worst tasks -- `valsartan_smarts` (+0.715),
`troglitazone_rediscovery` (+0.506), `sitagliptin_mpo` (+0.472),
`isomers_c9h10n2o2pf2cl` (+0.389), `celecoxib_rediscovery` (+0.351) -- carry
**2.43, or 52% of the total**. A rediscovery/similarity family (troglitazone,
celecoxib, thiothixene, albuterol, mestranol) accounts for much of the rest.

`qed` is already ahead of GenMol (-0.002) and `gsk3b` is within 0.041.

## The correction this amendment exists to record

`scaffold_hop` ranks **15th of 22, at +0.099, about 2% of the deficit.** It
received an entire development session -- an oracle-greedy reachability
diagnostic, a 249,455-label surrogate, a five-arm ten-run controller sweep, and a
macro-proposal ablation -- on the basis of a GenMol target of 0.936 that was
never GenMol's number. The published value is **0.628**, and COMPOSE's
oracle-greedy diagnostic had already reached 0.6285, i.e. parity, before any of
that work began.

Two lessons, both now enforced elsewhere:

1. Benchmark targets come from a checked artifact with an aggregate assertion.
   See [[benchmark-targets-from-artifact]].
2. **Rank the deficits before choosing what to develop.** Whichever task is in
   front of you is not evidence it is the costly one. A breadth sweep at 100
   calls over all tasks costs under a dollar and would have redirected the night
   in its first twenty minutes.

## Macro proposals: a real effect, on the task we did not expect

One counted oracle call buys a trajectory of L successive frozen-R_theta edits,
intermediates never evaluated. `L1` versus `mixedL` (L in {1,2,4,8}), everything
else identical, 100 calls, 2 seeds:

| task | L1 | mixedL | delta |
|---|---|---|---|
| jnk3 | 0.6580 | **0.7290** | **+0.0710** |
| scaffold_hop | 0.5322 | 0.5266 | -0.0056 |

The effect on jnk3 is far outside seed spread (L1 seeds 0.6670/0.6490). The
mechanism was motivated by scaffold_hop's structural landscape and does nothing
there, while helping a property-optimization landscape substantially. That
inversion is unexplained and should be understood before generalising.

**Cost caveat.** `mixedL` runs ~5x the wall clock of `L1` (30 min versus 6 at 100
calls), because every trajectory step visits a fresh state and each state costs a
full R_theta forward. At 23 tasks x 10,000 calls x 3 seeds this is prohibitive
without batched inference.

## Measured cost structure (supersedes the earlier estimate)

Measured, not assumed: **t_law = 6.74 s per state, 58% of a fiber call**;
per-apply 17.4 ms, and 6.74 + 285 x 0.0174 = 11.69 s reproduces the independently
measured fiber cost exactly.

This **retracts** the earlier claim in `AMENDMENT_PMO_SCAFFOLD_HOP.md`-era
analysis that `t_apply` dominates and that batching R_theta inference "buys
nothing". The forward pass is the larger half. Consequences:

- Batched R_theta inference is worth implementing, behind a flag, with bitwise
  parity asserted against the one-state path before any caller switches.
- Sampled rollouts are **1.7x** cheaper than full fiber enumeration, not the 22x
  previously stated. A T4 `future_h` round with sampled rollouts is ~62 hours,
  not ~5. Shortlisting, not sampling, is the lever there.

## Known gaps in this map

- **drd2 excluded.** Its TDC oracle is an unpicklable SVM with no frozen asset
  and no verified extraction. Every aggregate here is over 22 of 23 tasks and
  must be stated that way; GenMol's 18.362 is not the right comparator, 17.367 is.
- **gsk3b's frozen forest has never been parity-checked.** Gate A verified jnk3's
  to 0.000e+00; gsk3b's was assumed to follow. It is currently our second-best
  row (0.9453) and must be verified before the number is used.
- **These are 100-call floors** projected to 10,000 by the official `finish=True`
  rule. Every row rises with budget and rises unevenly, so the RANKING is
  trustworthy and the magnitudes are not. Tasks still climbing steeply at call
  100 are understated relative to plateaued ones.
- One seed per task. jnk3 and scaffold_hop have two.

## Next

Develop against the ranked deficit, not against a single row. The immediate
candidates are the five tasks carrying 52% of the gap, and the rediscovery/
similarity family as a group. `scaffold_hop`, `osimertinib_mpo`, `qed` and
`gsk3b` are regression checks, not development targets.
