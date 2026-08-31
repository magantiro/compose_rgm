# T4 development split — REGISTERED BEFORE THE CONTROLLER IS BUILT

Two remaining regimes account for essentially all of the gap to InVirtuoGen.
Everything else on the 30-cell table is at or inside the 0.70 kcal/mol docking
noise band and is NOT a development target.

## Regime A — deep constructive growth / delayed credit

The deficit cells, all with the same mechanism (route audit section 9):

| cell | gap to IVG |
|---|---|
| parp1 s1 d0.4 | +3.9 |
| parp1 s0 d0.4 | +3.6 |
| parp1 s0 d0.6 | +3.5 |
| 5ht1b s7 d0.6 | +3.5 |
| parp1 s1 d0.6 | +2.5 |
| 5ht1b s7 d0.4 | +1.8 |

**DEVELOP ON (winners already inspected, contaminated):**
    parp1 s0 d0.4,  5ht1b s7 d0.4

**HELD OUT — their winners must NOT be inspected:**
    parp1 s1 d0.4,  parp1 s1 d0.6,  parp1 s0 d0.6,  5ht1b s7 d0.6

The held-out set is WITHIN the same regime, which is the point: if a segment-
option controller developed on two landmarks also fixes four cells whose answers
were never opened, that is evidence the generic bridge problem is solved rather
than two landmarks fitted.

## Regime B — repair from an initially infeasible seed

Five uncovered cells, seeds starting below QED 0.6:

    braf s9 d0.4,  braf s9 d0.6,  braf s10 d0.6,  braf s11 d0.6,  5ht1b s8 d0.6

**HELD OUT ENTIRELY.** No winner inspection. Success criterion is coverage:
feasible candidates produced at all, not a score margin.

## NOT development targets

5ht1b s6 d0.4 (+1.0), jak2 s13 (+0.8/+1.0), jak2 s12 (+0.3/+0.4),
fa7 s4 (+0.1/+0.3). At or near docking noise; fixing the two regimes above
changes the aggregate without touching these.

## Done criteria, before any 30-cell run

1. parp1 s0 d0.4 and 5ht1b s7 d0.4 stop cycling; segment options produce deep
   constructive trajectories.
2. The SAME FROZEN controller improves the four held-out Regime-A cells.
3. braf d0.6 produces feasible candidates at all.

Then one clean 30-cell run from call zero under a single frozen configuration.

---

## PRECOMMITTED zero-oracle gate criteria (recorded BEFORE the run reported)

The mechanism gate PASSES only if ALL of the following hold across the full
40-edit budget on the two development cells:

1. grow / shrink / local stay DIRECTIONALLY SEPARATED for the whole budget.
2. the unstructured `mixed` control continues to show substantially more
   insert<->delete cancellation than `grow`.
3. `construct_ring` actually COMPLETES ring-building sequences -- net rings
   formed -- rather than merely surviving on its protected quota.
4. deep feasible states keep appearing; the population does not snap back
   shallow.
5. option survival is maintained.
6. LINEAGE DIVERSITY does not collapse. ESS is allowed to concentrate -- that is
   what SMC does when the potential is informative -- but if roughly 3-5
   ancestors end up generating essentially the whole frontier, the gate is
   INCOMPLETE regardless of how good displacement looks.

### Decision tree, fixed in advance

* **Pass + diversity healthy** -> run the 100-200 call docking test on the two
  registered development cells immediately.
* **Pass structurally but lineage collapses** -> fix ONLY the particle
  approximation (lineage-aware resampling / rejuvenation / per-ancestor
  offspring caps within each protected option). Do NOT touch the objective: it
  has finally demonstrated the right behaviour. Rerun the cheap gate, then dock.
* **Fails to sustain constructive or ring progress** -> inspect the OPTION
  GRAMMAR. Not another scalar weight.

### Docking bar before anything is frozen

A real improvement means clearly outside the 0.70 kcal/mol noise scale --
ideally >= 1 kcal/mol on at least one of the big-loss growth cells. On meeting
that bar the controller is FROZEN IMMEDIATELY and run on the held-out Regime-A
cells (parp1 s1 d0.4/d0.6, parp1 s0 d0.6, 5ht1b s7 d0.6) whose InVirtuoGen
winners were never inspected. That held-out step is what makes this science
rather than development.
