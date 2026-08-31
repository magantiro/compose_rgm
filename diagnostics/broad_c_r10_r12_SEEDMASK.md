# broad-C runs r10/r11/r12 — EXPLORATORY SEED-MASK DIAGNOSTIC

Cell 5ht1b_s7_d0.6. Launched 2026-08-27 02:48, stopped deliberately at round 0-1.
Credit bookkeeping was CORRECT in these runs (only executed requests credited).
They are preserved for one measurement they made cleanly.

## What they measured

The x0-frozen support mask is MATERIAL, not a theoretical caveat:

    CREDIT   requests sampled 145, credited 10  (7% reached credit)

    FUNNEL (r10)  linked  sampled 7  -> executed 5 -> feasible 1 -> docked 1
                  fused   sampled 3  -> executed 2 -> feasible 2 -> docked 2
    FUNNEL (r11)  linked  sampled 2  -> executed 1 -> feasible 1 -> docked 1
                  fused   sampled 10 -> executed 5 -> feasible 4 -> docked 4

Execution rate at the eventual state is ~50-70%: the semantic tuple was drawn
from what was legal at the SEED, then had to execute at a state several edits
away. Combined with feasibility and docking attrition, only ~7% of draws ever
produce a learning signal. Against a ~47-class semantic space that is far too
sparse to fit.

## Also observed (correcting an earlier reading)

Fused REACHED DOCKED ENDPOINTS here (2 and 4). In the earlier credit-bugged
runs (r20-22) fused reached zero docked endpoints while P(fused) climbed to
0.959. That asymmetry is therefore at least partly an artifact of the
contaminated credit, not a property of fused chemistry.

## What these runs may NOT be used for

Any claim about learned topology or about broad-vs-narrow docking performance.
They were stopped at round 0-1 and the sampling was still seed-conditioned.

Artifacts: volume compose-v4-artifacts,
macro_basin/episodes_5ht1b_s7_d0.6_pooled_broad_r{10,11,12}.json
