# Scale is cheap, coherence is not — and competence falls off at the scale boundary

12 witness-segment sources, matched RNG seeds, only `max_modules` differs (3 = production,
8 = the cap). Zero oracle calls. PMO kernel rdkit 2023.09.6.

| required K | best `simmax_ALL` | exact witness recoveries | best `simmax_LARGE` 3→8 |
|---|---:|---:|---:|
| **≤14** (n=6) | **1.000** | **2** | 0.324 → 0.458 |
| **≥17** (n=6) | 0.277 | **0** | 0.159 → 0.203 |

**The generator solves small macros — sometimes exactly — and cannot touch large ones.**
Two segments (`init_04` K=8, `init_02` K=9) are recovered at similarity 1.000, the witness
endpoint itself. At K>=17, across **333 large proposals at mods=8**, the best is 0.277 and
nothing is exact.

That dose-response kills the two remaining alternative explanations: the probe is not broken
(it registers exact hits where they exist) and the metric is not wrong (the same metric reads
1.000 and 0.13 from the same generator on different segments).

**The module budget is inert on coherence in BOTH regimes.** `P(K>=17)` rises ~4x everywhere
(0.015-0.031 -> 0.053-0.086) while median `simmax_LARGE` moves 0.273 -> 0.260 on small
segments and 0.125 -> 0.124 on large ones.

**Raising the budget can COST an exact solution.** `init_02` K=9 drops `simmax_ALL`
1.000 -> 0.483 at mods=8: mass shifts toward larger programs that overshoot an 8-primitive
target. Bigger is not free when the required macro is small.

Executable yield is 1.000 in all 24 arms; distinct endpoints rise; QED costs ~0.02-0.04.

## The statement

COMPOSE solves macros up to ~14 primitives, sometimes exactly. Above ~17 it reaches the
required scale on 1-2% of draws, and the large programs it does produce are structurally
unrelated to what is needed. The gap is not horizon, execution, validity, or module count --
it is **coherent connected-region replacement at scale**: the right NUMBER of regions (2 vs 2)
at 7 atoms where 19 is needed, retaining 64% of the source where 30% is required.
