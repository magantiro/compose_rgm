# Target-free Pareto control — held-in smoke result

12 sources, `potency_vs_developability` adopted by a 60-source geometry census,
K=8, held-out never opened. **Every instrument was frozen before these numbers
existed**, timestamps verified against zero `DONE` lines at commit time.
Lane branch `codex/compose-pareto-control`, result at `7deee51`.

## 1. Preference responsiveness — real, and imperfectly ordered

**Distinctness is unambiguous.** 3.83 of 5 distinct endpoints, **zero** sources
where all five preferences collapse to one molecule, and **P6 returns 3.75/5
distinct endpoints from a byte-identical branch point** with every branch
verified to start there.

**Ordering is weaker, and this is the honest limitation of the result.** Monotone
on only **1 source in 6**. Guided ρ(w, objective) = **+0.636** sits above the
preference-blind floor of **+0.432** without dominating it, and that floor is
noisy at n=12.

> Preferences produce **different futures, loosely ordered** by the requested
> tradeoff. **Five cleanly ordered Pareto regions is NOT supported by this
> smoke.**

Distinct SMILES was never the test — ordering was — and ordering came back
partial. Say so.

## 2. Held-in-scaled hypervolume

| arm | HV |
|---|---:|
| `verified_pref` | **0.938** |
| `greedy_pref` | 0.849 |
| `gen_rank@verified` | 0.308 |
| `unguided` | 0.191 |

12.7% of endpoints exceed `z*` (max excess 0.909), reported descriptively and
**unclipped** — `z*` is the held-in p99, a normalizer and not a cap.

## 3. Per-preference scalarized value — GUARANTEED SIGN, magnitude only

Mean **+0.1431**, 12W/0L. **The 12-0 is definitional**: greedy's action is
always in the shortlist and strict improvement never commits a lower `V_G`.
Reported as a magnitude and never as evidence.

## 4. Set-level hypervolume — the contrast that could have failed

Mean **+0.0890**, CI **[+0.0379, +0.1449]**, **10W/2L**.

The sign was free. Sources 000 (−0.001) and 010 (−0.073) went the other way, and
a registry keyed on the contrast alone had suppressed this comparison entirely
until it was rekeyed on (contrast, metric).

> **Verified control improves the endpoint SET, not merely each trajectory
> individually.** Pointwise improvement does not imply set-level improvement —
> five individually better points can enclose less dominated area — so this is a
> real result precisely because it could have come out negative.

**This contrast is NOT a pass condition for Pareto.** Exact-target recovery
already establishes that future-aware reasoning can matter; Pareto's job is
target-free reuse across preferences. If a larger development finds that greedy
preference control already covers the front and verified improves individual
scalarized outcomes without improving the set, that is a coherent scientific
outcome and not a failure. Figure 5 does not carry a second "planning beats
greedy" claim unless the data volunteer one.

## 5. Resources — three axes, never merged

`N_90`, trajectories to reach 90% of **pooled attainable** HV:

| arm | N_90 | censored |
|---|---:|---:|
| `verified_pref` | **2.00** | **0/12** |
| `greedy_pref` | 2.00 | **5/12** |
| `unguided` | — | **12/12 never reached** |
| both `gen_rank` arms | — | **12/12 never reached** |

Region coverage: verified 0.433, greedy 0.367, unguided 0.150.

Censored sources are reported as censored, never assigned the maximum budget.

**Cost is real and unflattering:** 249,640 unique evaluations and 393 kernel
calls per source; 612.2 unique evaluations per kernel call against a census
fiber width of 586, so the counter is sound. The ratio against `gen_rank`
remains **withdrawn as a cost claim** pending a matched comparison.

## Watch-items — the live risks did not materialise

W1 1.00 · W2 last productive step 5.0/6, no early stall · W3 spread ratio 0.427,
neither axis flat · W4 3.83/5 distinct, none collapsed · W5 1.00, no saturation
stall. The census's G3 binding share 0.76 and G4 potency reach 0.700 were logged
as live risks; **neither bit**.

## What this smoke CANNOT say

**P3 and P4 are withheld** — kernel ratios 1.492 and 2.388 against a 1.25 limit.
The matcher assumed 6 fresh kernel calls per trajectory, but unguided
trajectories from a shared root collide in the enumeration cache, so
`gen_rank` was **underfunded** — an error in the direction that flatters us.

> **`gen_rank`'s poor numbers must NOT be read as defeating that baseline.**

There is no admissible closed-loop versus generate-and-rank comparison in this
smoke. Fixing it means iterating the matcher until the ledger reaches target and
topping up `gen_rank` trajectories — additive, not a rerun of the COMPOSE arms.
