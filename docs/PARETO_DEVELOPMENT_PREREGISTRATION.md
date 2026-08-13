# Target-free Pareto control — development preregistration

**Status: DESIGN ONLY. NOT LAUNCHED.** Written after the held-in smoke
(`docs/PARETO_CONTROL_SMOKE_RESULT.md`, `7deee51`) and before any larger run.
The four questions, the instrumentation fix list, and the escalation decision
rule are fixed here.

## The framing this smoke licenses, and the one it does not

**Barred from Figure 5:**

> ~~"COMPOSE precisely sweeps the Pareto frontier according to user
> preference."~~

The smoke does not support it. Ordering was monotone on 1 source in 6, and
guided ρ = +0.636 sits above the preference-blind floor of +0.432 without
dominating it.

**Provisional caption, until a larger development says otherwise:**

> The same frozen molecular process can be recontrolled across target-free
> property preferences, producing a diverse Pareto set with **measurable
> preference responsiveness**.

"Measurable" is upgraded only by data, not by a better sentence.

**Pareto is not required to re-prove that planning beats greedy.** Exact-target
recovery already carries that. Pareto's job is target-free reuse across
preferences. `HV_verified − HV_greedy` is reported as a real falsifiable
contrast that may come out positive, zero, or negative — it is **not** a pass
condition.

---

## REVISED HIERARCHY — P3 is elevated to a primary conceptual comparison

The smoke's hierarchy under-ranked P3, partly because it was inadmissible at the
time. Inadmissibility was a **budget-matcher defect, not a weak question.**

COMPOSE's claim is not merely that changing a preference produces different
molecules. It is that **controlling the evolving molecular state online is
useful compared with generating candidates and ranking them afterward.** That is
precisely what P3 tests, and it is the question a reviewer will actually ask:

> Why do I need this whole control formalism? Why not generate a batch from
> `R_θ`, score them, and keep the Pareto set?

Architecturally the two differ in exactly one place:

```
generate-and-rank:  legality → plausibility → generate → purpose ONLY AT THE END
COMPOSE:            legality → plausibility → purpose FED BACK AT EVERY STATE
```

P3 asks whether that last box does anything. **It is capable of returning "the
closed-loop structure doesn't buy much here", and that falsifying possibility is
why it matters.**

| rank | question | what it establishes |
|---|---|---|
| **A** | **Q1 (P5/P6)** | *capability* — the same frozen process can be recontrolled under different target-free preferences, and those preferences produce meaningfully different futures. Without this, "Pareto control" is not real |
| **B** | **Q3 (P3)** | *structural advantage* — does closed-loop control of the evolving molecule beat a **fairly funded** generate-then-rank? Arguably more important than P2 for Figure 5 |
| **C** | **Q2 / P4** | *increment* — does richer future-aware control add more on top? Less load-bearing, because exact-target recovery already proves future-aware reasoning can matter |
| **D** | **Q2 frontier quality** | HV, **HV-AUC**, coverage, nondominated behaviour, preference coverage |

## Sequencing: P3/P4 are repaired BEFORE the larger development launches

**Do not launch the larger development with these comparisons still broken.**

1. Fix the generate-and-rank budget matcher.
2. Top up **only** the missing baseline computation, on the 12 existing smoke
   sources.
3. Recompute P3/P4 from the existing COMPOSE artifacts plus the corrected
   baseline. **Touch no COMPOSE outcome.**
4. Determine whether the repaired comparisons are meaningful under the frozen
   resource convention.
5. Then launch.

**P5 is not rerun.** It was never inadmissible — it is a valid PRIMARY
preference-responsiveness contrast and its result stands. Only P3 and P4 were
barred, and only because `gen_rank` was underfunded in the direction that
flatters COMPOSE. The correct response is not to discard valid COMPOSE
trajectories; it is to finish the comparator fairly.

### There is probably no single universal P3

Closed-loop control and generate-and-rank consume resources in structurally
different ways, so **one arbitrary matching convention must not decide the
story.** Report a **resource frontier**:

```
HV_closed-loop(B)   vs   HV_generate-and-rank(B)
```

under, separately: **unique expensive objective evaluations**;
**kernel / generative work**; **completed trajectories**; and wall compute as a
clearly secondary axis.

A plausible and fully reportable outcome is: *closed-loop control reaches much
better HV per complete trajectory, but uses considerably more objective
queries.* That is neither "P3 wins" nor "P3 loses" — it is the real structural
comparison. **If the P3 conclusion changes depending on which resource axis is
used, say so explicitly** rather than hiding it behind one budget number.

That axis is also exactly where `R_θ` plausibility shortlisting would later be
expected to improve things — which is why shortlisting is gated on this result
rather than run alongside it.

---

## The four frozen questions

### P1 — Preference responsiveness

Does increasing the potency preference systematically move outcomes toward
potency relative to developability? Uses the **already-frozen ordering/rank
statistic**, against the **preference-blind floor**, which is what makes it a
measurement rather than a description.

This is question one because **five different SMILES is not preference
control.** Distinctness was never the test.

### P2 — Frontier quality

Across the five preference-directed endpoints, per source and then aggregated
with the source as the independent unit: **hypervolume; coverage; nondominated
point count; HV-AUC.** Frozen nadir (held-in p5) and utopia (held-in p99); `z*`
remains a normalizer and never a cap, so endpoints that exceed it are reported
unclipped.

### P3 — Does future-aware control improve the *set*?

`HV_verified − HV_greedy`, source-clustered paired bootstrap. Declared in
advance as a contrast whose sign is free, and explicitly **not** a gate. The
smoke's +0.0890 [+0.0379, +0.1449] is the development value; two of twelve
sources went the other way.

Kept distinct from the per-preference scalarized contrast, whose 12W/0L **is
definitional** (greedy's action is always in the shortlist and strict
improvement never commits a lower `V_G`) and is reported as a magnitude only.

### P4 — What does control cost?

Three axes, **never merged into a universal efficiency number**:

```
HV  vs  complete trajectories
HV  vs  unique potency evaluations
HV  vs  kernel calls
```

Censored sources are reported as censored and never assigned the maximum budget.

---

## Instrumentation fixes — and nothing else

**The controller is not redesigned.** Only the three defects the smoke itself
demonstrated are repaired.

### 1. Exact batching, gated on identical trajectories

`objective_vector()` calls `oracle.margin_many([key])` with a list of exactly
one molecule. Measured locally, per molecule:

| step, as currently run | µs |
|---|---:|
| `oracle.margin_many([key])` at **batch 1** | 1258 |
| `QED.qed` | 1158 |
| `Crippen.MolLogP` | 275 |
| `MolFromSmiles` ×2 (the oracle re-parses internally) | 200 |
| Morgan FP ×2 + Tanimoto | 68 |

Batched, the SVM call drops to **303 µs (4.2×)**.

**Honest accounting of what that buys.** At 249,640 unique evaluations per
source, scoring is ~12 min of a ~58 min source; the 393 kernel enumerations at
~7 s each are ~46 min. So batching is **~7% of wall time**. It is required for
*counter honesty*, not for speed, and must not be sold as the optimization.

Two further free wins, both gated: `QED.qed` already computes Crippen ALOGP
internally, so `QED.properties(mol)` yields both and removes the separate
`MolLogP` call — **only** if ALOGP is asserted bit-identical, since otherwise
trajectories move. And `objective_vector` computes all three of P/D/S while
returning two.

Every change here is gated on a regression test asserting **identical
trajectories, scores, actions and `N_drd2`** against the smoke.

### 2. Separate counters

`drd2_batch_size: 1` and a shared call site currently make `N_drd2 ==
N_descriptor == N_all` by accident rather than by measurement. Split at their
own call sites, before any cost claim:

```
N_drd2_requests      N_drd2_unique      N_drd2_evaluator_batches
N_descriptor_requests                   N_scorer_batches
N_kernel_calls                          N_complete_trajectories
```

### 3. Fix the generate-and-rank matcher

P3/P4 of the smoke are inadmissible (kernel ratios 1.492 and 2.388 against a
1.25 limit) because the matcher assumed 6 fresh kernel calls per trajectory
while unguided trajectories from a shared root collide in the enumeration cache.
`gen_rank` was **underfunded in the direction that flatters COMPOSE**.

Fix per the already-frozen resource convention: iterate the matcher until the
ledger reaches target, and **top up only the missing `gen_rank` computation**.

> **Do not change COMPOSE because the baseline was underfunded**, and do not
> read the smoke's `gen_rank` numbers as defeating that baseline.

---

## Escalation: three levels, and the rule for reaching each

The current experiment is the right first experiment for proving target-free
preference control. It is **not** necessarily the optimal COMPOSE algorithm for
deliberately sweeping a frontier, and the distinction is worth preserving rather
than blurring.

For five fixed preferences it independently optimizes `U_{w_i}(x_H^{(i)})` per
trajectory. It does **not** optimize `HV({x_H^{(1)},…,x_H^{(5)}})`. That is the
first-principles reason perfect frontier sweeping should not be expected
automatically — and the smoke already shows the gap is real, since verified
control can improve an individual preference trajectory while reducing set
hypervolume.

| level | what it is | status |
|---|---|---|
| **1 — fixed preference control** | five predetermined weights, current design | **primary. Stays primary.** |
| **2 — aspiration / reference-point control** | five predeclared *property-space* targets `a_1…a_5`; terminal desirability from normalized shortfall (max-min / Chebyshev, matching the frozen goal algebra). Still target-**free**: these are regions of property space, not target molecules | conditional, not built |
| **3 — archive-aware outer control** | choose the next preference to maximise expected marginal hypervolume against the current archive | **not built, and probably beyond this paper** |

**The rule, fixed before the development reports:**

| development outcome | consequence |
|---|---|
| preference response **and** HV both strong | keep the controller. The one scaling follow-up is efficient reference-based shortlisting |
| HV strong, ordering still mediocre | **one** aspiration-point experiment (Level 2), preregistered as a distinct cleaner formulation of frontier targeting |
| both weak | **demote Pareto.** Do not build machinery to rescue it |
| fixed preferences cluster badly *despite* strong individual optimization | only then consider Level 3, as a motivated extension — never as a rescue |

Level 2 must be motivated as a cleaner formulation and preregistered before
running. It may **not** be introduced because a particular source or an n=12
result is annoying. Level 3 competes directly with archive-aware multiobjective
methods on their home turf and makes the paper more complicated; that is a cost
to pay deliberately or not at all.

---

## The hero visualization is the same-state fan

The quantitative experiment starts from a source. The **figure** should not.

```
x_0 → x_1 → x_2 → [x_3]  ──w_1──▶ …
                  [x_3]  ──w_2──▶ …
                  [x_3]  ──w_3──▶ …
```

One realized molecular history, branched five ways from a **byte-identical**
branch point. The smoke already supports this: P6 returned 3.75/5 distinct
endpoints from a verified-identical branch point.

> This exact molecular history already exists. Without changing the learned
> model, its future can be branched according to different desired tradeoffs.

HN-GFN can generate a Pareto set; pCoMole does preference-conditioned sequence
editing. Same-state branching is what fits *this* object, and it connects
Figure 4 to Figure 5 instead of letting Pareto read as an unrelated benchmark.

## What is explicitly NOT the goal

> ~~five weights → five perfectly monotonically ordered points~~

Molecules are discrete and the reachable front is irregular; adjacent weights
may legitimately map into the same basin. ρ ≈ 0.6 can be a respectable result if
the front is discrete, endpoints span the actual tradeoff, HV is strong, and the
preference-blind floor is substantially weaker. What matters is the guided-versus-blind
gap, spread across meaningful objective regions, nondominated coverage, whether
*extreme* preferences move endpoints in the expected direction, and how many
trajectories buy useful coverage.

## Two things this unlocks, in order

**The reference-law ablation now definitely runs here.** Pareto passed its
task-geometry gate and the smoke shows actual target-free behaviour, so the
preregistered primary host is live:
`docs/REFERENCE_LAW_ABLATION_PREREGISTRATION.md`. It may matter more than P3,
because it answers why learn molecular plausibility at all if control supplies
purpose.

**Then, and only if the capability is established: shortlisting.** Control
currently asks ~586 successors for their objective values at nearly every
decision — the brute-force form. The architecture suggests

```
legal support → plausibility shortlist via R_θ → expensive goal evaluation → control
```

i.e. legality → plausibility → purpose. If that retains most frontier quality at
a fraction of the objective evaluations, it is the distinctive Pareto story. It
is a **scaling follow-up gated on full-fiber control proving the capability
first**, not a substitute for proving it.
