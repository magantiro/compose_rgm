# Comparator scope-lock: every block, plus the Pareto gate

**Canonical. Every claim-bearing experiment must have an explicit comparator
rationale recorded BEFORE launch.**

## The policy, corrected

The slogan "published-number-first" was too absolute. Strong papers use a **mix**:
Edit Flows trains its *own* AR and Mask-DFM baselines under common
architecture/compute for its principal comparisons and quotes prior numbers
alongside, clearly marked as such; TD3B ships runnable implementations of CG, SMC,
TDS, PepTune and unguided baselines. **"Never run another model" was never the
lesson.**

> **Published-number-first when there is exact protocol alignment; official-native
> rerun when a load-bearing comparison requires matched conditions unavailable in
> published results; otherwise contextual only.**
>
> **Do not reconstruct a method solely to increase baseline count.**

Applied: **DDSBM** — do not rerun, exact benchmark alignment makes published
numbers ideal. **Pareto** — after the final controller freezes, a bounded attempt
at 1–2 strong external numerical comparisons; published numbers if exactly
aligned, else official native code through a thin adapter. **Novel capabilities**
— do not manufacture an external baseline when the matched internal counterfactual
answers the question more directly.

## The non-Pareto comparison map

| block | question | primary comparator | external numerical? |
|---|---|---|---|
| **reference law** | did `R_θ` learn useful molecular dynamics? | uniform canonical, empirical-family, learned-family/uniform-ID, empirical-family/learned-ID | **none needed** |
| **reference dynamics** | what behaviour does learned `R_θ` induce? | uniform + empirical-family processes | **none needed** |
| **conventional editing/transport** | can frozen COMPOSE solve an established external task? | **DDSBM** | **yes — published values** |
| **exact-target control** | does finite-horizon reasoning improve hard target attainment? | greedy, verified, similarity controls, `R_θ`-top1 | **no natural external task exists** |
| **dynamic retargeting** | is the realized state useful when the objective changes? | continue-from-`x₃`, restart-from-`x₀`, clairvoyant | **matched causal is better** |
| **hard support** | does enforcing admissibility during generation beat generate-then-filter? | exact support vs identical unconstrained controller + post-hoc filter | CDD/PRODIGY/ConStruct as **lineage** |
| **pathwise** | does constraining the *route* matter beyond endpoint admissibility? | endpoint-only vs pathwise-constrained | **matched causal** |
| **validity / edit expressivity** | variable-size restructuring with every state valid? | construction + operator-use characterization | external validity where directly comparable |
| **map reuse** | can prior search effort be reused for a later request? | best mapped state vs restart from `x₀` | **matched causal** |

**This is not weak baseline coverage.** For several blocks an unrelated external
model would be a *worse* control. `x₃ →ᴮ ⋯` versus `x₀ →ᴮ ⋯` directly answers
whether prior molecular progress has value; inserting DDSBM or GraphXForm there
would isolate nothing. Same for endpoint-only versus pathwise on the *same*
process — that is close to an ideal causal experiment.

### Established-task blocks inherit the benchmark's own metrics

For DDSBM we use **benchmark-native** metrics — logP `W₁`, QED MAD, SA MAD,
validity — never a bespoke COMPOSE score, and cite DDSBM's published values for
those same metrics. FCD and NSPDK are omitted because we are not claiming a full
reproduction of their distribution-modelling table, and **that selection was
frozen before any COMPOSE result existed**, which is what stops it being
cherry-picking.

If any of dataset / split / oracle / budget / metric differs materially, it is
**not** a numerical head-to-head.

### New-capability blocks earn rigor differently

There is often no published benchmark because **the capability is part of what
COMPOSE introduces**. Rigor then comes from: freezing the task before outcomes; a
metric that can genuinely fail; the nearest matched causal counterfactual;
held-out sources; source-level uncertainty; and **no rescue of a negative
result**. That is stronger than borrowing a vaguely related published metric.

---

# The Pareto gate

**Required. Ordering is binding.**

```
P0c resolves  →  final Pareto controller frozen  →  THIS MATRIX FROZEN  →  fresh panel launches
```

**DDSBM does not fill this slot.** It covers conventional *source-conditioned
endpoint transformation*. The Pareto block needs its own external comparator
analysis, and freezing the matrix **before** the panel launches is what stops a
comparator being chosen after our numbers exist.

## The eight methods

For each, record: **native task · oracle · budget · metrics · official-code
status · published-number compatibility · role.**

Role is one of `NUMERICAL` / `PUBLISHED_CONTEXT` / `CONCEPTUAL_LINEAGE`.

| method | already audited? |
|---|---|
| **HN-GFN** | yes — GPU-only, no released checkpoint, empty `BlockMoleculeDataExtended()` start, surrogate-budget asymmetry |
| **InversionGNN** | yes — not runnable as shipped, no LICENSE, 70× budget disagreement with HN-GFN |
| **OP-GFN** | yes — excluded on CC BY-NC-**ND** |
| **MOG-DFM** | yes — conceptual lineage |
| **AReUReDi** | yes — the only same-lab numerical candidate, gated on native/thin/author-validated |
| **pCoMole** | **`UNVERIFIED`** — OpenReview not machine-reachable, not on arXiv. A human task |
| **ParetoFlow** | **NEW — not yet audited** |
| **A-GPS** | **NEW — not yet audited** |

Six carry prior findings that should be **reused, not re-derived**. Two are new
work. `pCoMole` still needs a login someone has and the lane does not.

## Target

**AMENDED.** Include **every** method that achieves tier-1 exact alignment as a
*reported* numerical row — one COMPOSE run on a shared benchmark can buy several
legitimate rows at no retraining cost. **Cap only reruns: at most 1–2**
official-native reruns, and only where published numbers are unavailable and a
thin adapter suffices. Partial alignment stays contextual. Standards are never
lowered to add a row.

**Barred:** manufacturing ports; mixing incompatible published protocols. The
70× HN-GFN budget disagreement is the standing example — two papers reporting the
same method under protocols that differ by seventy times in oracle calls, where
quoting either beside ours would have been wrong.

**One or zero numerical comparators is an acceptable outcome.** The Pareto block
already carries a fair matched causal baseline in repaired P3/P4, which tests the
claim more directly than any external row: at matched kernel work, closed-loop
control beat fairly funded generate-and-rank **12/12** on hypervolume.

## Policy, unchanged

**Published-number-first.** Where an established benchmark matches exactly, run
COMPOSE alone and cite the authors' reported numbers.

**Novel COMPOSE capabilities keep using matched causal counterfactuals**, because
those test the claim more directly than an external leaderboard row. Retargeting,
pathwise control and front construction have no external method instantiating the
same question — that is why the internal contrast is primary there and not a
fallback.
