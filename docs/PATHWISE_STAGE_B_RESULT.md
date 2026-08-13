# Pathwise control — Stage B result

**Status: DEVELOPMENTAL.** The cLogP corridor family emerged from the
feasibility screen rather than being specified in advance, so this needs its own
held-out confirmation before it can be stated as a result. That caveat was
committed at `abab8ad`, *before* the run. Lane 2 branch
`codex/compose-pathwise-constraints`, result at `bcc4a40`.

## The premise, which the ring motif failed and this passes

Endpoint-only filtering can return a molecule that satisfies a requirement at
the end while having **violated it along the way**. The ring-system experiment
found that premise empirically false — motif destruction was absorbing, so
endpoint validity implied path validity and the distinction did not exist. The
cLogP corridor is different.

## PRIMARY — hidden-path rate

`P(x_H ∈ C AND ∃t<H : x_t ∉ C)`, over every eligible source, source-clustered.

| arm | | rate | 95% CI |
|---|---:|---:|---|
| `endpoint_greedy` | 16 / 24 | **0.667** | [0.458, 0.833] |
| `endpoint_verified` | 14 / 24 | **0.583** | [0.375, 0.792] |

**Two-thirds of sources produce an endpoint that passes inspection after
traversing a forbidden state.** The estimand had genuine room to be zero — the
ring motif returned exactly zero — and it isn't.

Secondary conditional fraction 16/24 and 14/23. Denominators nearly full, so the
small-denominator pathology the unconditional primary guards against did not
materialise. The guard was still correct to have.

## TERMINAL COST — controller parity, sign free

| | mean | median | 95% CI |
|---|---:|---:|---|
| `Δ^G` greedy parity | −0.105 | **0.000** | [−0.377, +0.099] |
| `Δ^V` verified parity | −0.023 | **0.000** | [−0.182, +0.134] |

**Both medians exactly zero, both intervals spanning zero.** This is the first
of the three predeclared outcomes — little or no potency cost. The negative
means come from a few sources, not a systematic tax.

## Support viability

1 of 24 support-tight at the frozen 0.10 threshold; per-source median retention
**0.798**; **zero** mask-empty states; one terminal failure. The predeclared
sensitivity analysis excluding the tight source gives rates 0.652 / 0.565 and
`Δ^G` −0.121, `Δ^V` −0.041 — **conclusions unchanged**.

## Barred from the results

- **Zero violations on the pathwise arms** is definitional, not a finding.
  Recorded as `mask_integrity: PASS`, never with a denominator.
- **Future-aware under the mask**: effect size +0.661 [0.428, 0.944], no sign
  test. Binary headroom is 0 over a denominator of 0 — a **CEILING, not a
  null**, since `pathwise_greedy` completed 24/24.

## The claim this would license, if confirmed

> Endpoint-only filtering accepts molecules that reached an acceptable state by
> an unacceptable route in two thirds of cases, and enforcing the requirement
> throughout the trajectory costs no terminal objective on the median source.

Also quotable, from the descriptive arm: under pure potency optimisation only
**17 of 24** endpoints land in the corridor at all — endpoint-only filtering
discards nearly a third of the work before hidden excursions even arise.

## OPEN DECISION — does this earn a main-text figure?

Passing a gate earns interpretation, not figure space. The bar set in advance
was whether the result is *memorable on its own terms*. Two-thirds hidden-path
rate at zero median cost plausibly clears it. Against that: it is developmental,
the corridor was screen-selected, and the paper's spine — known target, target
changes, no target — is already complete without it.

Not decided here.
