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

| | mean | median | 95% CI | W/L/tied |
|---|---:|---:|---|---|
| `Δ^G` greedy parity | −0.105 | **0.000** | [−0.377, +0.099] | 5/8/11 |
| `Δ^V` verified parity | −0.023 | **0.000** | [−0.182, +0.134] | 9/10/5 |

Both medians are exactly zero and both intervals span zero. The negative means
come from a few sources, not a systematic tax.

**An interval spanning zero is not evidence of equivalence.** It says the data
do not resolve the sign; it does not say the cost is small. `Δ^G` in particular
still permits a potency loss of −0.377 held-in IQR units, which would be a
material penalty. This section therefore reads as *unresolved*, not as *free*,
and the confirmation replaces it with a noninferiority test against a margin
fixed in advance — see `docs/PATHWISE_HELDOUT_CONFIRMATION_PREREGISTRATION.md`.

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

## The claim this licenses NOW, and the one it does not

Banked, as the development record:

> Pathwise enforcement eliminated a common hidden-trajectory failure mode, while
> development showed **no median potency loss and only a small estimated mean
> penalty** under verified control.

**Not** banked, and barred until the held-out noninferiority test clears:

> ~~"…at no cost."~~ / ~~"…essentially free."~~ / ~~"…costs no terminal
> objective."~~

The distinction is the whole reason the confirmation exists. Zero median plus a
wide interval motivates a noninferiority test; it does not substitute for one.

Also quotable, from the descriptive arm: under pure potency optimisation only
**17 of 24** endpoints land in the corridor at all — endpoint-only filtering
discards nearly a third of the work before hidden excursions even arise.

## DECIDED — this earns a held-out confirmation, and figure space is conditional

Passing a gate earns interpretation, not figure space. The bar set in advance
was whether the result is *memorable on its own terms*. The prevalence finding
clears it:

> Endpoint-only inspection misses forbidden intermediate states on roughly
> 60–70% of sources even though the final molecule looks acceptable — and A2
> measured those excursions at median depth 0.69 cLogP and median duration 4 of
> 6 steps, so they are not boundary jitter.

Against that: it is developmental, and the corridor family was selected *after*
the feasibility screen. So the confirmation is authorised and figure space is
decided by its outcome, not here:

| held-out outcome | consequence |
|---|---|
| prevalence replicates **and** cost clears −δ | **main Figure 6** |
| prevalence replicates, cost does not clear −δ | secondary; a main-text paragraph reporting the cost honestly |
| prevalence collapses | supplement; the pillar is not forced |

Design frozen in `docs/PATHWISE_HELDOUT_CONFIRMATION_PREREGISTRATION.md`.
**Pathwise remains an orthogonal capstone, not a fourth co-equal pillar** — the
main escalation stays known target → changed target → no target.
