# Lazy family-first sampler — the transition-level result, and what it obsoletes

**Same learned stochastic law, same exact chemistry, ~79 ms instead of seconds.**

Banked as the end of transition-level optimization. Nothing below is projected.

## The measurement

| metric | eager | lazy |
|---|---:|---:|
| median fresh transition | 5376–7665 ms | **78.6 ms** |
| mean | — | 91.1 ms |
| p90 | — | 134.0 ms |
| p95 | — | 173.2 ms |
| p99 | — | 391.7 ms |
| max | — | 403.6 ms |
| RDKit/semantic resolver calls per draw | ~1300 | **0.37** |
| family redraws per draw | n/a | 0.02 |
| coordinate rejections per draw | n/a | 0.17 |
| eager fallback fraction | n/a | **0.0%** |
| returned a valid edit | 100% | **100%** |

The eager median measured 5025, 7665 and 5376 ms across three runs on different
containers and state samples, so the speedup is a RANGE of roughly **65–90×**,
not a single number. The lazy side is stable at 78–85 ms.

Time split of a fresh transition:

    encode          55.5 ms   61%
    family masks    28.0 ms   31%
    resolver         3.0 ms    4%
    family draw      0.53 ms   1%

## What changed, conceptually

Nothing was made faster. The eager path computes the legal support of EVERY
operator family and then samples one:

    prepare all 8 families  ->  choose one

Cycle-close is 56.5% of that construction and carries 0.53% of realized family
mass, so it was being built roughly 200 times for every draw that used it. The
lazy path inverts the order:

    choose one  ->  prepare that one

Same model, same operator probabilities, same legal chemistry, same stochastic
transition law.

## Qualification

| gate | result |
|---|---|
| raw-head parity, 9 scorers | bit-identical, on and off support, 16 states |
| family base logits | bit-identical, 0.05 ms |
| superset containment | verified per family |
| analytic full-law equality | 1.8e-15 over 15,179 marks |
| execution parity | **8 families × 6 coordinates = 48/48** identical successors |

The eight families exercised for execution parity were `atom_delete`,
`atom_insert`, `atom_restate`, `bond_reorder`, `bond_reroute`, `cycle_attach`,
`cycle_insert` and `ring_system_restate`. The remaining three tables --
`grow_root`, `ring_system_grow`, `ring_system_delete` -- were NOT exercised
because they carry no legal marks on any test state; they are the always-empty
families, and a family with no legal action can never be sampled.

Two bugs the gates caught, both of which would have shipped a finite,
normalized, WRONG law:

* `bond_reroute` scored from base-class logits only. The runtime is
  `ContextualRingRestateFactorizedTraceletRateModel` ->
  `RelationalRerouteFactorizedTraceletRateModel` -> base, and the middle class
  adds a relational residual invisible from the base implementation. That family
  carries 27.5% of all probability mass.
* the eager fallback sampled the GLOBAL law after the family had already been
  drawn, redrawing the family and biasing toward whichever families dominate
  globally. It is now family-conditioned, and with lazy ring-restate it is no
  longer reachable.

## WHAT THIS OBSOLETES

**The "vectorization is capped at 1.08×" finding is STALE and must not be
carried forward.** It was measured when `is_rdkit_valid` was 40.2% of an
enumeration and the vectorizable array math was 7.1%. That profile no longer
exists. Encode is now 61% of a fresh transition, it is a neural forward pass on
one state at a time, and SMC runs 32 particles that frequently share states.

Deduplicating particle states and batching the unique encodes is therefore now
plausibly the LARGEST remaining lever, where before today it was pointless. Any
ceiling derived from the pre-lazy profile has to be re-measured rather than
reused.

## Reading of rung 3, stated precisely

Rung 3 converted 1 of 38 (2.6%), with the marginal hazard falling 29.7% ->
15.6% -> 2.6% and 37 of 37 failures showing zero particles ever entering the
region. That is **strong evidence that additional independent attempts have
sharply diminishing returns under the current H=24, N=32 controller**.

It is NOT a claim that more attempts cannot help in general, nor that those
molecules are unreachable under the underlying rewrite process. The official 800
is untouched, and a changed horizon or controller could behave differently.

## Next, and only this

Replace fresh-law enumeration inside SMC with the qualified lazy sampler while
RETAINING the existing law caches:

    cache hit   -> existing cached path, ~1.5 ms, never enters the sampler
    fresh state -> lazy exact sampler

Nothing about N=32, H=24, h_phi, resampling or the controller changes. Then run
the two sentinels -- one known success, one full-horizon extinction -- and
profile the new SMC from scratch, because the old profile is obsolete. The
number wanted is minutes per returned candidate, split across fresh lazy draws,
cache hits, h_phi, graph conversion and canonicalization, SMC bookkeeping, and
model startup.

Do not anchor on any projected end-to-end multiple. 54.5% of law requests were
already cache hits, so the 65–90× will compress by an amount that has to be
measured, and costs previously invisible behind the chemistry will surface.
