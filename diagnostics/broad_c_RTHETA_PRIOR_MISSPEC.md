# broad-C with R_theta semantic prior — MIS-SPECIFIED, STOPPED

Cell 5ht1b_s7_d0.6, r30-32. Launched 2026-08-27 03:42, stopped at round 0.

## Result: the prior was anti-correlated with plausibility

    248 draws, 181 classes
    ring:linked/6/C6/aromatic:  0 draws = 0.0%
        uniform-prior control:  1.0%
        narrow B:              33.3%

Top draws were all hetero-rich saturated rings (fused/7/C5N1O1/saturated,
fused/6/C3N1O2/saturated, fused/5/C2N2O1/saturated). The space became MORE
diffuse (181 classes vs 164 under the uniform prior).

## Cause

`rtheta_semantic_prior` built `allowed` as the spec's element SET and summed
R_theta over every insertion matching it. `match_growth_descriptors` filters
`atom_type in allowed_elements`, so:

    C6      -> allowed {C}      -> carbon insertions only
    C4N1O1  -> allowed {C,N,O}  -> carbon + nitrogen + oxygen insertions

rho grows monotonically with the SIZE of the element set. It measures
permissiveness, not plausibility, and gives all-carbon specs the smallest
possible mass by construction.

## What a correct rho has to respect

The spec is a STOICHIOMETRY, not an allowed set. rho must reflect the mass of
building THAT composition -- e.g. the mass of the specific first insertion the
spec requires, or a composition-weighted combination across the required
elements -- and must not reward specs merely for admitting more element types.
Normalising by |allowed| is NOT sufficient: it would still ignore which
elements the stoichiometry actually demands.

Untested and still open: whether R_theta's probability mass favours carbon at
all. Carbon has one (element, valence) class while S has three and P two, so
carbon is ~7.5% of legal atom_insert ACTIONS; whether its MASS is
correspondingly larger was never measured, and a correct rho is only useful if
it is.

Artifacts: volume compose-v4-artifacts,
macro_basin/episodes_5ht1b_s7_d0.6_pooled_broad_r{30,31,32}.json
