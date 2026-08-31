# broad-C, uniform semantic prior — CONTROL ARM (stopped early, partial)

Cell 5ht1b_s7_d0.6, r10-12. State-conditioned support, correct credit gating.
Stopped at rounds 1-2 of 9 deliberately. Retained as the UNIFORM-PRIOR CONTROL
against which the R_theta-prior version is to be compared.

## The measured failure, which motivated the next controller

    500 ring draws across r10-12, 164 distinct semantic classes

    ring:linked/6/C6/aromatic   drawn 5 times = 1.0%
    narrow B gives that action                = 33.3%

The move that wins this cell is drawn 33x less often under broad semantics.
And it is COMPOSITIONAL: B's -11.00 endpoint is a terphenyl, i.e. that action
fired 2-3 times inside one 6-step program. At 33% per slot that is routine; at
1.0% it effectively never happens.

    B r11 -11.00  FC(F)(F)c1cc(-c2ccc(-c3ccccc3)cc2)cc(N2CC[NH2+]CC2)c1  28hv 4 rings
    B r12 -11.30  FC(F)(c1ccc(-c2ccccc2)cc1)c1cccc(N2CC[NH2+]CC2)c1      27hv 4 rings
    C r10  -9.30  FC(F)(c1ccccc1)c1cccc(N2CC[NH2+]CC2)c1                 21hv 3 rings
    C r12  -8.00  FC(F)(F)c1cccc(N2CC[NH2+]NC2)c1                        16hv 2 rings

C tops out at ONE added benzene; r12 never grew beyond the seed at all.
C's most-drawn classes were linked/5/C5/saturated, fused/10/C10/saturated,
linked/7/C7/saturated -- each sampled more often than plain benzene.

## The diagnosis this supports

Not "broad semantics do not help" and not "fused is bad". Rather: broadening
from 3 to 164 classes under a UNIFORM prior diluted a compounding move below
the rate at which ~10 docking labels per round can learn it. The failure is
credit-assignment/sample-efficiency at fixed oracle budget, and it is a
property of the flat initialization, not of the semantic space.

Also measured here: credit rate 7-10%, bounded near its structural ceiling by
dock_per_round=10 against n_particles=64 (at most ~16% of particles can carry a
docking signal), so the support fix could not have raised it much.

Artifacts: volume compose-v4-artifacts,
macro_basin/episodes_5ht1b_s7_d0.6_pooled_broad_r{10,11,12}.json
