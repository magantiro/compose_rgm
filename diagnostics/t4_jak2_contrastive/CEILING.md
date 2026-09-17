# Route-assisted ceiling: can reward-guided COMPOSE exploit a known JAK2 basin?

Development lane. The structural basin came from JAK2 teacher routes; the exact teacher
program and endpoint are excluded. This is level 1-2 on the evidence ladder and is NOT an
autonomous result.

## Result

    BEST  -11.30
    CC1N(C(N)=O)CCN1C(=O)CC1Nc2ccccc2-c2ccnc3[nH]cc1c23
    SA 3.79   QED 0.625   similarity 0.629   radicals 0   no unstable motif

75 of 89 valence-correct candidates scored; median -9.90; 9 at or below -10.5.

| reference | score |
| --- | ---: |
| **this run** | **-11.30** |
| autonomous JAK2 record | -10.7 |
| known JAK2 teacher route | -11.6 |
| best in the whole historical archive | -11.7 |

So given the correct basin but not the answer, the machinery reaches -11.3, beating the
autonomous record by 0.6 and landing 0.3 short of the teacher route.

The winner is one aromatic-nitrogen position away from the archive's best molecule:

    found     CC1N(C(N)=O)CCN1C(=O)CC1Nc2ccccc2-c2ccnc3[nH]cc1c23    -11.30
    archive   CC1N(C(N)=O)CCN1C(=O)CC1Nc2ccccc2-c2cccc3[nH]cc1c23    -11.70

It was produced by an `element` intervention on a teacher subgoal, with the teacher
endpoint itself removed from the pool.

## The bug this run exposed, and the fix

The first attempt at this experiment reported -11.60. That number was wrong: the molecule
was a DIRADICAL. So were 42 of its 61 scored candidates.

The cause was not the generator. Measured radical rates:

| source | radical rate |
| --- | ---: |
| production generator, no intervention (4,000 archive molecules) | **0%** |
| `intervene_bond_order` | **100%** (25 of 25) |
| `intervene_element` | 54% (20 of 37) |
| `intervene_scale` | 29% (2 of 7) |

An atom signature is `(element, charge, implicit_h, neighbour_count)` and the valence
equation is `h = standard_valence + charge - row_sum`, where `row_sum` is the sum of BOND
ORDERS. `intervene_bond_order` changed bond orders and never touched hydrogens, so it
failed every time. `intervene_element` changed the element and kept the old hydrogen
count. `intervene_scale` adjusted by NEIGHBOUR COUNT rather than bond-order sum, so it
worked only when every changed bond happened to be single.

`instantiate_goal` did not catch any of it, and is not at fault: it verifies that the
built atom matches the DECLARED signature. A nitrogen declared with one hydrogen and one
single bond is faithfully built as a nitrogen radical, and the role counts as satisfied.
The executor was correct and the declaration was impossible.

`_rebalance` now restores the valence equation after every intervention. Retained roles
carry bonds outside the patch, so their hydrogens and neighbour counts move by the DELTA
inside it; created roles exist only inside the patch, so theirs are computed outright. An
intervention that would leave negative hydrogens -- raising a bond order on a saturated
atom -- now RAISES instead of silently emitting an open shell.

Guards: `_rebalance(sg, sg) == sg` on all 147 teacher subgoals, so it is an identity on
unmodified patches; and the corrected cloud contains 0 radicals in 93 molecules.

## What was inflated before this fix

Every eligible-yield number produced by the intervention machinery, because the fiber was
not checking radicals. The 73-of-81 JAK2 scale-sibling figure and the Stage-1 funnel both
counted open-shell species as usable. They are superseded.

## Still not shown

This is route-assisted. The basin was supplied. Nothing here shows the controller can
FIND such a basin without it, and the next rungs -- JAK2 routes held out, then fully
frozen autonomous -- are untouched.
