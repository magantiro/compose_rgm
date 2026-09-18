# Why autonomous search does not enter the strong JAK2 delta=0.6 basin

Five diagnostics were specified; three cost no oracle calls and they settle the question,
so the two that cost calls were not run.

## The strong basin is one structural family, and it is not ours

Every molecule at or below -10.5 that is FEASIBLE at delta=0.6 shares one architecture:
the root's methyl ester is REPLACED by an amide to a 1,2-diamine ring, and that ring
carries a urea or carboxamide on the distal nitrogen.

    -11.30  sim 0.629  CC1N(C(N)=O)CCN1C(=O)CC1Nc2ccccc2-c2ccnc3[nH]cc1c23
    -11.00  sim 0.639  CC1N(C(=O)O)CCN1C(=O)CC1Nc2ccccc2-...
    -10.90  sim 0.600  CC1=CN(C(=O)CC2Nc3ccccc3-...)CN1C(N)=O
    -10.80  sim 0.619  CC1(C)N(C(N)=O)CCN1C(=O)CC1Nc2ccccc2-...
    -10.70  sim 0.600  CC1CN(C(=O)CC2Nc3ccccc3-...)CN1C(N)=O
    -10.60  sim 0.619  CC1N(C=O)CCN1C(=O)CC1Nc2ccccc2-...
    -10.60  sim 0.623  NC(=O)N1CCN(C(=O)CC2Nc3ccccc3-...)C1

Seven of seven. What the autonomous run found instead KEEPS the ester and decorates it.

| feature | strong d0.6 | strong d0.4+ | autonomous docked |
| --- | ---: | ---: | ---: |
| amide linkage `N-C(=O)-CH2` | 7/7 100% | 21/23 91% | 13/193 7% |
| **ester retained** | 0/7 0% | 0/23 0% | **126/193 65%** |
| ring `N-C-N` | 7/7 100% | 18/23 78% | 7/193 4% |
| ring `N-C-C-N` | 6/7 86% | 16/23 70% | 1/193 0.5% |
| urea/carbamoyl on ring N | 5/7 71% | 14/23 61% | 5/193 3% |
| acyloxymethyl (our motif) | 0/7 0% | 0/23 0% | 22/193 11% |

## The basin is never PROPOSED, which is different from never chosen

Measured on the free pool before any selection: 2,400 raw programs from the root yield
365 feasible endpoints.

| feature | count | rate | strong basin |
| --- | ---: | ---: | ---: |
| ester broken | 83 | 22.74% | 100% |
| amide linkage | 3 | 0.82% | 100% |
| ring N-C-N | 2 | 0.55% | 100% |
| ring N-C-C-N | 2 | 0.55% | 86% |
| urea on ring N | **0** | **0.00%** | 71% |
| **amide AND diamine ring** | **0** | **0.00%** | **100%** |

The pieces exist separately and never co-occur. Treating the marginals as independent puts
the conjunction near 0.005%, about one in 145,000 raw proposals. The whole 13-round
campaign generated roughly 2,400 feasible endpoints, so the expected number of basin
molecules in it was about 0.1.

## There is no reward valley to cross, because there is no distance to cross

    root's own similarity to the basin      0.639 max, 0.618 mean
    best approach over 105 charged calls    0.681
    best among 365 free endpoints           0.702

The basin sits at Tanimoto 0.629-0.639 from the root, INSIDE the delta=0.6 fiber. The
search began at the root's own 0.639 and reached 0.681. It did not approach and then
leave; it never moved. So the "requires temporarily worse intermediates" hypothesis is
refuted without spending an oracle call, and the lookahead value `Q_3` is NOT the fix.

## What this does and does not license

CONFIRMED: the reward controller performs well inside the basins it visits -- adaptive beat
reward-blind selection in 10 of 13 matched rounds, its model-selected calls beat its own
random quota by 0.405, and the round's true best molecule sat at a median 2.3 percentile
in its pre-docking ranking of the pool. In round 10 it ranked the eventual -10.30 winner
1st of 168.

CONFIRMED: the reference process places essentially no mass on the coordinated
construction that enters the better basin.

NOT MEASURED: the IVG basin. The published paper reports jak2 docking scores but no jak2
SMILES, so `diagnostics/ivg_winners.json` legitimately has no jak2 cell and their basin
cannot be decomposed. Their lead-optimization oracle budget is also undocumented -- the
10,000-evaluation figure in that paper belongs to the PMO benchmark, a different
experiment -- so no call-efficiency comparison against them is supportable.

NOT MEASURED, and would cost calls: the reward profile along the teacher route, and the
basin-injection test. Both were made unnecessary for THIS decision by the three free
diagnostics, and the injection test remains the right confirmation if the q0 fix lands.

## The implied next move

Raise proposal mass on the coordinated conjunction, not on lookahead and not on the
reward model. The basin needs three changes together: replace the ester oxygen by a ring
nitrogen, construct the 1,2-diamine ring, acylate the distal nitrogen. The runtime pairs
at most TWO regions and multi-region endpoints were 0 to 7% of any pool.

---

# Factoring the teacher-program probability (3,000 programs, zero oracle calls)

A whole-program frequency cannot say WHICH conditional carries the smallness, so the
factors of `q0(Z|G) = q(R|G) q(A|G,R) q(C|G,R,A) q(z|...)` are estimated by conditioning
on sampled draws. 3,000 programs from the root, 41,591 realized variants, 1,610 feasible
endpoints at delta=0.6.

| factor | rate | detail |
| --- | ---: | --- |
| q(R): WHERE touches the ester group (slots 0-3) | **65.4%** | 1949/2978 bound programs |
| q(R): WHERE touches the ESTER OXYGEN (slot 1) | **55.4%** | 1649/2978 |
| q(A\|R): ester oxygen RETYPED TO NITROGEN | **0.77%** | 321/41591 variants; 48 feasible |
| q(amide \| R hit, feasible) | **6.6%** | 55/828 |
| q(diamine ring \| R hit, feasible) | **0.12%** | 1/828 |
| q(amide AND diamine ring) | **0/828**, < 0.36% by rule of three | the wall |
| q(urea on a ring N) | 0/1610, < 0.19% | never observed |

## What this rules in and out

**WHERE is not the defect.** Over half of all programs bind the exact atom the strong
basin retypes, which matches the historical 96.6% JAK2 WHERE coverage. Region selection
needs no redesign.

**The retype is not the defect either.** 321 variants perform the O-to-N retype and 48 of
them are feasible. **308 of the 321 come from `retained_element`** -- the retained-role
retyping capability added earlier in this project is doing exactly its job.

**Coordination is the defect.** Conditional on a program that hits the right atom AND
yields a feasible amide (6.6%), the diamine ring co-occurs in 0 of 828. The two decisions
are drawn as independent rare events, so the joint sits near 1e-4.

## The mechanism, concretely

`_variants` can only intervene on a program `synthesize_dynamic_program` already drew. If
that program did not happen to construct a ring at the ester position, no variant can add
one. So "build a ring here" and "retype this atom to nitrogen" have to coincide across two
independent samples rather than being one semantic intervention.

## Consequences for the fix

THREE CHEMICAL CHANGES ARE NOT THREE REGIONS. The move is a single dependency closure
around slot 1: retype that atom and build a ring incorporating it. Admitting arbitrary
3-region conjunctions would enlarge the search space without addressing this, and is
therefore the wrong fix.

What the measurement points to instead is sampling a structural replacement GOAL and then
generating its mutually dependent edits jointly -- and learning that coupling from route
structure across targets rather than hard-coding a piperazine operation for JAK2, since
route COMPONENTS are already known to transfer where whole templates do not (0 of 147
exact whole-patch overlap against 68.1% region-level coverage, 96.6% on JAK2).

## Two claims from the earlier section, corrected

- "zero support" overstated a zero count. 0 of 2,400 bounds the joint rate at roughly
  3/2400 = 0.00125 at 95%; it is a proposal-MASS problem, not proven absent support.
- "there is no reward valley" does not follow from the endpoint's Tanimoto distance.
  Similarity 0.629 to the root means the endpoint satisfies the delta=0.6 constraint; it
  says nothing about the rewards of intermediates along any particular route. Settling
  that needs the route-prefix scoring diagnostic, which costs oracle calls and has not
  been run.
