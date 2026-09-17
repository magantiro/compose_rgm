# Constructive proposal law and prospective future value

Zero oracle calls. Reads frozen `t4_strategy_reset` artifacts; writes no docking.

## Corpus

27,737 constructive dependency-region decisions extracted from 71 archived controller
runs, collapsed to **7,294 distinct** decisions (73.7% were duplicate serialisations of
the same decision across replicates and arms). 4,844 come from autonomous arms, 2,450
from the winner-bank arm. No decision appears under two targets, so leave-one-target-out
folds are clean.

| target | decisions |
| --- | ---: |
| parp1 | 2,361 |
| jak2 | 1,975 |
| 5ht1b | 1,662 |
| braf | 1,099 |
| fa7 | 197 |

FA7 is starved in the archive as well as at proposal time.

## q_theta: does the law rank the teacher's construction?

Leave-one-target-out, metrics averaged inside folds. Median menu 25 sites, 74 modes,
1,850 joint pairs.

| arm | site top-1 | site top-3 | median site rank | joint top-32 | median joint rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| no opinion | 0.040 | 0.120 | 12.0 | 0.017 | 924.5 |
| additive `A+B` | 0.212 | 0.374 | 4.3 | 0.517 | **31.0** |
| coupled, ridge 1 | 0.213 | 0.375 | 4.4 | 0.516 | 30.6 |
| coupled, ridge 0.01 | 0.232 | 0.393 | 4.8 | 0.513 | 32.4 |

Half of held-out constructions fall in the top 32 of 1,850 candidate pairs, against a
production sampler that reproduced 0 of 11,762.

**The interaction does not pay.** At n=97 that was a power limit; at n=7,294 with `|W|`
fitted up to 2.76 it is a well-powered negative. It is a statement about THESE features,
not about coupling: a bilinear form on 18 generic atom descriptors and 12 coarse mode
counts may not span the chemistry a coupling would express. Additive is the operating
law until a richer site basis is tried.

A categorical `(mode, site-context)` lookup scores a median site rank of 13 -- no better
than having no opinion -- once ties are broken honestly. Its apparent 0.309 top-1 was
entirely tie collapse. Features transfer where identity does not.

## h_phi: does anything rank lineages prospectively?

Behaviour-policy future value, NOT a committor: labels record what the historical search
found below a state under its own selection policy.

Label choice decides the answer, and two of the three obvious labels are traps.

| label | what it rewards |
| --- | --- |
| best improvement over all descendants within k calls | how many descendants the policy chose to generate. Descendant count alone scores AUC 0.851; stratifying by it collapsed the soft allocator 0.693 -> 0.556 |
| best improvement among the first m descendants | budget-invariant, but ranks by ROOM to improve: the fitted arm scores rho -0.395 against absolute quality, actively preferring bad nodes |
| **did any of the first m children beat the standing incumbent** | the decision a controller actually faces |

On the third label, per-fold AUC over 2,803 states / 178 frontier advances:

| arm | AUC | advances captured in top 10% |
| --- | ---: | ---: |
| `gap_to_incumbent` raw | 0.840 | **0.517** |
| learned `h_phi` (gap + score) | 0.845 | 0.488 |
| learned `h_phi`, all 16 features | 0.818 | 0.436 |
| current score | 0.610 | 0.154 |
| soft allocator weight | 0.595 | 0.209 |
| ancestral edits | 0.454 | 0.071 |
| lineage depth | 0.332 | 0.057 |
| chance | 0.500 | 0.100 |

`h_phi` clears the stated gate but adds nothing over one free column. Structure-only
features with no docking score score 0.495, i.e. chance: the signal is score geometry,
not chemistry.

**Lineage depth is strongly anti-predictive.** The earlier "winners accumulate 15
ancestral edits against 4-5 for failures" was hindsight champion ancestry. Prospectively,
deeper lineages are less likely to advance the frontier, which does not support
depth-aware parent allocation.

## Two defects found in this work, both silent

- A rank-k interaction `W = U^T V` has a stationary point at `U = V = 0`; fitted from a
  small random start the factors stayed at 4e-5 and the law silently became `A + B`.
  Held-out scores were byte-identical across rank 1-4 and two orders of penalty, so
  nothing downstream revealed it. The full-matrix form is convex and cannot do this.
- Pooling per-fold scores into one metric compares folds on incomparable scales. It
  depressed every fitted arm while leaving raw columns untouched, producing three
  readings where a model scored below a feature it contains -- impossible for a rank
  metric. Metrics are now averaged inside folds.

Both carry regression tests in `tests/test_constructive_prior.py`.

## What is NOT yet measured

The binding gate for JAK2 is autonomous constructive YIELD: whether sampling from
`q_theta` produces valid constructive proposals through the production compiler, against
the 0/11,762 baseline. Ranking a teacher's choice inside a menu is necessary and not
sufficient. No docking budget should move before that is measured, and it costs no
oracle calls.
