# Inverse-derived option demonstrations and proposal diagnosis

Status: development evidence. No autonomous search, new docking, or claimed
benchmark improvement. The broad executor and production controller are unchanged.

## What improved

`attempt_1/result.json` records 83 exact-replayed ring reconstruction examples
across 64 of 82 supported IVG winners (78.0% coverage of this supported subset).
All 83 accepted examples reconstruct the same canonical endpoint after optional
terminal-decoration restoration. The full original panel has 91 source/winner
pairs; nine without a supported saved witness retain their original exclusion
reasons. These are inverse-derived precursors, not the benchmark starting seeds.

Each example removes a peripheral ring with ordinary valid deletions, constructs
it with an existing pendant/fused ring program, and restores stripped terminal
substituents with ordinary one-neighbor births. Every forward and reverse state is
saved and replayed. This covers aromatic/nonaromatic C/N/O five/six rings and
carbonyl restoration without storing endpoints as production options. Other
chemistry remains available through the unchanged generic and ordinary channels.

Preparation took 22.732 seconds and 6,047 executor applications on local CPU,
with zero reference-law evaluations and zero oracle calls. Complete per-winner
units are hash-addressed and restartable. Their cache identities exclude fitting
code and use the preparation's scientific implementation closure.

Source roles were inherited from the earlier fixed retrospective split before
extracting examples. Winners shared with an excluded source could not enter
training. Exact precursor overlap was checked before fitting. Deduplication left
115 decisions: 80 training decisions across 11 sources, and 35 decisions across
four sources excluded from fitting. The latter include 24 complete-ring labels.

## What did not improve enough

`fit_1/result.json` records an unchanged 250-update proposal learner fit, with no
new molecular replay. The extra ring examples repair missing supervision, but
the neural actor still does not beat the simpler demonstration marginal:

| Source-balanced, excluded-source diagnostic | Balanced reference | Demonstration marginal | Neural actor |
| --- | ---: | ---: | ---: |
| Negative log likelihood, lower better | 5.848 | 3.813 | 3.995 |
| Mean demonstrated-option probability | 0.00934 | 0.05842 | 0.02429 |
| Top-label agreement | 0 | 0.139 | 0 |

The neural actor is not promoted. The fit took 0.526 seconds, and the fit plus
audit took 0.992 seconds. The 213-descriptor diagnostic menu is fixed independently
of winners and contains inapplicable distractors, so these are classification
metrics, not production recovery rates.

The factored float32 scorer was faster in its representative profile but failed
the original numerical-equivalence gate on three of 6,816 scores. Argmax ordering
was unchanged. The gate was not relaxed; inference and the reported comparison
use the original public actor forward. The failed optimization is recorded.

## The important sequence-level result

`route_prior_2.json` is the authoritative counterfactual on the four saved real
PARP1 option rows. It verifies exact persistent-slot continuity against the
compiled route and uses the existing measured HOW probabilities unchanged.
`route_prior_1.json` is retained as its predecessor without the extra exact-source
continuity assertion; both returned the same numbers.

The winner-informed marginal raises the pendant aromatic-ring option probability
12.744 times, but reduces each of the next three required options to 0.491 times
its original probability. The specified four-option exact prefix therefore moves
from 4.733e-12 to 7.122e-12, only 1.505 times higher. This is conditional on the
already remodeled linker and a fully mutable region at every option. It is not a
probability from the benchmark seed or a bound on all alternative routes to the
winner. The calculation costs 0.315 seconds and uses no chemistry or oracle calls.

Decision: do not scale either this actor or the static ring prior as the answer
to T4. Next learn coordinated option sequences and attachment/primitive choices
from complete executable demonstrations. Inverse removal of core carbonyl
insertions and recursive option decomposition are specific missing preparation
steps. Task-value guidance still needs outcomes from the deployed continuation
policy; successful teacher routes are not expected future values.

## Checks and operational state

Focused tests cover ring reconstruction, exact-state corruption rejection,
decoration restoration, source balancing, deterministic fitting, context
independence, and generic/unseen-option floors. No remote job was launched.
These files and implementation remain uncommitted development work; full
repository verification and the full controller milestone are not complete.
