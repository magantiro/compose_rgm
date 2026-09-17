# JAK2 32-call mechanism experiment: preregistration

Written and committed BEFORE any Stage-2 candidate is docked. Nothing below may be
changed after outcomes are seen; a deviation must be recorded as a deviation.

This is a MECHANISM experiment, not the benchmark claim. It does not test whether COMPOSE
beats IVG. It tests one thing:

> Does information from 16 designed interventions alter later choices in a way that
> produces better measured molecules?

## Arms

Primary control is the SAME controller at `h=1`. Both arms then share the COMPOSE
generator, the route geometry, all 16 Stage-1 observations, the Bayesian contrast
posterior, the candidate pool, the noise model and the acquisition implementation. The
only difference is myopic program value against depth-3 continuation value. A weaker
control would leave the comparison confounded by everything the two arms have in common.

## Budget: 32 calls

| stage | calls | content |
| --- | ---: | --- |
| 1 | 16 | shared matched block-intervention bundles across several global hypotheses |
| 2a | 6 | selected by the `h=3` controller |
| 2b | 6 | selected by the `h=1` controller |
| 3 | 4 | predeclared confirmation of decision-relevant extremes |

Both selectors commit their six candidates BEFORE any Stage-2 candidate is docked. If both
select the same candidate it is docked once and scored for both arms.

Uniform, route-prior and Dynamic-like selections over the same pool are LOGGED as
counterfactual choices but not docked, except where they coincide with an already-docked
candidate.

## Frozen controller settings

Fixed by prior zero-oracle measurement, not by tuning here.

| setting | value | set by |
| --- | --- | --- |
| inner rollout depth | 3 | credit-horizon audit: saturates at h=3; h=1 misses 17.2% of winner ancestors |
| outer planning depth | 1 | nothing measured supports recursive batch planning |
| rollout prior | measured length-2 option transitions | length-3 rests on 19 observations |
| candidate admission | `legal AND round_trips AND complement_preserved AND binds` | `edge_presence` is legal 147/147 and round-trips 72/147 |
| reference law | `q_ref` mixture, `eps_glob << eps_loc` | `q_legal` needs 10^85 draws; `q_block` needs 24 |
| route prior weight | weak and coarse | conditioning adds nothing; worse than uniform on BRAF/PARP1 |
| maneuverability term | none | adds nothing over score and gap (0.761 against 0.763) |
| docking noise `sigma` | 0.35 | prior; not fitted on this experiment |

## Success criteria, predeclared

PRIMARY: best confirmed Stage-2 latent utility under `h=3` exceeds that under `h=1`.

SECONDARY, reported whatever the primary shows:

- mean parent-relative improvement per arm;
- count of incumbent improvements per arm;
- whether `h=3` reaches a productive global basin that `h=1` does not;
- posterior over structural coordinates before and after Stage 1, and which coordinates
  moved;
- global against local composition of each arm's selections, and their continuation
  structure;
- how many `h=3` and `h=1` selections coincided;
- confirmed against single-docked extremes, since the objective is an extreme value.

NOT a criterion: statistical significance. Six against six cannot supply it, and claiming
it would be false. Nor is beating IVG a criterion for this experiment.

## Declared in advance as reasons this could fail without indicting the method

- Stage 1 may show no separation between coordinates, in which case Stage 2 has nothing
  to act on and the failure is in the intervention family, not the controller.
- `h=3` and `h=1` may select identical candidates, in which case the experiment is
  uninformative about depth and must be reported as such rather than as a tie.
- The candidate pool may not contain a productive transformation at all. Proposal support
  for productive JAK2 CONTENT is still unproven; only size support and block support are
  established.

## Evidence standing behind this design

| established | measurement |
| --- | --- |
| protected coordinated programs are necessary | winners are one-shot ~14-edit programs, ancestral 0, prefixes ineligible |
| program size is not sufficient | autonomous 16-17 edit programs score -7.2 to -7.8; corr(size, score) -0.059 |
| passive allocation cannot find discovery | 8 of 11 wins came from branches ranked 3rd-12th at decision time |
| intervention support exists on JAK2 | 73 of 81 scale siblings admitted, where binding offers 0 alternatives |
| siblings are reachable | closure-conditional 24 expected draws against 10^85 uniform |
| neighbourhood accessible on JAK2 | median 10.3 draws leave-target-out |
| depth can change a decision | mechanistic test, stable across 8 seeds |

| NOT established | status |
| --- | --- |
| the generator proposes productive JAK2 content | unproven; the support gate |
| depth-3 improves realised optimisation | historical test had 6 disagreement states, 4-2 in favour, underpowered |
| any of this beats Dynamic-v0 | untested; no prospective call has been spent |
