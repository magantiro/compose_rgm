# Controlled conditional execution contract

**Date:** 2026-07-20  
**Status:** interface and bounded scripted smoke complete; corrected backbone
unqualified; no training or benchmark launch authorized

## Decision boundary

This is a new conditional system, not another sweep of the developmental
events-1--12/events-13--24 controller.  It imports only validated unconditional
lessons: canonical molecular successors, validity-closed execution, explicit
virtualization of molecular self-transitions, and the productive operator/rate
lineage of the retained pancake model.  It explicitly rejects the unvalidated
P1 empirical-prior and P2 topology-mass modes.

The backbone interface accepts either `pancake_derived` or
`canonical_successor_native` weights, but it cannot promote them by naming
them.  Production use requires an external unconditional qualification result,
the exact checkpoint SHA-256, canonical-successor execution, and molecular
self-transition virtualization.  An unqualified backbone is usable only when
the caller sets the explicit smoke-only override; every such report remains
`large_benchmark_authorized: false` and `training_launched: false`.

## Frozen task

The production default is the exact GrIDDD-style QED task:

- starting-molecule QED in **[0.70, 0.80]**, inclusive;
- **20** candidate attempts per start;
- candidate QED at least **0.90**;
- Morgan fingerprint Tanimoto similarity at least **0.40** to that start;
- Morgan radius **2** and **2,048** bits, frozen in the protocol record;
- every start, failed candidate, stalled candidate, invalid candidate, and
  event-budget exhaustion retained in its declared denominator.

The complete 800-start benchmark remains unauthorized.  Its eventual success
denominator is all 800 fixed starts, not only starts that return a valid edit.

## Matched arms

| Arm | Learned rate input | Successor control | Oracle use |
|---|---|---|---|
| Direct | QED target | beta-zero shadow resampling | Canonical successors are scored but scores do not influence selection |
| Controller | classifier-free/base | lead-aware QED/similarity tilt | Scores influence successor selection |
| Combined | QED target | same lead-aware QED/similarity tilt | Scores influence successor selection |

Candidate-level random streams are identical across arms.  Arm identity is not
part of seed derivation.  Different selected successors may cause streams to
diverge later, which is a treatment effect rather than a seed mismatch.

## Exact oracle accounting

Each start/seed/arm receives:

1. one lead-eligibility QED call;
2. a fixed guidance allowance for each of 20 candidate slots; and
3. one terminal-candidate QED call for every slot, including failures.

At the production default of 48 guidance calls per candidate, this is exactly

`1 + 20 * (48 + 1) = 981 QED calls per start/seed/arm`.

Canonical duplicate successors are scored only once productively.  A scoped
ledger cannot overspend.  If a short or stalled trajectory cannot consume its
allowance, the remaining calls are real deterministic QED invocations on its
terminal state, labeled `unused_guidance_padding`; they do not influence
selection.  Invalid terminal candidates use the valid lead for the required
final padding call but remain invalid in the candidate denominator.  Reports
separate total, selection-influencing, padding, unique-state, and oracle wall-
time counts.  Padding therefore makes the budget exact without hiding control
opportunity starvation.

The fixed-start list may require one common pre-run QED screening call per lead.
Those calls are reported separately and are explicitly excluded from every
matched per-arm budget.

## Required outputs

Every arm reports:

- success over all start/seed attempts;
- candidate validity/connectivity and all-trajectory validity/connectivity;
- best feasible QED improvement and no-solution/stall/exhaustion counts;
- exact oracle ledgers and padding fraction;
- unique-candidate fraction and mean pairwise Tanimoto distance;
- anytime success and best feasible QED after every candidate slot;
- oracle and end-to-end wall time; and
- paired bootstrap comparisons for combined-versus-direct,
  combined-versus-controller, and controller-versus-direct.

## Bounded smoke result

`diagnostics/griddd_conditional_execution_smoke.json` uses one real neutral C/N/O/F
lead with QED 0.7435 and one real candidate with QED 0.9229 and Tanimoto 0.4074.
Candidate generation is deliberately scripted because no corrected backbone is
qualified.  Two candidate slots and four guidance calls per slot give exactly
11 QED calls per arm.  All three arms used exactly 11 calls, produced 2/2 valid
and connected candidates, retained one all-attempt start denominator, emitted
anytime/diversity/wall-time records, and produced the three paired comparison
records.  The smoke passed in about three seconds and launched no training.

This result validates execution and accounting only.  Its scripted success
values are not model efficacy evidence.

## Next gate

1. Freeze and qualify one corrected canonical-successor backbone derived from
   the productive pancake weights or a native canonical-successor run, with P1
   and P2 disabled.
2. Only after that qualification, attach or train the direct QED-conditioning
   head under a bounded pilot and run a small fixed-lead execution smoke through
   the real rewrite generator.
3. Require exact ledgers, no safety failure, bounded padding, and a positive
   paired signal before considering the 800-by-20 benchmark.

No old-controller sweep, large conditional training run, or GrIDDD benchmark is
authorized by this contract.

## Qualification-manifest integration

The conditional lane now consumes a versioned, fail-closed unconditional
handoff rather than accepting a caller-supplied qualification Boolean.  The
required format is
`compose_v4_canonical_successor_backbone_qualification_v1`.  It binds:

- a multi-state panel, held-out rate gate, and rollout gate that all passed;
- the qualified checkpoint SHA-256 and either pancake-derived or native
  canonical-successor lineage;
- canonical molecular-successor execution and virtual molecular self events;
- `empirical_mark_prior_mode="none"`,
  `ring_family_mass_mode="boolean"`, and `p1_p2_imported=false`; and
- for pancake-derived weights, the exact retained step-6,250 source SHA-256.

The subsequent QED-conditioned child must name the exact qualification
manifest hash and unconditional checkpoint hash in its own payload.  It must
also expose a one-dimensional frozen QED condition and repeat the same
canonical-successor, self-event, and no-P1/P2 assertions.  A hash, lineage, or
mode mismatch stops before molecule generation.

`scripts/run_griddd_real_rewrite_smoke.py` is the prepared one-lead real
rewrite gate.  It runs direct, controller, and combined arms through the fixed
exact-call ledger with one candidate and four guidance calls by default.  This
is six QED calls per arm after including the lead and terminal calls.  Its
default rollout is bounded to 0.4 operational time and four events.  Success
at this gate means only that real rewriting, safety checks, and accounting
execute correctly; it does not authorize an efficacy claim or the 800-by-20
benchmark.

At integration time the best available unconditional artifact is still the
one-state step-6,250 distillation smoke.  That artifact explicitly authorizes
only a cached multi-state microbenchmark, not backbone qualification.  The
runner therefore wrote
`diagnostics/griddd_real_rewrite_smoke_status.json` with
`phase="waiting"`, `launched=false`, and blocker
`qualification_manifest_missing`.  No model rollout or training was launched.

## Implementation

- `src/compose_v4/experiments/griddd_conditional.py`
- `scripts/run_griddd_real_rewrite_smoke.py`
- `scripts/smoke_griddd_conditional.py`
- `tests/test_griddd_conditional.py`
- `diagnostics/griddd_conditional_execution_smoke.json`
- `diagnostics/griddd_real_rewrite_smoke_status.json`
