# Canonical-successor distillation repair — 2026-07-20

## Decision

The smallest permanent unconditional repair is to distill the retained,
calibrated step-6,250 pancake marked-rate law onto the productive molecular
quotient. Keep its measured behavioral family rates; aggregate syntactic
Graft/bond-reroute aliases at the canonical molecular successor; and do not
renormalize virtual self/calibration mass onto the remaining chemistry.

This lane does **not** import the rejected P1/P2
`catalog_topology_local_support` scalar. The model uses
`ring_family_mass_mode="boolean"`. Small-ring and joint heterocycle-context
calibration are a later, separately gated extension.

## Why the existing quotient mask is not sufficient

The current feature compiler already performs two correct operations:

1. canonical molecular self-Grafts are absent from productive support; and
2. syntactic Graft aliases reaching one successor share a group id, and the
   teacher action score uses their log-sum-exp.

However, deleting the self matches from the conditional action softmax makes
the surviving Graft mass sum to one. For a legacy teacher with total hazard
`H`, family probability `p_g`, and productive conditional Graft mass `a_g`,
the correct productive Graft rate is

`H * p_g * a_g`,

not `H * p_g`. The same issue holds for the calibrated delete multiplier. A
plain mask therefore changes productive transition intensities even if it
removes every visible self-event.

## Exact target and objective

For every frozen teacher state/time pair:

- reconstruct the legacy root-free Graft support;
- evaluate its learned syntactic logits;
- sum all non-self alias logits by canonical successor;
- retain self mass as virtual teacher mass;
- multiply atom-delete family rate by `exp(-0.5)`;
- leave all other base-family rates unchanged; and
- set target productive total hazard to the sum of productive family rates.

The student uses the canonical quotient support. Its objective decomposes the
complete-successor Poisson-KL into:

1. Poisson-KL over productive family rates, which matches the corrected total
   hazard and preserves the pancake family law; and
2. a teacher-rate-weighted cross-entropy over distinct canonical Graft
   successors conditional on the Graft family.

This decomposition charges every student family rate exactly once and does
not transfer removed self mass to another successor. It is differentiable and
requires no molecular successor materialization inside the training loss: the
existing compiled `graft_successor_groups` table supplies the quotient.

## Immediate backtrack boundary

Canonical quotienting removes `A -> A` gauge events, but it cannot by itself
forbid a productive `A -> B -> A` pair because the model is Markov in the
current molecule. The permanent learning signal is the calibrated pancake
successor-rate target, whose retained rollout audit has very low immediate
backtracking. A trajectory-local safety wrapper additionally routes an
immediate return to the previous accepted canonical state into a virtual event.
It retains the original hazard and every other marked rate, so this is exact
CTMC thinning rather than conditional renormalization. Rejection counts are
first-class diagnostics; a trained checkpoint must make them rare rather than
using the wrapper to hide a bad rate law.

## Code and exact tests

- `src/compose_v4/experiments/canonical_successor_distillation.py`
  implements explicit canonical aggregation, calibrated pancake targets,
  differentiable quotient predictions/losses, and the history-aware exact
  thinning safety.
- `tests/test_canonical_successor_distillation.py` covers exact small-state
  equivalence, family-mass preservation, target projection, trainability, and
  non-renormalizing reverse-jump thinning.
- `scripts/smoke_canonical_successor_distillation.py` is the timed bounded
  distillation gate.

The focused test suite has five passing tests. On the labeled C8 chain, the
legacy support contains 42 syntactic root-free Grafts. Exact canonicalization
partitions them into 30 productive matches across four molecular successors
and 12 self aliases. The explicit-rate mass identity is exact.

## Quantitative tiny smoke

The deterministic one-state smoke is retained at
`diagnostics/canonical_successor_distillation_tiny_smoke.json`.

| Quantity | Initial | Final after 100 updates |
|---|---:|---:|
| Excess quotient Poisson-KL | 0.0177659 | 0.000000119 |
| Productive-family relative L1 | 15.37% | 0.0440% |
| Canonical-Graft relative L1 | 40.25% | 0.121% |
| Student total hazard | 0.6682 before correction | 0.57945 (target 0.57919) |

The excess objective fell 99.9993%. Target construction took 0.291 s;
100 CPU updates took 0.779 s (7.79 ms/update), giving a mechanical 500-update
projection of 3.89 s for this tiny case. Teacher mass balance error was
`8.94e-8`. All smoke gates passed, but this is a mechanism test, not a chemistry
or real-checkpoint result.

## Actual step-6,250 pancake smoke

The identical gate was then run from the selected state in the real retained
checkpoint (SHA-256
`47716924f7798ed24556c5aa8fb10c533c55dbd1f02f8f53a463cf2ad80ae2bf`).
The result is durable at
`diagnostics/canonical_successor_distillation_pancake6250_smoke200.json`.

The C8 state exposes a large, previously hidden renormalization error:

- raw teacher total hazard: 7.95489;
- productive quotient hazard after self/delete calibration: 2.16487;
- virtual rate mass: 5.79001;
- raw Graft family rate: 5.83493;
- productive canonical-Graft rate: 0.28605; and
- Graft self/virtual rate: 5.54888.

Thus only 4.90% of the checkpoint's Graft family intensity is a productive
molecular transition on this state. Removing the self aliases and normalizing
the other actions would inflate productive Graft rates by roughly twentyfold.
This matches the direction of the P1/P2 rollout's reroute thrashing and gives a
direct mechanistic reason to distill rates rather than masks.

The naïve quotient student began with excess quotient KL 4.74160,
productive-family relative L1 267.45%, and canonical-Graft relative L1
1,939.86%. After 200 head-only CPU updates, excess KL reached numerical zero,
family relative L1 was 0.00212%, Graft relative L1 was 0.00438%, and student
hazard was 2.16492 versus target 2.16487. Exact mass-balance error was zero.

Target construction took 1.226 s and 200 updates took 4.905 s
(24.53 ms/update), a mechanical 500-update projection of 12.26 s for this
one-state smoke. All cheap-pilot gates pass. This authorizes a cached multi-state
microbenchmark, not corpus-scale training or a chemistry claim.

## Launch gate and cache boundary

No corpus-scale training is authorized by the tiny smoke. Before a bounded
real pilot:

1. Load the actual step-6,250 selected pancake checkpoint and verify
   `ring_family_mass_mode="boolean"`.
2. Microbenchmark target construction and backward pass on a fixed small panel
   of cached carbon-tree states. Project the requested update count; reject any
   path approaching day-scale runtime.
3. Confirm exact family-mass identity and zero canonical self successors on all
   panel rows.
4. Require the frozen validation panel to improve productive-family rate error
   and canonical-Graft rate error without worsening pancake family accuracy.
5. Run only a cheap smoke rollout first. Promotion requires validity and
   connectivity 1.0, zero canonical self events, immediate backtracking no
   worse than calibrated pancake, and no regression in event/family mix or
   size behavior.

The raw corpus, compiled paths, ring catalog, source prior, and chemistry
support remain reusable. A real distillation target cache needs only state/time,
productive family rates, canonical Graft group rates, and provenance hashes of
the pancake checkpoint and calibration signature. It must use a new checkpoint
and metrics namespace. No path, executor, or ring-catalog rebuild is needed.

## Separately gated ring-context extension

Only after the base quotient pilot preserves pancake behavior may a second
extension add:

- action-dependent small-ring acceptance matching the retained `exp(-1.5)`
  calibration; and
- joint ring context such as topology class, cycle-size multiset, neighboring
  element/electronic roles, and adjacent heteroatom patterns.

That extension must compare support-conditioned rate mass to the corpus and
must not resurrect the rejected Boolean-to-topology absolute-mass scalar. It
has its own small-ring, fused/spiro/bridged, N-N/O-O, and aromatic `[nH]` gates.
