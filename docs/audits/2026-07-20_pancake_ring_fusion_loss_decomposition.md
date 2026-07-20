# Pancake fused-ring loss decomposition and root fix — 2026-07-20

## Decision

Kill earliest commutation as the immediate ring fix.  It preserves endpoints
but misses the preregistered support gate, expands placement competition, and
does not address the observed topology loss at rollout states.

The connected ring-system operator itself is correct.  The smallest supported
ground fix is an explicit learned hierarchy over ring topology class and cycle
sizes before complete-template selection.  That mode is now implemented as
`topology_cycle_hierarchical`, is zero-started, and leaves old checkpoints in
the exact existing `flat` mode unless explicitly requested.  No training was
launched.

Exact row-level support and score data are in
`diagnostics/pancake_ring_earliest_commutation_falsification32.json`.  The
cross-layer summary, input hashes, and rollout action counts are in
`diagnostics/pancake_ring_fusion_loss_decomposition.json`.

## Is a fused system one action or a sequence of rings?

One connected cyclic component is one atomic `RingSystemGrow`.  The compiler
removes graph bridges, joins fused, bridged, and spiro rings into connected
cyclic components, and emits one complete action per component.  Across all
4,000 frozen path records:

- all paths report `atomic_full_ring_systems`;
- none of the 9,946 ring-action component sets overlap; and
- teacher actions comprise 7,210 single rings (72.49%), 2,606 bridged/fused
  systems (26.20%), 128 spiro systems (1.287%), and 2 macrocycles.

Disjoint polycycles are different.  Each disconnected cyclic component gets
its own action.  Among 2,000 unique endpoints, 1,665 contain two or more
disjoint ring systems: 691 have two, 657 have three, 273 have four, and 44 have
five.  Therefore increasing the total number of ring actions can add several
separate single rings without increasing fused prevalence.

## Where fused probability is lost

### 1. Completed rollout actions already have the wrong topology mix

The preserved 600 calibrated rollouts contain 1,274 committed ring actions.
Their canonical before/after states recover the installed cyclic component
without ambiguity (zero reconstruction failures):

| action topology | teacher | calibrated rollout |
|---|---:|---:|
| single ring | 72.49% | 84.69% (1,079/1,274) |
| bridged/fused | 26.20% | 14.99% (191/1,274) |
| spiro | 1.287% | 0.314% (4/1,274) |
| macrocycle | 0.020% | 0% |

The generated bridged/fused action share is only 57.2% of the teacher share.
This loss occurs before final endpoint taxonomy; endpoint thresholding then
amplifies it.  It is not merely a fused-classification artifact.

### 2. Global catalog coverage is broad, but exact held-out patterns are not

The checkpoint contains 4,096 raw catalog entries and 2,074 deduplicated
production templates.  In the deterministic 32-row fused/spiro panel, exact
legal support is not small:

- original support: median 584 templates (range 158--1,274);
- earliest-commuting support: median 1,097 (range 228--1,592); and
- neither arm has a support set of five or fewer or a small-ring-only set.

However, only 26/32 teacher complete patterns have an exact production-template
representative.  The other six actions remain executable but have no exact
template under the checkpoint catalog.  On the 26 covered rows, median
conditional probability of the exact teacher template is only 0.00213 at the
original state.

Thus “the catalog has many fused templates” and “the held-out fused pattern is
represented and receives useful probability” are not equivalent claims.

### 3. The flat selector does not suppress the broad fused class on teacher states

At original late teacher states, median bridged/fused mass is 0.1584 under the
empirical prior and 0.3159 under the learned flat selector.  The learned/prior
ratio is 1.974.  A global fused logit boost is therefore contradicted by the
teacher-state score panel.

The discrepancy is state distribution and factorization.  The current model
uses one global query against 2,074 independent complete-template keys plus a
template prior.  There is no learned topology/cycle-size decision that shares
evidence across related fused templates.  It can fit broad fused mass at the
late teacher states yet commit only 14.99% fused actions on its own states.

### 4. Timing compounds the selector shift

Moving the exact teacher action earlier lowers median ring-family probability
from 0.5351 to 0.1046 at the action's native earlier time.  Holding time fixed
at the original value raises it to 0.2603, still below the original state.
Likewise, early-state learned fused mass is 0.1940 at native time but 0.3543
when scored at the original late time.

The model learned the compiler's late ring presentation.  Exact commutation
cannot by itself make those earlier states probable under the frozen model.

### 5. Placement exists; commutation makes its competition worse

All 32 teacher actions have empty interfaces and no atom insertions, matching
the structured v1 placement contract.  For every covered template, the exact
teacher placement is present in both states (26/26 original and 26/26 early).

But commutation expands the competing placement table from 193 total
placements to 643.  Per covered row, the median grows from 6 to 18.  Median
probability of the exact placement falls from 0.5904 at the original state to
0.1750 at the native early state (0.1956 even at controlled original time).
The failure is not a missing interface matcher; earlier underdecorated states
create more placement aliases and dilute the correct one.

## Preregistered scheduling gate

The 32 distinct paths all replay to their exact endpoints, and the selected
teacher action remains executable in all 32.  Twenty-nine actions move earlier
by a median 5.5 positions (maximum 14).  Nevertheless:

- median admitted bridged/fused prior mass increases only 1.186x, below the
  required 2x;
- degenerate support rows remain 0 to 0, so a 50% reduction is impossible; and
- small-ring-only rows remain 0 to 0.

Result: **FAIL; kill scheduling as the immediate ring fix.**

## Implemented causal fix

`FactorizedTraceletRateModel` now accepts
`ring_template_factorization="topology_cycle_hierarchical"`.  In that mode:

1. a shared head predicts one of the existing 130
   `(topology_class, cycle_sizes)` groups;
2. the existing template logits are normalized conditionally inside each legal
   group; and
3. group probability is therefore not multiplied by the number of templates
   in that group.

The group head is zero-initialized, so its initial residual is neutral over the
empirical group prior and receives gradients immediately.  The new
`ring_topology_only` training scope freezes the encoder, family timing, hazard,
Graft, exact-template keys, placement scorer, and electronic decoder.  This
isolates the diagnosed seam rather than reopening the coupled P1/P2 system.

Checkpoint loading and the training CLI carry the new mode explicitly.  Old
checkpoints default to `flat` and retain their prior state-dict shape.  The
bounded, not-yet-launched execution contract is
`recipes/tree_fcd_transfer_unconditional_ring_topology.json`.

This is aligned with the conditional-generator plan.  The mode is checkpoint
metadata understood by the ordinary rollout loader and compatible
initialization path, so a qualified topology-repaired unconditional backbone
can seed later conditional training.  Nothing here freezes the eventual
conditional model's parameters; `ring_topology_only` is merely the isolation
scope for this causal unconditional pilot.

## Required promotion gate

Do not promote this architecture on teacher loss alone.  After an explicitly
authorized bounded pilot, require a matched rollout audit showing:

- committed bridged/fused action share moves materially toward 26.20% without
  reducing total ring actions per molecule;
- cycle rank and fused prevalence improve together;
- small-ring, spiro, heterocycle, aromatic-role, and bond-order chemistry do
  not regress; and
- validity, connectivity, event-budget exhaustion, and throughput stay within
  the incumbent gates.

The new hierarchy cannot solve the six uncovered exact patterns by itself.
If those remain outcome-limiting after topology recovery, catalog truncation
or a compositional topology decoder is a separate, later problem—not a reason
to add held-out templates or install every ring as aromatic.
