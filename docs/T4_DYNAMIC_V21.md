# T4 Dynamic COMPOSE v2.1 pooled-channel development

## Scientific identity

- **Problem:** Dynamic-v0 is productive on shallow route-free transformations,
  while Dynamic-v1 has produced a substantially stronger 5HT1B development
  result but is not uniformly efficient. A fixed pre-generation channel split
  can waste finite oracle allocation.
- **Primary output:** two independently generated, exact-executed and
  non-oracle-filtered candidate pools, followed by one bounded allocation over
  their eligible novel union.
- **Claim under test:** post-filter arbitration can retain v0-like opportunity
  while adding v1's structured transformations. This is proposed and is not
  established by implementation, preflight or retrospective evidence.
- **Validation:** one shared configuration on PARP1 seed 0, FA7 seed 0,
  5HT1B seed 0, BRAF seed 1 and JAK2 seed 1 at delta 0.4, compared read-only
  with available Dynamic-v0, Dynamic-v1 and Full-146 results at matched call
  counts.
- **Support:** connected charge-preserving 2D persistent-slot molecular graphs,
  broad-organic represented elements, at most 40 heavy atoms, 32 primitives,
  eight blocks and one to three generic high-level modules. Stereochemistry
  and formal-charge changes remain outside support.

## Frozen controller recipe

For each selected parent, v2.1 independently attempts one shallow and one
structured proposal. The shallow stream uses the unchanged Dynamic-v0 proposal
implementation and its own RNG. The structured stream uses Dynamic-v1's generic
protected ring-path remodeling, joint substituted-ring construction and
context-conditioned binding implementation and a different RNG. Parent choice
and candidate arbitration use a third RNG. Advancing either proposal stream
must not advance the other.

Each channel may make at most 128 proposal attempts and retain at most 16
eligible novel endpoints per batch, with its own 45-second soft work bound.
The union therefore contains at most 32 endpoints before arbitration. The
charged batch retains at most 16. These proposal-work limits are deliberately
larger than one v0 pool and must be reported separately from oracle calls.

All candidates compile to the same program representation and use the same
exact executor. Exact validity, atom-count support, QED, SA, original-seed
similarity and canonical endpoint duplication are checked before arbitration.
Different routes to the same endpoint do not create additional query slots.
Intermediate task scores are never evaluated.

The allocator uses a deterministic bounded UCB-style channel score based only
on charged observations in the current run. If both channels have eligible
novel endpoints and at least two query slots remain, each receives one
exploration-floor selection. Remaining selections use smoothed parent-
improvement yield plus an uncertainty bonus, updated after every scored batch.
No protein or benchmark identity enters the allocator. When the eligible union
fits the query allowance, every endpoint is selected and no artificial ranking
decision is made. A channel with no eligible novel endpoint consumes no oracle
call.

All scored endpoints enter one common run-local archive. Shallow mutation and
recombination may refine routes first found by structured synthesis, and
structured synthesis may start from parents first found by shallow search. The
task-specific complete-route archive is empty at initialization.

## Zero-oracle preflight

The preflight is sealed before scoring and verifies:

1. the initial program library and complete-route archive are empty;
2. Full-146, Dynamic-v0 and Dynamic-v1 outcomes are absent from runtime inputs;
3. the shallow synthesizer implementation is unchanged;
4. shallow, structured and arbitration RNG streams are independent and exactly
   restorable;
5. fixed representative legal transformations from both channels compile and
   exact-replay in the pinned environment;
6. the combined pool, strict filter and allocator run deterministically on all
   five development cells, with strict-eligible yield recorded rather than
   used as an arbitrary stochastic launch gate;
7. an empty structured or shallow pool falls back to the other pool;
8. preflight makes zero oracle calls;
9. candidate, channel, filter, allocation and implementation provenance are
   complete.

The preflight does not require either isolated stochastic channel to find a
strict-eligible endpoint from every raw benchmark source within a fixed draw
count. That was the failed Dynamic-v2 gate and does not describe v2.1 runtime.

## Scored-run limits

- units: `parp1_0_r0`, `fa7_0_r0`, `5ht1b_0_r0`, `braf_1_r0`,
  `jak2_1_r0`;
- at most 1,000 docking calls per unit and 5,000 total;
- five concurrent single-CPU workers, no GPU;
- original replicate-0 search and docking seeds;
- strict endpoint similarity greater than 0.4, QED greater than 0.6, SA less
  than 4, and at most 40 heavy atoms;
- original competitive-plateau rule and round limits;
- no confirmation calls or automatic retry;
- a namespace and ledger distinct from v0, v1, v2, 69-only and Full-146.

Every generated proposal, rejection, eligible pool member, cross-channel
duplicate, allocator score, selected query, failed query, endpoint score,
archive update, channel credit, proposal time and score curve is durable and
resumable.

## Prohibited

- modifying or relaunching Dynamic-v2 or changing its failed gate;
- modifying, restarting or cancelling live Dynamic-v1;
- loading known routes, winner endpoints, comparator scores or target maps at
  runtime;
- target-specific channel settings or search rules;
- neural policies, MCTS, SMC, reference inference or unbounded program depth;
- any fourth cell, replicate, confirmation, delta-0.6 or PMO call;
- describing this answer-known five-cell development study as held-out or a
  complete T4 benchmark.
