# T4 Route-Distilled Dynamic COMPOSE

## Problem and output

Dynamic COMPOSE can express the successful JAK2 development transformation,
but its stochastic proposer assigns too little probability to the required
module, binding and parameter combination. The primary output of this
milestone is one shared, context-conditioned distribution over complete
executable programs:

\[
q_\theta(P,\xi,\mathrm{stop}\mid x,A_t).
\]

The generated object is a protected, bounded program over exact molecular
rewrite actions. The controller chooses relative structural roles and created
handles, not source-specific atom indices.

## Claim under test

Successful T4 routes can supervise a generic program proposer that raises the
rank and prospective recovery rate of Full-146-class transformations without
loading executable teacher routes at runtime. This is trained-on-T4
performance. It is not winner-independent or held-out T4 evidence.

## Data and support

- Training data: all 77 entries whose sole block is
  `compiled_complete_transformation` in the frozen 146-program bank.
- Representation: the current exact persistent-slot molecular state during
  teacher replay; route labels use relative input/created roles.
- Runtime support: the existing generic shallow and structured program
  language, up to three learned high-level decisions, 32 primitives, eight
  blocks, 40 active atoms, neutral charge-preserving edits and the declared
  T4 task adapter.
- Runtime exclusions: target and protein identity, benchmark seed identity,
  route IDs, endpoint molecules, absolute source atom addresses, target-route
  maps and executable stored teacher programs.

The 32-primitive limit covers all 77 frozen complete routes. This observation
does not establish that the learned proposer will recover them.

## Baselines and ablations

- Full-146 is the literal-route teacher/reference ceiling.
- Dynamic-v0, v1, v2.1 and v2.2 are read-only dynamic comparators.
- A global source-balanced marginal route prior tests whether molecular
  context adds value.
- The unchanged generic proposal is the no-distillation control.
- The runtime keeps an unchanged generic shallow exploration lane so learned
  imitation cannot silently reduce program support.

## Zero-oracle gates

### Gate 1: teacher ingestion and exact replay

Execute each complete teacher route from its declared source through the
production binder and executor. Report coverage and execution precision
separately. Launch requires 77 of 77 admitted traces and exact replay precision
of 1.0. Missing or ambiguous source assets block the gate; no route may be
silently replaced.

### Gate 2: compression and support

Segment each exact trace into recognized high-level decisions. Preserve every
uncompressed primitive as an explicit generic fallback. Report high-level
decision count, primitive count, compression ratio, dependency depth and
created-handle reuse per route. This gate reports coverage, precision and
abstentions. It does not claim a globally minimal decomposition.

### Gate 3: proposal probability and rank

Fit the policy on training-role routes only for any held-out diagnostic. For
the all-route performance arm, all 77 routes may be used. Compare conditional
teacher-decision negative log likelihood and rank against both the unchanged
generic prior and a source-balanced marginal teacher prior. The gate passes
only if the context-conditioned policy improves source-balanced teacher
negative log likelihood over both baselines without reducing the frozen
exploration floor. Report train and any source-heldout diagnostic separately.

Proposal rank is measured over the exact applicable candidate row exposed to
the policy. It is not autonomous endpoint recall unless the proposer actually
generates and exact-executes the endpoint without injecting the teacher.

### Gate 4: runtime and provenance

The serialized runtime checkpoint and launch payload are scanned and parsed to
prove that forbidden identifiers and executable teacher records are absent.
The preflight also verifies deterministic loading, finite normalized
probabilities, an unchanged shallow proposal stream, exact execution and zero
oracle calls.

## Prospective five-cell pilot

Only after the four gates are sealed, run 5HT1B seed 0, BRAF seed 1, JAK2 seed
1, PARP1 seed 0 and FA7 seed 0 with the existing replicate-0 controller and
docking seeds. Each cell has a 1,000-call ceiling. The run uses no plateau
stopping, no confirmation calls and no automatic retry. A failed charged query
remains charged and blocks implicit continuation until an explicit recovery
contract is recorded.

Report best eligible docking score at calls 50, 100, 200, 400, 600, 800 and
1,000 when available, plus exact final query counts, candidate shortfalls,
proposal and docking time, and final lineage. Compare at matched calls with
read-only prior artifacts.

## Evidence boundaries

Exact teacher replay is representability evidence. Retrospective option
likelihood is supervised imitation evidence. Only the sealed scored pilot is
prospective trained-on-T4 optimization evidence. Literal Full-146 replay is
never autonomous recovery, and no score match is guaranteed before the scored
experiment.
