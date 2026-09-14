# T4 Dynamic COMPOSE v2 implementation

## Scientific identity

- **Problem:** Dynamic-v0 has an efficient shallow route-free proposal law,
  while Dynamic-v1 adds needed protected transformations but may dilute that
  law when exposed too often.
- **Primary output:** a generic parent-specific choice between a shallow and a
  structured proposal channel, followed by one complete exact-executable
  molecular program.
- **Claim under test:** selectively invoked structured planning can preserve
  Dynamic-v0's early efficiency while improving transformation regimes that
  require protected ring construction or remodeling. This is proposed, not a
  completed result.
- **Validation:** one frozen configuration on 5HT1B seed 0, BRAF
  seed 1 and JAK2 seed 1 at delta 0.4, compared with Dynamic-v0, Dynamic-v1 and
  Full-146 at matched call counts. The user authorized this three-container
  scored launch after the focused and zero-oracle gates pass.
- **Baselines and causal ablation:** Dynamic-v0 is the shallow baseline,
  Dynamic-v1 is the always-exposed richer comparator, and Full-146 is the
  read-only stored-program ceiling.
- **Support:** connected exact persistent-slot molecular graphs, broad-organic
  supported elements, at most 40 heavy atoms, charge-preserving 2D graph edits,
  at most 32 primitives and eight blocks per complete proposal. Stereochemistry
  and formal-charge changes remain out of scope.

## Controller contract

Dynamic-v2 uses the existing Dynamic-v0 and Dynamic-v1 compilers and the same
exact executor. It adds only an interpretable planner gate and resumable
channel-credit state.

The default structured probability is 0.25 and is always clamped to [0.10,
0.50]. The gate may depend on generic molecular opportunity, parent-specific
duplicate/rejection/eligibility history, bounded global stagnation and
within-run channel productivity. It may not inspect a target name, benchmark
seed identity, known winner, Full-146 program, endpoint or score.

The shallow channel retains Dynamic-v0 synthesis and ordinary mutation and
recombination. The structured channel invokes Dynamic-v1 protected ring-path
remodeling, substituted-ring construction and contextual binding enumeration.
Both compile to the same program representation and execute under the same
work limits. Intermediate task-oracle evaluation remains zero.

At the start of each task-specific run, the complete-route archive is empty.
Only routes scored within that run may enter the mutable/recombinable archive.

## Required instrumentation

Each gated synthesis records the parent, generic gate features, structured
probability, sampled planner mode, high-level modules, binding and execution
metadata. Resumable state records per-channel and per-parent proposal,
execution, eligibility, duplication, docking and improvement counts, plus the
number of charged observations since the last global-best improvement.

## Acceptance criteria for implementation

1. The shallow branch delegates to the existing Dynamic-v0 synthesizer.
2. The structured branch delegates to the existing Dynamic-v1 structured
   synthesizer without loading a program bank.
3. The same generic gate applies to all molecules and is bounded to [0.10,
   0.50].
4. Parent stagnation raises structured probability, while both channels retain
   explicit probability floors.
5. Channel credit distinguishes execution/eligibility outcomes from measured
   docking improvements. Unscored and failed candidates never receive invented
   task rewards.
6. Snapshot and restore preserve gate state and future proposal decisions
   exactly.
7. A zero-oracle structural check produces eligible candidates through both
   channels and records complete planner provenance.
8. Focused tests, Ruff, formatting and whitespace checks pass. Do not run the
   repository-wide suite as an iteration gate.

## Scored-run limits

- three independent single-CPU workers, one per development cell;
- at most 1,000 docking calls per cell and 3,000 new calls total;
- the same replicate-0 controller seed, docking seeds, endpoint rules and
  competitive-plateau policy as v0/v1;
- no retries, confirmations, GPU or additional cells;
- a separate namespace and ledger from the still-running Dynamic-v1 study;
- Dynamic-v0, Dynamic-v1 and Full-146 are joined offline for reporting only;
  their outcomes are absent from the v2 runtime image and cannot influence
  search.

## Prohibited in this milestone

- modifying, restarting or cancelling any v0, v1, 69-only or Full-146 run;
- making more than the authorized three-cell v2 launch or any confirmation
  call;
- loading target-specific routes or winner-derived runtime payloads;
- target-specific gates, hyperparameters or horizons;
- neural policies, MCTS, SMC, future-value models or unbounded depth;
- evaluating the remaining T4 cells, delta 0.6 or PMO.
