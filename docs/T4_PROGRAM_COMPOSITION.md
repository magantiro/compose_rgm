# T4 protected program composition

## Identity and claim under test

- **Scientific problem:** useful T4 endpoint discovery can require several
  mutually dependent molecular changes that should complete before task-based
  pruning.
- **Primary output:** one exact, executable edit program compiled from a bounded
  variable number of reusable program components, with component bindings,
  dependency evidence, complete trace, endpoint and work receipt.
- **Claim under test:** adding a protected two-component proposal channel to the
  fast coordinated-program optimizer increases the density or quality of novel,
  eligible T4 endpoints at matched total work.
- **Setting:** winner-informed development on BRAF seed1 and JAK2 seed1 at the
  original-seed similarity threshold 0.4. This is not a held-out or matched IVG
  benchmark claim.
- **Control and ablation:** the retained fast program-only recipe is the causal
  control. The experimental recipe changes only the 0.25 allocation to protected
  composition. It introduces no predictor or reference-law call.
- **Support:** exact persistent-slot molecular graphs, the existing broad-organic
  vocabulary and executor, connected valid intermediates, at most 40 active atoms,
  at most 32 primitives and eight blocks per complete proposal. Stereochemistry
  and formal-charge editing remain outside the declared editing support.

## Architecture and frozen first experiment

The controller implementation supports compositions of 2 through configurable
K components with a stop decision after the second component. Primitive count,
block count, binding work and wall time are bounded across the whole composition,
not per component. The first experiment freezes K=2. This makes the architecture
extensible without treating untested deeper compositions as evidence.

Each component is a verified dependency-closed branch from the measured program
archive. It is rebound contextually on the exact predecessor. If a later binding
uses a persistent slot created by an earlier component, extraction retains the
created-atom handle and the compiled graph records the resulting dependency. The
complete composed program is replayed from its exact archived parent. No
intermediate task score or T4 endpoint gate is consulted.

The local structural comparison uses BRAF/JAK2 warm archives and search seeds
20260921 and 20260922:

| Setting | Control | Experimental |
| --- | ---: | ---: |
| Composition allocation | 0 | 0.25 |
| Maximum components | 2 | 2 |
| Programs per donor census | 8 | 8 |
| Candidate cap | 12 | 12 |
| Attempt cap | 128 | 128 |
| Wall cap per pool | 45 s | 45 s |
| Primitive/block cap | 32 / 8 | 32 / 8 |
| Reference calls and oracle calls | 0 / 0 | 0 / 0 |

Record every attempt, exact candidate and trace, eligible yield by channel,
canonical overlap between arms, changed-site count, component count, input handles
created by earlier components, proposal seconds and executor calls. Reusing a
previously scored endpoint is a duplicate, not a new candidate.

## Decision gate and scored pilot

The structural result is **positive** only if the composition channel produces at
least one novel eligible endpoint on both targets and at least four across the
four target/randomization pools, without exceeding 1.5 times the control's summed
proposal time. It is **negative** if no composition endpoint survives. Other
outcomes are inconclusive and stay local.

Only a positive result advances to docking. Before observing any new docking
scores, lock at most four candidates per arm and target from the completed local
pools using a declared deterministic channel/structure-stratified rule. Score all
distinct locked endpoints, then obtain one fresh evaluation of each target's
selected new champion and one fresh evaluation of its existing incumbent. Shared
endpoints consume one physical first evaluation. The hard ceiling is 20 new
docking calls, eight single-CPU workers plus one driver, no GPU, no automatic
retry and $10 reserved maximum.

The docking comparison keeps the original target receptor, preparation/search
protocol, seed-relative QED/SA/similarity gates and delta=0.4. Candidate locks,
docking seeds, ligand/pose hashes, receipts, failures, wall time and estimated
cost are immutable outputs. A first-score improvement is exploratory; fresh
observations and the full arm distributions remain separate.

Decision after docking:

- If the composition arm improves candidate quality with confirmation and no
  material throughput collapse, retain it and then test learned combination
  allocation.
- If it makes distinct molecules but no better scores, keep the general mechanism
  experimental and improve component/attachment choice rather than increasing K.
- If it mainly duplicates or fails, keep the fast single-program controller as
  default and do not increase composition depth.
- K greater than two requires a separately frozen comparison motivated by a
  specific residual failure.

No PMO execution, frozen-reference training, executor modification, hidden-oracle
screening or external benchmark-superiority claim is part of this milestone.

## Completed structural result

The final zero-oracle comparison is stored in
`diagnostics/t4_program_composition/comparison.json`. The first implementation
attempt produced no eligible composed endpoint. After adding bounded exact
feasibility checks before selecting a component, the composition channel produced
two novel eligible endpoints on BRAF seed1 and none on JAK2 seed1. Summed proposal
time was 64.99 seconds, compared with 62.35 seconds for the retained fast control
(ratio 1.042).

The preregistered gate therefore did not pass: only two composed endpoints were
produced in total and only one of the two targets had any. The result is
`NO_DOCKING_INCONCLUSIVE_STRUCTURAL_GATE`. No task oracle was called and the
20-call conditional scoring allowance was not activated. The general
composition implementation and its focused tests are retained, but composition
remains disabled in the frozen fast T4 controller. This negative result is a
development diagnostic, not evidence that all multi-program composition is
ineffective.
