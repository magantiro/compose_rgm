# COMPOSE v4

**Rewrite Generator Matching for validity-preserving molecular generation.**

COMPOSE v4 is a clean research implementation of a learned continuous-time
jump process on molecular graph rewrite rules. Molecules are complete states;
typed, executable rewrites are transitions; a neural model learns their time-
dependent firing rates with Generator Matching.

The project intentionally starts smaller than `compose_v3`. It reuses the
chemistry state representation and validity semantics that survived testing,
but does not inherit the old destruction-noising objective, oracle inversion
labels, beam/candidate-bank ring decoder, or MIS-centric reverse sampler.

## Start here

For a collaborator taking over the project, read
[`docs/HANDOFF.md`](docs/HANDOFF.md) first. The complete artifact map is in
[`docs/ARTIFACT_INDEX.md`](docs/ARTIFACT_INDEX.md), and the dated live workstream
is in [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md). The three canonical
HTML research plans are versioned in [`docs/research_plans/`](docs/research_plans/),
and the full historical “pancaking” trajectory that motivated the Graft
successor quotient is preserved under
[`docs/trajectory_diagnostics/legacy_prequotient/`](docs/trajectory_diagnostics/legacy_prequotient/).

The step-6,250 trajectory and FCD bundle is deliberately labeled legacy: it
predates the exact successor quotient and current semantic ring-support code.
It is diagnostic evidence for a repaired failure mode, not a result of the
corrected generator.

## Base formulation

For the legal action fiber `A_c(G)` at molecule `G` under condition `c`,

```text
(L_theta,t,c f)(G)
  = integral_[a in A_c(G)] [f(T_a(G)) - f(G)] q_theta,t(da | G, c).
```

The base implementation now includes a deterministic validity-closed runtime,
random legal trace compiler, conditional CTMC teachers, a whole-graph rate
network, canonical successor aggregation, and target-free ancestral sampling.
The production scaling path uses the stochastic-rewriting lift: one batched
graph encoding predicts a total hazard and normalized probabilities over
analytically legal rule matches. It instantiates only the sampled mark; the
exhaustive canonical-successor fiber remains a correctness oracle.

See [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) for the staged plan and
[docs/V3_REUSE_LEDGER.md](docs/V3_REUSE_LEDGER.md) for the reuse boundary. The
distinction from fragment assembly is explicit in
[docs/FRAGMENT_METHOD_BOUNDARY.md](docs/FRAGMENT_METHOD_BOUNDARY.md). See
[docs/GENERATOR_MATCHING_INTUITION.md](docs/GENERATOR_MATCHING_INTUITION.md) for
the relationship to flow matching and diffusion. The first held-out corpus gate
is documented in
[docs/CNOF_CONDITIONAL_GATE.md](docs/CNOF_CONDITIONAL_GATE.md).
The scoped literature claim and distinctions from Morph, Edit Flows, DDSBM,
graph grammars, and fragment assembly are maintained in
[docs/NOVELTY_POSITIONING.md](docs/NOVELTY_POSITIONING.md).
The universal micro ring semantics and implemented verified cycle/ear/electronic
tracelet hierarchy are specified in
[docs/RING_REWRITE_ARCHITECTURE.md](docs/RING_REWRITE_ARCHITECTURE.md).
The structured alkane-tree source, atomic bridge-reroute semantics, and required
reachability/ring-commitment gates are specified in
[docs/TREE_SOURCE_TRANSPORT.md](docs/TREE_SOURCE_TRANSPORT.md).
The exact meanings of source prior, teacher coupling, proposal support, learned
rates, sampling, and future guidance are summarized in
[docs/CURRENT_MODEL.md](docs/CURRENT_MODEL.md).
The evidence-based transfer from the sibling `rank_d500k` model that reported
FCD 9.96, including the frozen first tree-source quality recipe and covariance
diagnostic, is documented in
[docs/FCD_TRANSFER_FROM_RANK_D500K.md](docs/FCD_TRANSFER_FROM_RANK_D500K.md).

## Development setup

```bash
python -m pip install -e ".[dev]"
pytest
```

## Current status

- Molecular graph state and RDKit validity layer imported from v3.
- Minimal executable rewrite basis implemented.
- Checkpoint-compatible structured-source mode implemented: an explicit
  degree-bounded carbon-tree prior, primitive leaf-delete/regrow transport,
  atomic subtree Graft, and flexible-size grow/shrink-plus-Graft transport.
  Every compiler reaches the exact target through valid connected states; the
  production Graft paths commit each complete cyclic component with one atomic
  `ring_system_grow` event. The learned factorization contains no ring-ear or
  scalar-closure family. Ring marks use a hierarchical topology, exact legal
  scaffold-match, and valence-masked atom-label decoder rather than a flat
  enumeration of every topology-placement pair.
- Rewrite registry, hard-condition hooks, and successor-rate alias aggregation
  implemented.
- Exact null-to-target compiler reconstructs acyclic, charged, monocyclic,
  fused, bridged, and spiro fixtures through valid micro-rewrite programs.
- Exact inverse-program construction and randomized slot-permutation fuzzing
  implemented.
- Phase 0 stress gate passed: 10,020 validity-checked commits, 580 exact forward
  and inverse programs, and 580 canonical slot-permutation matches.
- Closed-form conditional progress CTMC and complete-successor rate Bregman
  objective implemented and numerically tested.
- Exact connected C/N/O/F action fiber over all six micro-rule families.
- Slot-quotiented, valence-factorized C/N/O/F fiber with exact successor-set
  equivalence and 21-30x measured evaluation speedup on representative states.
- Hierarchically normalized total-hazard, rule-family, and operand rate model.
- Permutation-equivariant whole-graph neural marked-rate model implemented.
- Dense marked-rewrite backend implemented: batched graph/message encoding,
  tensorized family/template/site masks, deterministic persistent data workers,
  pinned prefetch, BF16 H100 training, direct ancestral mark sampling, and an
  exact index-stable recovery stream. The measured steady-state batch-32 H100
  update is 0.023-0.032 seconds instead of 9-15 seconds for exhaustive fibers.
- Deterministic 16-process path compilation now stores exact byte-packed graph
  checkpoints instead of every dense intermediate, writes atomic proposal and
  transport shards, and resumes from a signature-validated manifest. On 54
  representative 40-slot Graft paths, interval-8 checkpoints reduced serialized
  storage from 14.62 MB to 1.38 MB (10.6x); the previous full 99,738-path cache
  was measured at 29.0 GiB. Modal runs compile on CPU before an H100 stage that
  refuses to start without a complete cache. The compile/load/resume workflow
  passes both local and real-volume Modal smoke tests.
- First end-to-end learned gate passed: 1,000 target-free rollouts were 100%
  valid/connected, 99.8% non-null, 99.5% on tiny reference support, recovered
  all reference modes, and reached total variation 0.0523.
- First held-out corpus gate completed on a 96/16/16 neutral C/N/O/F split. The
  selected late-time-calibrated checkpoint produced 100/100 valid, connected,
  non-null, unique, target-free samples; mean size was 10.05 versus 10.50 and
  mean cycle rank 1.39 versus 1.19 in the reference corpus.
- A full-scan 1,333-molecule C/N/O/F gate and matched corpus-marginal rewrite
  baseline are complete. The first prior tilt gave a mixed structural result.
  The follow-up topology-aware, trust-regularized generator uses deferred bond-
  order teachers and retains 100% valid/connected/non-null/unique sampling. It
  improves matched-prior FCD from 22.24 to 19.42, ring-size TV from 0.284 to
  0.054, cycle-rank TV from 0.254 to 0.094, QED from 0.511 to 0.539, and SA from
  5.727 to 4.771. Aromaticity remains the dominant measured bottleneck.
- Full result boundaries and diagnostics are recorded in
  [docs/TINY_RATE_GATE.md](docs/TINY_RATE_GATE.md) and
  [docs/CNOF_CONDITIONAL_GATE.md](docs/CNOF_CONDITIONAL_GATE.md). The scaled
  comparison is in [docs/SCALE16_GATE.md](docs/SCALE16_GATE.md).
- A block/ear tracelet compiler, finite C/N/O/F tracelet fiber, exact inverses,
  and validity-checked micro lowerings are implemented. The compiler builds a
  neutral carbon ring scaffold before ring-contextual atom/electronic
  refinement; it exactly handles monocyclic, fused, bridged, spiro, cage,
  macrocyclic, heterocyclic, aromatic, and partially unsaturated fixtures.
- The 1,067-molecule training split compiles in 11,952 visible events versus
  21,909 verified micro lowerings (1.83x path compression). A ring-frequency
  audit records both molecule-level topology prevalence and operator-mark
  frequencies; it is saved in `results/ring_tracelet_frequency_audit.json`.
- Tracelet teacher successors are verified to lie in the production marked
  fiber, and the factorized semantic ring decoder is integrated into both
  teacher scoring and ancestral sampling. The current gate is exact-support
  throughput followed by a fresh quotient-correct run; the archived step-6,250
  FCD and trajectories predate that correction and remain diagnostic only.
