# IVG winner paths: diagnostic only

2026-09-09. User-authorized offline analysis, not a controller change or training
run. COMPOSE remains the executable stochastic molecular editing framework.
The output here is an atom mapping and, where found, an executor-replayed path
from an actual benchmark seed to a released winner's supported 2D graph.
This tests constructive accessibility and exposes constraint/value barriers;
it does not test blind discovery, shortest paths, or docking causality.

Reuse the complete pinned census at
diagnostics/ivg_t4_census/census.json (SHA-256
2c06fdfe65fc31327d865381ccb8f4d2c40553c3925d9dc0a03386c22fda05bb).
Preserve all cell/run/co-best identities, deduplicating only identical
source/target pairs for computation. Use the corresponding original seed, not
a conveniently selected closer molecule. Initialize that seed once from its
original registry SMILES, then retain exact persistent-slot states throughout.
Do not reconstruct intermediate states from canonical SMILES.

SMILES are parsed for graph correspondence, never character-differenced.
Try a connected common-subgraph mapping with element matching and a separately
labelled topology-only mapping. Match formal charge. Mapping timeout and a
small target-directed search are diagnostic work allocations, not statements
of model support. Search failure is unresolved, never proof of unreachability.
An atom-count/cycle-rank lower bound is separate from the length of a found
witness, which is only an upper bound. Report mapping truncation/timeouts.

Use the existing Active8 semantic executor, at most 40 active atoms, one-neighbor
birth, exact charged-center preservation, no null or disconnected committed
states, and no endpoint pasting or whole-ring insertion. Primitive proposals
come only from differences to the labelled diagnostic target. They never enter
the optimizer or its priors. Every accepted path is replayed independently from
its saved exact source using the production action codec. Local RDKit version
is recorded; a local witness is not a claim of pinned-Modal-runtime equivalence.

For every path record actions, states, topology, QED, SA, original-seed similarity,
constraint failures and one-step region touch-filter eligibility. This is not
full context-invariant, macro-contract, learned-fiber or bundle admissibility.
Compare witness length
with the existing 16-edit horizon, without treating longer witnesses as proof
that no shorter path exists. Reuse the saved frozen PARP1 seed0 value snapshot
only for that cell, check its stored feature identities, and report raw predictor
values separately from feasibility-gated values. Other targets abstain from
task-value scoring because no applicable snapshot has been supplied. No R_theta
enumeration or fresh docking is authorized by this offline stage.

These inspected winners are development material. Do not extract a proposal
catalog, fit to their labels, or select a benchmark method against their
recovery and call the result held out. Any subsequent controller repair needs
a separately recorded decision and untouched evaluation. The same-generator
post-hoc baseline remains required for a causal guidance-performance claim.

Minimum checks: identity/SMILES-order invariance, a pendant/fused/ring-expansion
witness, charge-policy rejection, explicit incomplete search, and serialized
action replay. Focused checks only; no repository-wide suite for this diagnostic.
Run from a clean committed worktree, with absolute `--census`, `--snapshot` and
`--output` paths into the main workspace:
`PYTHONPATH=src python3 tools/ivg_winner_paths.py --help`.
One CPU worker, no accelerator, no docking; two one-second MCS searches and up
to 128 expanded states per mapping per unique seed-target pair. Expected wall
time is minutes, not a measured guarantee. Restart unit is a hash-checked pair
receipt; completed compatible pairs are replay-checked and reused. Results
include all pairs and an explicit failure category, not only recovered winners.
