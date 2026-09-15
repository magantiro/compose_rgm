# T4 strict-delta-0.6 structural-subgoal utility lock

## Scope

This zero-oracle milestone audits the 3,840 already frozen `attempt_2`
structural-subgoal proposals and locks a small panel for a possible later docking
diagnostic. It does not generate candidates, inspect teacher endpoints or scores,
call an oracle, launch Modal, or access a live run. The self-hashed protocol is
`configs/t4_delta06_structural_subgoal_utility_lock_v1.json`.

The generated object is an existing complete endpoint molecular graph. Candidate
traces are absent from the frozen proposal locks. Consequently, endpoint utility
cannot establish teacher recovery, route recovery, or executable candidate-route
precision.

## Eligibility and selection

The implementation recomputes validity, connectivity, canonical SMILES, heavy and
active atom counts, non-self status, quantitative estimate of drug-likeness (QED),
synthetic accessibility (SA), and source similarity under RDKit 2024.03.5. Morgan
similarity uses radius 2, 2,048 bits, and no chirality. An endpoint is eligible only
when similarity is strictly greater than 0.6, QED is strictly greater than 0.6, SA
is strictly less than 4.0, and active atom count is at most 40.

Within each frozen source and policy, the lock selects the lowest-rank unique
eligible endpoint. When another unique endpoint exists, it also selects the one
with maximum Morgan distance from the first. Distance ties use rank, canonical
SMILES, and attempt identifier in ascending order. Physical requests are
deduplicated by target, canonical molecule, docking seed, and exact evaluator
identity while retaining every source-policy membership.

The preregistered count checks are 22 memberships, 19 unique requests, eight cells
with requests, and seven cell abstentions. The command fails rather than publishing
if authoritative recomputation differs.

## Reproduction

Run from a clean committed worktree with Python dependencies pinned to RDKit
2024.03.5 and NetworkX 3.5:

```bash
PYTHONPATH=src:. uv run --offline --isolated --no-project \
  --with rdkit==2024.3.5 --with networkx==3.5 -- \
  python tools/t4_delta06_structural_subgoal_utility_lock.py
```

The command publishes the complete eligibility ledger, candidate lock, request
lock, abstention ledger, aggregate result, and interpretation under
`diagnostics/t4_delta06_structural_subgoal_utility_lock/attempt_1`.

## Launch boundary

Publication is not launch authorization. A later scored diagnostic requires a new
explicit authorization naming the exact physical SHA-256 of `request_lock.json`,
with a ceiling of 19 first-score docking calls and no retry, replacement, or
backfill. Even a favorable scored result would demonstrate endpoint utility only.
