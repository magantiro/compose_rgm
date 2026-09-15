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

## Authorized scored diagnostic result, 2026-09-15

The user subsequently authorized the exact 19-request lock at physical SHA-256
`23ccc7250d5f6ac20af3609047b39730a47ab414bba01cf8c7c0479ae2681659`
and payload SHA-256
`bc0b6ffdbd229e3ba6cf143f53fcc37003326138c43fa249aef629131360a15b`.
The launch used 19 independent Modal workers, one first-score request per worker.
All 19 requests completed. There were zero failures, retries, replacements, or
backfills. The immutable remote run identifier is
`e7bd3e04255a72bef7f48eae10791188a8af856f1d0d12abd49d96794374d398`;
the app identifier is `ap-3EmwfjNd1nfuWzZQsIS5op`. Exact worker call identities
are retained in
`diagnostics/t4_delta06_structural_subgoal_utility_launch_spawn.json`.

The reduced result is
`diagnostics/t4_delta06_structural_subgoal_utility_launch/attempt_1/result.json`
(physical SHA-256
`e717cc9bfccde01caf20b8cc1b31bc0e500711f96221ff1e797461605480c6d7`,
payload SHA-256
`b3e2189ebda86ecb9dc2b56068fefa7e8784820068957fc74c3be12044ff0f66`).
The deterministic comparison is the adjacent `review.json` (physical SHA-256
`ac7327bf65b409146518ea1af43c849982d33607839bf4c1cf10c5a6183319bb`,
payload SHA-256
`7a5439d769b0f06dc725e298140ec2d5bd74e529470ed335577bfa3e31408a38`).

For each scored cell, the descriptive cell best improved on the published source
reference in six of eight cells, tied in one, and was worse in one. Across the 13
scored cell-policy pairs, nine improved on the source reference, three tied, and
one was worse. No cell-policy best matched or improved on the frozen reported IVG
mean. These comparisons use lower-is-better docking scores. A positive reported
gain means `reference_score - best_locked_score` is positive.

This is prospective endpoint evidence under the exact bound QuickVina evaluator.
The IVG values are reported comparator means, not matched prospective reruns.
Candidate traces remain unavailable, so the result does not establish route
recovery, executable proposal precision, teacher recovery, or superiority to IVG.
