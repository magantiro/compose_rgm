# T4 compositional-generator prospective utility lock

## Outcome

The zero-oracle score-blind lock audited 2,560 candidates
from 20 predeclared cell-arm units. It selected
8 memberships and merged them into
4 unique prospective requests. There were
12 cell-arm abstentions and
4 cross-arm merged memberships.

No oracle, docking function, Modal job or live run was accessed. The request lock
is not launch authority. No teacher endpoint, teacher transformation, teacher
metric, task score, docking score or evaluation manifest entered selection.

## Artifacts

- `eligibility_ledger.json.gz`: every in-scope candidate, exact states/actions,
  recomputed descriptors, eligibility and selection exclusions
- `candidate_lock.json`: at most one selected membership per cell/arm
- `request_lock.json`: canonical requests with all arm memberships preserved
- `abstention_ledger.json`: all 20 predeclared cell-arm units
- `result.json`: aggregate and cell-arm findings

Request-lock payload SHA-256: `420a2b063f1309493c7b283559e517b183a40e9f982fafc12b7093570b3b5247`

Physical SHA-256 values at publication:

- `abstention_ledger.json`: `ffc51e0bfb634c5b2848de0aa73b4cf1f5416f80b7e5895e8b5981047717e4e3`
- `candidate_lock.json`: `d9b90389443a5906179d93adeaa654bdd65742f60ea6e7da4e27aaf10e2eef60`
- `eligibility_ledger.json.gz`: `06ff8f02aebe99ba014c81941aab25b92fc952f786db10266e00d6143c4a4c5b`
- `request_lock.json`: `3d4a1712cda185d13e5eb66e4b8bc47db7e85960854b4132bb8710bdf7cf4469`
- `result.json`: `3d036d9af1e383f264d5ee02b3d840b2d12d4a4d1dcb3772ee0da7330d8464e6`

## Interpretation and launch boundary

This lock establishes only prospective, score-blind endpoint eligibility and
coverage. It does not establish endpoint utility, teacher or route recovery,
autonomous optimization, or an InVirtuoGen comparison. Any later score run needs
explicit authorization naming the exact physical and payload SHA-256 of
`request_lock.json`, its exact call ceiling, and zero retry, replacement or
backfill.
