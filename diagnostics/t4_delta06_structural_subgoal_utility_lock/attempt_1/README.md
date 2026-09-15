# T4 strict-delta-0.6 structural-subgoal utility lock

## Outcome

The zero-oracle lock passed its preregistered count checks. All 3,840
frozen attempts were audited under RDKit 2024.03.5. The 3,141
completed endpoints yielded 49 unique
strictly eligible source-conditioned endpoints. Deterministic within-source/policy
selection produced 22 memberships and
19 deduplicated prospective docking requests across
8 of 15 cells.

The seven explicit cell abstentions are: 5ht1b_0, 5ht1b_2, braf_0, fa7_0, fa7_1, fa7_2, parp1_2.

No oracle, docking function, Modal job or live run was called. The request lock is
not launch authority. Candidate traces are unavailable in the frozen input locks,
so endpoint utility cannot establish route recovery or executable candidate-route
precision.

## Artifacts

- `eligibility_ledger.json.gz`: all 3,840 attempts, descriptors and exclusions
- `candidate_lock.json`: all 22 source/policy selection memberships
- `request_lock.json`: 19 deduplicated, currently unauthorized physical requests
- `abstention_ledger.json`: all 15 cells and all 30 source/policy statuses
- `result.json`: aggregate, per-policy and per-cell findings

Request-lock payload SHA-256: `bc0b6ffdbd229e3ba6cf143f53fcc37003326138c43fa249aef629131360a15b`

Physical SHA-256 values at publication:

- `abstention_ledger.json`: `8221deb9270649fd3da816711bda8a4880718c1c4a44ee1c55f966e72f4a27dd`
- `candidate_lock.json`: `cb99a77a0d721af50a44b85309f105787e18a31f1bc006c3c23be5e9893b4089`
- `eligibility_ledger.json.gz`: `b4128d30bfa2108346138e9cd3f30823d90b6df5bff04654cc53df3041c67cc0`
- `request_lock.json`: `23ccc7250d5f6ac20af3609047b39730a47ab414bba01cf8c7c0479ae2681659`
- `result.json`: `ab34ef5e065ffc7f13cb4a6a3bd0414c9f8bc5abc05cc78abb5554764da995a3`

## Required next authorization

To launch, the user must explicitly authorize the exact physical SHA-256 of
`request_lock.json`, a ceiling of 19 first-score docking calls, no retry, no
replacement and no backfill. Until then, the panel remains unscored.
