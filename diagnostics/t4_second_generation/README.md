# Measured second-generation preparation

Attempt_2 contains 49 distinct new target/molecule queries across the paired
score-ranked and score-blind program proposals. No second-generation molecule
has been docked. The previous measured champions are unchanged.

| Cell | Ranked: first preparation | Ranked: conditional mutation repair | Score-blind after repair |
| --- | ---: | ---: | ---: |
| JAK2 seed1 | 8 | 8 | 8 |
| FA7 seed0 | 3 | 7 | 8 |
| BRAF seed1 | 8 | 8 | 8 |
| 5HT1B seed0 | 6 | 8 | 8 |

Each column has an eight-candidate cap and 128-attempt limit per cell. The
first preparation exposed repeated selection of nonexistent contraction
parameters. The shared repair normalizes over the mutations currently available
in each complete program; it does not change executor validity or size support.
FA7's remaining ranked shortfall is retained. No cell-specific code was added.

Both arms start with identical paid histories. Score rank changes allocation;
score-blind selection ignores numeric score quality but retains the same
duplicate-exhaustion adjustment and uniform exploration floor. Size statistics
record the original construction source, measured parent endpoint, final endpoint,
peak intermediate size and both relevant deltas. Growth, contraction and neutral
remodeling remain supported; size is not the task objective.

Attempt_1: 67.06 seconds, 11,272 executor calls, 39 distinct locked queries.
Attempt_2: 81.85 seconds, 9,534 executor calls, 49 distinct locked queries.
Both used zero oracle calls and one local CPU process. The second attempt's
63 arm/candidate placements share 14 endpoint queries. Raw locked batches,
predecessor identities, exact snapshots, allocation records, failures and
implementation/input hashes are retained in each attempt.

Next: score attempt_2 without changing selection. A proposed maximum of 73
new calls includes all 49 first evaluations and up to 24 fresh champion and
first-generation incumbent confirmations. This exceeds the old allocation's
two remaining calls and has not been launched. Details and acceptance criteria:
`docs/T4_SECOND_GENERATION.md`.
