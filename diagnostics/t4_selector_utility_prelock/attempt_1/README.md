# T4 selector utility prelock

This prospective development lock adds 2 new score-blind
selector-ranked requests to the immutable delta-0.4 raw panel. The selector was
chosen using offline teacher/component diagnostics, so any later scored outcome
is development evidence rather than an independent final benchmark.

The existing raw request lock is unchanged. Its baseline-learned memberships
are the frozen raw counterparts. Candidate and physical request identities from
that lock were excluded before selecting the first eligible selector-ranked
candidate in each supported cell.

## Cell outcomes

- `5ht1b_0`: `abstain_no_eligible_baseline_learned_candidate`
- `braf_1`: `abstain_no_eligible_baseline_learned_candidate`
- `fa7_0`: `abstain_no_eligible_baseline_learned_candidate`
- `jak2_1`: `selector_candidate_locked`
- `parp1_0`: `selector_candidate_locked`

## Immutable request lock

- new call ceiling if separately authorized: 2
- request-lock physical SHA-256: `9802da3b356ae5f41460a29377640c9a0e3fb866f2e82c46ea581624074ba046`
- request-lock payload SHA-256: `6d49e2a397b662a4425a4a5ee0d17a6223d64c4ac3e0effa594dc8d8c1c52b29`
- candidate-lock physical SHA-256: `715302518749e3120041f48a515e1c94e0df13631a1f423f9aa06554d48b9a00`
- candidate-lock payload SHA-256: `b76b41c2d1400ccc115798cc21b02a5768c24de4eea381628cc8ae13c42fc549`

No docking or task score was inspected. No oracle or docking function was
called, Modal was not launched, and no live run was accessed. This request lock
is not launch authority. A later launch requires explicit authorization of the
exact request-lock hashes and exact call ceiling, with no retry, replacement or
backfill.
