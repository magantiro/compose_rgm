# T4 cold-start allocation replay v1

The smallest supported repair is a round-one, eight-call coverage rule:

- one small route-complete-region proposal;
- one medium route-complete-region proposal;
- three large route-complete-region proposals;
- one shallow proposal;
- one anchored-replacement proposal;
- one ordinary exploration proposal.

If a nonempty protonation-aware proposal pool is present, it receives the final
slot instead of ordinary exploration. Missing pools release their slots. The
policy contains no target, cell, molecule, route, endpoint, or score lookup.

## Evidence

The audit reproduced the original round-one selections exactly on six immutable
completed candidate pools: 5HT1B-1 at delta 0.4, JAK2-0/1/2 at delta 0.4, and
PARP1-1/2 at delta 0.6. Under the same eight-call budget, the repair:

- selected both answer-known 5HT1B-1 large-route probes at route ranks 6 and 9,
  reported at -12.8 and -13.0, while the original selector selected neither;
- preserved the original round-one best score on all six replays;
- preserved the exact original shallow and anchored expert-floor candidates on
  all six replays;
- retained representation of shallow, anchored, and route experts on every
  replay.

On PARP1-2, two original candidates tied at -9.9. The repair retained the
large-route member of that tie and replaced the randomly explored anchored
member, so the historical best score is preserved but not both tied molecules.

FA7-1 was not replayed. Its completed result is locally available and
hash-valid, but its round-one candidate-pool lock is absent. The audit records an
explicit abstention and the missing lock payload hash rather than reconstructing
or accessing a live run.

## Claim boundary

This is a zero-oracle retrospective selection replay. It establishes candidate
coverage and preservation under a counterfactual allocation. It does not show
that the newly selected candidates would reproduce their upstream scores under
the current evaluator, nor does it constitute prospective controller evidence.
The authoritative machine-readable result is `result.json`.
