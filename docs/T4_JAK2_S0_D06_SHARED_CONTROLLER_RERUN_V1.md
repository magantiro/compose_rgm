# T4 JAK2 seed-0 delta-0.6 shared-controller rerun v1

## Status

`PREPARED_NO_SCORED_AUTHORIZATION`. This revision is unlaunched and authorizes
zero docking calls. No Modal or live-run access was used to prepare it.

The prepared experiment is exactly one fresh `jak2_0` cell at delta 0.6 with a
49-charged-call ceiling. It starts from an empty scored archive, apart from the
newly charged root evaluation, and neither resumes nor reads the earlier JAK2
run. The controller seed is `2026091912` and the docking seed is `20260919`, the
current primary shared-controller settings.

## Frozen controller

The proposal mixture remains the three existing lanes:

- shallow, 480 draws at horizon 3;
- anchored replacement, 512 draws at horizon 3;
- the legacy `route_complete_region` lane, with the current 192 proposal pool,
  96 realization limit, 48-wide beam and expansion, and scale balancing.

Rounds one and two retain the current small, medium and large route-scale floors
and the expert floors. Batch size 8, four parents, parent exploration 0.3, two
ordinary exploration selections and `ProgramValue(penalty=1.0)` are unchanged.

The route lane is bound to the task-independent all-route checkpoint:

- physical SHA-256:
  `cb0d0bd0130b31f956c320ccf8171ce897f797506c8cf1a524a7f697c749865e`;
- payload SHA-256:
  `5476be572dd40ee3f068cc8f1df238eec54e23a82cb84f803905ac891701e07d`;
- training census: 77 routes, 147 regions and 137 templates;
- runtime target conditioning: false.

Deferred joint planning is excluded. It is irrelevant to the one-region JAK
routes and is neither a proposal lane nor an imported runtime implementation.
There is no known winner, endpoint, teacher route, historical score or reported
comparator in the runtime contract.

## Scored-call and recovery discipline

The JAK2 receptor identity, box, `compose_valid` endpoint support and delta-0.6
fiber are frozen. The first charged call is the root; six full batches of eight
reach the 49-call ceiling when candidates remain. A candidate exhaustion is a
reported terminal outcome, never grounds for replacement or backfill.

Every expert result, round query lock, per-query receipt, phase result and
checkpoint is written to the dedicated durable volume before continuation.
Reserved, missing or otherwise unresolved locked queries are charged after the
frozen deadline and are never resubmitted. Automatic retries are zero. Driver
continuations use durable `reserved -> running -> terminal` generations; an
ambiguous generation waits, and manual advance requires authoritative
confirmation that the prior driver call is terminal.

## Hash-bound preflight

The frozen identities are:

- contract payload SHA-256:
  `fed1f158caa3142eea4a0e861c4689d72addb32d37c665ef6eeb2771cc3492a3`;
- contract file SHA-256:
  `8ff98658f697e7c81cee41d09e821563fad8b612ba141b0903f0e14bfdfe5f0c`;
- Modal app SHA-256:
  `d3d85d4054916d05b27bbb4aaf9f6c0e7c19be8c68c47e98c4c3099416bfd92f`.

After the preparation commit, the read-only command is:

```bash
PYTHONPATH=src python3 tools/preflight_t4_jak2_s0_d06_rerun.py
```

It must report `PREPARED_NO_SCORED_AUTHORIZATION`, one cell, a 49-call ceiling,
zero retries, the three proposal lanes, 77 checkpoint routes, no deferred-joint
lane and the exact hashes above. A dirty or hash-drifted runtime input fails.

## Required separate scored authorization

The exact one-sentence authorization required before any launch is:

> I authorize one fresh JAK2 seed-0 delta=0.6 shared-controller rerun under contract payload SHA-256 fed1f158caa3142eea4a0e861c4689d72addb32d37c665ef6eeb2771cc3492a3, capped at 49 charged docking calls, with zero retries, replacements, prior-outcome reuse, winner injection, or runtime comparator.

Preparation, review, or preflight output is not a substitute for that sentence.
Only after it is explicitly provided may the hash-bound app launch mode be used.
