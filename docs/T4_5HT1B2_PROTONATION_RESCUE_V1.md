# T4 5HT1B-2 protonation-aware rescue v1

## Status

Prepared and unlaunched. No scored call is authorized by this document or by
the preparation contract.

The rescue is one fresh 49-charged-call run on `5ht1b_2` at delta 0.6. It keeps
the frozen shallow, anchored-replacement, shared route, endpoint-filter,
FiberControl and recursive-archive settings from the primary shared controller.
Its only scientific change is a fourth general proposal expert:
`protonation_aware_retained_subgraph`, using the explicitly versioned Editing-V3
`atom_protonation_restate` action.

The zero-oracle proposal gate produced 14 unique eligible endpoints across eight
Murcko scaffolds on the motivating root and exact-replayed all 1,078 attempted
programs there. This is proposal and feasibility evidence only. Docking utility
has not been measured.

The score-free cold-start replay separately froze the round-one allocation. A
batch of eight reserves one small, one medium and three large route proposals,
one shallow proposal and one anchored-replacement proposal. The last slot goes
to the protonation-aware expert when that pool is nonempty; otherwise it returns
to ordinary exploration. Missing route pools release their route slots. From
round two onward, the unchanged two-query exploration quota and learned
FiberControl allocation apply. The runtime derives this plan from actual
nonempty pools through `t4_cold_start_allocation_v1` and persists the plan in
the round lock.

## Scientific boundary

The new expert enumerates the complete admitted tertiary-amine protonation fiber
on any current parent, then composes the rewritten state with the same generic
shallow, retained-core and route-complete mechanisms. It receives no target
name, teacher endpoint, known candidate, comparator score or cell-specific
chemistry constant. The other three experts and the 49-call policy remain
available unchanged.

This is a prospective development rescue under one docking seed. A positive
result is not a replicated full-T4 result. A negative or exhausted result remains
part of the record.

## Fail-closed preflight

After the preparation commit, run:

```bash
/Users/rmaganti/compose_rgm_git/.venv/bin/python \
  tools/preflight_t4_5ht1b2_protonation_rescue.py
```

It must report `PREPARED_NO_SCORED_AUTHORIZATION`, the exact contract payload
and file hashes, four proposal experts, 14 preflight-eligible endpoints, a
49-call ceiling and zero retries. Any dirty or hash-drifted runtime input fails.

## Separate authorization and launch

Before launch, the user must explicitly authorize:

> One fresh 5HT1B seed-2 delta=0.6 run under contract payload
> `e15929bd80a67c3ce0e856f313d0e8bad4da042045e6010afae61d87e2f5d80e`,
> capped at 49 charged docking calls, with zero
> retry, replacement, prior-outcome reuse, endpoint injection or teacher
> injection.

Only after that authorization, substitute the exact payload hash printed by the
clean preflight and run:

```bash
/Users/rmaganti/compose_rgm_git/.venv/bin/modal run \
  modal_apps/t4_shared_retained_fiber_5ht1b2_protonation_rescue_v1_app.py \
  --mode launch \
  --authorization-payload-sha256 e15929bd80a67c3ce0e856f313d0e8bad4da042045e6010afae61d87e2f5d80e
```

The app rejects an absent or mismatched payload hash before spawning the driver.
It publishes every query lock before docking, charges unresolved locked queries,
never retries or backfills, and stores independent query and proposal receipts.

## Continuation and observation

The driver uses durable `reserved -> running -> terminal` continuation state.
It writes `reserved` before spawning and records the spawned call identifier
afterward. A missing or ambiguous state waits rather than spawning a duplicate.

Status is read-only:

```bash
/Users/rmaganti/compose_rgm_git/.venv/bin/modal run \
  modal_apps/t4_shared_retained_fiber_5ht1b2_protonation_rescue_v1_app.py \
  --mode status --run-id <RUN_ID>
```

Manual continuation is permitted only after independently establishing that the
prior driver call is terminal:

```bash
/Users/rmaganti/compose_rgm_git/.venv/bin/modal run \
  modal_apps/t4_shared_retained_fiber_5ht1b2_protonation_rescue_v1_app.py \
  --mode advance --run-id <RUN_ID> --confirm-prior-call-terminal
```

Never use manual advance merely because observation timed out.
