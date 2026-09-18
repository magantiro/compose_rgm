# PMO route-prior x FiberControl scored-pilot contract v1

## Outcome

The smallest prospective scored test is specified, but it is deliberately not
launchable and authorizes zero oracle calls. The authoritative self-hashed
contract is configs/pmo_route_fiber_scored_pilot_v1.json.

The pilot asks one narrow causal question on two qualitatively different PMO
objectives: do leave-one-family-out route proposals and online reward control
provide complementary gains over unchanged Dynamic-v0 proposals and blind
selection?

## Frozen factorial

- Tasks: GSK3B with fold 0 (all bioactivity routes held out) and Perindopril MPO
  with fold 2 (all MPO routes held out).
- Arms: v0 plus blind, 50/50 route/v0 plus blind, v0 plus FiberControl, and
  50/50 route/v0 plus FiberControl.
- Budget: 48 charged calls per task and arm. Sixteen initialization calls count,
  followed by four batches of eight. The total ceiling is 384, with zero retry.
- Candidate work: each round uses the same sixteen immutable initialization
  molecules. The v0 pool makes two unchanged v0 attempts per parent. The
  additive pool uses the same first v0 attempt and one route-prior attempt.
- Pairing: blind and FiberControl arms with the same proposal source see the
  identical immutable score-blind pool. All pools are locked before any score.
  Round one selections are also identical within a pair because FiberControl
  has no transformation outcomes yet.

This is intentionally a root-conditioned selection/proposal test. It does not
yet test descendant expansion or Dynamic archive refinement. A positive result
promotes the combined method to that separate experiment.

## Integrity, logging and resume

Every pool and selected query set is atomically published and self-hashed before
the corresponding oracle reservations. There is no candidate regeneration,
replacement or backfill after scoring. A receipt must join one-to-one to its
query lock. Failed or unresolved reservations remain charged and block automatic
continuation.

Each task-arm ledger records all 48 ordered observations, native reward, parent,
proposal source, exact program identity, online model features and prediction,
selection reason, random-number state, proposal/oracle/wall time, best and
top-ten curves and the exact AUC inputs. Checkpoints are written after
initialization, every pool lock, every query lock and every completed round.
Resume validates the contract, code, checkpoint, ledger, pool, query and random
state identities before doing work. Completed units are idempotent.

Status is designed to be queryable without advancing any random stream. It
reports phase, round, charged and remaining calls, pool size and shortfall,
current best/top-ten reward, timing and estimated remaining seconds.

## Decision rule

Promotion requires the combined route plus FiberControl arm to beat v0 plus
blind in both AUC48 and best reward on both tasks. It must also beat the better
single-factor arm by at least 0.02 AUC on one task and trail that comparator by
no more than 0.01 on the other, with a route-derived parent improvement on both
tasks.

Kill this revision if neither route proposals nor their combination with
FiberControl improves AUC over the relevant v0 comparator on either task. Any
leakage, lock mutation, execution mismatch, accounting overflow, retry or
pre-score candidate shortfall is also a hard stop.

## What remains before scoring

1. Implement the thin runtime against the frozen contract. No method choice is
   left to the runner.
2. Extract one sanitized fold bundle per task. A worker may not load the other
   fold checkpoints or any exact training trace.
3. Verify decision equivalence in the pinned PyTDC 1.1.15 and RDKit 2023.09.6
   environment and verify the GSK3B model asset without evaluating a molecule.
4. Generate and seal every score-blind candidate pool. Abort if any round has
   fewer than eight unique exact endpoints.
5. Pass focused candidate-lock, query-accounting, oracle-adapter and interrupted-
   resume tests. Commit the implementation, preflight and locks.
6. Record the scoped AGENTS milestone and obtain this exact authorization:

> I authorize the scored PMO route-prior x FiberControl pilot exactly under
> configs/pmo_route_fiber_scored_pilot_v1.json, with at most 384 charged calls,
> zero retries, and no task or arm outcome sharing.

The existing AGENTS.md is user-owned dirty state, so this preparation does not
modify it. The exact amendment text is stored inside the contract under
required_agents_amendment.

Prepare or verify the contract with zero oracle calls:

    .venv/bin/python tools/pmo_route_fiber_scored_pilot_contract.py prepare
    .venv/bin/python tools/pmo_route_fiber_scored_pilot_contract.py verify

There is intentionally no launch command in this revision.
