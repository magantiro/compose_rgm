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

## Zero-oracle implementation

The thin implementation is in
src/compose_v4/experiments/pmo_route_fiber_scored_pilot.py. It provides:

- split-clean single-fold runtime bundles;
- deterministic score-blind pool generation and atomic pool manifests;
- a fixed task-free PMO transformation feature adapter for FiberControl;
- paired blind/Fiber selection, immutable query locks and a charged query ledger;
- round checkpoints with exact model, archive and selection identities;
- interruption-safe replay and an RNG-free status reader;
- a mandatory sealed launch receipt for every non-synthetic evaluator.

The preparation tool intentionally exposes no scored action:

    .venv/bin/python tools/pmo_route_fiber_scored_pilot.py prepare-pools
    .venv/bin/python tools/pmo_route_fiber_scored_pilot.py preflight
    .venv/bin/python tools/pmo_route_fiber_scored_pilot.py status

Production pool generation and preflight must run in the pinned Python 3.11,
RDKit 2023.09.6 and PyTDC 1.1.15 environment declared in the contract. The
preflight verifies package and PyTDC source hashes plus the 27,791,877-byte GSK3B
asset before publishing a ready-for-authorization result. It never constructs or
evaluates an oracle.

## Sealed zero-oracle preparation result

The pinned preflight passed with zero oracle calls. All 16 score-blind pools were
sealed before any score was observed. Each pool contains 31 or 32 unique valid,
exactly replayed endpoints, exceeding the frozen eight-candidate minimum.

- candidate-lock manifest file SHA-256:
  `a10608e159b6d70ef5c30d1253a911fea9b43057b7b3f300b4ff02f2289dd0be`
- candidate-lock manifest payload SHA-256:
  `0908280ab5ebeada232013a4f46a6408fa1fddfc2fb665b5a78ef14048f220d7`
- preflight file SHA-256:
  `5a15cfc096f7e58f9f7d10caaedebcf288f1e72b033a1269eefac58265c0f747`
- preflight payload SHA-256:
  `986459abcc849dc7d84cf595a2ee489b10ac7a0adaae56187dde1fef97725379`

The decision remains
`ZERO_ORACLE_PREPARATION_PASSED_NOT_AUTHORIZED_TO_SCORE`. A scored run still
requires an AGENTS.md milestone amendment, the exact 384-call authorization and a
separately sealed launch receipt bound to these preflight and manifest file hashes.
