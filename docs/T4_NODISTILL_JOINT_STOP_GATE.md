# T4 NoDistill joint-constituent and STOP support gate

## Scope and claim

This is a standalone, zero-oracle support gate. It does not integrate a
controller into production search, fit a model, call docking, launch Modal, or
alter any prior T4 artifact.

- **Scientific problem:** independent generic constituent choices can collapse a
  bounded beam before a feasible multi-constituent program reaches STOP.
- **Primary output:** complete depth-2 through depth-4 protected edit programs,
  constructed on each actual successor and locked before evaluation.
- **Claim under test:** feasibility-aware joint ranking with explicit structural
  coverage produces nonzero diverse, T4-admissible support on the frozen
  `5ht1b_0` delta-0.4 source and improves valid yield or post-lock coarse teacher
  support over a matched independent constituent baseline without reducing exact
  replay precision.
- **Validation setting:** the primary near-capacity `5ht1b_0` source and the
  contrasting lower-capacity `parp1_0` source. These are inspected development
  sources, not held-out evidence.
- **Support:** complete exact molecular graph programs with at most 32 primitives,
  eight blocks, 40 active atoms, and two to four constituents. Stereochemistry
  and formal-charge changes remain outside the claim.

The governing self-hashed contract is
`configs/t4_nodistill_joint_stop_gate_v1.json`.

## Frozen arms

Both arms use the same task-blind generic Dynamic COMPOSE constituents and exact
executor. No route, template, teacher action, endpoint target, task or cell name,
docking score, objective value, or learned coefficient crosses the graph-only
generation boundary.

1. `independent_marginal` sums per-constituent primitive, heavy-atom, and
   cycle-rank costs. Its independent STOP preference favors depth two.
2. `coverage_joint_stop` ranks only completed joint programs using exact replay,
   resource margin, retained-interface continuity, and joint rewrite diversity.
   It assigns beam slots round-robin across the prospectively frozen axes: scale,
   delta heavy atoms, delta cycle rank, retained-interface class, and rewrite
   mode. The ordering is lexicographic and contains no fitted weights.

Every constituent is compiled on the actual predecessor graph. Every retained
prefix is re-extracted as one complete program and replayed from its original
source. Intermediate T4 admission and docking are prohibited. Prefixes at depths
two, three, and four are complete STOP candidates.

## Lock and evaluation boundary

Four deterministic process shards cover two sources and two arms. Each shard is
rerun independently and must reproduce the same full lock. The parent process
publishes all four locks and a physical-hash manifest before it opens the
historical teacher comparison. Actual T4 endpoint admission is applied only to
locked endpoints.

Teacher comparison uses only route-level coarse descriptors already present in
the immutable strategy-reset audit. Coarse support means a match in scale,
heavy-atom-change band, and cycle-rank-change band. Semantic support additionally
requires a completed mixed-mode candidate. The historical rows do not provide a
settled retained-interface label, so the evaluator reports that field as
unavailable and does not infer it.

The primary gate is strict:

- exact replay precision is 1.0;
- the proposed arm has nonzero valid complete STOP yield on `5ht1b_0`;
- its valid support spans at least two joint strata;
- it strictly improves valid yield, coarse teacher support, or semantic teacher
  support over the independent baseline;
- it has no lower exact precision;
- all locks are byte-deterministic across reruns;
- the 32-primitive, eight-block, 40-active-atom support is unchanged.

A failed condition records `FAIL`. It must not be repaired by changing the
threshold, adding a route, injecting a teacher candidate, broadening support, or
using task scores.

## Commands

Run focused checks:

```bash
.venv/bin/python -m pytest -q tests/test_t4_nodistill_joint_stop_gate.py
.venv/bin/python -m ruff check src/compose_v4/control/nodistill_joint_stop.py \
  src/compose_v4/experiments/t4_nodistill_joint_stop_gate.py \
  tools/t4_nodistill_joint_stop_gate.py \
  tests/test_t4_nodistill_joint_stop_gate.py
```

Run the predeclared gate:

```bash
.venv/bin/python tools/t4_nodistill_joint_stop_gate.py --workers 4
```

The authoritative output is
`diagnostics/t4_nodistill_joint_stop_gate/attempt_1/result.json`, paired with
`lock_manifest.json` and the four immutable candidate locks.
