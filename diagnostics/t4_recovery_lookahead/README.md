# One-step recovery lookahead

Status: launched; no outcome yet. This is one fixed-decision diagnostic, not a
multi-round optimizer run or docking experiment.

Generation revision: `3dc2d2d010b4` (full identity in the launch receipt).
Run: `9cbc9853afb9ab5692e8c4fb64c74d05822044df45f3f5ceb20b36ba0df024c5`.
Volume: `compose-v4-artifacts`; the three case paths and call IDs are in
`attempt_1/launch.json`.

All nine candidates at the saved second retention decision are included, not
just the known near-feasible state. Three workers prepare deterministic i::3
partitions. Cached repair products are reused only with exact state, input,
model/executor and enumerator compatibility. No source, option, region or
primitive support is changed in the saved prefixes. The additional recovery
probe considers all supported one-edit continuations after option completion.

The planning value is the best frozen-model desirability of a new eligible
one-edit endpoint, zero if none. This is maximum-value planning, not a Doob
expectation or calibrated future value. Known endpoints remain in support but
have zero new-acquisition value. The paired immediate and lookahead arms share
retention mechanics and randomness, and both receive the same best verified
repair after selecting their parents. This isolates the retention choice.

The original saved decision must replay before the counterfactual is interpreted.
Final QED, SA, original-seed similarity and medicinal-chemistry gates remain
unchanged. No docking is authorized, and a predicted improvement cannot be
reported as observed docking improvement. See `docs/T4_RECOVERY_LOOKAHEAD.md`
and `configs/t4_recovery_lookahead.json` for the prospective scope and limits.

Verification before launch: 11 focused tests passed in 18.14 seconds on the clean
launch tree. Strict preflight, touched-code lint/format, contract self-hash,
unchanged cached-enumerator identity and diff checks passed. The app has the same
17 pre-existing lint findings, with none added. Full-suite milestone verification
was not run. The app was deployed before durable spawning through t4_launch.

Read-only collection, using the clean launch source for molecular replay:

```sh
env PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:/private/tmp/compose-recovery-lookahead.2FMuIt/src:/private/tmp/compose-recovery-lookahead.2FMuIt:. OMP_NUM_THREADS=1 \
  .venv/bin/python diagnostics/t4_recovery_lookahead/fetch.py
```

The local chemistry overlay pins RDKit 2024.03.5, numpy 1.26.4 and scipy 1.13.1.
The collector binds its own hash, planner/executor module hashes and all input
hashes. It does not launch workers or spend oracle calls.
