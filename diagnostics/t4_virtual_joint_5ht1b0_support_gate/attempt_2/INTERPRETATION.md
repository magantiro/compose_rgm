# 5HT1B-0 binding-particle joint support gate

## Outcome

Binding particles close the measured teacher-constituent allocation gap, but do
not produce a valid joint target or recover any strong route autonomously. Across
particles 0 through 7, all nine teacher constituents are bound and selected at
least once. Particle 3 selects all nine simultaneously, so it contains all three
constituents for each of the three routes. Nevertheless, every particle records
zero valid depth-2, depth-3 and depth-4 STOP targets. The deterministic union
therefore remains at 0/3 exact endpoint recovery and 0/3 radius-2
transformation-equivalent recovery.

This is a downstream planning or final-target formation failure, not a binding
or top-48 allocation failure. The present telemetry does not distinguish whether
an exact three-constituent prefix is removed by the bounded planning beam or is
attempted and rejected at STOP validation. No compiler can receive the teacher
joint targets because no depth-2 through depth-4 target becomes valid.

## Per-particle results

Every call used the unchanged `VirtualJointRegionBudgets`: depth 4, beam 64,
expansion width 48, eight bindings per template, 16,384 binding visits, 4,096
planning expansions, 128 targets, 128 realizations, 65,536 total realizer
expansions, 4,000 per realization and 32 primitives. Each particle selected 9
local, 22 medium and 17 large constituents from the same 101-row bound census.

| Particle | Teacher selected | Routes with all 3 constituents | Exact endpoints | Valid STOPs d2/d3/d4 | Precision | Final/compiler abstentions | Planning / realization / realizer expansions | Proposal seconds |
|---:|---:|---|---:|---|---:|---:|---|---:|
| 0 | 7/9 | `88e87055` | 12 | 0/0/0 | 12/12 | 224/2 | 1,660 / 14 / 41 | 1.811 |
| 1 | 7/9 | `88e87055` | 13 | 0/0/0 | 13/13 | 224/2 | 1,719 / 15 / 44 | 1.991 |
| 2 | 8/9 | `2b0aff16`, `88e87055` | 15 | 0/0/0 | 15/15 | 223/1 | 1,791 / 16 / 45 | 2.023 |
| 3 | 9/9 | all three routes | 14 | 0/0/0 | 14/14 | 223/2 | 1,891 / 16 / 43 | 1.947 |
| 4 | 8/9 | `88e87055`, `bbd7db0d` | 15 | 0/0/0 | 15/15 | 223/2 | 2,152 / 17 / 46 | 2.106 |
| 5 | 7/9 | `88e87055` | 14 | 0/0/0 | 14/14 | 223/3 | 2,608 / 17 / 46 | 2.176 |
| 6 | 8/9 | `2b0aff16`, `88e87055` | 13 | 0/0/0 | 13/13 | 223/2 | 2,834 / 15 / 42 | 2.073 |
| 7 | 8/9 | `2b0aff16`, `88e87055` | 13 | 0/0/0 | 13/13 | 223/3 | 2,431 / 16 / 45 | 2.112 |

The route-complete constituent coverage is:

- `2b0aff16...`: particles 2, 3, 6 and 7;
- `88e87055...`: all eight particles;
- `bbd7db0d...`: particles 3 and 4.

All nine teacher constituents remain bound in every particle. Their identities
are joined only after every autonomous proposal call returns.

## Deterministic union

The eight particles produce 109 exact committed endpoint occurrences and 25
unique canonical endpoints after deterministic deduplication. All 25 are
depth-one endpoints. The union contains 19 unique program identifiers and 25
unique constituent combinations. Duplicate endpoint occurrences total 84.

New unique endpoint contributions in particle order are 12, 3, 4, 2, 2, 2, 0
and 0. Endpoint occurrence across particles is distributed as follows: 16
endpoints occur in three particles, two in five particles, two in six particles,
one in seven particles and four in all eight particles.

Exact realization precision is 109/109. There are 1,786 final-target abstentions
and 17 compiler abstentions, all `frontier_exhausted_abstention`. The union has
zero valid and zero retained depth-2 through depth-4 STOP targets. Exact and
transformation-equivalent ranks are null for all three routes.

Across the eight independent calls, budget use is:

- binding visits: 47,056 / 131,072;
- planning expansions: 17,086 / 32,768;
- retained targets: 126 / 1,024;
- realization attempts: 126 / 1,024;
- realizer expansions: 352 / 524,288.

No particle exhausts any global budget, and no realization reaches the
per-candidate 4,000-expansion cap. The negative result is therefore not explained
by exhaustion of the declared limits.

## Runtime and reproducibility

The first measured run took 49.785 seconds end to end. The eight proposal calls
took 1.811 to 2.176 seconds each; union and post-hoc analysis took 1.509 seconds.
Wall time is operational and machine-dependent, so it is stored in the separate
`runtime.json` sidecar and excluded from the deterministic scientific payload.

The main `result.json` reproduced byte-for-byte under an independent second
execution. Its payload SHA-256 is
`0da3acd8bcba6e5194ffb722763ed0462468206f43f177a07da4068ddfb4863d`.
The first operational runtime payload SHA-256 is
`75642762dde18d71fe735142315d191858c194ca916aca948c468dabf195be77`.

The measured command was:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:. \
/Users/rmaganti/compose_rgm_git/.venv/bin/python \
tools/t4_virtual_joint_5ht1b0_particle_gate.py \
--output diagnostics/t4_virtual_joint_5ht1b0_support_gate/attempt_2/result.json \
--runtime-output diagnostics/t4_virtual_joint_5ht1b0_support_gate/attempt_2/runtime.json
```

This gate made zero oracle, docking, Modal or scored-artifact calls. It supports
no docking-utility or prospective-optimization claim. Attempt 1 and its runner
remain unchanged.
