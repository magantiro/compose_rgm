# Arm C — created-atom rebinding (frozen before any scored call)

## The claim this arm can support

> Holding the operation sequence, payloads and source-atom bindings fixed, replacing the
> prescribed connections to newly created atoms with alternative legal bindings changes
> optimization performance by X.

It supports the value of **planned atom-level connections**, beyond the choice of which
edits to perform. It does **not** isolate the dependency scheduler, and it necessarily
changes the resulting molecular connectivity — those are boundaries of the question.

**A→C and C→B are NOT an additive decomposition of A→B** and will not be reported as one.
The score gap will **not** be divided by the affected fraction: adaptive search is not
additive that way.

## What is held fixed, and what moves

| | |
|---|---|
| proposal machinery | UNCHANGED `_channel_proposal` at commit `c7fdddca` — region-replacement families, jump-plan access, archived-program mutation and recombination all retained |
| construction success filter | RETAINED — the intervention runs only on a proposal arm A successfully constructed; a construction failure is recorded as a construction failure |
| operation sequence, length, block boundaries | preserved byte-for-byte |
| every payload field | preserved byte-for-byte |
| `{"input": i}` source bindings | preserved byte-for-byte |
| **`{"created": j}` operands** | **resampled uniformly over the currently live created atoms that leave the operation admissible** |
| `atom_insert` birth slot | never resampled (assigned by `fresh_slot`; `EditProgram` requires sequential handles) |
| the prescribed binding | stays ELIGIBLE wherever admissible; a step with one admissible binding is recorded as having no alternative, not as intervened |

Admissibility is decided by offering each candidate to `execute_program` exactly as
`execute_bound_program` does, so every semantic-admission, liveness, connectivity, valence
and capacity check is the production one.

## Failure is frozen and labelled

No admissible binding ⇒ `ProgramExecutionError`. No fallback to the prescribed program, no
replacement chain, no redraw. Stages recorded separately:
`rebinding` · `rebinding_reschedule` · `rebinding_replay`, distinct from an
original-construction failure.

## Randomness

`blake2b(run_seed : program_id : source_canonical_key : assignment : occurrence)`.
`occurrence` is a per-run proposal ordinal, because `program_id` is a CONTENT hash — without
it the same recipe on the same parent would replay one choice. Never Python's salted
`hash()`; never a draw from the arbitration or proposal streams. `identity=True` must
reproduce the prescribed program and advance nothing (asserted, 80/80).

## Development panel — PASS (zero oracle calls, rdkit 2023.9.6)

80 arm-A programs, snapshot insertion order, `score`/`static_score` never read.

| measure | value |
|---|---|
| identity check | 80/80 |
| completion, occurrence 1 | **80/80 (100%)** |
| completion, 640 attempts (8 occurrences) | **639/640 (99.8%)** |
| carry created-atom operands (applicable) | 45/80 (56%) |
| …with ≥1 alternative admissible binding | 41/45 (91%) |
| **canonical endpoint changed** | **38/80 (47.5%)**, self-reported flag matches direct measurement 38 = 38 |
| refusals over 640 attempts | `rebinding` 1 · `rebinding_reschedule` 0 · `rebinding_replay` 0 |
| candidate cap bound | 0 |
| cost | 85 ms / proposal |

Admissible bindings per step: `{1: 51, 2: 61, 3: 53, 4: 38, 5: 16, 6: 8, …, 34: 1}` —
199 of 249 steps (80%) have a choice. Rules carrying created operands: `atom_insert` 210,
`cycle_close` 32, `atom_delete` 7.

**Both stated failure modes are ruled out:** bindings are not almost always forced, and
completion does not collapse.

**Footprint, stated in advance:** ~45% of proposals carry no created-atom operand and are
UNCHANGED by construction. That is correct ablation behaviour — a proposal that does not
use the targeted mechanism should not be altered to manufacture contrast. Applicability,
endpoint-change and failure rates will be reported **from the scored campaigns**, since an
80-program panel does not establish them across an evolving run.

## Integration — the check that made this an experiment rather than a throttle

`add_measured_program` replays the STORED program through `compile_program_graph` +
`scheduled_program` and demands exact endpoint AND state-sequence identity.

    arm C admitted to archive        79 / 79      rejected 0
    arm A admitted (control)         79 / 79
    arm C endpoints differing from A 38
    of those, storing the PRESCRIBED program would have been REJECTED   38 / 38

So the unpatched path would have archived only proposals on which rebinding changed
nothing. v21 now rebuilds the program from the modified marks, recompiles the graph and
recomputes `program_size_profile` from it, so dependency metadata describes the modified
references; the wrapper proves the modified program reschedules to the executed order and
replays to its own molecule, so nothing is silently rescheduled.

Real proposal path (`propose_batch`, arm C on, zero oracle calls): **8/8 candidates carry
`ablation_stage`, 8/8 stored programs replay to their candidate endpoint and states.**
Only v21's proposal seam may rebind — internal construction inside `_channel_proposal` and
admission replay stay prescribed (asserted from the import graph).

## Defects found before launch

1. **created-handle index reuse** — the counter was derived from the live list, so after an
   `atom_delete` the next insert collided with a surviving handle. Caught by the
   pre-divergence admissibility invariant (1 violation → 0).
2. **`atom_delete` bookkeeping followed the prescribed mark**, not the rebound one, so the
   wrong atom was retired. Caught by `EditProgram.validate`. Fixing it took completion
   97.5% → 100% and removed every refusal; the earlier "2 rebinding failures" were this
   bug, not a property of the intervention.
3. **the archive stored the prescribed program** — see above, fatal, 38/38.

## Scored campaigns (unchanged from the A/B)

Six objectives × three seeds, first 1,000 charged calls, same pool target, same
initialization, same oracle accounting. Arms A and B are not modified.

    tools/launch_pmo_fibercontrol_targets.py --binding-rebind-arm \
      --targets albuterol_similarity isomers_c9h10n2o2pf2cl ranolazine_mpo \
                scaffold_hop celecoxib_rediscovery gsk3b

`--binding-rebind-arm` and `--uniform-chain-arm` are refused together, in the launcher and
again at `binding_arm_enabled()`; the label and volume carry the arm so no two arms can
share a namespace. The canary streams a per-round `REBIND` line and fails closed if
rebinding never changed a molecule.
