# One-step recovery lookahead

Status: completed. Short repair found a new eligible molecule, but lookahead
did not change the selected offspring in this paired decision. There were no
new docking calls. This is a fixed-decision diagnostic, not a multi-round
optimizer or a demonstrated docking improvement.

Generation revision: `d86c055d49070f2cb14ab050e5ff7b7c4029f351`.
Run: `6f6e72c73ac0fcbce1fa0e0449375929f8ee42888199345aee99cdd6324e9b35`.
Volume: `compose-v4-artifacts`; the three case paths and call IDs are in
`attempt_2/launch.json`. The failed first attempt remains separately preserved.

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
env PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:/private/tmp/compose-recovery-lookahead-v2.WOe9JP/src:/private/tmp/compose-recovery-lookahead-v2.WOe9JP:. OMP_NUM_THREADS=1 \
  .venv/bin/python diagnostics/t4_recovery_lookahead/fetch.py
```

The local chemistry overlay pins RDKit 2024.03.5, numpy 1.26.4 and scipy 1.13.1.
The collector binds its own hash, planner/executor module hashes and all input
hashes. It does not launch workers or spend oracle calls.

## Operational correction

All three first-attempt workers failed with `cached repair enumerator changed`
at root 0. The Python 3.12 local AST hash included the empty `type_params` field,
which Python 3.11 on Modal does not emit. The function source itself is identical
to the cached producer. It hashes to
`fed69b0fbb4368b649166b33424a716d7ae5ebe61e86440db060670f77e69b8f`.
The corrected guard hashes that exact function source, extracted using its source
span, rather than serializing interpreter-specific AST fields. Other cached-law,
input and exact-state checks remain unchanged. A regression test prohibits use
of AST serialization for this cache identity and checks that function-body changes
still invalidate the identity. This is not a chemistry, threshold or planning
change. First-attempt contract and all failure/runtime receipts are preserved.

Attempt 2 is the same scientific decision with this compatibility repair only.
Its receipts live separately under `attempt_2`; the collector defaults there.
No first-attempt repair rows completed, so only previously compatible earlier
diagnostic rows can be reused. Startup cost of the failed attempt is additional
and must not disappear from compute reporting.

## Result

`summary.json` is the authoritative input-hashed reduction. Across all nine
roots, 5,093 canonical root/product pairs contained 18 eligible pairs, including
16 new eligible molecules. Four roots had new eligible continuations; five had
any eligible continuation when known archive returns are included.

| Retention arm, followed by the same repair rule | First-slot probability on roots with new eligible repairs | Selected new eligible molecules |
| --- | ---: | ---: |
| Immediate | 0.789558 | 1 |
| One-step lookahead | 0.943562 | 1 |

Both selected `levels/01/attempt_01_01` and `levels/01/attempt_02_00`.
Only the former had a new eligible repair. Both arms therefore returned the
same molecule, not two distinct discoveries:

```text
Cc1nc(=O)nc(-c2cc(Br)c3n2-c2ccc(CN(C)C)cc2CNC3=O)n1C
```

The exact `atom_delete` witness removes fluorine at persistent slot 27.
QED is 0.650791, SA 3.334475 and original-seed similarity 0.492308. Cycle rank
stays four and ring-system count stays two; this is local repair, not new ring
construction. The remaining primitive budget is 39.

Its predicted docking is -8.819451, worse than the incumbent prediction
-9.148093. Even the best prediction across all 16 new eligible products is
-9.004462. These are frozen-model predictions, not docking observations.
The result demonstrates a useful repair from an already-retained state, but no
selected-endpoint advantage from lookahead under this paired seed. It does not
prove lookahead is generally ineffective. A cheaper proposed next comparison is
repair completion on already-retained states, preserving exploration; no such
optimizer run was launched here.

## Completed verification and compute

The source-hash repair passed four focused tests in 2.92 seconds on the clean
launch source. The collector verified all nine roots, sealed inputs, exact-state
and cache dependency identities, endpoint eligibility, archive novelty, original
selection probabilities within 1e-12, exact selected indices and selected-repair
executor replay. All 28 reused scoring/topology/codec dependencies matched.
Touched-code lint/format passed. No full-suite milestone qualification is claimed.

Successful worker elapsed times were 173.5, 191.8 and 249.9 seconds, about 4.2
minutes for the parallel worker phase, excluding deployment and collection.
Proposal counters were 54.6, 69.8 and 102.1 seconds. Six new laws were enumerated,
two old laws reused, and one complete prior repair row reused. Public executor
calls totalled 6,132. Peak RSS per Linux worker was approximately 4.11 million
KiB. These are elapsed/reservation metrics, not measured CPU utilization.
The failed first attempt adds 427.2 worker-seconds of startup; it is not omitted
or reported as a molecular failure. No new oracle calls, training or follow-up.
