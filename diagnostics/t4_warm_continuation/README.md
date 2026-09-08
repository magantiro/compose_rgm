# Guided warm continuation

Initial scientific source: `6fe5cdf3d650d16fb4f27b9804a4608d91e0693c`.
Attempt 2 source: `34aa672eee4cb5d60bfa92d4c5386bd455617c73`.
Endpoint-only amendment and ring auditor: `6cf5043b18dc84617c710a91491ca67846f68367`.
Prospective scope: `docs/T4_WARM_CONTINUATION.md` and the self-hashed
`configs/t4_warm_continuation.json`. Two additional rounds, at most 40 new
dockings, PARP1 seed0 d=0.4. This is guided-only development, not a matched
guidance comparison or an IVG benchmark claim.

## Reuse and verification before launch

`conversion.json` binds both original guided receipts and the exact conversion
implementation. `expected_warm_start.json` contains the original seed and all
20 evaluated states, scores, ancestry, and post-selection RNG state. Its
physical SHA-256 is
`2df69a5a21ee9658a1add7938252ff70425d61464cd0c08a9f34fe0b38f29f65`.
Conversion performs no molecular re-enumeration or oracle calls. It uses the
saved selected-step product, not reconstruction from canonical SMILES.

The conversion can be reproduced with `initial_archive(unseal(source_lock),
unseal(source_docking))` and `seal(output_path, archive)` from
`compose_v4.experiments.t4_warm_continuation` and `t4_matched_pilot`.
Exact source paths and hashes are recorded in `conversion.json`.

The scientific source passed 83 focused controller and launcher tests
(`focused.xml`). The separate offline reporter passed two tests
(`auditor_tests.xml`). New modules, tests, and launcher pass formatting and
lint. The legacy app retains the same 17 lint messages/counts as before and
was unformatted both before and after; no unrelated app-wide cleanup was made.
The user approved removing the broad-suite launch blocker on 2026-09-08, before
new oracle calls. The passing focused checks authorize this bounded development
launch alongside unchanged scientific/input/budget gates. The broad suite was
sent an interrupt after about 31 minutes but finished as the interrupt arrived.
Its actual result is 4,520 passed, 47 failed, 58 errors, two skipped, and one
xfailed in 1,902.66 seconds. All 105 nonpassing identities and messages match
the prior baseline (ignoring process memory addresses). The two prior
OS-permission failures passed with the required permissions. There are no new
failing identities, but the repository is not green.

`launch_plan.json` records the clean worktree, content-derived run ID,
deployment identity, resource census, timing/cost estimate, inventory, and stop
policy. Source/config/app files remain identical to the scientific source even
when later isolated reporting code or evidence is committed.

## Attempts

The continuation was spawned into the deployed app on 2026-09-08:
`fc-01M2005G8FSNYBXN3RM7P9Z0SA`. The content-addressed volume path is
`t4_warm_continuation/9aef7ea2d6b10c6bdd605d3dc6786a6cb57085cb102aae2e44da09743f62be97`.
`attempt_1/spawn.json` binds its exact launch manifest. Startup failed after
152.83 seconds in `canonical_slots`, before archive publication, molecular
proposal, or docking. The frozen runtime gates passed. Remote RDKit 2024 emits
`_smilesAtomOutputOrder` as a Python list with a trailing comma, which the JSON
parser rejected. The failure and all four remote startup receipts are retained.

The bounded repair parses this metadata with `ast.literal_eval` and verifies
that it is a complete integer permutation. Regression tests cover both formats,
malformed permutations, and equality of all 21 recovered archive records under
legacy formatting. This changes serialization compatibility, not states, region
draws, controller weights, scientific inputs, or oracle accounting. Resume the
same authorized two-round continuation with the full 40-call allowance still
unspent. There is no failed oracle batch or completed proposal work to replay.

### Attempt 2: compute ceiling, no new oracle calls

Call `fc-01M200RFJG3Z8G0W0CFRC19NMG`, volume prefix
`t4_warm_continuation/180696c8c1caf1b4fab2107fb2c9b49792a70372cf8d22664a1947972ec31cdd`.
Warm conversion passed and exactly matched the saved 21-record archive bytes.
Seven of eight parent units completed. The eighth exhausted the unchanged
20,000 public-executor ceiling. There is no candidate lock or new docking
batch. The 10,057 calls in the failure heartbeat describe only checkpointed
complete units; the full attempted-executor ledger contains 20,000 calls.
This distinction is verified in `attempt_2/ring_quality.json`.

`attempt_2/remote_inventory.json` records every downloaded artifact's SHA-256.
Complete parent units, startup/failure receipts, and the inventory are retained
in Git. The 29,829,279-byte attempted-executor ledger is retained locally and on
the named Modal volume, not duplicated in Git. All input hashes were checked
before the local audit. Reuse complete units after an explicit compatibility
and budget decision; do not re-run the seven completed parents or manufacture
a completed eighth unit from partial executor attempts.

### Ring-quality finding and endpoint repair

Authoritative local descriptor report: `attempt_2/ring_quality.json`, SHA-256
`3858aa16b85cedbef74693058c49981a776d5f7451209672ef00a30d2e01d15d`.
It covers 21 bundles and 25 canonically distinct emitted candidates from the
seven completed parents. Fourteen of 25 meet the existing T4 constraints;
three of six cycle-gaining candidates meet them. No candidate has a new docking
score. This is a partial preparation, not a completed round or oracle pool.

| Saved ring-growing product | Local SA | Original-seed similarity | Existing T4 constraints |
| --- | ---: | ---: | --- |
| Fused C6 program, parent 1 | 2.955 | 0.413 | Pass |
| Fused C6 program, parent 6 | 2.601 | 0.534 | Pass |
| Cyclize, parent 4 | 3.417 | 0.491 | Pass |
| Second pendant system, parent 0 | 4.221 | 0.440 | Fail SA |
| Generic bridged product, parent 2 | 4.753 | 0.253 | Fail SA and similarity |
| Generic bridged alternative, parent 2 | 5.112 | 0.301 | Fail SA and similarity |

All six pass QED >= 0.6. The limits remain SA <= 4 and similarity >= 0.4.
Both fused-program endpoints pass the existing exact-slot aromatic C6 witness:
four added carbon atoms close across one existing ring edge. Neither adds
bridgeheads or spiro atoms. The two generic bridged candidates each add five
RDKit bridgehead atoms; their zero QED structural-alert counts and empty legacy
gate reasons do not establish good chemistry. The pendant program constructed
a six-atom ring entirely in new slots, accumulating two extra cycles and two
extra ring systems from the original seed, but the endpoint fails SA.

These descriptors were computed locally with RDKit 2026.03.6 in about 1.10
seconds, without molecular generation or docking. The report binds the exact
input bytes, software, source code, SA scorer and fragment-score asset. The
remote runtime uses RDKit 2024.03.5 and must recompute production properties
before locking. No synthesis, stability, safety, or affinity validation is
implied. RDKit bridgehead/spiro definitions are documented in its official
[descriptor API](https://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html).

The prospective `t4_feasible_only_v1` amendment prevents candidates with an
existing T4 constraint violation from receiving docking slots, while keeping
rejected records in the audit and all intermediate trajectory support intact.
Generic remains active; there is no global ring blacklist or ring-count reward.
The old rank-all behavior remains available for unchanged historical tasks.
This repairs oracle spending, not the proposal process or the compute ceiling.

Verification: 59 focused tests passed in 6.47 seconds
(`endpoint_selection_tests.xml`). Formatting and lint passed for the new and
touched standalone code. The legacy app retains the same 17 pre-existing lint
messages, with no new messages. `git diff --check` passed. No broad suite was
repeated; the earlier full-suite result remains non-green as recorded above.
The endpoint repair is locally committed, not deployed. No retry, increased
executor budget, additional oracle calls, training, or push was performed.

## Result interpretation

Run `PYTHONPATH=src:. .venv/bin/python tools/t4_warm_audit.py <downloaded-run>`
to verify archive/lock/oracle bindings and report chemistry at the pool,
selection, and docking stages. The reporter handles interrupted preparation
explicitly and never treats completed parent units as a full oracle pool.
It compares per-parent topology fields with saved exact states and reports
cumulative graph counts separately from per-parent coherent displacement.

A cycle-rank increase is not by itself a fused-ring classification. A ring-system
increase caused by opening an old system is reported separately from cycle
construction. Docking remains unseeded, so small changes or repeated-structure
score differences cannot establish an affinity or guidance advantage.
