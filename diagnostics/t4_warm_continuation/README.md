# Guided warm continuation

Scientific source: `6fe5cdf3d650d16fb4f27b9804a4608d91e0693c`.
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

## Live attempt

The continuation was spawned into the deployed app on 2026-09-08:
`fc-01M2005G8FSNYBXN3RM7P9Z0SA`. The content-addressed volume path is
`t4_warm_continuation/9aef7ea2d6b10c6bdd605d3dc6786a6cb57085cb102aae2e44da09743f62be97`.
`attempt_1/spawn.json` binds its exact launch manifest. No additional run is
authorized by the focused-verification policy.

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
