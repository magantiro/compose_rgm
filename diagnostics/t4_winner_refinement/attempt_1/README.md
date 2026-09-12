# Winner-initialized refinement: completed, no improvement

The qualified resume completed with sixteen additional calls, nineteen across
the original assay. Best candidate and both fresh repeats: -13.0. The supplied
winner controls were -13.6 each, so the repeat-only difference is +0.6, worse.
The proposal stage produced 649 candidates, 638 unique, with fourteen selected
for first docking. Resume driver runtime was 232.14 seconds.

`resumed_result.json` is the unchanged remote result from source `0d965bf`.
Call: `fc-01M2BJYZ04VGA4S3J4JHSFQPY1`. Remote prefix:
`t4_winner_refinement/d8881faeffb165bcbe59b1a572d5a6c9b2ec394746875b72a1e8a85ec6d55fd1`.
It binds the recipe, qualified runtime, proposal work, source revision, scores,
exact states, and new/reused call counts. This is not an autonomous benchmark
comparison and does not prove the winner is locally optimal.

## Preserved initial failure

The clean `66a5f6eae3eab6361acecb0f2626cf1ae6264f30` launch used three
of its nineteen permitted docking calls. All three winner measurements were
-13.6 at declared QuickVina seeds 1701, 1702 and 1703. Each call used the existing
ligand preparation; this is not a claim of zero docking variance in general.

The driver then failed before candidate generation at a stale training-chain
process identity. Preserve `failure.json` unchanged. The gate was not relaxed.
The sealed `docked_controls.json` contains exact winner states, original-seed
eligibility, call timestamps, docking seeds and scores. These are paid control
measurements, not newly discovered molecules.

Remote call: `fc-01M2BJ3DV9T5RWV9R0XN9HERF0`.
Remote run: `t4_winner_refinement/fd2fe78e8fe3a6cc5f0dbdd9a3795c28bfa00762d9abc536e57501eb69e7cc92`.
Remote control/pose receipts:
`t4_winner_refinement/winner/fd2fe78e8fe3a6cc5f0dbdd9a3795c28bfa00762d9abc536e57501eb69e7cc92`.

The bounded resume uses the authenticated inference package already qualified
for the same checkpoint, with matching dependency sources and saved-law parity.
It must reuse these three controls and permit at most sixteen additional calls.
No candidate choice, proposal RNG, endpoint criterion or docking setting changes.
