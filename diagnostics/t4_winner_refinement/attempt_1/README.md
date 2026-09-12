# Winner controls completed; proposal initialization failed

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
