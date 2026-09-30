# T4 lead optimization: submitted ICLR 2027 Table 3

This page traces the **250-call, IVG/GenMol comparison in the owner-identified
ICLR 2027 PDF**, not the earlier 500-call GenMol experiment or the later unified
controller replication. It is an evidence map, not yet a fresh-run recipe.

The table's frozen local reduction is
`diagnostics/T4_FROZEN_RESULT_v1.json` at revision
`bc8cc4216516d7c5ba7181d2b6ac7d473d6b7b7a`. Its file SHA-256 is
`87584d49ceacff2b861f1479b283f47d58f9c60513e1fb63c61fc5f5d8230b9b`;
the submitted PDF and source identity are recorded in
[`experiments/paper/manifest.json`](../paper/manifest.json). To check the
historical file without making a docking call:

```bash
git show bc8cc4216516d7c5ba7181d2b6ac7d473d6b7b7a:diagnostics/T4_FROZEN_RESULT_v1.json | shasum -a 256
```

The JSON contains a sealed payload, thirty target/starting-molecule/threshold
rows, a source label for each row, charged-call counts, and a recorded
correction. Use the JSON rather than its accompanying Markdown summary: the
Markdown was not fully refreshed after a 5HT1B correction. The row source
labels distinguish the historical `panel` from `support_expansion`; they are
not interchangeable with results from the later unified-controller campaign.

## Reconciliation that must not be hidden

- The frozen reduction has 15 COMPOSE scores at similarity threshold 0.4 and
  **14 at 0.6**. Its FA7 starting-molecule-1, threshold-0.6 row has no
  COMPOSE score and records **zero charged docking calls**. The submitted
  Table 3 instead prints **-6.4**, equal to that row's starting-molecule
  score. The reduction does not establish this as a 250-call COMPOSE docking
  outcome. The exact table-producing fallback rule and score provenance need
  confirmation from the T4 run owner before full table reproduction is claimed.
- At threshold 0.6, the frozen JSON stores `sum_gap = -7.6`; summing its
  paired row gaps, or subtracting its paired score sums, gives **-8.5**. The
  artifact is sealed and must not be edited to conceal this discrepancy.
- The source records docking variability from repeated conformer generation.
  Recomputing these table values from saved rows would not re-dock molecules or
  establish that small per-row differences are reproducible.

## Fresh execution boundary

The saved rows alone do not supply a runnable fresh campaign. Fresh docking
requires the exact row-producing launcher and contracts, deployed image,
receptor and QuickVina2 identities, starting leads, checkpoint inputs, and
external per-cell receipts. The T4 handoff locates the shared campaign code in
`src/compose_v4/experiments/t4_fiber_campaign.py`, docking adapter in
`src/compose_v4/experiments/t4_docking_adapter.py`, and Modal entry points
under `modal_apps/`. Those paths do **not** by themselves identify the exact
authorized Table 3 launch. Do not substitute a similarly named later app or
run a new docking campaign as a repository-cleanup step.

The T4 production environment uses Python 3.11 and RDKit 2024.3.5. PMO uses
a different chemistry kernel. A result recomputed in the laptop environment
needs a parity check before it can be compared with either production run.
