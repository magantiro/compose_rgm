# Retrospective T4 evidence bundle

Start with `REPORT.md`; `verification.json` binds the original audit inputs and
code. No new docking or model fits were performed by this audit.

- `scored_rows.jsonl.gz`: lossless 35,895-record historical scored table, not
  independent samples and not a pre-split training dataset.
- `scored_rows.jsonl.manifest.json`: compressed/decompressed hashes and counts.
  The exact pack producer is in commit `5b3e228d`; a later formatting-only cleanup
  does not change the compressed records.
- `audit.json`, `search_audit.json`, `probes.json`, `route_comparison.json`:
  numeric evidence and explicit limitations.
- `recovery_manifest*.json`: remote source paths, retrieval status and hashes.

The larger raw checkpoint/query mirror remains at
`/Users/rmaganti/compose_rgm_git/diagnostics/t4_strategy_reset/20260916/raw`.
Its source volumes are `compose-v4-artifacts` and
`compose-t4-dynamic-v0-full-suite` under Modal profile `nitya`.

The audit scripts deliberately refer to the frozen code worktree
`/private/tmp/compose-t4-complete-region-policy-20260916` (revision `2114c405`).
Recreate that exact detached worktree if needed, rather than silently running
the audit against a different controller. `fetch_history.py` can recover remote
files, but run it into a new directory and verify against the preserved manifests;
do not overwrite this frozen evidence with a later live checkpoint.

Before model fitting, freeze source/lineage groups and derive features/components
only within training folds. Docking duplicates have nontrivial noise. Missing call
indices, unfinished results and observed failures must remain explicit.
