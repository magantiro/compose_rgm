# Pathwise gate repair: saved-action evidence

The new explicit `executable_intermediates_v1` mode reproduces **17/17** saved
ring-restatement products through each of `generic`, `restate` and `aromatize`,
versus **10/17** under the preceding full-screen-at-every-step implementation.
These are 17 actions checked through three channels, not 51 independent cases.

The seven prior rejections are completely accounted for:

| Pair prefix | Prefix step (zero-based) | Existing endpoint rejection |
| --- | ---: | --- |
| 044793ef3c40 | 0 | cumulene:C22 |
| 2e08e2ca8f17 | 0 | cumulene:C24 |
| 65ebee68b32b | 0 | cumulene:C16 |
| 91cea65cdf2d | 1 | isolated_ring:11 |
| b2a368725001 | 0 | cumulene:C5 |
| c5d37fe1a020 | 21 | isolated_ring:11 |
| f69b7777b66a | 0 | cumulene:C28 |

None has a pathwise rejection under the already existing pathwise predicate.
All seven remain unacceptable returned endpoints under the unchanged full
medicinal screen. The three PARP1 seed0/d=0.4 restatement cases passed before
this change and still pass; this repair does not itself improve that cell.

## What changed and what did not

The option kernel can now use the existing pathwise predicate for intermediate
products. The hierarchical T4 adapter explicitly screens completed candidates
before terminal reward or oracle allocation. Rejected completed candidates stay
in the diagnostic pool with separate benchmark and medicinal-screen reasons;
they can remain search states. Replay verifies the declared policy and endpoint
metadata. Old defaults/configuration serialization remain unchanged. New runs
must explicitly set `preparation.product_gate=executable_intermediates_v1`.

This is a gate-placement change, not a new ring operator, threshold, template,
learned policy or executor. The medicinal screen is an inherited heuristic,
not a chemical stability, synthesis or binding guarantee. These are graph-edit
paths, not synthetic reaction routes. Local/global selection, option support
contracts, kappa, generic, learned parameters and region/context guards remain.

## Evidence and verification

Authoritative producer: `8c0a6d4f2d7bf325f74f669d6b7378359a76a3e1`, clean worktree
`/private/tmp/compose-winner-paths.W1n7sh`. Scope was recorded prospectively in
`docs/T4_PATHWISE_GATE_REPAIR.md` at that revision.

- `audit.json`, SHA-256 `76a4a3609e9d119bbbca16bc562fb4496fcb77f583c3015946bc72334337106a`.
- Prior footprint audit SHA-256 `9c93f491837ebdce607fa587cd357348ec33d8f83f5c5347395861e1aafa3824`.
- Original winner audit SHA-256 `3b7fb79a541b94dd5421f30ffc0516c5cdc4d9316845cd40561d799ec517f922`.
- All exact input receipts are hashed in the audit. Seventy-six original source
  dependencies are verified unchanged; the repaired region code matches the
  preceding repair. Gate and integration changes have separate source hashes.
- Saved-action check: 1.650416417 seconds, 17 option-kernel executor applications,
  zero new searches, zero learned-law enumerations and zero docking calls.
  Kernel application counts are not total internal micro-executor counts.
- Focused clean-source tests: 70 passed, zero failed/skipped/errors, 10.104 seconds
  in the XML. Formatting and lint pass for all seven touched Python files;
  strict preflight and diff checks pass. Full repository suite was not run under
  the scoped T4 development policy; this is not a release milestone.
- Two initial fixture failures (wrong semantic rule name and uncontrolled WHERE
  draw) were repaired in tests before freezing this source, without changing
  production gates to obtain a pass. Initial lint caught a mutable test class
  attribute; it was moved into the fixture initializer.

Reproduce from that clean revision using the existing local environment:

```sh
PYTHONPATH=src:. /Users/rmaganti/compose_rgm_git/.venv/bin/python tools/preflight.py --strict
PYTHONPATH=src:. /Users/rmaganti/compose_rgm_git/.venv/bin/python tools/t4_restate_footprint_audit.py /Users/rmaganti/compose_rgm_git/diagnostics/ivg_winner_paths /Users/rmaganti/compose_rgm_git/diagnostics/t4_pathwise_gate_repair/reproduction.json --gate-repair-baseline /Users/rmaganti/compose_rgm_git/diagnostics/t4_restate_footprint/audit.json
```

## Interpretation and next step

Measured: the misplaced screen explains these seven singleton-action failures.
Synthetic production-executor tests also show two-step traversal through a
cumulene and a large ring, lazy/eager agreement, retained invalid/frozen guards,
and zero terminal reward/oracle selection for unacceptable completed candidates.

Not measured: positive learned probability on these actions, complete guided
winner-path discovery, better docking or pinned-Modal-runtime equivalence.
The audit uses singleton diagnostic mass, not R_theta. Local RDKit is 2026.03.6,
whereas the T4 runtime is pinned to 2024.03.5. The existing task-value snapshot
has not passed local feature equivalence, so it was not used for predictions.

The next scientific question remains whether task guidance recognizes productive
continuations across option boundaries and sufficient remaining horizon. Use a
new explicitly scoped preparation comparison in the pinned runtime, not another
winner-directed search or an automatic docking campaign. No deployment, training,
docking or push occurred in this repair.
