# Editing V2 P50-v2 prospective decision

Status: frozen prospective training decision. This document does not authorize a
run and does not alter the completed P50 result.

## Decision

Retain connected-nonleaf atom deletion in the Process-V2 legal rewrite support,
but do not treat `editing_v2_active8_v1:atom_delete:connected_nonleaf_death` as a
balanced load-bearing training cell in P50-v2.

P50-v2 balances the remaining 16 cells with 200 exact source-unique examples per
cell. It forbids source reuse anywhere in the 3,200-example pilot stream and uses a
learning rate of `3e-4`. The optional deletion cell receives no balanced P50-v2
coefficient and no learned-capability claim. A later production-calibration stage
may include it only at an explicitly frozen natural mass or after a larger,
diverse teacher set is acquired.

The executable fiber is unchanged. This is a training-role decision, not a support
withdrawal or an executor change.

## Evidence ledger

Measured:

- The current Gate 0 evidence contains 2 admitted train transitions for
  connected-nonleaf deletion, versus 725,683 leaf deletions. The decision artifact
  self-hash is
  `613259c4421c848fbea2eb0c68d8d2e5183bab4d27266b61b4245d25aed7e748`,
  and its file SHA-256 is
  `cf6d8aaa2cfeeebecee3f2a62695c92cdf9b0ae0aa7c644216f54578bc186607`.
- The completed P50 selector repeated those two connected-nonleaf sources to fill
  189 scheduled cell slots. Its selection self-hash is
  `e2327db068de863a0ea6809b74ee1bc37f6e55932009659f86d4819125c768ff`.
- The completed P50 result was `NO_GO`. Overall validation successor negative log
  likelihood improved, but six observable cells regressed and three crossed the
  catastrophic-regression threshold. The result self-hash is
  `be712efd7c4c979c6ab92bf708f375b30d125c73281da1f46f994efbcdc939c7`.

Inferred from the frozen operator semantics:

- A valid connected-nonleaf deletion removes a nonarticulation atom. Removing its
  incident cycle edges until it becomes a leaf, followed by ordinary leaf deletion,
  reaches the same final graph. The direct operator therefore improves path length;
  it does not add molecular reachability within the currently admitted cases.
- Retaining legal support preserves that possible efficiency gain. Repeating two
  sources under equal-cell balancing estimates neither a realistic transport law
  nor a diverse capability objective.

Proposed for later evaluation:

- Measure whether direct connected-nonleaf deletion materially improves E3 or E4
  path efficiency under the frozen edit budgets.
- If it does, acquire a diverse, source-separated teacher panel before promoting
  the cell to a load-bearing learned capability.

## Frozen P50-v2 delta

- optional path-efficiency cell:
  `editing_v2_active8_v1:atom_delete:connected_nonleaf_death`;
- balanced cells: 16;
- examples per balanced cell: 200;
- total examples: 3,200 in 50 batches of 64;
- selection: deterministic, exact source-unique, without replacement;
- source reuse across cells: forbidden;
- candidate reserve: at most 1,024 exact sources per balanced cell;
- optimizer: unchanged AdamW recipe except learning rate `1e-3` to `3e-4`;
- objective, support, model architecture, seed, batch size, hazard exclusion,
  validation partition, and thresholds: unchanged.

The previous P50 remains an authoritative negative diagnostic under its original
policy and commit. It is not relabelled or overwritten.
