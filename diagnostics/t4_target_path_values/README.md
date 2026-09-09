# Saved winner paths contain immediate-score valleys

2026-09-09. Computed on the three already inspected PARP1 seed0 d=0.4
development witnesses, using their exact stored states. No paths were searched,
no learned laws were enumerated, no models were fitted, and no docking was run.

| Saved path length | Similarity-decreasing edits | Largest decline from an earlier best | T4-infeasible prefix steps |
| --- | ---: | ---: | --- |
| 23 | 7 | 0.208258 | 11-22 |
| 25 | 8 | 0.178711 | 11-24 |
| 27 | 10 | 0.130739 | 12-26 |

The score is the current target-recovery diagnostic's fixed mean Morgan and
atom-pair Tanimoto similarity, not SMILES text distance or docking. On the
23-edit path it rises to 0.516915 at step 12, falls to 0.308656 at step 17,
and eventually reaches the exact target. All three terminal canonical 2D
identities match their destinations and satisfy the recomputed T4 endpoint
constraints. This scores saved states under the pinned chemistry runtime;
it is not a new pinned-runtime action replay.

The analysis scored all 78 stored states with no exclusions in **0.937724 s**
after imports/input selection. This is scoring time, not end-to-end search or a
speed comparison with the cancelled recovery attempt. Software: Python 3.12.9,
NumPy 1.26.4, RDKit 2024.03.5, CPU arm64 macOS, float64 score arithmetic.
No randomness or accelerator was used.

## Decision and limitations

These particular paths require temporary regressions in the immediate score.
That does not establish that every possible path does, that a stochastic
one-step controller cannot traverse them, or that the learned law supports the
saved marks with useful probability. No action-rank/probability comparison was
performed. The witness was found with knowledge of its destination and remains
development evidence.

Do not promote the provisional `next_option_heuristic` implementation to the
main controller based only on lower planning cost. Keep immediate guidance as
an explicit cheap ablation. The main practical direction is option-aware
lookahead with retained alternative trajectories across option boundaries,
and task feedback from the actual oracle. Infeasible intermediates cannot be
discarded solely because they currently fail endpoint requirements. Completion
of one option is not proof that a longer feasibility excursion is finished.

No ring menu was added or removed. Existing construction programs remain
optional proposal channels alongside generic atom/bond editing; the draft's
particular ring recipe is not a scientific requirement. Its claimed benchmark
scores have not been revalidated here. In particular, repeated independent
benchmark runs are replication cost, not extra sequential budget available to
one optimizer run; the draft's sixfold budget claim needs a separate protocol
audit before reuse.

## Provenance and reproduction

Scientific code: `e37eb97704c9e19360952203cf78cf11b55fd4bf`.
Prospective scope: `docs/T4_WORKSHOP_STRATEGY_REVIEW.md` at that revision.
Result: `result.json`, physical SHA-256
`2e75be234b4ecedffe6cdfd639f4d49c1d32684427f090c5035d54c17c2610c6`.
The result binds the source audit, all three compressed witness receipts,
scoring implementation and SA assets by physical hash, and retains every
per-step score, endpoint property, exact-state hash and preceding mark.

The clean committed diagnostic worktree passed **7 focused tests**, zero
failures/errors/skips, in 3.03 s. XML receipt: `focused_tests.xml`, SHA-256
`c6dc2814a0bd1c588e0c68586e5fb0e8e29ad3f98b5a773906423defcb8ce959`.
Ruff lint and format checks and diff whitespace checks passed. No unrelated
repository-wide suite was run; this is not release qualification.

```sh
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 \
  /Users/rmaganti/compose_rgm_git/.venv/bin/python tools/t4_target_path_values.py \
  --audit /Users/rmaganti/compose_rgm_git/diagnostics/ivg_winner_paths/audit.json \
  --audit-sha256 3b7fb79a541b94dd5421f30ffc0516c5cdc4d9316845cd40561d799ec517f922 \
  --output /Users/rmaganti/compose_rgm_git/diagnostics/t4_target_path_values/result.json
```

Run from a clean source tree; the output must be outside it. The isolated
chemistry overlay is operational, not a fully portable environment definition.
The repository's frozen Modal chemistry versions identify the reproducible
dependencies; no Modal job was needed for this analysis.
