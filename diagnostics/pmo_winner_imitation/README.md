# Answer-informed winner reconstruction

Measured locally on 2026-09-11 (UTC), using the pre-outcome
[protocol](../../docs/PMO_WINNER_IMITATION.md) and existing verified witnesses.
No new docking or PMO oracle calls were made.

## Result

Both the learned and initial untrained rerankers reconstructed the same known
public winner from all five starting states. Every returned trajectory was
independently replayed through the production executor to its exact canonical
endpoint. Coverage was 5/5 for each arm and endpoint correctness was 5/5 among
each arm's returned reconstructions. No source was excluded.

| Starting state | Verified teacher edits | Untrained edits | Learned edits |
| --- | ---: | ---: | ---: |
| current_best | 55 | 56 | 56 |
| original_root_0 | 38 | 39 | 38 |
| original_root_1 | 41 | 42 | 41 |
| original_root_2 | 40 | 42 | 40 |
| original_root_3 | 41 | 47 | 41 |

Preparation took 6.94 seconds. Fitting took 1.77 seconds (600 updates); fitting,
auditing and both sets of rollouts together took 15.87 seconds on one CPU thread.
Combined preparation and train-command time was 22.80 seconds, excluding code
development and checks. Recorded next-action top-1 agreement improved from
40/215 (18.6%) to 198/215 (92.1%). Perfect imitation was not achieved or required
for endpoint recovery. The teacher routes are not certified shortest paths.

## Interpretation and limits

This establishes executable, closed-loop target reconstruction for these five
source alignments. The reranker reads the actual state, aligned target and each
valid candidate, not a teacher action, route cursor or recorded suffix. No
endpoint is inserted into the proposal pool.

However, the proposal mechanism receives the target's exact persistent-slot
correspondence and restricts alternatives to target-difference-directed edits.
The untrained 5/5 result shows that this privileged proposal mechanism already
solves these examples. Learned ranking reduced total steps from 226 to 216 but
did not improve recovery. This is not evidence that imitation caused recovery,
that unknown high-scoring molecules can be discovered, or that a trained
controller generalizes to new targets. All five demonstrations are training
examples, and the public molecule is a previously documented manual
transcription, not a newly obtained authenticated IVG trajectory.

The frozen reference model, executor and production local/global controller
were not changed. This separate diagnostic is not the reference-conditioned
KL controller, does not enumerate its full support, and makes no exact-Doob or
representation-invariance claim. It uses primitives, not a new ring catalog.

The next useful development question is how to recover useful edit preferences
when the winner/alignment is withheld. More ring operators are not required to
reconstruct these particular examples. A score-guided run must measure that
separate problem without interpreting this target-informed result as discovery.

## Artifacts and verification

- `config.json`: source identities, prepared-panel hash, software, balancing and
  fixed fitting/rollout configuration.
- `result.json`: complete outcomes, step decisions, successful replayed states,
  losses, source/implementation hashes and model hash. The execution records
  base revision `53e2451` and its then-uncommitted, hash-bound diagnostic code.
- `panel.pt` and `model.pt`: retained locally, ignored by Git. The checkpoint
  includes model, optimizer, random states and configuration. Rebuild with
  `tools/pmo_winner_imitation.py prepare`, then `train` in the pinned environment
  recorded in the JSON. The preparation stage does not allocate a GPU.
- `tests/test_winner_imitation.py`: three focused tests passed in 2.15 seconds
  for valid execution, differentiability, no teacher replacement, budget and
  exact endpoint handling. Ruff lint passed; formatting was applied to the test.

The repository-wide release suite was not run. This bounded diagnostic is not a
completed production-controller milestone. No remote job was launched.
