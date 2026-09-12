# Persistent PMO archive pilot, 2026-09-10

User approved one paired development run on perindopril MPO, 100 unique oracle
queries per arm including four objective-blind initial molecules. No docking,
prescreening, winner inputs, reference-model training, or full-PMO claim.

Implementation is isolated in `/private/tmp/compose-pmo-archive-pilot`, branch
`pmo-archive-pilot`, clean scientific revision `8ef5eb220435` (full identity in the
spawn receipt). No branch has been pushed; the main worktree's unrelated changes
are preserved. Design and mathematical framing:
`/private/tmp/compose-pmo-archive-pilot/docs/PMO_ARCHIVE_PILOT.md`.

The optimizer retains scored exact molecular states indefinitely within the query
campaign. Both arms use rank-biased canonical parent selection with exploration.
The adaptive arm additionally changes region/option probabilities using realized
archive-improvement credit, including discounted credit to ancestor options.
The balanced arm does not. The full option support and primitive kernel are the
same. No beam retention or fixed three-option ancestry cutoff remains in this lane.

This is an archive-valued, budgeted semi-Markov problem with a proposed empirical
allocation policy. The KL/floor calculations and execution-support claims have
explicit definitions; the policy is not an exact Bellman solution or a demonstrated
SOTA method. The metric is the 100-query mean of zero-padded top-ten archive scores.
One pair is a development diagnostic, not generalization evidence.

Two CPU containers, 8 GiB each, at most 100 unique queries and 300 option attempts
per arm, 75-minute operational timeout, no automatic retries. Completed endpoints
are durably locked and immediately scored. Ordinary writes use periodic commits;
oracle starts and results force durability. Thirty-second heartbeats expose
progress. Timeouts and unresolved oracle attempts remain incomplete/blocked.

Verified before launch: ten distinct relevant cases passed across focused runs
(probability and canonical-mass invariants, delayed credit, exact option execution,
interruption/resume, oracle charging, saved-law compatibility, and shared-context
parity). An initial tuple/list resume-format mismatch was fixed and rechecked.
Ruff and strict preflight passed. No full repository suite ran, and no controller
milestone is declared complete.

## Zero-query setup profile

`cached_setup_profile.json` binds the pre-optimization revision `ebf4f252a381`.
`cached_setup_profile_optimized.json` binds `ceaf245539b1`. Both use the same saved
production-law file and exact first source, choose its largest-release region,
and perform no new model enumeration or oracle evaluation.

Measured locally, cold option setup decreased from 1.696 to 0.970 seconds;
warm repeated setup decreased from 0.743 to 0.032 seconds. Both expose 155 regions,
37 options, and 29 executor applications. The repair computes the exact region
context once per option row rather than separately for every option. A focused
test compares every option state against separately constructed contexts.
This does not measure end-to-end speedup or eliminate model-enumeration cost.

The reused law lives at
`compose-v4-artifacts:pmo_macro_probe/e68821e364011d20d2fdb8aec23b0084e7fd9f74b9cb124d5be225ce4674fe32/perindopril_mpo__post_hoc__0/laws/8d63a3d3d0e801f146f11f5d36592562efbd1a4cd63bf3e3e36534c7a008f227.json`.
Only compatible deterministic laws may be reused; prior oracle labels are excluded.

## Operations

App: `compose-pmo-archive-pilot`, function `archive_case`.
Deploy from the clean worktree, then use `tools/pmo_archive.py launch`; never
`modal run --detach`. The spawn receipt under this directory is the authority for
which cases actually started. An image deployment alone is not a scientific run.

```sh
PATH=/Users/rmaganti/compose_rgm_git/.venv/bin:$PATH \
PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 \
/Users/rmaganti/compose_rgm_git/.venv/bin/python tools/pmo_archive.py status \
  --output /absolute/path/to/spawn.json
```

Durably spawned at approximately 2026-09-10 20:37 UTC, after successful deployment:

- Run: `029a27392680e2e57bd9aa9567b005057f9c68fb5450807a768cc80e8e05c892`.
- Balanced call: `fc-01M26GKK0WFSW8BEJ3GGFN1Z84`.
- Adaptive call: `fc-01M26GKK84TGHTJKA0NAQ9N2Z5`.
- Receipt: `029a27392680e2e57bd9aa9567b005057f9c68fb5450807a768cc80e8e05c892/spawn.json`.
- Remote prefix: `compose-v4-artifacts:pmo_archive_pilot/<run>/<case>/`.

No final result has been recorded at launch. Read final findings from the actual
case outputs, with the complete query ledger and provenance. Do not relaunch this
receipt to poll status.
