# Feedback-round development record

## Authorized run, 2026-09-08

The user approved equal 2,500-public-executor-call shares for eight parents,
within the existing 20,000 total ceiling. The single feedback round is now
spawned into deployed `genmol-t4-opt`, with no automatic next round:

- Scientific code: `bdb933948ec6b9f5bbfd0b1c14cce26b7a6a8ef6`.
- Call: `fc-01M2100VP8YHZP8DJMTSCMBBMC`.
- Volume: `compose-v4-artifacts`.
- Prefix: `t4_feedback_round/7b3ff467f806ff38058d520a9aee8273caabe3be0a9e164678f7428ed2f323ff`.
- Contract self-hash: `9fdf7b46c149e49b36eb6196700129af5c89bb2f973304b49dd0f7924ade90c8`.
- Spawn receipt: `attempt_1/spawn.json`, SHA-256
  `3388164b3f463a32a3bf81a364e39910bbcc5dba434446e6d16dedf9821bfce3`.

All three remote source files matched the contract's physical hashes before
launch. No feedback namespace existed. Strict preflight passed in the clean
worktree, followed by deployment and `tools/t4_launch.py --feedback-round`.
The launcher's own clean-source check also passed. Unrelated concurrent
scaffold/decoder work was excluded and preserved in the shared workspace.

Launch verification at the scientific revision: 77 passed in 8.59 seconds,
covering parent-budget cancellation, retained exact outputs, feedback import,
warm continuation, matched-pilot arithmetic, endpoint selection and saved
docking. `launch_focused.xml` SHA-256:
`3ff0eef6f4c90212d8432272f4217c585d099aec3ef6b23fbfc57b7df7857fb6`.
Ruff checks on touched standalone modules/tests and diff checks passed.
The unrelated non-green full suite was not repeated.

The offline auditor now starts after the archive's actual event, so it does
not require a fictitious completed round 2. Three focused auditor tests passed
in 2.20 seconds, recorded in `audit_tests.xml`. This reporting-only change is
not part of the serialized scientific image and does not regenerate proposals.

Results are pending. The starting best is -9.3 at 33 calls. At most 20 new
calls are authorized. Share exhaustion is finite-budget censoring, not
chemical invalidity or evidence that an incomplete compound program succeeded.

## Earlier preparation record

Implementation revision: `01c17cf8ebeb` (full identity available in Git).
The single twenty-call round was authorized in chat. Its intra-round executor
allocation is still awaiting the user's choice. The contract is deliberately
unsealed; no deployment, proposal, or docking has been launched for this task.

Implemented exact-state import of all 33 evaluated molecules plus the original
seed. Preserve their scores and ancestry, distinguish the partial diagnostic
from a completed optimizer round, and use a deterministic, source-hash-derived
fresh-episode RNG. Reuse the existing locked-round runner, frozen inputs,
controller, constraints, and original-seed similarity. One round only, at most
20 new calls, no implicit retries or automatic continuation.

Clean-worktree verification at the implementation revision:

```sh
PYTHONPATH=src:. /Users/rmaganti/compose_rgm_git/.venv/bin/python -m pytest -q \
  tests/test_t4_feedback_round.py tests/test_t4_warm_continuation.py \
  tests/test_t4_matched_pilot.py tests/test_t4_endpoint_selection.py \
  --junitxml=/Users/rmaganti/compose_rgm_git/diagnostics/t4_feedback_round/focused.xml
python3 tools/preflight.py --strict
```

60 passed in 6.68 seconds. JUnit artifact SHA-256:
`d5af9cc0a9492c3298d37bb4ffcc068b1992e533e5f3719147c7caf3128f06d3`.
Ruff check and format check passed on the new module, shared continuation
runner, launcher, and new tests. `git diff --check` passed. The legacy Modal
app was not reformatted. No full-suite or remote-runtime pass is claimed.

The clean worktree excluded concurrent scaffold-construction and ring-fiber
edits. Those edits were preserved in the shared workspace. Local commits are
unpushed. The prior best score remains -9.3 after 33 calls; these tests are
software verification, not new optimization evidence.

Pending choice: retain the shared 20,000-call proposal ceiling or explicitly
allocate 2,500 calls per parent, retaining committed valid outputs and marking
unfinished trajectories. The latter would change finite-budget search effort,
not the executor, admissible molecular support, or primitive transition law
at completed decisions. It is not implemented or implicitly authorized.
