# Verification status

This scoped evidence commit is not a completed milestone.

- `tests/test_virtual_joint_region_proposer.py`: 16 passed in 2.29 seconds.
- Ruff on both diagnostic tools: passed.
- Python byte compilation on both diagnostic tools: passed.
- `git diff --check` on the scoped paths: passed before staging.
- The primary result and both allocation diagnostics reproduced byte-for-byte.
- Black with `--target-version py312`: did not pass because it would reformat
  both diagnostic tools.

The primary `attempt_1` result was already sealed with the exact runner SHA-256
before the Black check. The runner and result are intentionally preserved
byte-for-byte so the recorded implementation identity remains reproducible.
Formatting the runner and regenerating a separately versioned artifact is left
for a later authorized allocation revision. This limitation is not waived and no
milestone-completion claim is made.
