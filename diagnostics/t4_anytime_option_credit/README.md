# Anytime option credit: implementation verification

Implemented the opt-in `anytime_options_v1` return policy in the existing
hierarchical planner. Completed simulations back up the best acceptable
new-candidate utility along each suffix, rather than only the exact last edit.
Low-value intermediate completions do not terminate planning. Already archived
molecules receive zero new-candidate utility. Final endpoint screens and the
16-edit budget remain unchanged. Old defaults/configuration serialization remain
compatible; the new policy requires `executable_intermediates_v1`.

Scientific implementation commit: `54fa85fa9a7466314005d5210227593e84ebe19b`.
Prospective scope/limitations: `docs/T4_ANYTIME_OPTION_CREDIT.md` at that revision.
This is a new development planning objective, not a claimed fix preserving the
old terminal-only return law or a reinterpretation of the old negative run.

The clean-source focused suite passed **45 tests**, zero failures/errors/skips,
in **9.610 seconds**:

```sh
PYTHONPATH=src:. /Users/rmaganti/compose_rgm_git/.venv/bin/python -m pytest -q tests/test_anytime_option_credit.py tests/test_task_search.py tests/test_t4_task_search.py tests/test_pathwise_option_gate.py tests/test_lazy_reference.py --junitxml=/Users/rmaganti/compose_rgm_git/diagnostics/t4_anytime_option_credit/focused_tests.xml
```

XML SHA-256: `75c4257df180a83bc266d563faf2ae3d1861e8ed8f649c38fe6bd3e9af4ba772`.
Clean worktree: `/private/tmp/compose-winner-paths.W1n7sh`. Local test runtime:
Python 3.12.9, NumPy 2.5.3 and RDKit 2026.03.6. Strict preflight, diff checks,
Ruff lint and formatting passed for all five touched Python files. The unrelated
full repository suite was not run under the scoped T4 development policy.
No release milestone, deployment or docking campaign is claimed.

Covered: early useful completion followed by a poor final edit; later improvement
through an initially low-value completion; no propagation of already-past rewards
into future-only suffixes; no partial returns on interruption; lazy and adaptive
floors/KL; old behavior and config identities; exact replay, frozen-round models,
endpoint exclusions and completed-parent reuse in the T4 adapter.

These fixtures use declared synthetic rewards where needed. The separate
`../t4_ring_chain_probe/result.json` measures prescribed ring/local/ring capability
and real frozen-predictor diagnostics, not autonomous use of this return policy.
A matched learned-reference preparation is still needed to test actual utility.

Only the isolated temporary chemistry overlay was installed for the empirical
probe. The project environment and unrelated scaffold/model changes were
preserved. No pushing, training or new oracle calls occurred.
