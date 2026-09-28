# Local fragment runtime integration verification

This is implementation verification, not a new benchmark result. The normal task
entry point is [experiments/fragments](../../experiments/fragments/).

`verification.json` indexes the exact hashes of test outputs and copied parity
receipts. `parity/` preserves the six identity-selected attempted slots, their
source/asset/adapter provenance, and exact comparisons. Five task variants were
run from minimal source exports without Git metadata or private worktrees.

- `focused.xml`: 62 passing focused tests, including all five exported runtimes.
- `focused_attempt1.xml`: earlier negative source-export provenance assertions,
  retained. Git had attributed plain exports to an enclosing checkout; repaired
  without changing a sampler, metric, fixture, or chemical support.
- `full_suite.xml`: collection failure in the pre-existing Python 3.11 protocol
  introspection test. The full repository suite did not pass.

Touched-code Ruff, formatting, and `git diff --check` passed. Strict local
preflight passed at `e8a4f3e76036` with zero mounted source/config drift. Local
implementation commits are `c6125456` (preserved source/fixture identities) and
`e8a4f3e7` (runtime interface, tests, and guide); neither was pushed. Tests ran
before these commits. Every saved parity receipt's adapter hashes was checked
against the committed implementation and matched, so the hashes, not an
enclosing or pre-edit Git revision, identify the tested code.

The earlier motif packaging failure is retained in
`runs/fragments/parity_v1/.motif_extension.pending-yb9c5tor/` locally. It produced
no molecule or benchmark number. The missing, unchanged prior module was then
included in the captured dependency closure.

Commands used (from the repository root, in the pinned local environment):

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:scripts .venv_pinned_chem/bin/python -m pytest tests/test_fragment_runtime.py tests/test_fragment_evidence.py tests/test_fragment_locked_panel_selection_v1.py tests/test_fragment_generation_local.py -q --tb=short
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONPATH=src:scripts .venv_pinned_chem/bin/python -m pytest tests/ -q --tb=short
PYTHONPATH=src .venv_pinned_chem/bin/python -m compose_v4.experiments.fragments verify --recompute-intervals
PYTHONPATH=src python3 -m compose_v4.experiments.fragments assets --assets local_assets/fragments
```

The recorded focused run additionally used `--junitxml` and a new durable
`--basetemp=runs/fragments/verification_20260927_v2`. Do not reuse that base-temp
path: pytest clears an existing base-temp directory. Preservation/reduction:

```bash
python3 tools/summarize_fragment_runtime_verification.py --test-artifacts runs/fragments/verification_20260927_v2 --output diagnostics/fragment_runtime_integration_v1
```

That command refuses to replace this report. The individual parity receipts are
the authority for execution configuration and input hashes; copied test outputs
record a source export rather than inventing a Git revision. No full-panel
generation, cloud launch, oracle call, training, manuscript edit, or publication
was performed. Full raw-panel replay, public asset access, and cross-platform
equivalence remain outside the verified boundary.
