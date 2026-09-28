# Fragment saved-metric reduction

Produced locally with:

```bash
PYTHONPATH=src .venv_pinned_chem/bin/python -m compose_v4.experiments.fragments tables --recompute-intervals --output diagnostics/fragment_repository_integration_v1
```

`tables.json` is the machine-readable reduction and `tables.md` is its readable
form. `provenance.json` binds inputs, actual implementation bytes, environment,
configuration, working-tree state, and output hashes. This candidate was executed
before its integration commit, so its recorded base revision is accompanied by
the implementation hashes and explicit uncommitted state, not presented as a
clean frozen launch.

This reproduces numbers from saved per-prompt metrics, including all four
prompt-bootstrap comparisons. It does not regenerate molecules or reevaluate
the external comparator columns. The exact historical input artifacts are
unchanged. See the [task instructions](../../experiments/fragments/README.md) and
[verification record](../../docs/FRAGMENT_REPOSITORY_INTEGRATION.md).
