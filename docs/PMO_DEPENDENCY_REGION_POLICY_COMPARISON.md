# PMO dependency-region policy comparison

## Outcome being tested

This zero-oracle development gate asks whether generic graph conditioning and an
explicit dependency-region factorization improve held-task-family probability
and rank for complete PMO programs. It does not test PMO objective performance.

The generated object evaluated here is an exact complete executor-supported
action trace within the existing 32-primitive support. All three policies use
the same generic action support. They differ only in conditioning and
factorization:

1. source-balanced generic marginal;
2. flat graph-conditioned autoregressive ranker;
3. graph-conditioned region/dependency hierarchical ranker.

The frozen three-fold split holds out whole PMO task families and every lineage
within those families before fitting preprocessing or categorical heads.

## Candidate evaluations

The comparison reuses the 18 previously generated task-blind source panels. No
new proposal is generated. Two measurements remain separate:

- generic-panel recovery, containing only the original autonomous candidates;
- teacher-injected reranking, which adds held-family teacher endpoints to the
  same panels solely to measure proposal rank.

The second measurement is not autonomous recovery. The current checkpoints
score exact candidate action traces but do not generate typed primitive
parameters or legal bindings. Therefore a positive ranking result cannot yet
authorize a scored PMO run.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python -m tools.pmo_dependency_region_policy_comparison \
  --output diagnostics/pmo_dependency_region_policy_comparison/attempt_1
```

The runner verifies both physical and payload hashes, publishes atomically, and
records zero oracle calls, the current code revision, runtime, peak memory and
the fitted fold checkpoints. It refuses to overwrite an existing attempt.

Focused verification:

```bash
.venv/bin/python -m pytest tests/test_pmo_dependency_region_policy_comparison.py
.venv/bin/python -m ruff check \
  src/compose_v4/experiments/pmo_dependency_region_policy_comparison.py \
  tools/pmo_dependency_region_policy_comparison.py \
  tests/test_pmo_dependency_region_policy_comparison.py
```

The repository-wide suite is deferred until the milestone boundary, in
accordance with the repository testing contract.

## Measured result

The authoritative corrected artifact is `attempt_2`. Across 6,143 held-family
teacher decisions, the common-factorization negative log likelihood was 6.056
for the marginal, 5.905 for flat graph conditioning and 5.794 for the
dependency-region hierarchy. On the 2,792 decisions belonging to routes within
the runtime length support, the corresponding values were 6.258, 6.149 and
6.044.

The pre-existing generic panels contained 559 unique complete endpoints but
recovered zero exact or radius-2 teacher transformations. On the eight of 18
panels that had at least one teacher route within the unchanged runtime support,
teacher-injected mean rank improved from 10.125 to 8.625 to 5.000, and top-10
coverage improved from 0.375 to 0.750 to 1.000. Teacher injection coverage was
only 8/18 and is reported explicitly.

This is positive evidence for dependency-region conditioning as a candidate
ranker and negative evidence that ranking the current generic proposal support
is sufficient. No policy was selected and no scored pilot was authorized. The
next required capability is autonomous, legal site/parameter generation.
