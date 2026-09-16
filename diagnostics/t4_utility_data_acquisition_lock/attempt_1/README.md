# T4 utility data-acquisition lock, attempt 1

This directory freezes four score-blind docking requests, two for `fa7_1` and
two for `parp1_2`. All four candidates come from immutable zero-oracle pools,
replay exactly, pass the unchanged strict delta-0.4 endpoint eligibility rule,
and remain outside the v3 cross-fold connected component.

The structural gate simulation reuses the exact v3 loaders, grouping filter and
support planner. Conditional on all four later calls returning finite values and
forming one non-tied pair per new cell, the frozen corpus would contain 51 rows
after the unchanged 37-row conflict removal and support FA7, JAK2 and PARP1 over
seven held-source strata. The simulation persists no hypothetical score.

`request_lock.json` is not launch authority. Its status is
`UNAUTHORIZED_NOT_LAUNCHED`, retries and backfill are forbidden, and a later
authorization must name the final candidate-lock and request-lock physical and
payload hashes while authorizing exactly four calls. Acquired labels are
training-data inputs only; prospective evaluation requires a separate lock.

Final authorization identities:

- `candidate_lock.json`: physical SHA-256
  `d790389696d729f19ce617b3dcf4afddf0f3ce06110882a1b628889e0c354249`,
  payload SHA-256
  `8b5acdbfca42a122d6a882d48e6bbe00468052c913f79e806b6dbb7164685471`;
- `request_lock.json`: physical SHA-256
  `e11441244c98d919d42315dcafbdede300bbb6232bdc5a911896bdb3d1010d05`,
  payload SHA-256
  `f78985599a7ff5e533384ac7d61c2cc77f7c6cf94e064f6a6ec0e3ed692bc279`.

Rebuild with:

```bash
PYTHONPATH=src .venv/bin/python -m tools.t4_utility_data_acquisition_lock
```
