# Bounded actual-score continuation

Prospective recipe: `docs/PMO_REPLACEMENT_CONTINUATION.md` and the self-hashed
`configs/pmo_replacement_continuation.json`. The prepared artifact contains all
five preceding warm roots and all four completed replacements, with exact
slot-addressed state and historical label provenance. No new oracle calls were
used for preparation.

Local engineering verification: 21 focused tests passed in 8.16 seconds on
2026-09-11, pinned RDKit overlay, local Python 3.12. These cover old/new particle
arms, shared-task identity, archive completeness, resume, oracle candidate locks
and charging, replacement execution, and duplicate/concurrent persistence.
Command:

```
env PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/rmaganti/compose_rgm_git/.venv/bin/python -m pytest -q tests/test_pmo_store.py tests/test_option_particle_driver.py tests/test_pmo_archive_pilot.py tests/test_pmo_locked_batch.py tests/test_region_replacement_probe.py tests/test_region_replacement.py
```

Ruff and `git diff --check` passed for the touched code. No repository-wide
suite or new local neural qualification is claimed. The existing historical
branch-policy process-identity test failure remains recorded in the preceding
replacement report; its gate was not changed. The qualified production runtime
will enforce its own exact input and numerical identities remotely.

The persistence fixture confirms one barrier instead of two for an overdue
immutable save, and no second periodic commit after a waiting heartbeat observes
a completed barrier. This is an engineering count, not a remote timing result.
Both mandatory oracle barriers remain unconditional. Scientific results are
pending a clean-source deployment and durable launch.
