# No intermediate similarity penalty

Status: launched, result pending. No performance conclusion yet.

Generation revision: `eabe4addefb8ce2f0665a6d8ff4cf5b3cc417039`.
Run: `232389884cb5ee6fa95b846d0f7e0e7098cab08e0995617f02951e142fdd9050`.
Modal call: `fc-01M24VJ2ZMBJ4JXA7BG9KJFSVW`.
Volume: `compose-v4-artifacts`, path recorded in `attempt_1/launch.json`.

One new arm, at most 30 option attempts, zero docking. The three previous
constraint-recovery arms are reused, not rerun. `control_reuse.json` binds the
old and new revisions and verifies unchanged legacy beam code after removing
the exact opt-in additions, with unchanged scientific dependencies and inputs.

Only intermediate ranking changes: no direct original-seed similarity penalty.
The QED/SA penalty, task snapshot, local/global region prior, option prior,
generic channel, kappa, incumbent preservation, and final eligibility remain
unchanged. No inference about IVG's internal trajectories is required.

Verification: 11 focused tests passed in 33.75 seconds on the clean launch tree
(score isolation, final gate, launcher, incumbent, replay/resume and beam
behavior). Strict preflight, touched-code Ruff lint/format, and diff whitespace
checks passed. The existing app's 17 lint findings are unchanged; no new app
finding was introduced. The repository-wide suite was not run and milestone
qualification is not claimed. Deploy completed before the durable spawn.

Collect progress/result with pinned RDKit 2024.03.5:

```sh
env PYTHONPATH=/private/tmp/compose-t4-chemistry.hizM8Y:src:. OMP_NUM_THREADS=1 \
  .venv/bin/python diagnostics/t4_no_similarity_penalty/fetch.py
```

The collector never launches work. On completion it reuses the existing
constraint-recovery audit, verifies locked products, exact roots, shared first
proposals, model/input identities, and the new score formula, and records input
hashes with the comparison. Internal predictions are not new docking scores.
