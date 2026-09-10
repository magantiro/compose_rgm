# Small PMO controller-transfer probe

User authorized a few PMO tasks on 2026-09-10. This is a new development
experiment, not a reopening of the old frozen PMO pilot, a full benchmark,
reference-model training, docking, or a claim of superior performance.

Question: does objective feedback at completed macro boundaries improve search
over the same frozen executable process? Output: canonical scored molecules,
exact persistent states, and replayed primitive trajectories. Support remains
connected 1..40-heavy-atom graphs, existing vocabulary/charge policies and no
stereochemical claim. Initial sources are an explicitly neutral achiral
projection of an existing development cohort, selected for structural diversity
without objective labels. This is not standard full-PMO initialization.

Three tasks span similarity (`albuterol_similarity`), composite property
optimization (`perindopril_mpo`) and learned activity (`jnk3`). The same four
roots, option support, WHERE/WHAT prior, primitive executor and reference model
are shared by both arms. The current completed-option beam is reused. Guided
retention reads actual budgeted PMO scores; post-hoc retention cannot access
them until its root's search is locked. No docking surrogate or new value-model
training is used. This isolates completed-option retention, not every component
of the T4 optimizer, and does not claim an exact Doob transform. The existing
KL=1 and exploration floor are unchanged.

Four independent roots, depth three, width two, two branches, two paired seeds,
two arms, three tasks: twelve independent cases. Each has at most forty complete
option attempts and forty-four distinct oracle calls including all initial,
ranking and returned molecules. Canonical repeats are free within a case;
labels are never free across cases. All attempted calls have durable locks and
receipts. Neither PMO objective is altered or supplemented with T4 endpoint
constraints. `generic` stays active. Source projection/exclusions are recorded
before scoring and do not redefine the generator's support.

Compare full short-prefix best/top-ten curves, per-root improvement, paired-arm
differences, option completion, realized structural change, diversity, exact law
and executor counts, wall time and oracle failures. Do not extrapolate forty-four
calls into a 10,000-call AUC. A sparse-activity null is inconclusive at this
budget. These tasks have prior development exposure; neither fresh replicate
seeds nor this recipe make them held-out tasks. A full benchmark needs a frozen
controller and a separate authorization.

No compatible results exist in the new remote namespace. Historical PMO arms
used different source states, controller code, prescreen information or
accounting and cannot substitute for this paired result. Existing frozen
oracles and generator inputs are reused. JNK3 uses the already parity-checked
2048-feature forest; the two other objectives use PyTDC 0.3.6. All input hashes,
source revision and runtime versions are bound in the contract and results.

Compute: twelve one-CPU, 8-GiB containers, no GPU, no retries, 2,400-second
per-case timeout. Historical complete-option costs were about 22-29 seconds;
forty attempts imply approximately 15-30 minutes per case allowing overhead.
Estimated total cost $1-4 is an operational estimate, not a guarantee. Twelve
timeouts cap reserved compute at eight CPU-hours. Each root and primitive
prefix checkpoints independently. Heartbeats every 30 seconds and per-oracle
scores expose progress. Timeout is incomplete, not a negative scientific result.

Prelaunch verification: five new focused tests passed in 2.37 seconds, including
the existing ten-molecule JNK3 parity fixture. Seven unchanged beam/cache tests
passed in 18.92 seconds. Ruff and whitespace checks passed. No full suite was
run, and this development launch does not declare the controller milestone done.

Focused checks cover oracle charging/resume, candidate locks and feedback
timing, unchanged beam retention, and launch bounds. No broad regression suite
is an iteration gate for this bounded development probe. This does not waive
the milestone/release verification requirements. Clean committed source and
strict preflight are required; deploy the PMO app, then use durable `spawn`.

```sh
PYTHONPATH=src:. python3 tools/pmo_probe.py prepare --output /tmp/pmo-probe-unused
python3 tools/preflight.py --strict
modal deploy modal_apps/pmo_macro_probe_app.py
PYTHONPATH=src:. python3 tools/pmo_probe.py launch --output /path/to/diagnostics/pmo_macro_probe
PYTHONPATH=src:. python3 tools/pmo_probe.py status --output /path/to/run/spawn.json
```
