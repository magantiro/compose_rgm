# PMO route-prior plus FiberControl pilot

## Question and claim boundary

This prelaunch asks whether generic structural knowledge distilled from successful
programs can improve the proposals available to an online, objective-driven PMO
controller. The intended controller is:

```text
route-distilled or unchanged proposals
-> exact COMPOSE execution and validity
-> blind or online FiberControl selection
-> native PMO reward
```

The scored comparison is a two-by-two factorial on Celecoxib rediscovery, GSK3B
and Perindopril MPO. Each route proposer must exclude the exact evaluated task and
every shared lineage before fitting. Scores never transfer between tasks or arms.
The old-proposal plus blind arm is the unchanged autonomous control. The
old-proposal plus FiberControl arm isolates reward adaptation. The route-proposal
plus blind arm isolates proposal support. The combined arm tests complementarity.

This revision authorizes zero oracle calls. It does not claim that the PMO route
export is already an executable proposal policy.

## Fixed prospective envelope

- Tasks: `celecoxib_rediscovery`, `gsk3b`, `perindopril_mpo`.
- Arms: old proposals plus blind selection, route proposals plus blind selection,
  old proposals plus FiberControl, route proposals plus FiberControl.
- Proposed later ceiling: 48 charged calls per task and arm, 576 total, no retry.
- All 16 task-independent initialization scores count inside each 48-call budget.
- PMO rewards are mapped to FiberControl's lower-is-better coordinate by
  `control_score = -reward`; reported curves remain native higher-is-better PMO
  reward.
- Candidate attempts are matched within each proposal-source comparison.
- Route checkpoints may contain no task, family, route, endpoint, SMILES, source
  graph, absolute assignment or executable teacher program.

## Mandatory zero-oracle gate

Before scoring, the actual runtime sampler must generate from each immutable
initial source under equal old/route attempt budgets. Each accepted candidate
must exact replay and be a novel valid molecule. Publish coverage, precision,
unique yield, failures, work and wall time. A leave-one-task-out checkpoint is
required for every scored task. Teacher candidates may not be injected.

The current audit intentionally fails this gate. The sealed PMO route export has
6,143 primitive decision rows, but it explicitly reports no fitted actor, no
module-count targets and no binding-prototype targets. It is training supervision,
not a production proposer. Building a four-arm scored run on top of it now would
mislabel unavailable support as an implemented method.

Run the audit with:

```bash
/Users/rmaganti/compose_rgm_git/.venv/bin/python \
  tools/pmo_route_fiber_prelaunch.py
```

The authoritative output is
`diagnostics/pmo_route_fiber_pilot/prelaunch_v1.json`. The next revision must fit
split-clean production checkpoints and pass the equal-attempt actual-sampler gate.
Only then may a separately locked and explicitly authorized scored pilot run.
