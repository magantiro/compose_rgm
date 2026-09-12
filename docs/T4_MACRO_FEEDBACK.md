# Bounded end-to-end macro feedback development run

Authorized 2026-09-10 after the user requested a full feedback loop rather than
more isolated ring/repair diagnostics. This supersedes only the earlier
four-level zero-oracle probe's search schedule, not its chemistry or model.

Problem: select and chain executable molecular transformations using actual
docking feedback. Output: complete molecules with exact persistent states,
primitive witnesses, region/option ancestry, and oracle receipts. Claim tested:
the existing local/global and macro machinery can support a longer adaptive
optimization episode. This is inspected PARP1 seed0 delta=0.4 development, not a
new benchmark or evidence of superiority to IVG. Frozen executor, <=40 active
atoms, charge-preserving achiral support, R_theta, kappa, WHERE, WHAT, generic
channel and option semantics are unchanged. No winner input or prior tuning.

One episode: ten synchronized feedback rounds, eight proposal parents, two
complete option attempts per parent, and at most four new eligible docking
attempts per round (40 total). The source ledger contains all 67 prior calls.
Initial parents include four original-seed trajectories and the four best
distinct observed archive molecules. Ten rounds allow up to ten consecutive
option decisions; 110 remaining primitives at initialization prevent bookkeeping
from truncating any registered option. A saved descendant keeps its remaining
budget, lineage and cumulative ancestry, never a refreshed primitive horizon.

Reuse the existing one-level `run_search` execution and exact replay. Proposal
workers do not load or use the task guide. At each round, freeze a candidate
pool, fit the unchanged `DockingValue` recipe to all strictly prior labels, and
allocate up to four docking slots: one predicted-quality candidate, two diverse
structural/fingerprint candidates, one uniform exploration candidate. Pending
eligible outputs remain available in subsequent rounds. Global canonical
deduplication excludes every previously attempted oracle identity, including
failures. No surrogate threshold rejects an otherwise eligible candidate.

Docking uses the existing evaluator and unchanged endpoint gate. Intermediate
states need only the executable-product gate. After all batch results are
saved, update the task-value fit and select eight next parents: the original
seed, the best observed endpoint, two existing KL/floor-guided draws, three
diverse representatives and one uniform draw. Diversity first prefers an
unrepresented (cycle rank, ring systems, ring atoms, heavy-atom band) signature,
then Morgan max-min distance. This allocates finite search effort, not support;
no ring direction, type, atom composition or winner structure is forced.
Ineligible states remain eligible as search parents, with the existing soft
QED/SA-only heuristic and no direct intermediate seed-similarity penalty.
There is no learned Q(o), calibrated future-value or exact Doob-law claim.

The guide is a limited heuristic, not a gate or uncertainty estimate. Actual
docking changes archive selection and the next guide; three of four oracle
slots do not select by its predicted mean. Within-round docking order cannot
change that round's locked candidate pool. A useful same-generator post-hoc
baseline remains the prior macro-beam implementation; this single development
episode does not estimate a causal guided-vs-post-hoc improvement.

Compute: one 1-CPU driver plus at most eight 1-CPU, 8-GiB proposal containers.
The generator runtime stays cached in warm containers; each parent unit and
primitive prefix is restartable and published. Reuse prior exact-law caches
only after their existing dependency checks. Historical complete-option work
cost about 22-29 seconds/attempt. 160 attempts imply roughly 1.0-1.3 CPU-hours
plus startup, or around 15-30 minutes elapsed allowing stragglers; this is an
estimate, not a guarantee. Driver ceiling 7200 seconds, worker ceiling 1200
seconds; timeout means incomplete, not a scientific negative. No automatic
oracle retry and no further episode after round ten. No GPU or reference-model
training. Report actual work, failures and incomplete intervals separately.

Progress: driver heartbeat every 30 seconds, per-worker phase/primitive counter,
per-docking score and best-so-far, and per-round options, eligibility, cycle and
ring-system changes, depth, proposal time, label count and selected ancestry.
Every docking has durable started/result receipts; a started row without a
result blocks implicit redocking. Complete scientific units are reused.

Focused acceptance only: feedback timing, exact-state continuation, canonical
deduplication, complete-option reuse, oracle lock/once-only accounting, and the
parallel launch boundary. Existing unrelated suites are not a development gate.
Clean committed source, strict preflight, deploy then durable `t4_launch.py`
spawn remain mandatory. No claim that the whole controller is qualified.

Implementation verification: six focused tests passed in 19.85 seconds (source
ledger, feedback timing/completed-round resume, exact-state budget continuation,
launch bounds, existing primitive replay/resume, existing once-only docking).
No repository-wide regression suite or new molecular capability panel was run.

## Completed run

Started 2026-09-10 from clean commit `4367c0fa75ef` (local branch
`t4-macro-feedback`, worktree `/private/tmp/compose-macro-feedback-dev`). The
main `region-resampling` worktree retains unrelated user-owned changes; the
scientific launch did not serialize those changes.

- Modal call: `fc-01M255Z3HBDG1GNX61GCYPSR1W`
- Volume: `compose-v4-artifacts`
- Run: `t4_macro_feedback/3e8730df1962c80fcc8cfad050010c76a2e533f6191cd19b4bb25206ffc57c11`
- Source best observed score: -9.9; source calls: 67; maximum additional calls: 40.
- Source implementation is committed, unpushed. Completed ten rounds with 27
  new docking calls and no docking failures; best -10.6. No automatic continuation.
- Collected run evidence and retrospective diagnosis:
  `diagnostics/t4_macro_feedback/README.md`. This is warm-start development,
  not a fresh benchmark or an IVG-level result.

Read progress without launching anything:

```sh
.venv/bin/python tools/t4_feedback_status.py \
  diagnostics/t4_macro_feedback/3e8730df1962c80fcc8cfad050010c76a2e533f6191cd19b4bb25206ffc57c11/spawn.json
```

Add `--collect` to retain completed round summaries and their physical hashes;
add `--details` to also collect saved proposal and task-value ledgers.
Detailed primitive witnesses, law caches, proposal attempts, locks, and docking
receipts remain on the volume. Do not relaunch to query status.
