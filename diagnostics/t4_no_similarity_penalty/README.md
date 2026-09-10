# No intermediate similarity penalty

Status: completed. Removing the intermediate similarity penalty changed the
sampled paths but did not produce an eligible new endpoint in this bounded run.
There were no new docking calls and no observed docking improvement.

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

## Result

Authoritative computation: `summary.json`, produced by `fetch.py` from the
locked volume artifacts, with physical input hashes, reused-control checks,
producer identity and software versions. Generation completed at
2026-09-10T05:22:10.851210+00:00. All 30 attempted options completed and were
replay-verified; all 30 output molecules were canonically distinct.

| Intermediate retention | Completed | New eligible | Ineligible-to-eligible transitions | Median seed similarity |
| --- | ---: | ---: | ---: | ---: |
| Post-hoc control, reused | 30 | 0 | 0 | 0.271 |
| Terminal-only guide, reused | 30 | 0 | 0 | 0.271 |
| Graded QED/SA/similarity, reused | 30 | 0 | 0 | 0.286 |
| Graded QED/SA, no similarity penalty | 30 | 0 | 0 | 0.300 |

The new arm generated 15 molecules absent from the prior graded arm and shared
15 with it. Retention probabilities were nonuniform at all four levels, with
KL values 0.482, 0.787, 0.816 and 0.772. Thus the ablation was active, not a
no-op. It completed 18 options from ineligible parents without recovering an
eligible endpoint. Of 30 outputs, 29 failed similarity, 11 failed QED and 14
failed SA; these counts overlap. Final constraints were not weakened.

Five parameterized construction programs completed: two fused five-membered
aromatic additions and three pendant five-membered additions (two aromatic,
one nonaromatic). Each increased cycle rank by one relative to its immediate
parent. None was eligible. Across all options, six increased cycle rank by one,
five decreased it by one and 19 left it unchanged. Relative to the fixed search
root, cycle-rank deltas were -2:2, -1:6, 0:17, +1:5; ring-system deltas were
-1:2, 0:21, +1:7. Ring-system count is not cycle rank, and opening a fused
system can increase the former without constructing a new cycle.

Intended released fraction ranged 0.032..0.828 (median 0.172). Realized largest
connected changed fraction per completed option ranged 0.032..0.207 (median
0.069); cumulative change from the fixed root reached 0.310. These are different
quantities, not evidence that a large selected region was fully rewritten.

The same near-feasible molecule seen in the prior diagnostic was generated:
similarity 0.3970588, QED 0.71346, SA 3.37046. It was not retained or continued.
The earlier `diagnostics/t4_repair_neighbors/summary.json` records 11 eligible
one-edit successors from this state (10 absent from the archive and one return
to the incumbent). Availability of those repairs does not mean this controller
discovered them, nor that they improve docking. The present best predicted
score was -9.04662, worse than the incumbent's prediction -9.14809; neither is a
new docking observation. No candidate is authorized for docking by this result.

Interpretation: removing the penalty alone did not solve eligibility recovery
on one inspected development parent and one random seed. It does not establish
that the penalty is necessary or that recovery-aware lookahead will succeed.
A next diagnostic could give promising ineligible states a short, task-scored
recovery continuation before discarding them. That is a proposed follow-up,
not part of this completed run or an already validated remedy.

## Restart and verification

The heartbeat counter reset during execution. The captured app log records a
preemption at 01:15:48-04:00, consistent with the observed resume. Completed
option records were reused, not relaunched as a second experiment. The app log
is corroboration rather than a per-call infrastructure trace; see
`attempt_1/lifecycle.json`.

The result's 220.6 proposal seconds and 372.1 elapsed seconds cover only the
final resumed invocation. Its 9 fresh laws, 8 prior-cache hits, 2 saved hits,
397 memory hits and 1,020 executor calls also exclude interrupted work. The
whole-job compute total is incomplete; do not compare these counters with the
control totals to claim acceleration. The chronological run took longer.

The completed-artifact audit passed, including exact roots and first proposals,
frozen model/input hashes, endpoint eligibility, score isolation, canonical
deduplication, option completion/replay records, and retention probability/KL
checks. Two focused ablation tests passed again in 9.61 seconds; collector Ruff
lint/format and diff checks passed. No full-suite or milestone claim is made.
All null outcomes are retained. No follow-up experiment was launched.
