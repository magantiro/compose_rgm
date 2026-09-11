# COMPOSE controller: current decision record

Updated 2026-09-11. Active goal: reproducible improvement over the actual T4 and
PMO champions, then verified matched external comparisons. Supporting chemistry,
known-winner reconstruction and mathematics are not benchmark success.

## Champion and exact regime

- PMO perindopril, exposed prescreened development: new best **0.649519052838329**,
  previously 0.6030226891555273. Uniform donor proposals found it in the latest
  locked batch; contextual replay reached 0.6365315497232666. This is a warm
  proposal-component result, not replicated full-controller improvement or
  official PMO top-ten AUC. The new molecule uses a 39-primitive scaffold
  replacement, +8 heavy atoms and +1 cycle rank. Exact graph, SMILES, operands,
  costs and receipt audit: `diagnostics/pmo_edit_replay/report.json`.
- Strongest matched full-controller PMO evidence: broad-reference/donor mixture
  reached 0.6030226892 versus 0.5595028849 for the unchanged reference baseline.
  A fresh seed reached 0.5869734365 versus the same baseline. Identical original
  starts and donors, six option rounds, 156 and 149 new calls. First gain:
  four-edit side-chain replacement; replication: 47-edit structural replacement.
  `diagnostics/pmo_donor_comparison/README.md` and its two reports.
- Historical labels: 249455 prescreen evaluations plus **732 development physical
  calls** through the latest batch. The bank is our recovered historical asset,
  not external labels. Its legacy provenance remains incomplete. Do not call
  this no-prescreen. The separate no-new-prescreen controller remained at
  0.5222329679 (`diagnostics/pmo_learned_proposal/report.json`).
- T4: preserve both historical controller lineages. The workshop table selects
  compact B in 11 cells and an older 200-call controller in 15, with four failures.
  It is not one frozen 500-call controller or matched mean-of-three result.
  `diagnostics/controller_baselines/t4_lineage.json` and
  `diagnostics/t4_combined_table.json` document this. PARP1 s0 d=0.4 B scores
  -10.9/-10.6/-10.7 at 100 calls each. The broader warm run reached -11.0 at
  40 new plus 94 historical attempts, without replicated superiority. Incumbent
  redocks: -9.8 in one diagnostic, -11.8/-12.3 in the latest. Docking noise and
  the answer-known -13.6 reconstruction remain separate evidence.

## Latest experiment and decision

`docs/PMO_CONTEXTUAL_EDIT_REPLAY.md`: nonparametric complete-edit proposal from
33 positive historical donor edges spanning 29 parents. Same 16 exact current
parents, original 100 donors and 64 attempts per arm. Replay retains 20% uniform
exploration. No endpoint predictor, public winner, or reference retraining.

| Metric | Uniform | Replay |
|---|---:|---:|
| Best | 0.649519 | 0.636532 |
| Archive top-ten mean | 0.597691 | 0.605168 |
| Unique improving offspring | 3 | 9 |
| Completed / distinct molecules | 40 / 40 | 46 / 29 |
| Proposal seconds | 20.358 | 26.104 |

63 new calls, 50.413 seconds total, 0.024 actual oracle seconds. Local CPU,
no Modal worker or GPU. All unique compiled endpoints scored after locking.
Audit verifies sealed receipts, locks, exact graph identities, arm membership,
charged query union and reproduced best/top-ten values. Three focused tests
passed in 2.71 seconds; touched-code lint passed. No full suite or milestone
completion claimed. Executed local source is hash-bound, uncommitted.

**Primary replay-vs-control criterion failed.** The secondary top-ten gain does
not authorize promoting or replicating this unchanged recipe. Retain the useful
new states. Raw result: `/private/tmp/compose-pmo-edit-replay-20260911a/result.json`.
Source and receipts: content-addressed volume prefix in
`diagnostics/pmo_edit_replay/README.md`.

Stopped earlier comparisons: archive parent selection retained quality but no
arm exceeded 0.603023 (89 calls, 349.619 seconds). Refit endpoint preferences
avoided damage but selected no improving offspring (30 calls, 46.672 seconds).
A separate 51-query audit scored that entire frozen pool: best 0.593296, so no
selector could recover a new champion from it. This is a pool-specific result,
not a global ceiling on uniform proposals. See `diagnostics/pmo_archive_branching`
and `diagnostics/pmo_program_choice`.

T4 donor transfer stopped: 31/32 programs completed; 22 endpoints failed benchmark
gates and seven were known. Two novel candidates docked -7.5/-7.6. Four repeat
controls bring cost to six calls, 114.772 seconds. Useful feasible coverage failed,
not primarily compilation. `diagnostics/t4_donor_probe/report.json`.

## Next decision and constraints

No job is running. Goal is active and unmet.

Next changed hypothesis: fixed donor memory cannot directly reuse newly evolved
components across branches. Compare evolving proposal memory to frozen memory,
with identical starts, actual-score parent allocation, and broad reference
editing in both. Only an arm's own requested discoveries update its memory,
at round boundaries. Implemented and preparing to launch under
`docs/PMO_EVOLVING_MEMORY.md` and `configs/pmo_evolving_memory.json`.
Same top 16 exact parents, 116 common initial donors, six rounds, at most 192 new
calls. Both use 50% broad reference proposals and 50% rank/uniform donor programs;
only donor-memory updates differ. At most 29 workers plus one driver, $10 cap.
Ten focused memory/worker/driver tests passed in 4.10 seconds; lint passed.
Primary pass requires exceeding initial 0.649519 and concurrent fixed-memory
best. A null stops the unchanged recipe; a pass earns one fresh-seed replication.

Preserve generic/reference chemistry and local/global editing. Donor programs
use the unchanged valid-state executor but need not have positive probability
in the frozen neural tables: no original-R_theta Doob, kappa=1 or canonical-law
invariance claim applies to that channel. No endpoint gates on internal states.

The earlier future-value head used witnessed-path maxima and mismatched
primitive/option horizons and remains rejected
(`diagnostics/pmo_option_particles/GUIDE_AUDIT.md`). A replacement must estimate
actual continuation value under its stated policy/horizon. An endpoint predictor
or compiled program is not future value.

Match task, oracle, initialization, all history/training labels, budgets and
replicates for external comparisons. Known-winner development is not blind
discovery. At most 30 Modal containers within declared caps; scientific launches
use clean committed source, deploy and durable spawn. Reuse paid trajectories;
focused checks, no repeated full suites, unrelated cleanup or excessive commits.
