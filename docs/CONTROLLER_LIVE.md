# COMPOSE controller: current decision record

Updated 2026-09-11. Active goal: reproducible improvement over the actual T4 and
PMO champions, then verified matched external comparisons. Supporting chemistry,
known-winner reconstruction and mathematics are not benchmark success.

## Champion and exact regime

- PMO perindopril, exposed prescreened development: new best **0.672021505032247**,
  versus **0.6580812002646271** in the latest fixed-memory control, starting
  identically from 0.649519052838329. Six rounds, 109 new calls, 619.577 seconds.
  Top-ten means 0.6502614449 versus 0.6482515168. First-run criterion passed;
  replication required. The best sequence was hydroxyl addition then generic
  ring opening, not a newly selected donor. All six completed new-donor uses
  were worse than their parents. Do not claim a demonstrated transfer mechanism.
  Audit: `diagnostics/pmo_evolving_memory/report.json` and README.
- Previous component batch best **0.649519052838329**,
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
- Historical labels: 249455 prescreen evaluations plus **841 development physical
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

## Previous experiments and decisions

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
completion claimed. Executed local source is hash-bound; implementation and
reports were subsequently preserved in commit f47e852.

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

Completed PMO call: `fc-01M291A3D6D26JKTAH17S4CCPB`, run
`e111b2aac585acaf2c5c8d6338599f6a53856db9e3c5a7132b9cd77e53292189`.
Deployed clean source `eda6e8963d7f`, strict preflight passed. Deployment 89.413s.
Durable prefix `compose-v4-artifacts/pmo_evolving_memory/<run>`; local spawn
`/private/tmp/compose-pmo-evolving-memory-runs/<run>/spawn.json`.
Goal active and unmet. No job currently running. Preparing the earned fresh-seed
replication with identical starts/memory, seed 20261003. No concurrent T4 job.

Current hypothesis: fixed donor memory cannot directly reuse newly evolved
components across branches. Compare evolving proposal memory to frozen memory,
with identical starts, actual-score parent allocation, and broad reference
editing in both. Only an arm's own requested discoveries update its memory,
at round boundaries. Implemented and launched under
`docs/PMO_EVOLVING_MEMORY.md` and `configs/pmo_evolving_memory.json`.
Same top 16 exact parents, 116 common initial donors, six rounds, at most 192 new
calls. Both use 50% broad reference proposals and 50% rank/uniform donor programs;
only donor-memory updates differ. At most 29 workers plus one driver, $10 cap.
Ten focused memory/worker/driver tests passed in 4.10 seconds; lint passed.
The first run exceeded initial 0.649519 and concurrent fixed-memory best.
Replication contract 4af1aa1662cd04165e5463ada7eddc017345f53e3d456cf43e5a579b7b0ec3a9
uses seed 20261003. Exact initial parent/donor equality and the 109 cache-only
labels were verified; three memory/worker/round-lock tests passed in 2.37 seconds,
with touched-code formatting and lint passing. A repeated lead supports the
comparison; null or reversal leaves the first gain seed-dependent.

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
