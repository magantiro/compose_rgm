# Local-guidance replication: no best-score improvement

The unchanged fresh-seed comparison is complete. Both arms remain at their
initial best **0.6747477697966318**. The first run's guided advantage did not
repeat. Stop the unchanged mixture; preserve its observed 0.6835298931 molecule,
but do not promote the local-guided controller as a replicated improvement.

| Quantity | Broad baseline | Local-guided mixture |
| --- | ---: | ---: |
| Best | 0.6747477698 | 0.6747477698 |
| Archive top-ten mean | 0.6724616451 | 0.6728082152 |
| Completed / attempted options | 47 / 64 | 47 / 64 |
| Unique generated candidates | 47 | 47 |
| Mean pairwise Morgan distance | 0.5753731474 | 0.5288093382 |

The slight guided top-ten gain is secondary and does not meet the declared
best-score criterion. Guided diversity is lower. This is one exposed warm
replication, not a matched external PMO AUC result or a rejection of all
task-guided editing.

Same sixteen original parents, 116 donors, fixed 1044-label model and four-round
recipe; seed 20261009. The original proposal exclusions are unchanged. The 85
previously paid labels were available only on requested queries, not as initial
parents or added proposal exclusions. No model training, executor change or
public endpoint injection occurred.

Cost: **76 new physical queries**, 403.889795 seconds driver wall,
360.656347 proposal seconds, 106 distinct workers, 1603.020434 summed worker
seconds and 285.262764 summed neural-law seconds. Recorded oracle-function
time is 0.097181 seconds, excluding orchestration. The guided channel predicted
31810 pooled candidates across 22 local draws. Historical accounting becomes
**2217 development physical calls**, plus the separately reported 249455
prescreen calls. The reserved cap was $10 and 30 CPU containers, no GPU;
actual billing has not been reconciled independently.

`report.json` binds all material result/source/receipt inputs, full configuration,
software and executed source. The audit reproduced archive allocations and
proposal identities with zero numerical replay error, checked candidate scores,
donor memory and query accounting. Fourteen focused checks passed before launch;
the existing audited reporting command completed without new model or oracle
calls. No full-suite or overall-milestone completion is claimed.

Executed commit: `c7818ce74c2d28ebc79c426ebd93662e84373bf0`.
Contract: `8c5554584534f3d535edd870c5fa7ce489625ae848b586ed706151b8cfede439`.
Run: `83d84e96ce3d961d09816b3a7829103a1b608380147416c6b0996e9643806b77`.
Durable call `fc-01M29BN6BVQQF5PDR2A9PNKKP7` returned complete.
Raw receipts: `/private/tmp/compose-pmo-local-guidance-replication-runs/<run>/`.
Volume: `compose-v4-artifacts`, prefix `pmo_local_guidance_replication/<run>`.
The exact source snapshot and spawn receipt were uploaded into this namespace.
