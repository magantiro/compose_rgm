# Direct docking of the latest saved proposals

2026-09-10. The user approved obtaining real docking feedback on the latest
proposals after the frozen surrogate ranked the known IVG endpoint below the
incumbent. This is one saved-batch development diagnostic, not another molecular
search, training run, or automatic optimizer round.

Dock all 16 canonically distinct eligible products absent from the frozen
51-call archive in the complete nine-root recovery census. No surrogate ranking,
winner similarity, new chemistry filter, backfill, or winner-informed proposal
is used to select this set. These are enumerated one-edit repairs of blind
search outputs, not 16 autonomous selected offspring. Preserve that distinction.
The previously tested retention policies selected only one of these endpoints.

The platform remains the frozen executable molecular process over connected,
charge-preserving, achiral graphs with at most 40 active atoms. Bind the exact
saved states, production primitive witnesses, all source locks and prior archive
by hash. Re-evaluate QED >= 0.6, SA <= 4, original-seed similarity >= 0.4 and the
unchanged medicinal-chemistry screen before freezing the complete docking lock.
Keep every source record in its immutable input artifact. A gate or identity
disagreement fails rather than silently changing the selected batch.

Use the existing OpenBabel/QuickVina2 evaluator, PARP1 receptor and box, cpu=1,
exhaustiveness=1 and 10 modes. Preserve the inherited unseeded protocol; record
OpenBabel version and executable/receptor hashes. Each candidate is attempted
once, including failures. Started-without-result rows forbid implicit retries.
Maximum new calls: 16; prior source calls: 51. A completed batch gives 67 calls
in this diagnostic lineage, not a completed adaptive optimizer campaign.

One CPU container, 2 GiB, no GPU or R_theta initialization, zero retries,
1200-second timeout. Historical 20-molecule batches took 47-50 seconds docking;
these larger products may differ. Expect roughly a minute of docking plus image
startup/deployment. The reservation envelope is 1/3 CPU-hour and 2/3 GiB-hour;
this is not an invoice or wall-time guarantee. Each result and the running best
are published immediately; heartbeat interval 30 seconds. Existing complete rows
are reused, not redocked. No other jobs are stopped or changed.

Report observed best score, all failures, surrogate ranking agreement, diversity,
and structural changes against the actual parents and original seed. Compare
SMILES as molecular graphs, not character edit distance; use the known winner
only for an explicitly answer-known structural diagnosis after candidate locking.
No winner enters generation, selection, fitting, or the discovered best score.
No assumption that a missing ring implies improved docking.

Before launch: focused lock/accounting/retry/resource tests, pinned chemistry,
clean committed source and strict preflight. Deploy the T4 app, then use
`python3 tools/t4_launch.py --proposal-docking`. No full-suite milestone claim.

## Launch status, 2026-09-10

Implementation is committed at `2f02ba9` and `af28010`. On the clean
`af28010` worktree, the three proposal-batch tests and two existing durable
docking/retry regression tests passed (5 tests, 4.23 seconds). Strict preflight
reported zero mounted drift; `git diff --check` passed. No prior
`t4_proposal_docking` namespace was found on the Modal volume.

The permission reviewer rejected `modal deploy` because it uploads application
source to the external Modal service. No deployment, spawn or new docking call
occurred. Explicit approval for that source upload is required before retrying;
do not bypass the rejection through another launcher. The authorized batch
remains the same 16 identities. No surrogate or controller change has been made.

The user subsequently approved the source upload and fixed batch. Deployment
and spawn from the already-tested clean `af28010` commit then succeeded. All 16
attempts completed with no failures in 113.866 container-seconds; observed best
was -9.9 versus inherited -9.7. See
`diagnostics/t4_proposal_docking/README.md` and its hashed raw receipts and audit.
The earlier permission block is resolved. No additional run was launched.
