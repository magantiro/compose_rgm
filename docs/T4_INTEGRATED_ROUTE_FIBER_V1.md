# Integrated route-distilled FiberControl, JAK2 delta 0.6

## Frozen question

Can one unchanged autonomous controller combine shallow COMPOSE, generic
anchored replacement, and a leave-JAK2-out route-distilled complete-region
expert, then use only current-cell docking observations to allocate its calls
and improve the shared archive?

The generated object is one complete executable COMPOSE program. Shallow and
anchored programs contain at most three protected modules. A route proposal
contains one to four complete address-free dependency-region decisions and at
most 32 realized primitives. The route checkpoint stores structural templates
and a balanced marginal fitted after excluding all JAK2 routes. It stores no
source graph, endpoint, route identifier, target name, or executable teacher
program.

## Prospective protocol

The same controller runs on `jak2_0`, `jak2_1`, and `jak2_2` at delta 0.6.
Each cell has 49 charged calls: one root and six batches of at most eight.
Every round expands four measured parents with all three experts in independent
single-CPU workers, deduplicates canonical endpoints, and locks the full pool
plus the selected queries before docking. The first two rounds select at most
one endpoint from every available expert to seed reward learning. Thereafter
the shared `ProgramValue` model selects six calls and the remaining two are
random exploration. No expert retains a permanent quota.

Every lock and completed round is committed to the persistent Modal volume.
Failures are charged and are neither retried nor replaced. The app logs the
best score, round best, eligible and selected endpoint census by expert,
selection-source census, and timing after every batch.

An interrupted cell fails closed if a query lock exists without a terminal
result. It does not automatically resume or redock. Manual reconciliation of
the charged-call ledger is required before any separately authorized recovery.

## Evidence and claim boundary

This is winner-informed development. It is not held-out evidence. Its route
prior excludes JAK2, but the anchored operation was motivated by answer-known
JAK2 analysis. Published IVG results and historical COMPOSE outcomes are
read-only offline comparators and never enter runtime selection.

The run tests the integrated performance hypothesis. It does not isolate the
causal contribution of FiberControl relative to post-hoc selection on the same
new generator. A matched causal ablation is required before making that claim.

## Commands

```bash
modal run modal_apps/t4_integrated_route_fiber_app.py --mode launch
modal run modal_apps/t4_integrated_route_fiber_app.py --mode status --run-id RUN_ID
```
