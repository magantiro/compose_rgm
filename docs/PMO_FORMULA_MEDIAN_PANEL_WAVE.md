# PMO formula/median panel wave

## Problem and output

This bounded wave applies the successful panel-informed executable-program
recipe to two structure-defined PMO objectives: `isomers_c7h8n2o2` and
`median1`. The output is ten exact-replayed COMPOSE programs per task and a
complete query ledger for their endpoints plus one predeclared task-specific
generic neutral root per task.

This is answer-known white-box development. The public task formula and public
IVG top-molecule panels influence the locked candidate structures. It is not
held-out discovery or autonomous search.

## Frozen procedure

1. Deterministically encode the two predeclared generic neutral roots into 48
   persistent slots; hash-verify the upstream revision, panel identities, and
   comparator CSV identities.
2. Canonicalize all candidate structures. Require neutral, at-most-40-atom,
   canonical-unique support. For the isomer task, independently require RDKit's
   molecular formula to equal `C7H8N2O2` for every candidate.
3. Compile the first candidate from its task root. Compile each subsequent candidate
   as an anchor-plus-suffix program, with one recorded direct-root fallback only
   if the bounded suffix route is unavailable.
4. Exact-replay all 20 endpoints before scoring. Abort rather than drop or
   replace a failed program.
5. Score root and ten endpoints once per task with pinned PyTDC 0.3.6. Never
   retry an ambiguous started query.
6. Compute official top-ten AUC at budget 10,000, frequency 100 and
   `finish=True`; compare exact margins against both IVG regimes.

The ceiling is 22 new local CPU oracle calls. Support remains neutral
charge-preserving graphs with 40 active atoms in 48 persistent slots, no
stereochemical claim, at most 96 primitives and at most two program blocks.

## Prelock failure and corrective decision

The first zero-oracle structural attempt reused the previously charged unrelated
root. The C7 task's first target remained unresolved after the bounded element
and topology searches; the best partial state retained a sulfur atom from the
much larger source. No task score was requested. The exact diagnostic is stored
in `diagnostics/pmo_formula_median_panel_wave/prelock_failure_0001.json`.

Because neither the contract nor a curriculum had been committed or sealed and
zero oracle calls had occurred, the root choice was corrected before the frozen
candidate lock. Each task now uses a predeclared generic neutral root of similar
scale and topology. The root is still a charged query and is not selected after
observing its task score. This changes the structural starting context, not the
candidate panel, score labels, comparator, metric, support, or call ceiling.

After the first curriculum lock, the scoring entry point correctly stopped
before the first query because it called an adapter factory specialized for QED
and frozen-forest objectives. That factory could not dispatch these general
PyTDC tasks. This zero-query implementation defect is stored in
`diagnostics/pmo_formula_median_panel_wave/prequery_failure_0002.json`. The
scoring adapter was replaced with the existing general pinned-PyTDC dispatcher,
and the first curriculum remains immutable but invalidated for scoring. A new
curriculum must be sealed under the repaired implementation before any query.

## Baselines and interpretation

The external baselines are the official three-run mean IVG no-prescreen and
prescreen AUC values pinned in the contract. A task wins only if its measured
AUC is strictly greater. Negative margins, parse failures, route failures and
oracle failures remain reportable outcomes.

Positive results would show that COMPOSE can exactly execute a richer,
task-appropriate structural curriculum under this disclosed development
regime. They would not show that the structures were independently discovered
or that a general controller would recover them without formula/panel access.

## Acceptance

- Both tasks and all upstream/oracle identities validate.
- All 20 canonical-unique supported programs exact-replay from their frozen task roots.
- The curriculum is sealed from clean committed code before scoring.
- Exactly 22 query receipts complete without retry or replacement.
- Results preserve molecules, evidence roles, metrics, timings, hashes and
  exact margins.
