# PMO Complete-Program Decoder Gate

## Status

This milestone implements a CPU-only, zero-oracle complete-program decoder. It
does not authorize a scored PMO run. The legal WHERE/HOW result was rerun
unchanged from clean committed revision `61f816b`, preserved as `attempt_2`, and
its runtime checkpoint is now pinned as the authoritative decoder input.
Legal-action `attempt_1` remains a non-authoritative implementation preview.

Generation is partitioned into the nine frozen teacher-free source/fold cases.
Each case publishes live operational progress, one immutable candidate shard
and a separate compute receipt. At most nine single-CPU Modal workers run with
zero automatic retries. Completed cases are independently reusable, while a
failed or interrupted case can be recomputed only under the same sealed task
identity. The final lock is reduced in frozen source-manifest order only after
the full shard census and all input identities validate.

The first deployment preflight at revision `0bb4bb5` failed before app creation
because NumPy 2.5.3 has no Python 3.11 wheel. It launched no worker and produced
no remote PMO artifact. The corrected runtime uses the repository's established
Python 3.11 Modal base with its pinned NumPy 1.26.4, SciPy 1.13.1 and RDKit
2024.03.5 environment. The failure is preserved in
`diagnostics/pmo_complete_program_decoder/implementation/deploy_failure_0.json`.

## Scientific question

The earlier held-task-family action gate evaluated WHERE/HOW only after the
teacher rule was supplied. This gate asks whether that ranking signal improves
autonomous complete-route recovery when rule and stopping choices are also made
without teachers.

Both arms use the same fold-specific `balanced_generic_marginal` distributions
for `primitive_rule` and `program_control`:

1. `uniform_where_how` ranks canonical successors uniformly within each exact
   legal Active8 rule fiber.
2. `learned_where_how` ranks the same canonical successors with the fold-matched
   legal-action checkpoint.

The comparison therefore isolates learned WHERE/HOW. It does not select the
flat or hierarchical sequence policies after observing their held-fold results.

## Generation contract

Generation operates on nine unique `(source graph, held fold)` cases derived
from the frozen 18 lineage panels. The source manifest contains no task family,
lineage identity, teacher action, route, endpoint, or score.

For every live prefix the decoder:

1. enumerates and exact-executes every supported canonical Active8 rule fiber;
2. retains at most 32 successors per rule;
3. combines the marginal rule probability with the uniform or learned
   within-fiber probability;
4. applies the shared marginal stop/continue probability;
5. retains beams of width 1 or 8;
6. locks ranked route and endpoint candidates at depths 8, 16, 24, and 32.

Programs are force-terminated at 32 primitives. A completed program abstains if
it exceeds eight dependency components or 40 active atoms. Candidate route
identity is the canonical rule sequence plus canonical molecular-state
sequence, not raw slot-address payloads.

The candidate lock is deterministic and contains zero teachers, task identity,
task scores, oracle calls, and timestamps. Runtime and memory measurements are
written to separate shard and aggregate receipts so they cannot change candidate
identity. Authoritative generation also refuses to launch unless the
implementation, contract, documentation, source manifest and focused test files
are tracked, committed, and unchanged. Their physical hashes and the Git
revision are recorded in the lock.

## Evaluation contract

Evaluation is a separate process. It validates the candidate lock and its
generation receipt before loading either teacher artifact. It reports the
following at output cutoffs 1, 5, 10, and 32:

- exact canonical route recall;
- exact endpoint recall and reciprocal rank;
- radius-2 transformation recall and reciprocal rank;
- exact endpoint and radius-2 emitted precision;
- source coverage and unique endpoint yield;
- proposal, fiber, execution, component-abstention, time, and memory telemetry.

All 18 lineages and 184 routes are reported. The eight lineages and 106 routes
inside the declared 32-primitive/eight-component runtime support are evaluated
again as a separate primary scope. Task families are balanced before aggregate
metrics are averaged.

At beam width 8 and depth 32, the preregistered scientific gate requires learned
WHERE/HOW to have nonzero radius-2 recovery, strictly higher radius-2 recall and
reciprocal rank than uniform, no lower exact-endpoint recall, and strictly
higher radius-2 recall in at least two of three held folds. Exact route recovery
is reported separately. Even a passing result cannot authorize a scored PMO
pilot.

## Commands

The workflow is intentionally split:

```bash
PYTHONPATH=src python tools/pmo_complete_program_decoder.py prepare-sources
modal deploy modal_apps/pmo_complete_program_decoder_app.py
PYTHONPATH=src python tools/pmo_complete_program_decoder_parallel.py launch
PYTHONPATH=src python tools/pmo_complete_program_decoder_parallel.py relaunch --case-index N
PYTHONPATH=src python tools/pmo_complete_program_decoder_parallel.py status
PYTHONPATH=src python tools/pmo_complete_program_decoder_parallel.py collect
PYTHONPATH=src python tools/pmo_complete_program_decoder_parallel.py collate
PYTHONPATH=src python tools/pmo_complete_program_decoder_parallel.py evaluate
PYTHONPATH=src python tools/pmo_complete_program_decoder_parallel.py final-status
PYTHONPATH=src python tools/pmo_complete_program_decoder_parallel.py download
```

The legacy monolithic `generate` command fails closed under the authoritative
durable-shard contract. `prepare-sources` remains local and zero-oracle. The
Modal evaluator cannot start until all nine source shards and the independent
collation call are complete. `collect` downloads only immutable completed shards
and preserved failure records, so it is safe to run while other cases remain live.
`relaunch` accepts only a case with a locally sealed failure record, reuses the
exact original task identity and allows at most one explicit manual relaunch.

## Claim boundary

A passing gate would establish autonomous recovery under the declared generic
operator, horizon, component, source-corpus, and held-task-family support. It
would not establish PMO score improvement, universal route expressibility, or
authorization to call a PMO oracle.
