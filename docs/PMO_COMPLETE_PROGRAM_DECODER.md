# PMO Complete-Program Decoder Gate

## Status

This milestone implements a CPU-only, zero-oracle complete-program decoder. It
does not authorize a scored PMO run. Authoritative execution is intentionally
disabled in `configs/pmo_complete_program_decoder_v1.json` until the upstream
legal WHERE/HOW result is rerun unchanged from clean committed source and its
runtime checkpoint is pinned in this contract. The current legal-action
`attempt_1` artifact is non-authoritative and is permitted only for focused
implementation fixtures.

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
written to a separate receipt so they cannot change candidate identity.
Authoritative generation also refuses to run unless the implementation,
contract, documentation, and focused test files are tracked, committed, and
unchanged. Their physical hashes and the Git revision are recorded in the lock.

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

After the authoritative legal-action input is pinned and the contract is
rehash-sealed, the workflow is intentionally split:

```bash
PYTHONPATH=src python tools/pmo_complete_program_decoder.py prepare-sources
PYTHONPATH=src python tools/pmo_complete_program_decoder.py generate
PYTHONPATH=src python tools/pmo_complete_program_decoder.py evaluate
```

In the current implementation-only state, `generate` and `evaluate` fail
closed. `prepare-sources` is safe but is not required for focused unit tests.

## Claim boundary

A passing gate would establish autonomous recovery under the declared generic
operator, horizon, component, source-corpus, and held-task-family support. It
would not establish PMO score improvement, universal route expressibility, or
authorization to call a PMO oracle.
