# Continuation implementation gate

This directory records a local engineering result, not a learned-generator or
docking benchmark. Governing prospective specification:
`docs/CONTINUATION_CONTROLLER_IMPLEMENTATION.md`.

## What is implemented

- A remaining-budget-indexed exact reference backup with explicit compute limits.
- The qualified kappa=1 tilt, its exploration mixture, and actual mixed-law KL.
- Separate no-support, zero-terminal-mass, and budget-abstention statuses. A
  partial tree never supplies a guidance ranking.
- An option-conditioned production-executor adapter with complete persistent
  slot, charge, hydrogen, context, option-phase and lineage keys.
- One fixed-bundle trajectory sampler with per-step probabilities, executable
  action receipts, and cost counters. Counterfactual branches are not emitted
  as sampled offspring. Q(M) and Q(o) are supplied by the caller, not redesigned.

The exact solver is a bounded specification/reference implementation. It is not
a scalable MCTS system. The existing audited T4 population app was not changed
or redeployed. No docking-specific continuation estimator was trained or
qualified. The runtime adapter delegates to the production learned-law evaluator,
but the numerical gate below deliberately does not load a learned checkpoint.

## Computed engineering result

Run `PYTHONPATH=src .venv/bin/python tools/continuation_gate.py`.

`engineering_gate.json` binds source/code hashes, the explicit reference law,
configuration, exact initial persistent-slot state, and complete sampled action
trace. One hand-declared two-step fixture branches from an acyclic molecule:
both first-step products have zero immediate ring reward; one can close a ring
at the next step. The terminal predicate is cycle rank >= 1, not similarity to
an answer molecule.

On this explicitly synthetic reference, the exact success probability changes
from 0.50 under the reference to 0.95 under continuation guidance with 10%
reference exploration. The mixed decision KL is 0.4946319372 nats. Four executor
applications establish the two branches and their endpoints. The sampled
two-step path is complete and executable. These are mechanism/implementation
checks on one deliberately constructed case, not empirical molecular discovery
rates, coverage estimates, or evidence of beating an optimizer.

The separate two-state arithmetic fixture gives the same continuation value
0.709096 with and without caching. Memoization reduces reference expansions
from 31 to 2 and terminal evaluations from 32 to 2 on this graph. This is not a
production speedup claim; real state reuse and enumeration costs remain to be
measured under the frozen model.

## Verification status

Focused command:

```sh
.venv/bin/python -m pytest -q tests/test_continuation.py tests/test_option_continuation.py tests/test_option_selector.py tests/test_region_rewrite.py
```

All 92 tests passed. The new tests cover independent path summation, terminal
tilting, delayed credit, remaining-budget dependence, alias refinement, cache
equivalence, explicit failures, malformed inputs, production-evaluator
delegation, and real-executor integration. Focused Ruff lint and formatting
checks passed. The full suite ran once on the frozen implementation candidate:
4,348 passed, 46 failed, 58 errored, two skipped and one expected failure, in
1,883.74 seconds. All 29 new tests and all 92 focused tests also passed in this
full run. Failures occurred outside the changed test files, including catalog
identity, artifact-provenance and slot-safety checks. This observation does not
prove that every failure is pre-existing. No unrelated repair or gate relaxation
was made. Repository-wide verification remains non-green, so this is not a
completed scientific milestone or deployment authorization.

`full_suite.xml` preserves the raw report; `verification.json` records every
failure, the base revision, and the exact tested patch hash. The implementation
was committed as `644a4bf2648b`; the engineering receipt was rerun at that commit
and has SHA-256
`6d8fc20f483a3f81138b17d077ebc557b572a77eaf8bc83358e94973e355687b`.
Final focused Ruff lint, formatting, and `python3 tools/preflight.py --strict`
passed. The source patch passed `git diff --check`. The raw pytest XML contains
trailing whitespace in captured tracebacks and is preserved byte-for-byte;
the final staged whitespace check passes with only that raw artifact excluded.
Preflight established zero mounted
source/configuration drift; it does not supersede scientific verification.

## Reusable remote inputs

On 2026-09-07, read-only Modal inspection located the existing model inputs on
`compose-v4-artifacts`. `model_RUN_PATHS.json` was downloaded from volume path
`/editing_v2/r_theta_run/run_inputs/RUN_PATHS.json`. Its SHA-256 is
`585286e0fb6dd8fb6b5a050dc3e5f8d1c7e782b105df114cbcb14cabbb5b2bc2`,
identical to the earlier audited T4 receipt. This manifest binds the source,
Gate 0 and materialized scorer locations; no corpus or checkpoint was recreated.
The `/artifacts` prefix in its paths is the container mount point, not part of
the volume-relative path used by the CLI. Origin/access basis: existing internal
COMPOSE run artifacts in the user's configured workspace. This read-only check
does not independently rehash every referenced asset or authorize training.

## Cheapest informative next experiment

1. Profile one reusable bundle with the actual frozen learned law and an explicit
   executor-application ceiling, with no docking. Report completed versus
   budget-abstaining decisions; do not price T4 from this tiny fixture's timing.
2. On a split-first, independently sourced development panel, compare reference
   plus endpoint reranking, immediate-value guidance, structural guidance, and
   continuation guidance at a declared binding compute budget. Keep support,
   options, source draws, and objective fixed. Test broad ring construction,
   restructuring and constraint repair, not target-molecule imitation.
3. Only after proposal efficiency and a task-value signal are established, run
   a small paired docking experiment with the same counted warm-start information
   and oracle budget for both arms. A single unlabeled first round cannot test
   learned docking continuation. Twenty calls can audit mechanics, not establish
   statistical benchmark superiority.

For a development comparison on the already inspected PARP1 cell, the existing
`diagnostics/t4_three_level_option_audit.json` contains the common 20-call history
and oracle provenance. Reuse all of that history in both arms instead of
redocking it, and charge those 20 observations to each arm's reported budget.
For example, 20 new calls per arm would mean a 40-call budget per arm and 40
new physical calls, in addition to the 20 already spent. Retain the infeasible
observation as well; do not filter the warm start down to successful molecules.
This is a proposed development design, not a new launch authorization or a claim
that 20 labels suffice to qualify a docking continuation estimator.

IVG winners remain descriptive/development references or separately labeled
oracle/path diagnostics. No winner molecule, fragment, docking label, similarity
reward or macro-weight tuning entered this gate. Their inspected cells cannot
serve as untouched final evaluation. No new cloud scientific job or docking call
was launched for this implementation gate.
