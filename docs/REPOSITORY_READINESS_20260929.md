# Repository readiness repair

Date: 2026-09-29. Base: `8aa27f7f3bb8935dd7b1d8d8b48a92dfba2b294f`.

## Objective and scope

The user authorized repairing the test suite, assessing whether its expectations
are valid, and preparing the repository for other researchers to use. COMPOSE's
scientific identity remains the learned executable stochastic rewrite process
over complete molecular graphs. Its primary output is the marked transition law
and the molecular successors induced by exact execution. The paper evaluates
reference learning, constrained construction, program selection, molecular
editing, and feedback-driven optimization against their declared controls.
Representation, chemistry, size, split, baseline, and evaluation support remain
those of each versioned experiment.

This milestone concerns local software correctness and usability. It authorizes
test and implementation repairs, packaging, and documentation. It makes no new
benchmark claim and authorizes no publication, scored experiment, training,
docking, cloud deployment, or recovery of live campaigns. Existing worktrees,
user changes, frozen scientific results, and authorization identities are
preserved.

## Acceptance

1. The suite collects in the documented supported Python environment.
2. Every observed failure is assessed against the intended public behavior and
   scientific contract before changing either implementation or expectation.
   Repairs retain tests for validity, support, leakage, provenance, accounting,
   determinism, and failure handling.
3. Tests use small offline fixtures. A required external asset is documented and
   distinguished from an ordinary software defect. No failing test is hidden by
   a broad exclusion or a relaxed gate.
4. Focused checks pass for each repair; the final candidate receives repository
   verification and one complete suite run with its real exit status captured.
5. Installation, import, CLI help, and available offline benchmark workflows
   work from a source export independent of historical working directories.
   The PMO and T4 integration dependency identified in the handoff remains
   explicit until their producer code is integrated and tested.
6. The record reports the actual final passes, failures, errors, skips, runtime,
   environment, source revision, and unresolved input access requirements.
7. Changes are inspected and committed in scoped local units. No push.

## Verification environment

Core checks use Python 3.11 with RDKit 2024.3.5, Torch 2.4.0, NumPy 1.26.4,
SciPy 1.13.1, and NetworkX 3.3. PMO oracle checks, where applicable, require
their distinct RDKit 2023.9.6 environment and do not score molecules during
this milestone. The existing laptop `.venv` uses Python 3.12 and is not the
paper chemistry environment.

The other agent's 4,688 passes, 231 failures, and 101 errors describe a reported,
time-limited run on a different branch. They are not this checkout's baseline
and do not by themselves prove that every failure was pre-existing.

## Decision log

- Begin by reproducing collection failures and identifying failure families.
  Diagnose on the current checkout, with focused tests during repairs.
- Judge readiness using installation and usable experiment interfaces as well
  as test assertions. A passing assertion that does not protect meaningful
  behavior is not sufficient evidence.
- Added the missing PyYAML runtime dependency and a pinned Python 3.11 core/test
  installation recipe. The isolated local environment passed dependency
  compatibility checks; it is not a substitute for the separate PMO oracle
  kernel.
- Replaced tests that relied on Python's private typing internals or validated
  only their own mock arithmetic. The QED sampler tests now observe the actual
  local transition probabilities, while executor aggregation tests require
  real many-to-one mark aliases and fail on interface drift.
- Preserved the historical E6 receipts and tested their refusal under changed
  source identities. Fresh bounded fixture audits test current behavior without
  re-sealing or promoting the old scientific artifacts.
- Corrected a readiness audit that ignored its requested repository root and a
  training-report consistency check where float32 reductions crossed an input
  extremum by up to two ULPs. Frozen scientific thresholds remain strict.
- Removed a full-checkout recursive scan from a source-identity test; its
  replacement still refuses omitted serialized inputs. Removed a vacuous
  golden-path assertion and consolidated the cycle-rank definition.
- The first full-suite diagnostic on this checkout was deliberately interrupted
  during a CPU-bound section: 2,586 passed, no failures, in 15m46s of pytest
  time. This is a partial run, not suite sign-off. The final clean-export run
  and any remaining findings must be recorded before completion.
- `ruff check src/` currently reports 494 legacy findings. Touched Python
  files are checked separately. Mass formatting hash-bound historical sources
  would change their physical identity and is outside this repair.
