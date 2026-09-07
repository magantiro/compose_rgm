# Budget repair and sampled continuation

Scientific source candidate: `69e4ffb6fbdfb9d281dfe87895493118d925e747`.
Budget repair: `cf3cd56`. Changes are local and unpushed.

## Outcome and interpretation

The diagnostic now propagates computational cancellation past legacy chemical
rejection handlers and accounts for interrupted enclosing executor calls. Actual
ring-restate lowering is covered by the regression tests. No executor semantics,
R_theta weights, region selection, option prior, macro contracts, KL limit, or
live T4 optimizer were changed.

The new opt-in `estimator="sampled"` follows complete reference suffixes under
a fixed sample allocation. It caches exact augmented-state rows and deterministic
terminal evaluations, preserves reference support, uses simultaneous Monte Carlo
bounds, and abstains on incomplete or inconclusive comparisons. Planning and
committed-path random streams are separate. This is not MCTS, an exact Doob
transform, or evidence that a task surrogate is calibrated.

`engineering_gate.json` and its physically hashed `inputs.json` are the
authoritative engineering receipt. All six checks passed. The synthetic prefix
tree used 154 sampled reference expansions versus 682 exact expansions, with the
same controlled decision. It used 64 terminal evaluations versus 2,048. This
fixture deliberately has deterministic terminal classes and is not a molecular
performance benchmark. Sampling was slower in this tiny, cheap Python fixture
(0.0765 s versus 0.0179 s); fewer expansions are not a measured wall-time win.

On the separate hand-declared molecular fixture, both arms used four executor
applications and returned the same controlled decision. There is no measured
chemistry-cost saving here. The mathematical and executor tests do not establish
better T4 discovery, feasible molecule yield, ring coverage, or proposal time.

## Recovered failed production profile

`prior_profile_audit.json` binds every downloaded payload by SHA-256 and retains
the original launch, failure, and progress records. Source volume:
`compose-v4-artifacts`, path
`/continuation_profile/dcdac938b77b7440e0c214bcb197252d1e0c4dd9707720f18d1b336a3177b443`.
The complete payloads remain on that volume and in the local download under
`diagnostics/continuation_profile/attempt_1/`; they are not regenerated.

Computed inventory: 28 completed laws and 28 completed rows, including one
nine-successor root. Twenty stored rows at step eight have empty support. This
describes those precursors, not general ring reachability. The run's progress
reports 2,000 public executor calls, but only 1,999 receipts were saved (1,411
inside enumeration, 588 in option validation). The missing receipt was not
invented. The repaired meter tests exact receipt coverage.

Stored content hashes and aligned payloads were verified without new chemistry
or docking. This is not a direct evaluator-equivalence certificate. Before
cross-revision reuse, validate the narrow scientific dependencies and bounded
direct equivalence. Missing cache rows must remain explicit misses, not empty
reference laws. Do not repeat the exhausted exact tree as a speed benchmark.

## Verification and reproduction

Focused checks: 136 passed in 9.97 seconds, recorded in `focused.xml`:

```sh
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_continuation.py tests/test_option_continuation.py \
  tests/test_sampled_continuation.py tests/test_continuation_profile.py \
  tests/test_option_selector.py tests/test_region_rewrite.py tests/test_t4_audit.py
OMP_NUM_THREADS=1 .venv/bin/python tools/sampled_continuation_gate.py
python3 tools/preflight.py --strict
```

Touched-file Ruff lint and format checks passed. Strict clean-mounted-tree
preflight passed at the scientific source candidate. `tools/repo_audit.py`
completed; it is a reference inventory, not a scientific correctness gate.

Repository-wide verification was started once on the frozen candidate and is
still pending. It has already reported failures/errors, so no full-suite pass or
completed scientific milestone is claimed. A duplicate focused invocation that
included the unchanged, long-running feasibility-frontier test was terminated;
that test remains in the full suite. No test was weakened or silently skipped.

## Remaining gate

This is an engineering implementation, not a production-qualified efficient
controller. Full reference-row construction remains expensive. A fixed K per
root successor may still be too costly, rare terminal events may yield repeated
abstention, and spending the executor budget on planning can prevent completion
of later committed steps. K=32 is an engineering setting, not a selected T4
hyperparameter. Before promotion, use the frozen learned model on an independent
development panel, match the declared binding compute budget and baselines, and
measure full-program completion, diversity, realized chemistry, and wall time.
No new training, production deployment, or docking was launched in this repair.
