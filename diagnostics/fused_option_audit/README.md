# Stateful fused option: local integration evidence

Source revision: `1be2706772b9bc576fa3df84592fa8f7770abe20`.
Contract: `docs/FUSED_OPTION_INTEGRATION.md`.

The one predeclared reference-only trajectory completed benzene to naphthalene
through five primitive edits: four single-neighbor carbon insertions and one
bond closure. Independent exact-slot checks passed: four new atoms, one new
six-carbon aromatic cycle sharing the original edge (3, 4), cycle-rank gain
one, and unchanged ring-system count. Every sampled intermediate was a complete
valid executor product. No endpoint constructor was called.

This is **model-free engineering evidence**, not learned-model performance.
Uniform neutral primitive weights, the inherited 300/20 proposal cap, seed 0,
and one fixed synthetic whole-region bundle were used. This fixture omits
correlated RingCore tracelets. It does not evaluate Q(M), Q(o), discovery
coverage, realistic candidate diversity, docking, or superiority over another
ring method. Zero docking calls and zero lookahead/terminal-objective calls.

The initial reference row had 12 equally weighted oriented-edge states, produced
by six distinct physical insertions. The sampled edge was (3, 4). Its remaining
four rows each had one admissible mark. The audit required ten total public
executor applications, five raw-law enumerations (162, 196, 224, 285, and 330
candidate rows), and 0.10638020900660194 seconds measured inside the audit,
excluding interpreter/import startup. That timing is not a production R_theta
estimate. Exact states, probabilities, support funnels, and all executor
receipts are in `result.json`.

## Verification and unresolved boundary

The focused dependency suite passed: 120 tests, zero failures/errors/skips.
All eight touched Python files passed Ruff lint and format checks. Strict
preflight reported zero mounted-source drift. The repository reference audit
completed as an inventory, not a scientific acceptance gate. Input and
configuration hashes in the numerical artifact were independently rechecked.

One full repository suite completed on the frozen source revision outside
the sandbox to permit the documented native shared-memory/multiprocessing
tests: 4,423 passed, 47 failed, 58 errors, two skipped, and one expected failure
in 1,851.37 seconds. All 19 new fused-option/audit tests also passed in the full
suite. All 105 current nonpassing identities were present in the previous full
suite; the two previously documented native-runtime failures now pass. Matching
identities does not establish identical causes or waive gates. The repository
milestone remains non-green.

The implementation remained at its exact committed revision during the full
suite. Raw XML is retained byte-for-byte, including traceback whitespace that
fails the artifact-level diff whitespace check. Nothing has been pushed.

An initial unit-test assumption that symmetric benzene would yield different
completion values across edges was false: the declared fixture completed from
every edge. The test now explicitly verifies that this symmetry earns no
guidance preference. A separate, declared asymmetric synthetic reference tests
delayed credit; it is not a chemistry benchmark or a change to the production
law. Tests also cover cap-induced exclusion, invalid products, empty mutable
edge support, interrupted rows, sparse slots, program-state identity, exact
path replay, reference-only behavior, and sampled-planner reproducibility.

The new option remains opt-in and is not enabled in the T4 optimizer. Generic,
the eleven-step BUILD_RING_SYSTEM, default option-prior behavior, Q(M), R_theta,
and kappa remain unchanged. The next scientific step is a separately frozen,
capped production-reference cost/support audit. No such remote run, guided
chemistry comparison, or new docking experiment has been launched here.

## Commands

```sh
python3 tools/preflight.py --strict
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python tools/fused_option_audit.py
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_fused_option.py tests/test_fused_option_audit.py tests/test_option_selector.py tests/test_option_continuation.py tests/test_continuation.py tests/test_sampled_continuation.py tests/test_continuation_profile.py tests/test_continuation_replay.py tests/test_ring_construction_probe.py tests/test_slot_safety.py::test_experiment_infrastructure_is_slot_safe --junitxml=diagnostics/fused_option_audit/focused.xml
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q --junitxml=diagnostics/fused_option_audit/full_suite.xml
```

The numerical command refuses to overwrite this existing audit. Reuse the
artifact and verify hashes instead of recomputing unchanged molecular work.
