# Two controlled construction witnesses

Source revision: `ea9655143d1e2fdefd38490ef9975b719522a860`.
Prospective scope: `docs/RING_CONSTRUCTION_PROBE.md`.

## Finding

Both fixed requests produced the requested aromatic six-carbon ring through
the unchanged production editing executor from one synthetic benzene source.

| Request | Endpoint | Committed path steps | All executor applications | Cycle-rank delta | Ring-system delta | Wall time |
|---|---|---:|---:|---:|---:|---:|
| Pendant | biphenyl | 10 | 34 | +1 | +1 | 0.0916 s |
| Fused | naphthalene | 5 | 5 | +1 | 0 | 0.0228 s |

Both paths retain the initial atom identities and connectivity, add exactly
one aromatic ring, and contain only complete, connected, charge-preserving
supported molecules. Exact new-slot witnesses distinguish a separate attached
six-cycle from a six-cycle sharing an existing edge. A negative test creates a
nonadjacent two-anchor closure with cycle-rank gain one and ring-system gain
zero; this is rejected as a fused-edge witness. Counts alone are insufficient.

All 39 entered executor applications have saved source/action/product receipts;
only 15 belong to the final two paths. The pendant constructor also evaluates
alternatives. Candidate enumeration and rejected or unused work are not hidden
by reporting only the final path lengths. The hard cap was 256 executor calls
per request; neither request reached the cap or the safety deadline.

`result.json` SHA-256:
`57d2a7c8be89a163532df562acdf4ba1acbb266b27f6b1580121ddc934331beb`.
It binds the exact configuration, all tracked source/config input hashes,
software versions, exact graph states, receipts, and timing. Input/config hashes
and stored acceptance were revalidated after publication.

## Meaning and limits

These are **executed engineering witnesses**, not discovery results. Two of two
preselected fixture requests passed the independent endpoint checks. This is
acceptance on those requests, not a population-level coverage estimate.

The probe uses uniform neutral-primitive weights and the pre-existing descriptor
constructors. It does not load R_theta, use the sampled continuation controller,
exercise Q(M), or sample the balanced Q(o). Its primitive enumerator omits
correlated RingCore tracelets. The timings cannot predict production proposal
time, and the two different construction tasks cannot rank methods fairly.
No winner structures, docking calls, training, cloud work, or macro tuning were
used. Validity does not establish synthetic accessibility or improved binding.

The integration distinction matters: `annulate` is currently a one-step closure
option. `build_fused_ring_exact` is an existing multi-step descriptor constructor,
not a registered fused compound option in the region path. Similarly, the
pendant helper tested here is not the fixed eleven-step `BUILD_RING_SYSTEM`.
Neither was substituted into the frozen controller. A new program integration
needs an explicit program/support decision and an audited probability law.

## Verification

Run the probe with:

```sh
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python tools/ring_construction_probe.py
```

Focused tests: 10 passed in 1.98 s (`focused.xml`). Dependency checks: 44 passed
in 1.87 s (`dependency_tests.xml`), including exact receipt/cancellation tests,
option-prior invariants, and the experiment-infrastructure slot-safety scan:

```sh
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_ring_construction_probe.py tests/test_continuation_profile.py \
  tests/test_option_selector.py \
  tests/test_slot_safety.py::test_experiment_infrastructure_is_slot_safe \
  tests/test_macro_engine.py::test_layer2_annulate_and_append_are_mutually_exclusive \
  tests/test_macro_engine.py::test_build_ring_system_is_a_program_over_existing_macros
```

Ruff lint and formatting passed for the three new Python files. Strict preflight
passed at the source revision with no mounted-tree drift. The prior full suite
remains non-green, as recorded in `../sampled_continuation/verification.json`;
it does not include this standalone module's new tests. Those were checked
separately. No existing controller, executor, configuration, or test file changed.
No completed scientific milestone or production promotion is claimed.

Next: make an explicit integration decision, then measure the frozen
option-conditioned production law on a bounded bundle. An independent panel
and matched baselines remain prerequisites for a chemistry-performance claim.
