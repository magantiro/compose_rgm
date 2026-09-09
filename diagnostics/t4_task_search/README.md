# Hierarchical task search: first development check

2026-09-08 (America/New_York). No new oracle calls or Modal jobs were launched.

## Outcome

**Computed:** the fixed chemistry predictor passes its chronological development
gate on the saved PARP1 seed0, d=0.4 archive. This is a reason to evaluate the
controller prospectively, not evidence that it improves docking.

| Quantity | Result |
| --- | ---: |
| Prior counted docking attempts | 51 |
| Next-round observations scored | 31 |
| Initial-round observations excluded for warmup | 20 |
| Mean absolute prediction error | 0.554972 |
| Training-mean baseline error | 0.823446 |
| Within-round concordance | 0.751678 |
| Non-tied observed pairs | 149 |
| Predictor-check wall time | 1.186696 s |
| New oracle calls | 0 |

Rounds 2, 3 and 4 contributed 13, 13 and 5 scored molecules, respectively, after
training on 20, 33 and 46 earlier labels. All 31 eligible observations were
scored. Pairs share molecules and are not independent trials. This cell was
already inspected during development. These numbers are neither sealed-test
performance nor uncertainty calibration; they do not establish accuracy on
unseen ring-building trajectories. No recipe or threshold was changed after
this check.

The authoritative artifact is [value_check.json](value_check.json), physical
SHA-256 `ad61461bfd724ee563e09f0f1fad81725f4c55fed9ff3944d3908b08aaec3ea4`.
It contains every prediction, exclusion, training snapshot, input hash, software
version and configuration. The original saved archive has physical SHA-256
`803741324ac0e30bff96f9ad36809c6d08ff8334a76a1cc1927a7296a91daf30`.

## Implemented and checked

Code revision `5001e68bd2a2aee4f7559ed5c1da2d09029d1006` adds an opt-in
WHERE/WHAT/HOW planner, exact-state option-boundary transitions, a round-frozen
chemistry predictor and prepare-only T4 integration. The production generator,
executor, existing option support and legacy controller remain in place.

Sparse adaptive rollouts retain full reference rows. Completed returns propagate
across region, option and primitive choices with suffix importance correction.
Reference exploration floors and the KL bound remain explicit. The approximation
is finite-sample biased and is not claimed to be exact Doob control. Primitive
products are cached across options; each option still applies its own contracts
and conditional law.

**Verified:** 77 focused tests passed in 4.87 s in a clean detached worktree of
that exact revision. Tests include reversed delayed utility on a small synthetic
hierarchy, actual executor grow-to-cyclize transitions, full-row/floor/KL checks,
interruption handling, chronological leakage rejection, snapshot round trips,
cache parity and prepare-only integration. The synthetic tests are not docking
experiments. [focused_tests.xml](focused_tests.xml) has SHA-256
`5d443844509f1472aa2eaed76f6488def33e6b1e9fa7de4aeffdb5cb47eee439`.

Commands, with `PYTHONPATH=src:.` and the repository Python environment:

```sh
python tools/preflight.py --strict
python -m pytest -q tests/test_task_search.py tests/test_docking_value.py tests/test_t4_task_search.py tests/test_option_continuation.py tests/test_ring_program.py tests/test_continuation.py tests/test_sampled_continuation.py --junitxml=diagnostics/t4_task_search/focused_tests.xml
python tools/t4_task_value_check.py --output diagnostics/t4_task_search/value_check.json
```

Preflight passed with zero mounted-source drift. Ruff checks and formatting
checks passed for all ten touched non-Modal Python files. The Modal app compiled;
its whole-file Ruff check retains 17 pre-existing findings, independently checked
against predecessor `9596f22`. The new branch's import-order finding was fixed
after the measurement. A subsequent non-vacuous-path assertion in the preparation
test passed in a focused one-test rerun. Neither follow-up changes predictor
computation, so the saved scientific check was not repeated.

The repository-wide suite was not run for this bounded development change under
the explicit T4 policy in AGENTS.md. This is not a release milestone.

## Remaining boundary

**Not yet measured:** learned-reference production preparation throughput,
completed rollout coverage under the 512-call planning allowance, guidance
contrast on real trajectories, or improved docking. The current new-schema
preparation path has not run on Modal. Parent progress receipts are available,
but durable restart publication and a cross-option lock verifier/launcher
contract remain required before remote use. Do not pass its v2 candidates to
the legacy single-bundle docking verifier.

The next useful action is one bounded, prepare-only production audit that checks
whether useful terminal evidence reaches early choices within the fixed compute
budget. Then freeze candidates and the matched comparison before spending fresh
docking calls. A failed or all-reference audit must be reported, not called a
controller improvement. No automatic multi-round campaign is authorized by this
result. The previously observed best docking score remains -9.7 at 51 calls.
