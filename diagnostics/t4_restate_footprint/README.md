# Ring-restatement footprint repair: verified locally

2026-09-09. The empty-footprint defect is repaired in commit
`36df44cfc844` (full revision in audit.json). No deployment, training, molecular
search, learned-law enumeration or docking was performed.

## Result

- All 17 saved `RingSystemRestate` actions now report exactly their changed-bond
  endpoints. Before: zero admitting regions for every action. After: at least
  one admitting and context-preserving region for all 17 actions.
- A single deterministic, context-preserving region was selected per action.
  Under an explicitly diagnostic singleton reference law, 10/17 actions
  reproduce their exact saved products through each of `generic`, `restate`
  and `aromatize`. Seven still return no successor downstream of the touch
  filter. They are not counted as full option successes.
- All three PARP1 seed0/d=0.4 winner-path restatements pass the option checks
  and exactly reproduce their saved products in every tested channel:

| Pair prefix | Saved prefix length | Admitting, context-preserving regions |
| --- | ---: | ---: |
| `5705946f25f3` | 15 | 49 |
| `b776cfbb4080` | 15 | 52 |
| `76f129c70c87` | 17 | 40 |

This verifies removal of this implementation blocker. It does not demonstrate
learned R_theta support, controlled path probability, blind discovery, docking
improvement or pinned-Modal-runtime equivalence. The local runtime remains
RDKit 2026.03.6, not Modal's 2024.03.5.

## Safety and remaining issues

The repair recognizes the typed `RingSystemRestate.changes` payload. It also
rejects each changed context-context bond, including a frozen bond hidden
alongside a valid local change in a composite action. Existing primitive
admission and post-execution context checks are retained. No region expansion,
executor change, horizon change, kappa change or learned-parameter change was
made. Generic remains available.

Source inspection identifies an additional remaining issue to review:
`OptionContinuationKernel._clean_product` applies the full `med_chem_gate.is_valid`
predicate at every intermediate, while that gate module separately defines
`is_executable` for pathwise use. The full predicate includes cumulene and
isolated-ring preferences. This repair does not alter either predicate or its
placement. A subsequent gate-placement change requires a separately recorded
decision, including acceptable endpoint protections. The previously measured
16-edit horizon and delayed-feasibility issues also remain.

## Checks and provenance

Strict preflight passed in the clean committed worktree
`/private/tmp/compose-winner-paths.W1n7sh`, excluding unrelated dirty scaffold
and model work. These focused tests passed: 68 tests, zero failures/skips/errors,
2.50 seconds:

```sh
PYTHONPATH=src:. python -m pytest -q tests/test_region_restate_footprint.py tests/test_region_rewrite.py tests/test_option_continuation.py
```

The saved-action recheck took 1.601 seconds and 17 option-kernel executor
applications. Three channel checks per action share the same cached physical
product. These counts do not represent learned-law enumeration or the internal
micro-steps of composite lowering. All 78 non-repaired source dependencies of
the original audit were hash-verified unchanged. Exact saved arrays and actions
were reused; no intermediate was reconstructed from canonical SMILES.

```sh
PYTHONPATH=src:. python tools/t4_restate_footprint_audit.py /path/to/ivg_winner_paths /path/to/t4_restate_footprint/audit.json
```

The runner requires clean committed source and an output directory outside that
worktree. Audit SHA-256:
`9c93f491837ebdce607fa587cd357348ec33d8f83f5c5347395861e1aafa3824`.
Focused-test receipt SHA-256:
`834b150ecff9361a647a45d4f797bd31020cedca549d05de89baef445acbf08e`.

New test/audit files pass Ruff lint and format checks. The new production
statements pass range formatting; the existing module retains five unchanged
legacy lint findings (four UP037 and one I001) and pre-existing formatting debt.
Baseline/current lint comparison found no new findings. `git diff --check`
passes. No unrelated repository-wide suite was run under the scoped T4
development policy; no release milestone is claimed.

The original winner audit and its negative findings remain unchanged. No
winner-derived templates, rewards or learned weights were introduced.
