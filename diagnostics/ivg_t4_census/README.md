# Full released similarity-constrained T4 winner census

## Question and scope, 2026-09-08

The user asked why the controller does not expose the existing five/six-member
ring builders and why the comparison covered only a small IVG winner snapshot.
This record answers those questions through code inspection and a structural
census. It changes no generator, controller, executor, gate, or experiment.

COMPOSE remains an executable stochastic molecular-rewrite process. This
diagnostic's output is an endpoint inventory, not model output, an optimization
experiment, a reachability demonstration, or a causal ablation. No claim of
superior docking or universal molecular support is tested here.

## Computed results

The pinned public release contains 30 similarity-constrained T4 CSVs: five
targets, three starting molecules per target, and two similarity thresholds.
All 87,448 rows were considered. All 90 observed optimization runs have at least
one feasible winner. The 200 co-best source rows reconcile to 101 distinct
cell/run/structure records and 91 globally distinct canonical molecules.
The old `diagnostics/ivg_winners.json` was only a five-cell snapshot and remains
unchanged.

Among the 91 unique endpoints:

- 329 of 382 symmetrized-SSSR rings (86.1%) have five or six members.
- 86 molecules contain heterocycles; 87 contain multiple ring systems.
- 58 contain fused rings; none have RDKit bridgeheads.
- Six contain spiro rings. All six belong to 5HT1B source-index-0 cells, whose
  starting molecule already contains a spiro center. Endpoint comparison does
  not prove preservation of that exact center or a particular edit history.
- All contain an aromatic ring, but 43 also contain a saturated ring and 52
  contain an unsaturated nonaromatic ring. These molecule-level labels overlap.
- None exceeds 40 heavy atoms; 14 carry nonzero net formal charge. These are
  partial support checks, not proof of admission or charge-preserving paths.

| Target | Unique endpoints | Five/six-member rings / all rings | Heterocyclic endpoints | Fused endpoints |
|---|---:|---:|---:|---:|
| 5HT1B | 19 | 88/93 | 19 | 12 |
| BRAF | 16 | 49/49 | 15 | 2 |
| FA7 | 18 | 50/52 | 14 | 6 |
| JAK2 | 20 | 79/101 | 20 | 20 |
| PARP1 | 18 | 63/87 | 18 | 18 |

More rings is not a general objective. The 20 BRAF cell/run/co-best records
include 17 with lower cycle rank than their corresponding starting molecule
and three with equal rank; none has higher rank. For PARP1, 18 of 19 such
records have higher cycle rank and one has equal rank. These denominators
retain ties and are not run-balanced estimates. The report stores every source
structure and endpoint delta, separately from aggregate unique-molecule counts.

## Verified implementation gap

Code inspected at `c9dfdae`, unchanged by this reporting-only work:

- `control/option_selector.py` exposes ordinary macro options for one primitive
  step. A `cyclize` or `annulate` label does not itself grow a ring precursor.
- `build_ring_system` exposes the fixed eleven-step program, not the complete
  descriptor-builder interface.
- The opt-in `control/fused_option.py` grows four carbon atoms across an existing
  carbon edge to construct an aromatic six-member ring. It is not a general
  five/six-member heterocycle builder.
- `control/macro_engine.py` already has `build_ring_system_exact` and
  `build_fused_ring_exact`, with size, composition/stoichiometry, electronic
  state, and ring-scoped refinement parameters. The T4 region path in
  `modal_apps/genmol_t4_opt_app.py` does not call these interfaces or expose
  those parameters. Their deterministic ranked execution is not automatically
  equivalent to the frozen stochastic inner controller.

Thus macro reintegration was partial at the program-capability level, despite
restoring the option layer. Generic search remains active; this is a gap in
efficient structured proposal channels, not proof that every missing endpoint
is unrepresentable.

Proposed next scope, not implemented or launched here: expose the existing
size/composition/state construction contracts through explicit stochastic
pendant/fused programs and ring-scoped refinement, while retaining the
local/global region allocation, generic option, frozen base law, KL bound,
exact state provenance, and valid primitive execution. No winner-derived ring
catalog, learned option prior, or docking-tuned weights are justified by this
census. The existing ring-expansion diagnostic is a separate capability, not
a replacement for this construction interface.

## Reproduction, provenance, and verification

Source: [IVG public results](https://github.com/invirtuolabs/InVirtuoGen_results),
revision `b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb`.
Selection follows its `in_virtuo_reinforce/evaluation/results_table_lead.py`:
reported similarity strictly above the cell threshold, QED strictly above 0.6,
SA strictly below 4, then maximum reported positive docking magnitude per
optimization-run `seed`. Filename `idx` instead identifies the starting
molecule. All canonical co-best ties and their source rows are preserved.

`census.json` binds all 30 CSVs, the upstream selector and license, local seed
registry, analysis implementation, and shared ring taxonomy by SHA-256. Each
download also matches its pinned upstream Git blob. No full upstream code is
executed. The raw files and exact license remain in the ignored `.cache/ivg_t4/`
directory; immutable URLs are embedded in the report. The license's NC-SA
introductory notice and NC body conflict, so no commercial permission or
unqualified redistribution right is inferred.

Recomputed Morgan-radius-2/2048-bit similarities to the registered starting
molecules agree with all selected reported similarities to at most
`5.551115123125783e-17`. This check does not replace the upstream feasibility
rule. Ring counts use RDKit 2026.03.6; graph cycle rank and ring-system counts
remain separate. Scores are reported IVG outcomes, not redocked or
oracle-parity-verified measurements.

```sh
PYTHONPATH=src .venv/bin/python tools/ivg_t4_census.py --download
.venv/bin/python -m pytest -q tests/test_ivg_t4_census.py tests/test_t4_winner_comparison.py
.venv/bin/python -m ruff check tools/ivg_t4_census.py tests/test_ivg_t4_census.py
.venv/bin/python -m ruff format --check tools/ivg_t4_census.py tests/test_ivg_t4_census.py
```

All 10 focused tests passed in 2.47 seconds (`focused.xml`). Ruff lint,
formatting, JSON round-trip, upstream hashes, and diff checks passed. No
repository-wide suite was run for this diagnostic; no controller milestone is
being declared complete. Analysis code is committed at `df46ad2`; unrelated
concurrent work was preserved and is outside the hashed analysis dependencies.
There were zero additional oracle calls, model evaluations, or training runs.
A second local run produced a byte-identical report, SHA-256
`2c06fdfe65fc31327d865381ccb8f4d2c40553c3925d9dc0a03386c22fda05bb`.

This census covers all released **similarity-constrained T4** cells, not
unconstrained T4 or every other IVG task. All inspected cells are development
evidence. They cannot subsequently serve as untouched final-test evidence for
a method selected using this inspection. Structural resemblance does not
establish docking causality, synthesizability, or an executable path.
