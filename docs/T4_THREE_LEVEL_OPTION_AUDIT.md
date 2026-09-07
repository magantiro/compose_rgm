# T4 three-level option audit contract

Date frozen: 2026-09-06

Session label: `compose_iclr`

## Identity and claim boundary

The scientific problem is efficient lead optimization with an executable
stochastic molecular editing process. The primary model output remains a
state- and time-dependent marked rate law over complete legal primitive
rewrites. The controller changes the proposal law, but every accepted state is
still produced by the production executor as a complete supported molecule.

The central claim tested by this audit is narrow: adding the existing
macro/program option layer between region selection and primitive realization
can make constructive ring and scaffold transformations reachable without
changing the frozen generator, the qualified region controller, or the
Kullback-Leibler trust region. A better docking score in one 20-call round is
not required and cannot establish benchmark superiority.

The validation setting is the existing GenMol Table 4 `parp1` seed-0 cell at
Tanimoto similarity threshold 0.4. The causal reference is the frozen raw
region-proposal path on the same cell. Historical score references are prior
COMPOSE at 200 and 500 calls and the published comparators recorded in
`docs/HANDOFF_CODEX_2026-09-05.md`; their larger budgets are not matched by this
mechanism audit.

The declared generated-object support is the editing-V2 broad-organic
element/valence vocabulary, connected molecular graphs, at most 40 active atoms
in 48 persistent coordinate slots, charge-preserving edits, and no
stereochemical claim. QuickVina's receptor and ligand preparation support is a
downstream evaluator limit, not a restriction on the generator's support.

## Frozen factorization

```text
Q(M | x,z) -> Q(o | x,M,z) -> q(w | x,M,o)
   WHERE            WHAT              HOW
```

- `Q(M | x,z)` is the qualified scale-balanced `mu_exec` controller with no
  T4 task-value tilt. The null QED-trained `V_z` is not reused.
- Exactly one option is sampled for each region draw. Frontier width never
  creates another draw from either `Q(M)` or `Q(o)`.
- `Q(o | x,M,z)` is an untrained, applicability-aware prior. It is uniform
  over represented semantic-purpose groups, uniform over applicable variants
  within a group, and mixed with a per-option uniform exploration floor.
- `generic` is always applicable and retains the qualified multi-step raw
  region rewrite.
- Each ordinary macro option conditions one primitive transition through the
  existing `MACRO_FAMILIES`, `proposal_support`,
  `macro_action_distribution`, and macro-local contract machinery.
- `BUILD_RING_SYSTEM` is one indivisible compound option with the exact program
  `scaffold_extend x8 -> append_system x1 -> restate x2`. Its intermediate
  states remain executable search states but are not oracle candidates. Only
  complete 11-step endpoints are emitted.
- No option teleports to an endpoint. Every transition is executed by the same
  production executor and checked for validity, connectivity, frozen-context
  preservation, and the 40-active-atom support bound.
- Shared enumeration changes arithmetic only. Oracle candidates are sampled,
  committed particle states under `q(w | x,M,o)`, not every enumerated member
  of an option's legal support.

## One-round configuration

The audited launch is exactly one 20-call round:

| field | frozen value |
|---|---:|
| target / seed index / delta | `parp1` / 0 / 0.4 |
| random seed | 1000 |
| lineages | 8 |
| region draws per lineage | 3 |
| particles per region draw | 4 |
| dockings per round / total budget | 20 / 20 |
| maximum frontier width | 8 per bundle |
| emitted representatives | at most 3 per bundle |
| generic horizon | 16 primitive steps |
| ordinary macro horizon | 1 primitive step |
| `BUILD_RING_SYSTEM` horizon | 11 primitive steps |
| `kappa` | 1.0 |
| primitive exploration floor | 0.1 |
| region exploration floor | 0.2 |
| option exploration mixture | 0.1 |
| existing macro temperature / exploration | 2.0 / 0.15 |

The macro weights are inherited from the existing option machinery. They must
not be adjusted after inspecting docking outcomes.

## Required audit output

The authoritative JSON must bind the clean code revision and SHA-256 hashes of
the seed manifest, frozen generator checkpoint, generator run manifest,
committor, QuickVina binary, and receptor. It must report:

- every selected `(parent lineage, region, option)` bundle and its normalized
  `Q(M)` and `Q(o)` probabilities;
- selected, candidate, and docked counts by option;
- intended released-region scale and realized total/coherent structural change;
- ring-system, graph-cycle-rank, heavy-atom, and element deltas;
- added C/N/O backbone atoms, terminal halogens, and sulfur;
- globally unique candidate count, docked uniqueness, and mean pairwise Morgan
  distance;
- proposal and docking wall time;
- halted or incomplete compound programs and docking failures.

The diagnostic is positive only if the selected constructive options produce
measured constructive topology or backbone outcomes rather than merely terminal
halogen/sulfur decoration. A null or negative result is retained unchanged. No
docking result will be used to tune the option prior or macro weights.

## Launch and persistence

Run from a clean committed tree only:

```bash
python3 tools/preflight.py --strict
modal deploy modal_apps/genmol_t4_opt_app.py
python3 tools/t4_launch.py --budget 20 --n-seeds 1 --deltas 0.4 --arms macro_prior
```

Do not use `modal run --detach`. The cell writes a complete round receipt to the
`compose-v4-artifacts` Modal volume after the round and the local launcher writes
the deployed call identity to `diagnostics/t4_spawned_three_level.json`.

## Prelaunch verification record

On 2026-09-06, the focused dependency checks completed successfully:

- 64 option-selector, region-controller, and T4-audit tests passed;
- 76 macro-engine tests passed in the focused run, with one known slow frontier
  test excluded after an earlier attempt exceeded eight minutes;
- that slow test subsequently completed during the repository-wide run and was
  not among its failures;
- Python compilation, focused Ruff lint, Ruff formatting, and
  `git diff --check` passed for the changed dependency closure.

The corrected repository-wide command, `.venv/bin/python -m pytest -q`, ran to
completion in 2,303.40 seconds: 4,313 passed, 48 failed, 58 errored, 2 skipped,
and 1 was expected to fail. No listed failure or error was in the new option
selector, region receipt test, T4 audit test, or macro-engine test. The dominant
errors were frozen RingCore/catalog or panel-sampler identity drift and Active8
pipeline fixture setup; other failures were in unrelated historical modules.
This is not a green repository-wide gate and must not be reported as one. It is
retained as a negative verification result. The bounded T4 audit proceeds only
because its focused dependency closure is green and the repository contract
explicitly forbids delaying a bounded scientific run for unrelated cleanup that
does not block correctness, reproducibility, or recovery of the current run.
