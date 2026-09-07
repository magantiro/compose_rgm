# T4 endpoint comparison and search design note

Date: 2026-09-07. Role: development diagnostic and proposed framework, not a
new optimizer, training authorization, docking experiment, or final benchmark.

## Evidence and scope

The problem is efficient objective-directed search over COMPOSE's executable
molecular editing process. The model output remains a marked rewrite law; the
search output is a set of complete candidate molecules and their executed paths.
The proposed claim to test is better feasible-objective discovery per oracle
and compute budget, not merely more valid intermediates. The present endpoint
comparison cannot test that claim.

Support remains the frozen broad-organic vocabulary, connected graphs, at most
40 active atoms, fixed valence and charge policies, and no stereochemical
claim. Atom count below 40 alone does not prove full representability or
reachability. Graph validity is not synthetic accessibility or physical
synthesizability. The docking oracle is also not experimental affinity.

`comparison.json` records input hashes, exact code revision, analysis-script
hash, software versions, scoring-data hash, selection, and limitations. Run:

```sh
PYTHONPATH=src .venv/bin/python tools/t4_compare_winners.py
```

All five existing PARP1 seed-0 delta-0.4 snapshot winners are included, not just
an illustrative favorite. Their exact SMILES and reported scores were matched
to raw CSV rows; their QED, SA, and seed similarities were independently
recomputed and agree within 1e-6. This is a locally hashed historical snapshot,
not the paper's mean over repeated optimization runs. Its original upstream
revision and complete license/provenance lineage are absent from the local
receipt. No new oracle calls were made. Historical redocking is contextual
reported evidence, not a newly verified calibration experiment.

### Computed endpoint differences

| Endpoint | Heavy atoms | Cycle rank | Ring systems | Aromatic rings | Recorded docking score |
|---|---:|---:|---:|---:|---:|
| Seed | 19 | 3 | 1 | 2 | not compared |
| Current audit best feasible | 27 | 4 | 2 | 2 | -8.9 |
| Five IVG snapshot winners | 30–33 | 5–6 | 2–3 | 3 | -13.6 to -13.4 |

The drawing in `structures.png` shows an additional separate six-membered ring
in the current result. The snapshot winners instead have a second fused-ring
system, altered central-ring size (eight or nine rather than seven in this
SSSR decomposition), and different linkers/substituents. These are endpoint
descriptions, not inferred atom correspondences or verified edit sequences.

The current best has SA 3.951 against the maximum 4.0; the winners span
2.757–3.780. Thus, even our smaller candidate nearly exhausts the SA allowance.
Simply making molecules larger or adding rings is not an adequate objective.
Neither these descriptors nor the 2D drawings establish which changes caused
the docking difference. Budgets also differ: ours is a 20-call mechanism audit.

**Measured:** constructive separate-ring growth occurred in the locked audit.
**Not established:** useful fused-ring construction, recovery of a winner's
chemistry, its executor path, or a competitive search algorithm.

### Historical MCTS is not a selection result

`modal_apps/pmo_control_app.py` contains PUCT with progressive widening, horizon
selection, and return backups. The local tournament contains 12 runs across six
PMO tasks, not the new T4 controller. Eleven receipts have ledger counts that
differ from charged calls. Cache/priming treatment must be reconciled before
their headline comparisons can be trusted. The local BO tournament's `runs`
array is empty. These facts establish that a tree-search implementation was
tried, not that MCTS works, fails, or loses to BO under a matched protocol.

The completed T4 round uses batched controlled particles, not PUCT or MCTS.
Its committor predicts structural establishment; it is not a validated
option-phase-conditioned T4 continuation-value model.

## First-principles framework (proposed, not implemented)

### 1. Optimize the actual experimental budget

For single-objective T4, take utility as negative docking score and seek the
expected best feasible *evaluated* candidate under fixed oracle and compute
budgets. Retain the original feasible seed as a baseline if nothing improves.
For PMO, use its frozen top-10 learning-curve objective instead. Multiobjective
tasks need their own frozen set-valued criterion. These objectives are not
interchangeable.

An exact budgeted decision state would include the population/archive, all
oracle observations, uncertainty state, active trajectories, and remaining
budgets. A molecule-only value function cannot represent that entire problem.
The hierarchy is a tractable factorization, not a proof of global optimality.

### 2. Preserve the executable hierarchy

Keep `Q(M|x,z) -> Q(o|x,M,z) -> q(w|x,M,o)` with generic permanently supported.
Macros specify permissible transitions and termination, not endpoint templates.
Within a selected bundle, the planning state must include exact slot state,
frozen context, lineage, option phase, and remaining horizon. Canonical endpoint
identity can deduplicate oracle evaluations, but cannot by itself key legal
actions or continuation values. Different paths reaching the same SMILES may
still have different permitted futures.

The present audit freezes Q(M), the untrained balanced option prior, R_theta,
and kappa=1. Its KL reference is the option-conditioned law, not the unrestricted
primitive law. A future planner that reallocates visits among regions/options
changes effective outer search allocation and requires an explicit ablation;
it cannot be presented as a computationally equivalent implementation.

### 3. Separate current value from continuation value

Let `f_z(x)` estimate utility of docking x now. Let `h_t(s)` estimate useful
terminal outcomes reachable from augmented state s with its remaining budget.
Ranking the present molecule by f alone can discard prefixes that first need
several growth, closure, and restatement steps.

For a fixed supported reference bundle process R and terminal weight
`G_z(x) = 1_feasible(x) exp(beta U_z(x))`, an ideal reference continuation is

```text
h_T(s) = G_z(x(s))
h_t(s) = sum_{s'} R_t(s'|s) h_{t+1}(s')
q*_t(s'|s) = R_t(s'|s) h_{t+1}(s') / h_t(s), when h_t(s) > 0
```

This is the finite-horizon terminal-weighted reference process. It is neither
an exact solution of the whole budgeted oracle problem nor a guarantee that a
per-step KL bound of one holds. If h is zero, the terminal event is unreachable
under that reference and horizon; define abstention/fallback explicitly.
Our approximate, bounded tilt and structural committor must not be called this
exact task-conditioned transform. A noisy learned h preserves executor validity
but can badly hurt efficiency and discovery. Estimator calibration matters.

COMPOSE's useful property is that every committed intermediate is already a
complete molecule. It can be evaluated, continued, or reused without first
completing a token string. Complete does not mean valuable now: a valid but
currently unattractive prefix may have useful futures. Search-state retention
and expensive-oracle allocation should remain distinct.

### 4. Treat MCTS as a candidate estimator, not the framework

A sparse graph planner could reuse promising prefixes and back up continuation
returns, using progressive widening rather than expanding every legal action.
But a sampled shortlist and visit-based selection generally change the current
transition law. Neither is automatically an exact acceleration of the frozen
KL controller. Preserve exact law tests for genuine cache/batch optimizations;
label a new approximate planner as an algorithm change.

Without useful cheap rollouts or a calibrated continuation estimate, MCTS can
spend the limited docking budget learning a huge, shallow tree. Conversely,
bounded particle rollouts may be sufficient when options already coordinate
the difficult steps. The present evidence does not resolve this tradeoff.

Recommendation: compare bounded controlled particles against a sparse,
option-level graph planner, holding generator, admissible support, population,
oracle protocol, and total compute budgets fixed. Include the same-generator
post-hoc baseline to isolate in-loop guidance. Any task continuation estimator
must be fitted and selected on separate development/calibration data only and
its contribution ablated separately. No such training is authorized here.

### 5. Test capabilities without copying winners

The inspected cell is development evidence. Avoiding exact winner SMILES in
training is not enough to make later measurements on this cell an untouched
test: model, program, and search-design choices can all adapt to it indirectly.

Build a separately sourced, split-first capability panel spanning attachment,
annulation, ring-size changes, aromatic/saturated and heteroatom combinations,
linker changes, shrinkage, and mixed local/global rewrites. Balance these axes
explicitly rather than mining the five winners into a ring catalog. Source
scaffolds, target motifs, and reusable templates must respect the declared
train/calibration/test separation. Do not use similarity to an IVG winner as a
reward, stopping condition, or hyperparameter-selection rule.

Measure executor-valid complete trajectories, constructive endpoint precision
as well as coverage, constraint margins, actual versus intended change,
component/scaffold novelty, diversity, all oracle calls (including prescreens),
and proposal time. Reachability needs an actual executed witness; path existence
alone is not practical discovery probability. Reserve a fresh untouched panel
for the frozen final comparison. No claim of beating every IVG task follows
from a valid-state representation or from this endpoint inspection.

## Next safe action

Profile the existing proposal path on independent development inputs. The audit
spent 2,208.4 seconds proposing and 34.3 seconds docking; having only 57 emitted
candidates does not bound how many legal successors were materialized and
scored internally. Profile before assigning a percentage of time to any cause.
Then specify the capability panel and matched planner ablations before paid
experiments. This note changes no executor, checkpoint, prior, parameter,
support limit, run contract, or benchmark gate.

## Verification and provenance receipt

Analysis implementation revision: `2352c33`. JSON SHA-256:
`a19435b9bfdfa11cfaefe62d8dea454744498b3b1bc8e7173f749cf9b5390ac9`.
Figure SHA-256:
`dcfc2c7f2c6011b93b5e7de4ae6d9b0ee79164bf8b4989632fda048b4228303e`.
Two consecutive runs at that revision produced byte-identical JSON and PNG.
All seven depictions were visually inspected. Raw-record matching, descriptor
agreement, and JSON round-trip checks passed in the analysis command.

Focused tests:
`.venv/bin/python -m pytest -q tests/test_t4_winner_comparison.py tests/test_ring_taxonomy.py`
passed all six tests. Focused Ruff lint and format checks passed; the initial
import-order lint findings were corrected. This standalone diagnostic did not
change production dependencies. The repository-wide suite was not rerun; the
prior non-green result remains recorded in `docs/T4_THREE_LEVEL_OPTION_AUDIT.md`.
This is not a declaration that the broader scientific milestone is complete.

External context checked against the authors' [paper](https://arxiv.org/html/2509.26405)
and [results repository](https://github.com/invirtuolabs/InVirtuoGen_results).
Those references do not repair the missing immutable upstream identity of the
historical local CSV. Numerical comparisons here derive from the explicitly
hashed local inputs, not a paper-table transcription.
