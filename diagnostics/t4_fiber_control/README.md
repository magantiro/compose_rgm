# FiberControl: the autonomous development lane

What lives here, what each artifact is allowed to claim, and what it is not.

## The question this directory exists to answer

    Given a rich pool of exactly feasible COMPOSE program endpoints, does docking
    feedback improve which of them we choose to dock next?

Not "can COMPOSE reach -11-class JAK2 chemistry" -- that is settled and it is
route-assisted (`../t4_jak2_contrastive/DELTA_LANES.md`, -11.32 +/- 0.04 against an
archive best of -11.00). This lane asks whether reward-adaptive control finds comparable
chemistry *without* the route.

## The bar, pinned to its lane

This project has already been burned once by comparing a delta=0.4 molecule against a
delta=0.6 target (`../t4_jak2_contrastive/DELTA_LANES.md`). So the comparator is written
down per lane, per seed, with its statistic named.

InVirtuoGen's reported JAK2 means, read from the published table. Seed rows there are
labelled `seed score / QED / SA`, which identifies each cell unambiguously:

| seed row | cell | delta = 0.4 | delta = 0.6 |
| --- | --- | ---: | ---: |
| -7.7 / 0.725 / 2.89 | jak2_0 | -10.2 +/- 0.8 | -9.7 +/- 0.3 |
| **-8.0 / 0.712 / 3.09** | **jak2_1** | -10.5 +/- 0.3 | **-10.4 +/- 0.1** |
| -8.6 / 0.482 / 3.10 | jak2_2 | -10.2 +/- 0.2 | -10.3 +/- 0.2 |

delta=0.6 is not materially harder for them than delta=0.4 on this target.

**Oracle calibration.** IVG reports the jak2_1 seed itself at -8.0. Our docking returned
-8.00 and -8.10 for that molecule on two separate calls, so the two oracles agree on this
cell to within our own single-call reproducibility. That is the only direct evidence we
have that the scores are comparable at all, and it is worth re-checking per target.

**A statistic that is NOT the same number.**
`configs/t4_frozen_program_benchmark_v2.json` carries `ivg_reported.jak2_1.mean = -11.57`
from `run_bests [-12.6, -11.3, -10.8]` at delta=0.4. That is the mean of per-run BESTS
over released rows, not the paper's reported mean of -10.5 +/- 0.3. Both are real and they
differ by more than a unit. Any comparison must name which one it is measured against;
the paper table is the citable bar and the released-row best-of is the harder internal one.

## The controller

`P*(tau) ∝ P0(tau) · 1[tau queryable] · exp(beta · J(tau))`

`P0` is the production program process; the indicator is the exact T4 endpoint gate,
evaluated free; `J` is the best endpoint reward found inside the oracle budget.
Feasibility constrains the SUPPORT and never enters as a reward term -- measured,
`corr(psi_1, docking) = +0.424`, so higher future feasibility goes with worse binding
(`compose_v4/control/feasibility_value.py`). An explicit STOP exists so a high-reward
boundary state may be terminal.

Runtime: `compose_v4/control/fiber_control.py`, `compose_v4/experiments/t4_fiber_campaign.py`,
`compose_v4/experiments/t4_fiber_expansion.py`. Tests: `tests/test_fiber_control.py`,
`tests/test_fiber_expansion.py`.

## Artifacts

| file | what it is |
| --- | --- |
| `paired.json` | the paired adaptive/reward-blind campaign, every round and every counted call |
| `PAIRED.md` | the report generated from it by `scripts/t4_fiber_paired_report.py` |
| `width_probe.json` | pool width against wall clock for the sharded expansion |

## Why the arms are paired rather than run separately

Single-call docking reproducibility on this target is about half a unit: the benchmark
root returned **-8.00 and -8.50** under identical requests, and later **-8.10**. So a
best-so-far gap smaller than that is not an ordering, and two separately-run arms give
n=1 each.

Both selectors therefore run in one process against ONE shared pool per round. The union
of their picks is docked once, each arm is charged its full batch, and each sees only its
own outcomes. Every round is then a matched pair -- two batches drawn from one candidate
list, differing only in whether reward information was used -- which supports an exact
sign test over rounds.

Two biases run AGAINST the adaptive arm and are deliberate:

- the pool grows from the union of both frontiers, so the blind arm proposes from parents
  the adaptive arm discovered;
- a molecule both arms pick is docked once and its single score goes to both, so docking
  noise cannot manufacture a difference on shared picks.

## What is deliberately absent

- **No `Q_3`.** The receding-horizon branch value is implemented
  (`t4_fiber_expansion.branch_value`) and unused, so that one-step selection is tested
  before a second mechanism shares the result. `h=1` here means one-step SELECTION; the
  program itself still carries up to three modules, and multi-region options are first
  class.
- **No generic docking surrogate.** The reward model describes the DECISION -- parent
  context, regions, program family, retained/created/deleted roles, margins -- not
  arbitrary molecules. A whole-space surrogate was measured at 0.498 against 0.404 for
  random on held-out JAK2 siblings.
- **No route information in the decision loop.** No JAK2 teacher route, endpoint or next
  edit is read. Routes are frozen into the reference process only.

## Reading a result honestly

A flat or negative result is informative here, and `PAIRED.md` separates the three causes
the plan named:

- **generation** -- was anything better than the incumbent in the pool at all;
- **reward model** -- where the arm's picks landed in the REALIZED ranking of that round;
- **acquisition** -- `corr(predicted endpoint, realized)`; near zero means selection was
  effectively blind whatever the rule says.

Any champion must be confirmed at replicate docking seeds before it is quoted, as the
-11.32 was (four seeds, sd 0.04).
