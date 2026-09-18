# FiberControl: the autonomous development lane

What lives here, what each artifact is allowed to claim, and what it is not.

## The question this directory exists to answer

    Given a rich pool of exactly feasible COMPOSE program endpoints, does docking
    feedback improve which of them we choose to dock next?

Not "can COMPOSE reach -11-class JAK2 chemistry" -- that is settled and it is
route-assisted (`../t4_jak2_contrastive/DELTA_LANES.md`, -11.32 +/- 0.04 against an
archive best of -11.00). This lane asks whether reward-adaptive control finds comparable
chemistry *without* the route.

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
