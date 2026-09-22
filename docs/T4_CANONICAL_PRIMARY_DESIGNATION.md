# T4 canonical controller — prospective primary designation

**Written and committed BEFORE any cell of this experiment has been launched or any
result observed.** Its only purpose is to fix, in advance, which arm is the claim and
what each outcome would mean, so that the answer cannot be chosen after the fact.

Status: PROSPECTIVE. Authority: this file plus the three sealed arm contracts
`configs/t4_canonical_shared_controller_{a,b,c}_v1.json`.

---

## The designated primary

**Arm C is the prospective primary T4 controller.**

Arm C is: one frozen controller configuration carrying

1. shallow/local executable programs (`shallow`),
2. the existing coordinated structural program families (`structured`,
   `anchored_replacement`), at a fixed shared allocation,
3. state-dependent adaptive support expansion driven only by a generic search
   statistic, and
4. the same FiberControl archive and docking-feedback loop as every other arm,

run unchanged over all 30 cells (15 published T4 held-target seeds × δ ∈ {0.6, 0.4}).

The claim it is intended to earn:

> COMPOSE uses a single frozen controller across T4, combining local executable
> programs, coordinated structural programs, and state-dependent adaptive support
> expansion within the same validity-closed molecular process.

## What is frozen

Target molecule, receptor/oracle and δ are benchmark INPUTS. Everything else —
proposal vocabulary, draw counts, region law, completion law, batch size, parent
count, exploration rate, expert floor, value penalty, docking seed, expansion ladder
and expansion trigger — is a module constant of
`src/compose_v4/experiments/t4_canonical_controller.py`, identical on every cell.

State-dependent routing is allowed. **Target-name routing is not**, and this is
enforced structurally rather than by convention:
`assert_no_target_name_routing` parses every runtime module and fails if a target
token reaches the test of an `if`, a `while`, a conditional expression, an `assert`,
a comparison, a boolean operator or a comprehension guard. `tests/test_t4_canonical_controller.py`
runs it over the whole runtime closure and carries mutation controls proving it can fail.

## The ablation, and what it isolates

| arm | proposal vocabulary | adaptive rule | cells | charged calls |
|---|---|---|---|---|
| A `local` | `shallow` | no | 15 @ δ=0.6 | 3,750 |
| B `coordinated` | `shallow`, `anchored_replacement`, `structured` | no | 15 @ δ=0.6 | 3,750 |
| C `adaptive` | `shallow`, `anchored_replacement`, `structured` | **yes** | 30 @ δ∈{0.6,0.4} | 7,500 |

Total authorized: **15,000 charged docking calls.**

FiberControl, the archive, the executor, initialization, per-cell RNG seeds, budgets
and endpoint constraints are IDENTICAL across arms. Therefore

- **A vs B isolates coordinated structural programs.**
- **B vs C isolates adaptive use of structural support.**

Arm C's candidate pool on any round is a strict superset of the pool arm B would have
had on that round: the expansion's endpoints are UNIONED with the ordinary pool, never
substituted for it.

## The adaptive trigger, declared in advance

Arm C expands whenever a round's ordinary proposal produced fewer than
`EXPANSION_TRIGGER_MIN_ELIGIBLE = 4` distinct eligible endpoints. The trigger reads
one generic search statistic and nothing else — not the cell, not the target, not the
round index, not the incumbent score. At a threshold of 1 it would reduce exactly to
the shipped "the pool came back empty" condition; 4 additionally covers the marginal
rounds that survive on a single lucky draw (a control cell has been measured surviving
round one on 3 eligible endpoints out of 7,420 produced).

Expansion spends CPU and never an oracle call. It is bounded on every axis:
ladder `[960, 1920, 3840]`, at most 6,720 extra draws per event, 5,400 s wall per
event, stopping at 4 new distinct eligible endpoints.

## What each outcome would mean

Declared now, so that none of them can be reinterpreted later.

- **C completes all 30 cells with no `candidate_exhaustion`.** The single frozen
  controller covers the panel. This is the result the claim needs; it is about
  COVERAGE and is independent of whether C's scores beat any comparator.
- **C exhausts on some cells.** The adaptive rule is insufficient, and the exhausted
  cells are a scoped negative to be reported cell by cell with the bound that stopped
  the expansion. This is a publishable negative, not a reason to add a per-cell rescue.
- **B ≈ A.** The coordinated structural lanes contribute nothing measurable at this
  budget on this panel. That is a real negative about the coordinated channel and must
  be reported as the headline of the A/B comparison, not buried.
- **C ≈ B on cells where B did not exhaust.** Expected, and not a failure: the
  expansion is designed to be INERT on healthy rounds. The B-vs-C comparison is
  informative only on rounds that actually triggered, and the trigger count per cell is
  logged for exactly this reason.
- **C worse than B anywhere.** The expansion perturbs healthy search. This would
  falsify the "inert on healthy rounds" design intent and would be reported as such.

## Pre-declared reporting rules

1. **Docking is not reproducible across runs.** `qvina02` is seeded, `obabel --gen3D`
   is not; one T4 seed molecule has been measured at −7.5 / −8.30 / −8.8 across three
   runs. All arm comparisons here are WITHIN this run. Per-cell margins against
   `diagnostics/T4_FROZEN_RESULT_v1.*` or against a published baseline are NOT sound at
   the per-cell level and will not be quoted as such.
2. **No splicing.** The pre-existing mixed-provenance panel
   (`diagnostics/T4_FROZEN_RESULT_v1.*`) is preserved unmodified as the DEVELOPMENT
   EVIDENCE that motivated adaptive support expansion. No historical rescue or
   support-expansion score enters the canonical table.
3. **Seed-only rows are not successes.** A cell whose best molecule is its own docked
   seed is reported as producing nothing.
4. **The reconciled charged-call count comes from round locks**, which are immutable,
   not from checkpoints, which are overwritable. A lock above the last checkpointed
   round is an interrupted round: its queries are debited in full and never re-docked.
5. **Arms are reported together.** A is not quoted without B and C.

## Deliberate exclusions

- **The route-distilled `route_complete_region` expert is excluded.** Every checkpoint
  of it in this repository is a leave-one-target-out fit over the 77 locked T4 routes
  (`split_audit.split == "leave_one_target_out"`). Using one across the panel trains on
  the benchmark's own answers for four targets of five; using a different one per
  target IS target-name routing. There is no leave-all-targets-out checkpoint, because
  removing all five targets removes the corpus. The coordinated role is therefore
  filled by the task-independent `structured` sampler.
- **No protonation-specific channel.** The cell that motivated one was later measured
  to be reachable by ordinary excision.
- **No new macro families.** The controller uses the existing validity-closed executor
  and the existing program families only.
