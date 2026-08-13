# Workstream C — Pathwise Constraints: Protocol

**Status:** `SMOKE_HELD_IN` — stage A executed, **G1 FAILED**, lane stopped
**Branch:** `codex/compose-pathwise-constraints`
**Held-out data opened:** NO

> **OUTCOME (2026-08-12).** The premise gate failed. Of 42 rollouts on the
> unconstrained support, 19 broke the protected motif and **0 recovered**;
> `endpoint_valid_path_invalid` was **0/6** sources and **0/19** endpoint-valid
> rollouts against a preregistered 0.10 threshold. Motif destruction is
> absorbing at H = 6, so endpoint validity implies path validity and
> endpoint-only filtering is sufficient here.
>
> **The protocol below was NOT modified after seeing that result.** The motif
> rule, horizon, label semantics and panel are exactly as frozen before the
> run. See `DECISION_LOG.md` for the explicit list of tuning moves considered
> and rejected. Full numbers in `HANDOFF.md`.

---

## Claim

> Because every COMPOSE state is a complete molecule and every transition is
> executable, an exact labeled-subgraph requirement can be imposed on the
> support of the entire controlled process. Endpoint-only filtering can return
> a valid final molecule after traversing states the requirement forbids;
> support masking makes those states unreachable by construction, and control
> can then optimise inside the reduced feasible set.

## Non-claims

- Not that constrained control beats unconstrained control on the objective.
  The mask can only shrink the reachable set; the interesting number is what
  it costs, not that it wins.
- Not that the protected motif is a medicinally meaningful pharmacophore. It
  is a mechanically derived ring system. The claim is about the *machinery* of
  pathwise constraint, not about chemistry knowledge.
- Not that future-aware control beats greedy under the mask. See
  "What is not a measurement" below — that comparison has a guaranteed sign.
- Nothing about held-out generalisation. This lane stops before any held-out
  panel.

---

## The constraint

At every committed state `x_t`, `t = 0..H`:

> the protected motif `M` embeds into `x_t` as an exact atom/bond-labeled
> subgraph.

- **Atom label:** atomic number **and** aromaticity flag **and** formal charge.
- **Bond label:** aromatic, or the exact bond order.
- **Embedding:** subgraph monomorphism (the standard reading of `M ⊆ x_t`),
  not necessarily induced. Adding a bond between two motif atoms does not
  remove the motif; removing one does.
- **Outside the motif:** unconstrained. Substituting or deleting non-motif
  atoms is exactly the room left to act.
- **Unparseable state:** counted as a violation. A state COMPOSE cannot read
  is not a state that demonstrably contains the motif.

Explicitly **not** fingerprint similarity, **not** Tanimoto, **not** endpoint
recovery, **not** Murcko scaffold string equality.

Implementation: `src/compose_v4/experiments/pathwise_constraints.py`.
`Chem.MolFragmentToSmarts` was rejected as the query builder: it emits bare
`[#7]` primitives, so it drops the formal charge and would silently let a
charge-changing edit pass. The module writes every primitive explicitly and
self-checks each derived query against the molecule it came from.

## The motif rule — `largest_ring_system_v1`

The largest fused ring system of the **source**: connected components of the
ring-bond graph, taken with exact labels. Ties broken by atom count, then ring
count, then the sorted atom-index tuple, so the rule never depends on RDKit
iteration order.

The rule reads the source molecule and nothing else — no trajectory, no
successor set, no objective value, no arm outcome.

---

## Frozen inputs

| Object | Identity |
|---|---|
| Process-V2 chemistry / kernel | `canonical_successor_result`, `TIME_POINT=0.5`, `slots=48` |
| `R_theta` | `runs/run_v2_01` selected checkpoint on volume `compose-v4-artifacts`. **Not retrained, not modified.** |
| Goal language | `B = P AND D`, verbatim from `modal_apps/retarget_intervention_app.py` |
| Normalizers | `diagnostics/retarget_goal_language_normalizers.json` `187d1ccc…` |
| Oracle | `artifacts/oracles/drd2_svm_v1` npz `7c9224c1…` |
| Pool | `training_source_keys` (held-in, 96 094) from `…reserve_ids.json.gz` `ba9270fa…` |
| Panel | `diagnostics/pathwise_constraints_smoke_panel.json`, `panel_sha256 8b47a0e4…` |

The goal is **borrowed, not invented**. `B = P AND D` is used rather than `D`
alone because the committed retargeting calibration records developability as
a ceiling (greedy 28/30, binary headroom 0), while potency at 0.5 was reachable
on 10/30. The local census agrees: 32.6% of held-in sources already satisfy `D`
at step zero but only 2.3% satisfy `P`. Potency is the binding term and is
where the headroom lives.

---

## Panel construction

Held-in only. `training_source_keys`, shuffled under seed **20260813** —
deliberately different from the retargeting cohort's 20260812, whose first 57
shuffled entries are already spent — then the first `N` eligible sources in
scan order. Nothing ranks or prefers a source.

**Eligibility (source-only, outcome-independent):**

| Criterion | Value |
|---|---|
| parses | required |
| heavy atoms | 18–38 (reused from the frozen retargeting band) |
| motif atoms | ≥ 6 |
| motif fraction of molecule | 0.20–0.70 |
| heavy atoms outside the motif | ≥ 8 |
| already satisfies `B` at step zero | excluded |
| member of the retargeting cohort | excluded |

### The criterion deliberately NOT applied

The workstream brief lists "the frozen kernel offers both motif-preserving and
motif-destroying legal successors" as an eligibility condition. **It is not
used here.** That property is the numerator of the vacuity gate. Selecting
sources on it would guarantee a non-vacuous mask by construction and convert
the headline measurement into a definition — the exact failure shape this
project hit three times. It is measured and reported per source instead.

`tests/test_pathwise_constraint.py::test_eligibility_never_reads_a_successor_set`
pins the function signature so a future edit cannot quietly reintroduce it.

---

## Arms

All arms share the same frozen `R_theta`, source set, goal, horizon `H=6`, and
one enumeration cache per source. **Masking costs no kernel calls**: the kernel
returns the full legal support and the mask is a predicate on the returned
keys, so a masked and an unconstrained arm visiting the same state pay for one
enumeration between them.

### Stage A — no lookahead rollouts

| Arm | Support | Policy |
|---|---|---|
| `unconstrained_greedy` | full | greedy on `u_B` |
| `endpoint_only` | full | 6 stochastic goal-directed rollouts, keep motif-valid **endpoints**, return best `u_B` |
| `pathwise_greedy` | masked | greedy on `u_B` |
| `pathwise_stochastic` | masked | 6 stochastic goal-directed rollouts, best `u_B` |
| `mask_only_sampling` | masked | sample `R_theta`, **no goal** |

### Stage B — remaining-budget lookahead

| Arm | Support | Policy |
|---|---|---|
| `unconstrained_verified` | full | argmax `V_G` under strict improvement, re-plan |
| `pathwise_verified` | masked | same, with every lookahead rollout also masked |

The stochastic policy samples `R_theta` restricted to the top-`SHORTLIST=3` by
`u_B`. `SHORTLIST=1` **is** greedy, so the stochastic arms degenerate to the
greedy arms rather than forming a separate tunable policy family.

### Budget asymmetry, stated in the direction it cuts

`endpoint_only` gets 6 rollouts and a best-of-6 selection; `pathwise_greedy`
gets one trajectory. That asymmetry **favours the arm this workstream argues
against**, which is the safe direction. `pathwise_stochastic` exists so the
utility comparison also has a strictly budget-matched form: same rollout count,
same policy, differing in exactly one thing — the mask.

---

## Primary metrics

1. **`endpoint_valid_path_invalid`** — trajectories whose endpoint passes the
   motif check but which passed through a state the constraint forbids.
   Reported on two denominators: per-source on `unconstrained_greedy`, and over
   every endpoint-valid rollout `endpoint_only` was willing to return.
2. **`removed_fraction`** — the share of legal successors the mask deletes,
   measured along **unconstrained** states so the mask is not scored on states
   it selected itself.
3. **Paired terminal utility, `pathwise_stochastic` − `endpoint_only`** — the
   price of the guarantee at matched budget and matched policy.

## Secondary metrics

Feasible-trajectory completion rate; endpoint motif validity per arm; `B`/`P`/`D`
success per arm; marginal kernel calls per arm in run order (the shared cache
means a later arm pays less, so the order is recorded).

---

## What is NOT a measurement — the explicit gate

This project has been bitten three times by statistics whose sign was fixed
before any data existed (`docs/RETARGETING_SAME_PREFIX_DESIGN.md`,
"Three instrument defects in one experiment"). Two quantities in this lane have
that shape and are **barred from the results table**:

### 1. "The pathwise arms had zero violations"

Definitional. A masked arm cannot commit a violating state; that is the
construction, not a finding. It is checked as a **bug detector** — a non-zero
count sets the shard to `INVALID_INSTRUMENT` and voids the run — and is never
given a denominator, an uncertainty, or a p-value.

### 2. `pathwise_verified` ≥ `pathwise_greedy`

Greedy's action is always in the verified candidate set and strict improvement
never commits a lower `V_G`, so by induction the verified landing cannot be
worse on the lexicographic utility. Binary success is a threshold of the worst
margin, so **binary success inherits the guaranteed sign too** — this is why
the retargeting lane's "greedy 28/30, verified 28/30, headroom 0" was a ceiling
rather than a null.

Admissible from that comparison, and only these:

- the **magnitude** of the utility gap, expressed against the typical movement
  the objective shows over the horizon;
- the **top-1 disagreement rate** — non-circular, because it says the
  controller would *act* differently;
- **binary headroom over the denominator of sources where `pathwise_greedy`
  actually failed**. If that denominator is 0, the result is a **CEILING, not a
  null**, and must be reported as such.

### The test that enforces this

`tests/test_pathwise_analysis.py::test_analyser_FAILS_the_vacuity_gate_when_nothing_ever_violates`
drives the analysis script over a synthetic graph in which the motif is never
violated and asserts the gate returns **FAIL**. A gate that can only say PASS
is not a gate.

---

## Gates and stop rules

| Gate | Rule | Consequence |
|---|---|---|
| **G0** mask integrity | any committed violation on a masked arm | run is `INVALID_INSTRUMENT`; fix the mask, discard results |
| **G1** constraint non-vacuous | `endpoint_valid_path_invalid` < 0.10 on **both** denominators | **STOP.** Endpoint-only filtering is already sufficient; there is no trajectory-level claim |
| **G2** room to act | mean `removed_fraction` ≤ 0.02, or ≥ 0.95 | **STOP.** Vacuous mask, or infeasible task under this support |
| **G3** feasible improvement | no pathwise arm improves `u_B` over the source on any source | **STOP.** No feasible path can improve the objective |
| **G4** distinguishable | `endpoint_only` and `pathwise_stochastic` land identically everywhere | **STOP.** No trajectory-level claim |
| **G5** planning subclaim | future-aware adds nothing under the mask | keep the pathwise-guarantee claim, **drop** the planning-under-constraint subclaim |

**Anti-tuning rule.** If G1 fails, the motif rule is *not* changed until one
bites. The brief is explicit: "Do not keep changing motifs until one makes the
desired arm win." A failed G1 is a reportable result about the frozen kernel —
its legal support largely preserves ring systems — not a prompt to re-roll.

---

## Allowed calibration

- Eligibility constants may be set from the **source-only** census yield. No
  successor enumeration, trajectory, or arm outcome enters that census, so it
  cannot bias the gate. (In fact the constants were fixed before the census was
  run and gave a 74.9% yield, so none were adjusted.)
- The horizon may be checked once on held-in smoke if `H=6` proves too short to
  expose any violation. One check, recorded in `DECISION_LOG.md`.

## Forbidden adaptations

- Changing the motif rule until an arm wins.
- Retraining or modifying `R_theta`, the kernel, or the operators.
- Adding kernel-derived criteria to source eligibility.
- Opening any held-out panel.
- Reporting either definitional quantity above as a finding.

---

## Known instrument risks and their detectors

| Risk | Detector | Outcome on the stage-A run |
|---|---|---|
| Local RDKit (2025.09.6) perceives aromaticity differently from the Modal image (2024.3.5), silently widening or narrowing the protected pattern | the app re-derives the motif from the canonical start key and voids the shard if the geometry disagrees with the frozen panel | **PASS on 6/6** — no drift |
| Canonicalisation moves the source out of its own motif | `preserves_motif(smarts, start_key)` asserted before any arm runs | PASS on 6/6 |
| The mask leaks | G0, checked on every masked arm's realised trajectory | **PASS — 0 leaks** |
| An arm is truncated mid-run by the cost cap | the cap is checked **between** arms only, so an arm is complete or absent | never approached (max 58 of 360) |
| The app cannot be launched because `modal run` uses an interpreter without RDKit | two AST tests forbid chemistry imports at app module scope | caught before any spend; fixed |
| Two panel sources carry unusual valences (`[PH]`, `[SH4]`) | not removed — outcome-independent rule | source 5 (`[SH4]`) is one of the two violators |

---

# ADDENDUM — Reversible-constraint feasibility census (salvage lane)

**Status:** `DESIGN_ONLY` at the time of writing. **This section was committed
BEFORE any family was measured.** Commit order is the evidence.

**Scope, as authorised:** a feasibility census only. No controller run, no
stage B, no held-out data. The ring-system negative is CLOSED and its
threshold, horizon, ring definition and source set are not reopened.

## Why a second family at all

The ring-system premise failed for a *specific* reason: a broken fused ring
system was never rebuilt, so violation was **absorbing** and endpoint validity
implied path validity. The pathwise/endpoint distinction only has teeth when
violation is **reversible** — when a trajectory can leave the feasible set and
return. That is a property of the constraint family, not of COMPOSE, so it is
worth one bounded check across families chosen for reversibility.

## The three families, declared in advance and ranked in advance

Implemented in `src/compose_v4/experiments/pathwise_reversible_families.py`.

| Rank | Family | Constraint at every committed state `x_t` |
|---|---|---|
| **A** | `A_undesired_motif` | no undesired reactive group present |
| **B** | `B_physchem_corridor` | `cLogP(x_t)` inside the frozen held-in interquartile range |
| **C** | `C_size_corridor` | heavy-atom count inside the frozen panel eligibility band |

**Precedence rule, fixed now:** if more than one family passes, advance the
**first in the order A, B, C** — never the one that looks most favourable.

### Where every threshold comes from

No number below was chosen by me. Each is read from an artifact that was
already frozen for another purpose.

- **A** — a fixed list of 14 standard medicinal-chemistry reactive/structural
  alerts (acyl halide, sulfonyl halide, aldehyde, anhydride, Michael acceptor,
  epoxide, aziridine, nitro, azide, isocyanate, thiol, peroxide, hydrazine,
  N-nitroso). Written from chemical motivation. Not derived from, filtered by,
  or checked against any COMPOSE trajectory.
- **B** — `[p25, p75]` of cLogP loaded **at runtime** from
  `diagnostics/retarget_goal_language_normalizers.json`, whose own status is
  `HELD_IN_NORMALIZERS_NO_THRESHOLD_SELECTED`. Reading it at runtime rather
  than transcribing it means the corridor cannot be quietly nudged. The
  interquartile range is the summary that already existed in that file; it was
  not picked for how many states it would flag.
- **C** — `HEAVY_ATOM_BAND = (18, 38)`, reused verbatim from the frozen
  eligibility band already used by the retargeting cohort and by this lane's
  own panel. No new number at all.

**Choosing a corridor because it produces violations is the failure mode this
census guards against.** The defence is that the corridors are pre-existing
canonical summaries, loaded not typed, and committed before measurement.

## Metrics, per family, on held-in unconstrained trajectories

1. mid-path violation frequency;
2. **fraction of violating paths that RETURN to a feasible endpoint** — the key
   number, and the one that was 0 for the ring system;
3. absolute count of endpoint-valid / path-invalid trajectories;
4. median legal-support retention under the mask;
5. frequency of mask-empty states.

Metrics 1–3 are computable from the **already-committed stage-A shards** (42
unconstrained-support trajectories, 294 committed states) with **no new
compute**. Metrics 4–5 need enumerated successor sets, which the shards do not
store, and would require a new Modal run.

## Viability criteria, fixed before the census runs

A family PASSES only if **all** hold:

| # | Criterion | Threshold |
|---|---|---|
| V1 | violation is non-vacuous | at least one violating trajectory, and not every trajectory violating |
| V2 | **violation is reversible** | **≥ 10%** of violating trajectories return to a feasible endpoint |
| V3 | enough events to study | **≥ 20** endpoint-valid / path-invalid trajectories |
| V4 | mask leaves room | median support retention meaningful, mask-empty states rare |

V4 is not evaluable from the committed shards. A family that clears V1–V3 is
reported as **CONDITIONAL PASS pending V4**, with the bounded run that would
settle it costed — never as a full pass.

## If no family passes

**Pathwise constraints leave the main paper entirely.** The ring-system
negative goes to the appendix as a mechanistic finding about the frozen
kernel's legal support. **No fourth or fifth predicate is searched.** That
bound is the point of this lane.

---

# ADDENDUM 2 — Stage A2: cLogP corridor prevalence / viability

**Status:** `DESIGN_ONLY`. **Committed, NOT LAUNCHED.**
**Scope:** family B only. No other family. If A2 fails, no fourth predicate.

> **Family B was selected for follow-up AFTER the three-family feasibility
> census because it alone exhibited the intended reversible-excursion
> mechanism. Stage A2 is developmental follow-up, not independent confirmation
> of the phenomenon.**

The stage-A verdict remains **FAIL** and is not retroactively revised.

## What A2 asks, and why it is a different question

The 20-event bar answered "did this pool yield 20 events". With only 21
source-feasible trajectories it demanded the phenomenon appear in nearly every
one. The scientific question behind it was:

> are recoverable excursions frequent enough **and spread across enough
> distinct molecules** to support a causal, source-level pathwise-control
> experiment — and does the corridor mask leave a controller anything to act on?

## Frozen, unchanged from stage A

| Object | Value |
|---|---|
| corridor | `[2.3689, 4.4522]` — held-in cLogP IQR, **read at runtime**, never transcribed |
| horizon | `H = 6` |
| `R_theta` | the same frozen checkpoint |
| violation / recovery | the same definitions (`state_is_feasible`, `audit_trajectory`) |
| rollout law | sample `R_theta` within the top-3 by `u_B`, goal `B = P AND D` |

**No threshold is adjusted.**

## Sample design — the source is the independent unit

**12 NEW held-in sources × 6 stochastic rollouts.** New sources, not more
rollouts on the same six: `diagnostics/pathwise_a2_panel.json`,
`panel_sha256 79845e8d…`, seed 20260814, disjoint from the stage-A 6, the
retargeting 30, and the held-out reserve (all asserted in code and in tests).

**72 trajectories are 12 observations.** Rollouts from one molecule are
repeated measures. Every headline figure is reported per source with a
**source-clustered bootstrap** (resample sources, keep their rollouts
together). Trajectory-level incidence is reported because the criteria are
stated in those terms, but it is **never given an interval of its own** — a
trajectory-level interval would be about √6 too narrow and would make a
phenomenon carried by two molecules look like a population fact.

### One eligibility criterion is new, and it is applicability, not tuning

A2 requires `cLogP(x_0)` **inside** the corridor. A source starting outside can
never "leave and return", so it cannot produce the event being counted. In
stage A this was not required and it cost half the sample: **21 of 42
trajectories, and 3 of 6 sources, started outside the corridor** and dropped
out of the denominator after the fact. The corridor itself is unchanged; this
reads `x_0` only; it makes every A2 source contribute.

Panel headroom is reported so a reviewer can see the panel is not stacked
against the corridor edges (observed range: 0.01–1.86 below, 0.23–2.07 above,
on a corridor 2.083 wide).

## Metrics

Trajectory level: violation incidence; return rate among violators; total
endpoint-valid / path-invalid events.

**Source level (primary):** number and fraction of sources with ≥1 event;
events per source; largest single-source share; all with clustered bootstrap.

Excursion geometry: **depth and duration**. Duration is the count of
consecutive states outside the corridor — a multi-step excursion is budget
spent inside a forbidden region, which is exactly what endpoint-only filtering
cannot see.

**V4, which stage A could never evaluate:** legal-support retention under the
corridor mask, and mask-empty frequency, measured on every state the
*unconstrained* rollouts visited.

## Criteria, fixed before the run

| # | Criterion | Threshold |
|---|---|---|
| **V3** | event yield | ≥ **20** endpoint-valid / path-invalid events (retained verbatim) |
| **V4a** | mask leaves a choice set | median support retention ≥ **0.10** |
| **V4b** | mask rarely empties | mask-empty states ≤ **5%** of visited states |
| **V5a** | **source spread** | ≥ **1/3** of sources (≥ 4 of 12) have ≥ 1 event |
| **V5b** | no single molecule dominates | largest source ≤ **50%** of all events |

### Justifying V5a from downstream feasibility, not from current numbers

**The rule, before the derivation:**

> **Stage A and Stage A2 do not share an eligibility denominator. Stage A2
> restricts to sources for which excursion-and-return is logically observable;
> therefore Stage-A source prevalence must not be used as an estimate of the A2
> pass criterion.**

Anyone comparing "2 of 6" against "1/3" is comparing two different
denominators. The threshold below is derived without reference to either.

**Derivation.** The eventual causal experiment is source-level: per source,
endpoint-only handling versus pathwise masking. A source with no possible
endpoint-valid / path-invalid excursion is **non-informative** — both arms
return the same molecule and it contributes nothing to the contrast.

Using this project's own panel sizing (`RETARGETING_SAME_PREFIX_DESIGN.md`:
20–24 development, 60–80 held-out confirmatory), and requiring ~20 informative
sources for a stable paired estimate:

- 60-source confirmatory panel → `p ≥ 20/60 = 1/3`;
- 24-source development panel → `p ≥ 8/24 = 1/3`.

Both routes land on **1/3**, so that is the threshold. Neither consults a
stage-A number.

**Disclosure, because a reviewer will check the coincidence.** Stage A's
family-B events came from 2 of 6 panel sources = 0.333, numerically equal to
this threshold. Per the rule above the two are not comparable: stage A did not
require `x_0` inside the corridor and only 3 of its 6 sources were
source-feasible, so half its panel could never have exhibited the phenomenon.
Under A2's own definition — where every source is source-feasible by
construction — the stage-A analogue is **2/3 = 0.667**, twice the threshold.
V5a is set at half the previously observed value, not at it.

### V5b

With ≥ 4 event-sources, one molecule supplying more than half the events would
mean the clustered bootstrap is effectively driven by `n ≈ 1`. The cap makes
V5a robust to a single hyper-productive outlier. (Stage A's top source held
exactly 0.500, but with only 2 event-sources that statistic carries no
information.)

### V4a / V4b cannot have been fitted

**No cLogP-mask measurement exists anywhere in this lane.** Stage A masked on
the ring motif, not the corridor. These two thresholds were therefore set with
no corresponding data in hand, from what a downstream controller needs: a
median state that still offers a real choice set, and a dead-end rate low
enough that the experiment measures control rather than dead-end accounting.

## A2 must be able to fail cleanly, and here is how it does

Any one of these closes pathwise constraints **for good**:

- **too few excursions** → V3 fails;
- **excursions concentrated on one or two molecules** → V5a or V5b fails;
- **a mask that strangles the support** → V4a fails;
- **a mask that empties the support** → V4b fails.

Each failure mode is exercised by a test in
`tests/test_pathwise_a2_prevalence.py`, which drives the analysis script to a
FAIL verdict on synthetic shards for every criterion separately. A gate that
can only say PASS is not a gate.

## Costed plan — NOT AUTHORISED TO RUN

```bash
PYTHONPATH=src:. MODAL_PROFILE=rahul-94866 \
  modal run --detach modal_apps/pathwise_corridor_prevalence_app.py --sources 12
# verify `modal app list` shows `ephemeral (detached)` before walking away
```

| | Estimate |
|---|---:|
| sources | 12 |
| rollouts | 6 per source (72 trajectories, **12 observations**) |
| kernel calls / source | ~35 (range 25–50) |
| cost basis | measured stage A: 123 s startup + 14.3 s per call |
| **container-hours** | **≈ 2.1** (range 1.6–2.8) |
| wall time | ~11 min at 12 parallel containers |
| per-source circuit breaker | 200 kernel calls |

Cheaper than stage A per source because A2 has no lookahead arms. The mask
census is free: it reads states the rollouts already enumerated.

## Why this run is deferred

Every A2 trajectory would be generated under the frozen `R_theta` that lane 1
is currently deciding whether to discard. Spending ~2 container-hours now risks
measuring a model that is about to be replaced. **Pathwise is upside; lane 1 is
load-bearing.** The protocol and runner are committed ready to execute and are
not launched.

---

# ADDENDUM 3 — Stage B: corridor-constrained potency control

**Status:** `DESIGN_ONLY`. **Committed, NOT LAUNCHED.**
**Panel:** `diagnostics/pathwise_stage_b_panel.json`, `panel_sha256 2d993508…`

## The question

> When terminally acceptable trajectories can pass through forbidden
> intermediate states, what is the **cost and benefit** of enforcing the
> constraint throughout molecular evolution?

**Not** "can the mask achieve zero violations". That is guaranteed by
construction and stays barred from the results table, checked only as a bug
detector.

**Task:** increase DRD2 potency subject to `2.3689 ≤ cLogP(x_t) ≤ 4.4522` for
all `t`, `H = 6`.

**Objective provenance:** DRD2 potency alone (goal `P`), frozen in the
retargeting lane. Chosen because it is independently motivated and already
frozen, **not** because it maximises the pathwise effect. No objective search
was performed.

## The four primary arms — a 2×2

| | navigate myopically | navigate with lookahead |
|---|---|---|
| **enforce at `t=H` only** | `endpoint_greedy` | `endpoint_verified` |
| **enforce at every `t`** | `pathwise_greedy` | `pathwise_verified` |

All four run the **same code path** (`build_stage_b_arm`), dispatched from a
declared `(mask, endpoint_only, controller)` triple. A terminal-cost difference
between `endpoint_greedy` and `pathwise_greedy` therefore cannot be an artefact
of two separately written policies — the failure that let the retargeting
lane's "verified arm was secretly greedy" survive two runs.

`unconstrained_potency` is recorded as **DESCRIPTIVE ONLY**, never enters a
causal contrast, and is the arm whose visited states define the retention
census. **No fifth causal arm.**

## Estimands

### PRIMARY — hidden-path RATE (unconditional)

> `P( x_H ∈ C  AND  ∃t < H : x_t ∉ C )`

**Every eligible source stays in the denominator.** It cannot collapse.
Answers: how often does endpoint-only optimisation actually produce an endpoint
we would accept despite an invalid trajectory?

### SECONDARY — hidden-path FRACTION among accepted endpoints

> `P( ∃t < H : x_t ∉ C  |  x_H ∈ C )`

The intuitive reading, but **its denominator is controller-dependent**: an arm
that rarely delivers an acceptable endpoint can post a dramatic rate on a
handful of trajectories. **The denominator is reported alongside it every
time.** `test_conditional_fraction_can_be_dramatic_on_a_tiny_denominator`
constructs exactly that case (2/2 = 1.0 conditional against 2/12 = 0.167
unconditional) and checks the report exposes both.

**Both are reported separately for greedy and verified endpoint-only control
and are never pooled**, because the controller changes which endpoints become
acceptable and pooling would mix two different denominators.

Both can be zero. The ring-motif family returned exactly zero.

### TERMINAL COST — controller parity

- `Δ^G = U_P(pathwise_greedy) − U_P(endpoint_greedy)`
- `Δ^V = U_P(pathwise_verified) − U_P(endpoint_verified)`

Each pair shares a controller and a source and differs **only** in where the
constraint is enforced, so the sign is free.

**Three outcomes are declared informative in advance. None is a failure:**

1. **little or no potency cost** → strongest pathwise result;
2. **moderate potency cost** → still meaningful; the constraint genuinely
   restricts molecular evolution;
3. **large cost or frequent support collapse** → pathwise control works
   formally but is practically too restrictive for this corridor — a real
   finding.

**There is no expectation that pathwise beats endpoint-only on potency.**
Framing that as the goal would be suspicious, since the mask can only shrink
the reachable set.

### FUTURE-AWARE — guaranteed sign, therefore not a primary claim

`U_P(pathwise_verified) − U_P(pathwise_greedy)`. Verified contains greedy's
action and overrides only on strict improvement, so the direction is fixed
before any molecule exists.

**No sign test. No "verified never loses".** Reported: effect size,
top-1 disagreement, and **constrained-performance recovered** (planning gain as
a fraction of the potency the mask cost under greedy parity, undefined and
omitted where the mask cost nothing). Binary headroom is given over the
denominator of sources where `pathwise_greedy` actually failed; **headroom 0
over a denominator of 0 is a CEILING, not a null**, and the subclaim closes
exactly as the retargeting lane closed its own.

## Panel — 24 new held-in sources

Seed 20260815. Disjoint from the 6 stage-A, 12 A2, and 30 retargeting sources,
and from `reserve_source_keys`; asserted in code and in tests.

**Eligibility is source-only and EXCURSION-BLIND:** parses · heavy atoms in the
frozen band · `cLogP(x_0)` inside the corridor · potency headroom at step zero.
**Nothing about whether a molecule previously produced an excursion.**
Selecting on excursion propensity would make the primary estimand true by
construction; `test_stage_b_eligibility_is_excursion_blind` pins the signature.

**The source is the independent unit.** All intervals are source-clustered
bootstraps.

## Support-tight sources — predeclared before the run

A2 found per-source median retention spanning **0.048 to 0.917**, which a
pooled median of 0.573 conceals. Therefore, fixed now:

- a source is **SUPPORT_TIGHT** when its median retained legal-successor
  fraction is **< 0.10** — the viability threshold already in use, **not a new
  number**;
- retention is measured along the **DESCRIPTIVE** arm's states, so the
  classification cannot depend on any constrained arm's outcome;
- **all 24 sources stay in the primary intention-to-treat analysis**;
- the **fraction of support-tight sources is reported**;
- a **predeclared sensitivity analysis excluding them** is reported alongside;
- **the threshold is never redefined after seeing which arm suffers.**

### Source-level diagnostics, preregistered

- retained-support fraction per source;
- **mask-empty rate per source**, not pooled;
- number of feasible actions at each step;
- whether a terminal failure is attributable to `no_legal_successor`,
  `empty_after_mask`, or `controller_stopped_with_support_available`.

## Costed plan — NOT AUTHORISED TO RUN

```bash
PYTHONPATH=src:. MODAL_PROFILE=rahul-94866 \
  modal run --detach modal_apps/pathwise_stage_b_app.py --sources 24
# verify `modal app list` shows `ephemeral (detached)`
# do NOT wrap the client in `timeout`: killing it cancels the detached job
```

| | Estimate |
|---|---:|
| sources | 24 |
| arms | 4 causal + 1 descriptive |
| kernel calls / source | ~120–150 (two lookahead arms dominate) |
| cost basis | measured: 123 s startup + 14.3 s per call |
| **container-hours** | **≈ 12–15** |
| wall time | ~45 min at 24 parallel containers |
| circuit breaker | 400 calls/source |

**This is 9–11× the A2 spend** and the largest run in this lane. The panel is
source-sharded and the driver skips committed shards, so it may be authorised
in halves (`--start 0 --sources 12`, then `--start 12 --sources 12`) with no
wasted work.

## Scope limits

No held-out confirmation is designed. If stage B works, that decision comes
after. Stage A remains permanently **FAIL**, and A2 remains developmental
follow-up rather than independent confirmation that cLogP corridors are
special — family B was selected because it showed the effect.

---

## External comparator evidence (from the baseline-qualification lane)

**GraphXForm's action masking is genuinely pathwise, but only over valence,
atom type, atom count and bonding legality. Its ring-size and bonding-pattern
constraints are a TERMINAL filter
(`molecule_evaluator.py::infeasible_by_special_constraints`, asserted on
`mol.synthesis_done`), and there is no SMARTS or substructure matching anywhere
in that repository.**

The strongest published graph-editing comparator therefore instantiates this
lane's `endpoint_only` arm, not its pathwise arm. COMPOSE's capability is real
and unmatched.

**This raises the importance of G1 rather than lowering it.** With G1 failing,
the capability distinction is **academic in this regime**: if trajectories that
break the motif never come back, neither COMPOSE's pathwise mask nor
GraphXForm's terminal filter changes the returned molecule. The honest framing
separates *capability* (COMPOSE has it, GraphXForm does not) from *consequence*
(on this panel at this horizon, it does not change the answer).
