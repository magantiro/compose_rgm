# Workstream C — Pathwise Constraints: Protocol

**Status:** `DESIGN_ONLY` (no Modal run has been launched)
**Branch:** `codex/compose-pathwise-constraints`
**Held-out data opened:** NO

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

| Risk | Detector |
|---|---|
| Local RDKit (2025.09.6) perceives aromaticity differently from the Modal image (2024.3.5), silently widening or narrowing the protected pattern | the app re-derives the motif from the canonical start key and voids the shard if the geometry disagrees with the frozen panel |
| Canonicalisation moves the source out of its own motif | `preserves_motif(smarts, start_key)` asserted before any arm runs |
| The mask leaks | G0, checked on every masked arm's realised trajectory |
| An arm is truncated mid-run by the cost cap | the cap is checked **between** arms only, so an arm is complete or absent |
| Two panel sources carry unusual valences (`[PH]`, `[SH4]`) | they came from the training corpus under an outcome-independent rule and are **not** removed; flagged as a limitation |
