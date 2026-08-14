# Audit: what controller machinery already exists

**Audit only. Nothing was implemented, retrained or rerun.** Commissioned
because `griddd_conditional.py` (93 KB), a value-guided SMC controller, an RTB
trainer and a frozen `h_φ` ensemble were discovered *after* we had designed a
new `h_φ` stack from scratch.

**Headline: substantial machinery exists and is good. None of it is directly
reusable for the QED task without a change we must name honestly.**

---

## Q3 · Data hygiene — CLEAN, and better than expected

```python
official_griddd_lead_set_id = "griddd_release_qed_800_exact_unresolved"
exact_griddd_leads_available: bool = False
```

`GridDDBenchmarkFairnessContract` **explicitly recorded that the official 800
could not be obtained**, and carries an invariant that *raises* if an
`...unresolved` set is ever marked exact-and-available. It independently
arrived at the same task spec we froze today — QED [0.70, 0.80] → [0.90, 1.00],
Tanimoto ≥ 0.40, Morgan r2 / 2048 bits, 20 candidates, all-start denominator —
and separated "protocol A, GrIDDD-comparable" from "protocol B, COMPOSE
internal."

**No official Jin source influenced any of this work.** The official set remains
inferentially unconsumed.

**Today's work resolves that flag.** `data/jin/qed_test.txt` is now in the repo,
800/800 verified. The `_unresolved` marker can finally be retired — a genuine
contribution from today back into this older lane.

## Q1 · What is actually implemented — four separable pieces

| piece | what it is | state |
|---|---|---|
| **`h_φ` ensemble** | 4 seeds, frozen weights, sha256 pinned, 7,351 labels over 1,295 states | trained, `FROZEN_WEIGHTS_NO_THRESHOLD_SELECTED` |
| **value-guided SMC** | Feynman–Kac particles, hard fiber constraint, resampling | implemented, **run with `value_twist: None`** |
| **best-first "ceiling"** | search + label generator | run, 0.3333 |
| **RTB trainer** | Relative Trajectory Balance policy fine-tuning | implemented |

The architecture we sketched is **not** new to the repo. It was reached from the
SMC side rather than the rejection-sampling side.

## Q4 · What the existing `h_φ` predicts — the load-bearing finding

```
vector: [e_y, e_z, e_y − e_z, e_y ⊙ e_z, sim(y,z), budget one-hot]
input_dim 1033 · budget_max 8 · encoder: FROZEN R_θ _encode_batch global_state
```

The shape is exactly right — **frozen encoder plus a small learned head**,
budget-conditioned, goal-conditioned, and it even predicts `sim(y,z)`.

> ⚠️ **But `z` is a TARGET MOLECULE embedding, not an objective-space region.**

It answers *"can I reach **this molecule** in `b` steps?"* — the **exact-target
recovery** controller from banked Experiment 3A. The QED task needs
`g_z(x) = 1[QED(x) ≥ 0.9 ∧ Sim(x,x_src) ≥ 0.4]`, which is **reachability to a
region in objective space**.

**These are different conditioning variables.** The weights are not reusable
as-is. What *is* reusable, and valuable: the feature contract, the frozen-encoder
pattern, the budget one-hot, the label pipeline, and the freezing discipline.

## Q5 · How it was trained, and one incompatibility

`h_φ` learns from teacher labels — `teacher_top1_agreement` **0.76–0.80**,
`value_regret` 0.16–0.18, `bce` 1.03–1.30, `harmful_override_rate` **0.13–0.16**.
Decent, not excellent: it overrides the teacher harmfully about **one time in
seven**.

> ⚠️ **The RTB trainer fine-tunes the POLICY, not a value.** *"Trains a policy
> (init from frozen Lineage B) toward the reward-tilted target via Relative
> Trajectory Balance."*

That **violates the frozen-`R_θ` constraint**. Our design is `R_θ` frozen with
control applied as `R_θ·h_φ`, and "enlarge the controller, never `R_θ`." RTB
produces a *different policy*. It is a legitimate method — but it is **not** the
controller we specified, and its results cannot be reported as frozen-`R_θ`
control.

## Q2 · The "33 % ceiling" — the word is falsified

`best_first_ceiling_success_rate: 0.3333`, and in the same file
`success_rate: 0.5`.

> **SMC beat the "ceiling" by 50 %. It is not a ceiling.**

Exactly the P0c lesson again: an object named a ceiling that is not an upper
bound. It should be called the **best-first search reference**, nothing more.
**Do not inherit the word.**

## Q7 · What was actually measured — with the caveat that matters

`smc_panel12_zerocalib_b1000.json`:

| | |
|---|---|
| success | **0.5000** (6/12) — same success EVENT, different protocol |
| mean best feasible QED | 0.8869 |
| population diversity | 0.6186 |
| target / similarity floor | QED 0.9 / Tanimoto 0.4 — **our task** |
| `griddd_reference_success_rate` | 0.451 *(recorded by them)* |
| `value_twist` | **None** |
| **`budget_per_lead`** | **1000** |
| particles | 32 |

**Three caveats, and the third is decisive:**

1. **12 leads.** Tiny. Six successes.
2. **`value_twist: None`.** This is **V0 with no learned twist at all** — a
   reward-difference potential. The learned-value version was never the thing
   that scored 0.5.
3. **Budget 1000 per lead, against the protocol's 20 candidates — a 50×
   advantage.** This is **not** a GrIDDD-comparable number and must never be
   placed beside 45.1 %.

**And a fourth:** the base checkpoint is **Lineage B**
(`/private/tmp/lineage_b_checkpoint/`), **not** our frozen `R_θ`
(`runs/run_v2_01/R_THETA_CHECKPOINT.pt`). Different base model.

## What this tells us, net

**The good — stated at the strength the evidence actually supports.**

An earlier draft of this audit said *"the failure was the controller, not
COMPOSE."* **That overstates it**, because the run used Lineage B rather than
frozen `R_θ` and a 1000-evaluation budget rather than 20 candidates. The
rigorous statement is:

> **The QED target is demonstrably reachable by existing COMPOSE-family search
> machinery. Policy B's 0/320 should therefore NOT be read as evidence that the
> executable rewrite space lacks QED-improving trajectories.**

That is enough, and it is all we need. We are **not** claiming our current
`R_θ` can already attain 50 %.

**Same event, not the same protocol.** The old SMC run shares our *property
target and success event* — QED ≥ 0.9 at Tanimoto ≥ 0.4 — but **not** our
evaluation protocol. Never describe it as "the same task" unqualified.

**The gap.** Nothing existing gives us a **region-conditioned, frozen-`R_θ`,
budget-honest** controller:

| have | need |
|---|---|
| `h_φ` conditioned on a target **molecule** | conditioned on an objective **region** |
| SMC on **Lineage B** | on frozen **`R_θ`** |
| success at **budget 1000** | at **20 candidates** |
| RTB **policy** fine-tuning | **frozen** `R_θ` + learned control |

## The one decision this audit supports

**Reuse the pattern, not the weights.**

Take the feature contract, the frozen-encoder design, the budget one-hot, the
label pipeline and the freezing discipline. **Retrain the head against a
region-valued terminal event** rather than a target-molecule embedding. That is
**one precise missing component**, not a new research programme.

**Keep the SMC implementation available but do not start there.** For a QED
success region with `h_φ ∈ [0,1]`, the rejection sampler proven exact in Stage A1
is far simpler and preserves the learned kernel exactly. **SMC earns its
complexity only when acceptance collapses** — which is a measurable condition, so
measure it rather than assume it.

**Do NOT simultaneously start** a new value architecture, a new SMC algorithm,
archive-aware Pareto search, QED shaping, and five-objective optimization. The
five-objective work reuses whatever survives this QED cycle.

## Naming hazard — flagged, not fixed

`griddd_conditional.py` contains **zero GrIDDD code** — only numpy, rdkit and
`compose_v4`. Lane 5 already flagged this on 2026-08-13. With a genuine tier-1
GrIDDD comparison now in the repo, the collision is a provenance accident
waiting to happen.

**Do not refactor now.** Rename or deprecate cleanly once the object is settled.
