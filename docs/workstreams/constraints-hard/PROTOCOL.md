# Protocol — hard structural constraints (Experiment A)

> # ⛔ `SUPERSEDED` — 2026-08-13
>
> **This internal experiment design is withdrawn and must not be implemented.**
> It is retained, not deleted, because its negative result is real and recorded.
>
> **Two independent reasons:**
>
> 1. **The yield contrast was folded into Lane 2.** The lead accepted this lane's
>    own §13 recommendation. Lane 2 already has the right causal structure and
>    had already measured the contrast at n = 6. Building a parallel harness
>    would have been a second broad optimization lane.
> 2. **The protected object failed its feasibility census.** The Bemis–Murcko
>    scaffold is a median **78.95%** of heavy atoms across the 96,094-source
>    held-in pool, leaving a median **5** editable atoms; **22.03%** of sources
>    are eligible. See `SCAFFOLD_FEASIBILITY.md`, which **stands as a result**.
>
> ### The scaffold stop is binding
>
> **No smaller core may be substituted.** Not a "Bemis–Murcko-lite", not a
> reduced ring core, not a pharmacophore, not a hand-tuned protected core mined
> from the held-in corpus. Having learned that the full scaffold is too
> restrictive, *any* smaller core would be chosen precisely because it leaves
> enough room to act — selection on the outcome, and task shopping in the
> project's own vocabulary.
>
> **No new constraint may be invented from our own failed scaffold result.**
>
> ### What remains live
>
> `CONSTRAINT_SEMANTICS.md` (project record), `SCAFFOLD_FEASIBILITY.md` (a
> standing negative), and this lane's only remaining job:
> **`EXTERNAL_HARD_CONSTRAINT_AUDIT.md`**.
>
> Sections 1–13 below are historical. Read them for the semantics freeze in §4
> and the sign-guarantee handling in §8; do not read them as a plan.

---

**Status: `SUPERSEDED` (was `DESIGN_ONLY`). Nothing here was ever run.** No Modal
launch, no GPU, no claim-bearing compute. Held-out never opened.

**Base commit:** `f6146d7`. **Branch:** `codex/compose-constraints-hard`.

---

## 1. Claim

> COMPOSE can impose hard structural requirements by restricting the exact
> canonical successor support at inference time, while keeping the learned
> molecular reference process fixed.

Formally, with `F(x)` the canonical legal molecular successors of `x` and a hard
predicate `C`:

```
F_C(x) = { y in F(x) : C(y) = 1 }
```

The same `R_theta` and the same control machinery operate on `F_C(x)`. The
architectural chain is **exact legality → hard admissibility → learned
plausibility → purpose**, with no objective-specific update to `R_theta`.

## 2. Non-claims

- ❌ Not that COMPOSE invented scaffold preservation. Prompt-MolOpt^P (NMI 2024)
  and InVirtuoGen (ICLR 2026) do hard user-specified preservation; MolEditRL
  (ICLR 2026) does soft structure-preserving editing; ConStruct (NeurIPS 2024)
  guarantees structural properties throughout graph-generation trajectories.
- ❌ Not persistent atom identity. See `CONSTRAINT_SEMANTICS.md`: the provable
  invariant is `LABELED_SUBGRAPH_PRESENCE_INVARIANT`, and the phrasings implying
  atom tracking are barred.
- ❌ Not a **trajectory**-constraint claim for structural cores. Lane 2 measured
  `recovery_rate = 0.0` over 19 breaking rollouts: for a structural core,
  destruction is **absorbing**, so endpoint validity implies path validity and
  endpoint-only filtering does not admit hidden path violations. **That question
  is closed. This protocol does not reopen it.**
- ❌ Never that COMPOSE defeats a same-lab method. See `SAME_LAB_LINEAGE.md`.
- ❌ Never "without retraining" — barred project-wide. Say "with no
  objective-specific update to `R_theta`".

## 3. Frozen inputs

| object | binding |
|---|---|
| reference process | frozen `R_theta`, `runs/run_v2_01/R_THETA_CHECKPOINT.pt`, `load_state_dict(strict=True)` |
| executor | `editing_v2_semantic_rewrite_system` — the frozen Active8 Editing-V2 public runtime |
| kernel | `production_successor_kernel.canonical_successor_result`, unmodified |
| protected object | atom- and bond-labeled Bemis–Murcko scaffold of `x_0`, rule `atom_and_bond_labeled_bemis_murcko_scaffold_v1` |
| eligibility bands | reused verbatim from Lane 2 (`SCAFFOLD_FEASIBILITY.md` §2.4) |
| viability criteria | Lane 2's frozen Stage-A2 set (§7) |
| objective | Lane 2's already-frozen source-conditioned editing objective — **not a new oracle** |
| evaluator | Lane 3's pinned shared evaluator (RDKit 2024.03.5) |
| accounting | Lane 3's `oracle_accounting.py` three counters |

**Nothing in this lane retrains, fine-tunes, or reweights `R_theta`.** The
constraint is a filter on the enumerated successor list plus renormalization.

## 4. The predicate, and where it is injected

`C(y) = 1` iff at least one exact atom- and bond-labeled subgraph embedding of
the source's protected core exists in `y`. Label semantics — element, formal
charge, bond order/aromatic class **in**; implicit-H count, stereochemistry,
isotope **out** — are frozen in `CONSTRAINT_SEMANTICS.md` §6, with the reasons.

**Injection point (this is not negotiable — Probe 4):**

```
F(x)   = result.batch.successors            # exact legal fiber, kernel untouched
F_C(x) = [s for s in F(x) if C(s.key)]      # hard admissibility
         renormalize probability over F_C(x)
```

`RewriteSystem(constraints=)` **cannot** be used: it raises, and
`canonical_successor_result:557-567` converts any executor exception into a fatal
`ProductionSuccessorKernelError`. Renormalization is mandatory —
`validate_successor_batch:297-302` rejects a batch that does not sum to 1.
`SuccessorBatch.is_terminal` already represents the mask-empty case.

**The predicate implementation must REUSE Lane 2's
`pathwise_constraints.fragment_smarts` / `preserves_motif`, not reimplement
them.** Only the derivation rule differs (Bemis–Murcko core vs largest ring
system). This is a merge-time dependency on
`codex/compose-pathwise-constraints`; it is recorded in `handoff.json` as an
unmet prerequisite.

## 5. Panel

**Source is the independent statistical unit.** Multiple seeds from one source
are repeated measures.

- **Pool:** held-in `training_source_keys` only (96,094). Held-out reserve is
  structurally not opened.
- **Eligible subpopulation:** 21,166 sources = **22.03%**.
- **Selection:** first `n` eligible sources in seeded shuffled scan order. No
  ranking, no preference, **no successor enumeration, no arm outcome**. Same
  construction as Lane 2's `pathwise_select_*_panel.py`.
- **Development n = 24**, matching Lane 2's Stage B. Held-out is a separate
  authorization and is **not** requested here.

**Eligibility is frozen from starting-state applicability only** and may never
reference a controller outcome. No source may be replaced because its scaffold
result is inconvenient — the `no_replacement_rule` pattern from
`diagnostics/retarget_heldout_panel.json`.

**Mandatory disclosure:** the panel is drawn from a biased subpopulation —
molecules with comparatively small cores and heavy decoration. The 22.03%
applicability rate is reported as a result, not hidden.

## 6. Arms

All four share source, objective, controller class, horizon, legal executor,
oracle budget and resource accounting. The mask is the **only** difference
between arms 2/3.

| # | arm | support | relation |
|---|---|---|---|
| 1 | `unconstrained_verified` | `F(x)` | context |
| 2 | `posthoc_scaffold_filter_verified` | `F(x)`, filter at the end | **primary comparator** |
| 3 | `hard_scaffold_mask_verified` | `F_C(x)` | **primary arm** — declared `SUPPORT_ABLATION` |
| 4 | `hard_scaffold_mask_greedy` | `F_C(x)` | optional sensitivity, only if nearly free |

**Primary contrast: 3 vs 2.** Is it better to keep the process inside the
acceptable state space than to generate trajectories and discard unacceptable
endpoints afterward?

Arm 3 must be registered with `comparison=SUPPORT_ABLATION` and the mask
declared as the single preregistered support-differing field
(`successor_kernel.py:308`). Registering it as `LAW_ONLY` would be false.

## 7. Preregistered gates

Reused verbatim from Lane 2's frozen Stage-A2 criteria. **Not re-derived, not
widened.**

| gate | criterion | threshold |
|---|---|---|
| **V3** | non-vacuity: core-violating legal alternatives exist | ≥ 20 events |
| **V4a** | pooled median support retention `\|F_C(x)\|/\|F(x)\|` | **≥ 0.10** |
| **V4b** | mask-empty rate | **≤ 0.05** |
| **V5a** | source spread of nonvacuous opportunity | **> 1/3** |
| **V5b** | single-source share of events | **≤ 0.50** |

**Measure nonvacuity; do not select for it.** Sources are not filtered to those
already exhibiting both preserving and violating moves.

Prior expectation from Lane 2's smaller core: retention mean 0.774 / median
0.945 removed-complement, 0/42 mask-empty. The BM core is a median 12 atoms
larger, so retention will be **lower** and V4a is the gate most likely to bind.

## 8. Metrics — per source, source-level paired uncertainty

**Primary:** feasible-output yield at fixed resources (arm 3 vs arm 2).

**Secondary:** terminal objective improvement; support retention; mask-empty
frequency; nontrivial edit count and diversity outside the core;
scaffold-relative edit locality; source applicability; chemical-envelope
fidelity.

**Resource, all three counters, never merged, each efficiency claim naming its
axis:** `unique_valid_canonical_evaluations`, `oracle_requests`,
`evaluator_calls`, plus kernel calls and complete trajectories.

### The guarantee that is not a result

Arm 3 retains the core in **100%** of committed states. **True by construction.
Recorded as a construction check, without a denominator. Never reported as an
empirical success.** Mathematical reason and adversarial fixture:
`CONSTRAINT_SEMANTICS.md` §7; test: `tests/test_hard_scaffold_constraint.py`.

The measurement is what the **unconstrained** arm does and what the constraint
**costs**.

## 9. Stop rules — all admissible outcomes

Stop and report rather than retune if: the mask removes nearly all useful
successors; constrained trajectories frequently hit empty/tight support;
objective improvement collapses; only trivial edits remain outside the core
(**at risk — median 5 editable atoms**); applicability is narrow (**already
triggered at 22.03%, disclosed**); or post-hoc filtering performs equally well at
much lower cost.

**Do not add another predicate if the protected-core census fails.**

## 10. Forbidden adaptations

Widening the eligibility bands; hand-picking a sub-scaffold after seeing that the
full one is inconvenient; conditioning eligibility on controller outcomes;
sweeping a knob and reporting the favourable value; altering external
architectures, search or fine-tuning; porting sequence methods to molecular
graphs; modifying Lane 2's cLogP corridor or confirmation; reviving the withdrawn
hand-driven GraphXForm action loop.

---

## 11. Costed Stage-2 development plan

### Cost basis — measured, not estimated

From Lane 2's Stage A shard (`diagnostics/pathwise_constraints_smoke.json`,
`cost` block), 5 arms at `H = 6` on 6 sources:

| quantity | value |
|---|---|
| kernel calls per source | mean **51.0**, range [43, 58] |
| wall seconds per source (all arms) | mean **852.6**, range [619.9, 1163.3] |

≈ **170 s per arm-source**. Enumeration dominates; the mask *reduces* the scored
set, so the masked arms are not more expensive per state.

### Projection

| item | value |
|---|---|
| arms | 4 |
| sources | 24 |
| per-source, 4 arms | ≈ 682 s (4/5 × 852.6) |
| **serial CPU** | **≈ 4.5 CPU-hours** |
| fan-out | 24 containers, 1 source each |
| **wall time** | **≈ 12–20 min** |
| CPU per container | 2.0 (matching the retargeting/Pareto apps) |

**Add the Stage-1 fiber census**, which enumerates fibers along unconstrained
rollouts on the same 24 sources: ≈ 1 arm-equivalent ≈ **1.1 CPU-hours**,
≈ 5 min wall.

**Total ≈ 5.6 CPU-hours, ≈ 25 min wall at 24-way fan-out.** This is a cheap
experiment. Cost is not the reason to hesitate; the scientific overlap with Lane
2 (§13) is.

### Uncertainty on the projection

Honest band: **0.7×–2.5×**. Lane 2's per-source range already spans 1.9×, the BM
mask changes the scored-set size in a direction that has not been measured, and
`H` for this experiment is not yet fixed. Treat 5.6 CPU-hours as a planning
number, not a quote.

### Durability requirements — binding before any launch

1. **Per-arm partial checkpointing.** `PARALLEL_WORKSTREAMS_AND_HANDOFF.md`
   records that `modal_apps/pathwise_stage_b_app.py` writes a durable shard per
   *source* but no per-*arm* partial, so a preemption costs the whole task. That
   fix is marked **UNBLOCKED and BLOCKING**. This lane's app must ship per-arm
   partials from the start.
2. **Durable per-task shards** written before return; the aggregate
   reconstructible from shards.
3. **`--detach` on any Modal run.** A server-side `drive()` does not survive
   closing the laptop; check the State column before calling a run safe.
4. **No session-specific paths.** All commands take explicit arguments.

### Sequenced plan

| step | gate | authorization |
|---|---|---|
| **S0** | executor semantics verdict | ✅ **DONE** |
| **S1a** | model-free protected-core census | ✅ **DONE** (`SMOKE_HELD_IN`) |
| **S1b** | fiber census, V3/V4a/V4b/V5a/V5b | ⛔ **BLOCKED** on Gate-0; needs a Modal run + lead authorization |
| **S2a** | predicate implementation + adversarial fixture + unit tests | ready once Lane 2's `fragment_smarts` merges |
| **S2b** | 4-arm development run, n = 24 | **requires lead authorization** |
| **S2c** | external comparison (GraphXForm, Prompt-MolOpt^P) | requires Lane 3 artifacts + separate authorization |
| **S3** | held-out confirmation | **not requested** |

**This lane stops at S2a.** S1b and everything after need the lead's per-run
approval.

---

## 12. Unresolved risks

| risk | severity | status |
|---|---|---|
| **Overlap with Lane 2's Stage A** (§13) | **HIGH** | needs a lead decision, not more work |
| Only trivial edits outside the core (median 5 atoms) | **HIGH** | unmeasured at fiber level; V4a will expose it |
| Aromaticity relabelling of the core causing false violations | **MEDIUM** | predicate-level; census must report the rate |
| BM mask retention below the 0.10 floor | **MEDIUM** | plausible given +12 atoms vs Lane 2 |
| Gate-0 blocks all local fiber work | **MEDIUM** | diagnosed, queued, not this lane's to fix |
| Panel is a biased subpopulation | **MEDIUM** | mitigated by disclosure |
| `fragment_smarts` reuse depends on an unmerged branch | **LOW** | recorded in `handoff.json` |
| RDKit 2025.09.6 locally vs 2024.03.5 pinned | **LOW** | flagged in the artifact |

## 13. The question the lead must answer

**Is Experiment A's overlap with Lane 2's Stage A the intended division of
labour?**

Lane 2 already ran, at n = 6, the arm structure this protocol proposes, and
recorded the primary contrast under the name `endpoint_only_return_failure` with
a correct free-sign argument: `endpoint_only` returned nothing on 2/6 sources
while the masked arms returned on 6/6. Their G1 FAIL closed the *trajectory*
framing; it did not touch the *yield* framing.

What this lane genuinely adds is a larger and stricter protected object, a panel
size that makes the contrast a rate rather than an existence proof, and the
external comparison. Whether that is worth a separate experiment — or whether it
should be folded into Lane 2's next stage — is a scope decision above this lane.

**Recommendation: fold the yield contrast into Lane 2's existing arm harness
rather than standing up a parallel one**, and keep this lane's distinct
contribution to the Stage-0 semantics freeze, the protected-object census, and
the baseline matrix. That avoids a second broad optimization lane, which the
charter explicitly warns against.
