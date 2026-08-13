# Stage 1 — protected-core feasibility census

**Status: `SMOKE_HELD_IN` for the model-free half; `DESIGN_ONLY` for the
fiber-dependent half (BLOCKED — see §4).** Held-out never opened.

**Base commit:** `f6146d7`. **Artifact:**
`diagnostics/constraints_hard_scaffold_census.json`. **Producer:**
`scripts/constraints_hard_scaffold_census.py`.

---

## 0. Summary for the lead

Three things were measurable without the kernel, and all three matter:

1. **The full Bemis–Murcko scaffold is too large to protect on the median
   source.** Across the **entire 96,094-source held-in pool**, the BM core is a
   **median 78.95%** of heavy atoms, leaving a **median of 5** editable heavy
   atoms. Only **22.03%** of sources are eligible under the reused frozen bands.
2. **Lane 2 already measured the fiber-level version of this constraint** on a
   smaller protected object, and its numbers bear directly on our gate. Most
   importantly: **motif destruction is absorbing** (`recovery_rate = 0.0` over
   19 breaking rollouts), which *kills* the trajectory-constraint framing for
   structural cores but *sharpens* the hard-admissibility-yield framing this
   lane was chartered for.
3. **The fiber half of the census cannot run locally.** Gate-0 authentication
   fails here for the already-diagnosed non-portable reason, and the model is
   constructed *from* the authenticated source, so there is no partial local
   path. Reproduced, not routed around.

**Recommended verdict: PROCEED, but with the protected object re-scoped and the
claim's applicability stated at 22%.** Reasoning in §5.

---

## 1. What the model-free census measured, and what it cannot

The census needs only RDKit. No `R_theta`, no successor kernel, no oracle, no
Gate-0 chain.

**Can establish:** whether the frozen protected object is a *usable* core —
non-empty, not the whole molecule, leaving structure to edit; and the source
applicability rate.

**Cannot establish:** anything about the successor fiber — support retention
`|F_C(x)|/|F(x)|`, mask-empty rate, or the frequency of legal core-violating
alternatives. Those are §4.

This split is why a partial answer exists at all, and the boundary is enforced in
the artifact's own `scope` field.

## 2. Results

### 2.1 Held-in developability cohort (n = 30)

`diagnostics/retarget_calibration_cohort.json`, `cohort_sha256 e402e318…`,
status `FROZEN_BEFORE_CALIBRATION_OUTCOME_INDEPENDENT`.

| quantity | value |
|---|---|
| eligible | **6 / 30 = 0.200** |
| core fraction, all sources | median **0.7524**, mean 0.7385, max 0.9677 |
| core fraction, eligible only | median 0.6236, range [0.4231, 0.6552] |
| editable atoms outside core, all | median **5.0**, mean 7.07, min 1 |
| BM core minus largest ring system | median **+12** atoms |
| scaffold parent-index cross-check | **0 failures / 30** |

### 2.2 Whole held-in pool (n = 96,094) — the number that matters

`diagnostics/editing_v2_matched_validation_reserve_ids.json.gz`, key
`training_source_keys`. `reserve_source_keys` is deleted from the loaded payload
before the sweep, so the held-out reserve is structurally not opened.

| quantity | value |
|---|---|
| parsed | 96,094 / 96,094 |
| **eligible** | **21,166 = 0.2203** |
| core fraction | **median 0.7895**, mean 0.7535 |
| editable atoms outside core | **median 5.0**, mean 6.37 |

Ineligibility reasons (not mutually exclusive):

| reason | count | fraction |
|---|---:|---:|
| `core_fraction_outside_band` | 68,872 | 0.7167 |
| `too_few_editable_atoms_outside_core` | 65,091 | 0.6774 |
| `heavy_atoms_outside_band` | 9,767 | 0.1016 |
| `core_too_small` | 2,300 | 0.0239 |
| `empty_scaffold_acyclic_source` | 1,481 | 0.0154 |

The 30-source cohort was representative, not an artifact of its selection.

### 2.3 A bug this census caught, recorded because the cross-check is the point

The first implementation computed the BM core by pruning terminal non-ring atoms
to a fixpoint. That **disagreed with RDKit on 16 of 30 sources**, because
Bemis–Murcko keeps **exocyclic atoms double-bonded to a scaffold atom** — ring
carbonyl oxygens. The pruning-only rule dropped them.

Corrected in `murcko_atom_indices`; `scaffold_parent_index_crosscheck` now
compares against `MurckoScaffold.GetScaffoldForMol(mol).GetNumHeavyAtoms()` on
every source and reports **0 failures**. The uncorrected run reported eligibility
8/30 and core fraction median 0.7524; the corrected run reports 6/30 and 0.8000.
Both are recorded so the correction is auditable rather than invisible.

**Do not remove the cross-check.** It is the only thing standing between this
lane and a protected object that is silently not the one we named.

### 2.4 Thresholds are reused, not minted

All four bands are taken **verbatim** from Lane 2
(`src/compose_v4/experiments/pathwise_constraints.py`, branch
`codex/compose-pathwise-constraints`, tip `bcc4a40`):

| constant | value | Lane 2 name |
|---|---|---|
| `MIN_CORE_ATOMS` | 6 | `MIN_MOTIF_ATOMS` |
| `CORE_FRACTION_BAND` | (0.20, 0.70) | `MOTIF_FRACTION_BAND` |
| `MIN_FREE_ATOMS` | 8 | `MIN_FREE_ATOMS` |
| `HEAVY_ATOM_BAND` | (18, 38) | `HEAVY_ATOM_BAND` |

Lane 2 applies them to the **largest fused ring system**; we apply them to the
**Bemis–Murcko scaffold**, which is a superset (ring systems *plus* linkers *plus*
exocyclic double-bonded atoms) and here a median 12 atoms larger. Applying an
unchanged band to a strictly larger object is the **conservative** direction.

> **The bands must not be widened to make this census pass.** A 0.70 upper bound
> failed on 71.7% of the pool. That is the measurement. Raising it to 0.85 would
> be minting a threshold from the result the project rule exists to prevent.

---

## 3. Prior fiber-level evidence from Lane 2 — the decisive input

Lane 2's Stage A (`diagnostics/pathwise_constraints_smoke.json`,
`SMOKE_HELD_IN`, `panel_sha256 8b47a0e4…`, 6 sources, 42 rollouts, protected
motif = largest fused ring system, goal `B = P AND D`, `H = 6`):

| quantity | value |
|---|---|
| rollouts that broke the motif | **19 / 42** |
| broke **and recovered** by the end | **0** — `recovery_rate = 0.0` |
| `endpoint_valid_path_invalid` (their G1) | **0 / 6** sources → **G1 FAIL** |
| mask removes, fraction of successors | mean **0.2257**, median 0.055, max 0.9988 |
| mask removes, fraction of reference mass | mean **0.3444**, median 0.1908 |
| states with empty masked support | **0 / 42** |
| `endpoint_only` selection failed | **2 / 6** sources |
| `b_success` by arm | `endpoint_only` 4/6; `pathwise_greedy` 6/6; `pathwise_stochastic` 6/6 |
| price of the guarantee (paired, n=4) | mean **+0.0119**, median −0.0019, [−0.0367, +0.0880] |
| cost | **51** kernel calls, **852 s** mean per source |

### What this means for us, stated carefully

**Lane 2's G1 FAIL does not fail our experiment — it selects which of the two
questions is live.**

Their G1 asked: *does endpoint-only filtering let trajectories sneak through
forbidden intermediate states?* Answer: **no**, because for a structural core,
destruction is **absorbing** — nothing that breaks the ring system in `H = 6`
steps ever restores it. Endpoint validity therefore implies path validity, and
the *trajectory-constraint* claim is dead for structural cores. Lane 2 closed it
correctly and we must not reopen it.

Our chartered primary contrast is the **other** question: *does endpoint
filtering **waste** the budget?* Absorption makes this question sharper, not
weaker. If breaking is absorbing and 19/42 unconstrained rollouts break, then
endpoint filtering **discards ~45% of completed trajectories**, and Lane 2's own
`endpoint_only_return_failure` block records `endpoint_only` returning nothing at
all on **2 of 6** sources while the masked arms returned on 6/6.

Lane 2 wrote the admissibility argument for us, and it is correct:

> *"free sign: endpoint_only could have matched the masked arms on every source.
> Nothing in the construction prevents it — it fails only when its unconstrained
> policy spends the whole budget outside the feasible set"*

That is a genuine falsifying range. **Feasible-output yield at fixed resources is
our primary metric, and its sign is free.**

### The honest cost of this overlap

At n = 6 this is an existence proof, not a rate — Lane 2 says so explicitly. But
it means **Experiment A as chartered is substantially a scale-up of a
measurement Lane 2 has already taken at smoke scale on a smaller core**, not a
new experiment. The lead should decide whether that is the intended division of
labour. What this lane genuinely adds:

1. a **larger, stricter protected object** (BM scaffold, median +12 atoms), where
   the mask removes strictly more than Lane 2's mean 22.6%;
2. **source-level paired uncertainty at a panel size that makes it a rate**;
3. the **external comparison**, which Lane 2 never attempted.

---

## 4. The fiber half is BLOCKED. Reproduced, not bypassed.

Everything in the charter's Stage-1 list that involves `F(x)` — unconstrained
legal fiber width, hard-mask retained fiber width, retained-support fraction,
mask-empty rate, frequency of legal core-violating alternatives, property
headroom — needs the successor kernel.

**The kernel cannot be constructed locally.** Verified, not assumed:

```
$ python3 docs/workstreams/constraints-hard/probes/gate0_local_authentication_repro.py \
      --local-runtime <local_runtime>
BLOCKED (expected): ProcessV2T1PanelError
  the Gate-0 PASS names another Active8/process input
```

This is the already-diagnosed defect in
`docs/PARETO_PARITY_ENVIRONMENT_STATUS.md`: `source_index_sha256`
(`editing_v2_process_v2_gate_zero.py:593`) hashes a body whose first field is the
**absolute Active8 mount path**, so byte-identical content mounted locally
authenticates against the wrong string. Content was independently verified
identical — completion and sentinel digests both match.

**There is no partial local path.** The model is built *from* the authenticated
source: `build_process_v2_score_revised_scratch_runtime(source, …)` in
`modal_apps/retarget_intervention_app.py:185`. Failing to open the source means
there is no model, hence no marked law, hence no fiber. The legal mark set is
structural (it comes from `masks` in `_action_tables`, not from the learned
logits), so a *weight-free* fiber census is conceivable in principle — but it
would still need the architecture and vocabulary that the authenticated source
supplies, so it does not unblock anything today.

Per the standing instruction, this is reported rather than routed around. The
chain is closed by a canonical-JSON framing guard and a decision self-hash;
three integrity guards deep is a signal to stop.

**Consequence: the fiber census must run where the tree is mounted at the path
the digest names — i.e. on Modal — and that requires the lead's per-run
authorization. Nothing has been launched.**

---

## 5. Verdict and the one decision the lead must make

### The stop rules, evaluated against what we have

| charter stop rule | status |
|---|---|
| mask removes nearly all useful successors | **not triggered** — Lane 2 measured mean 0.226 removed on a smaller core; BM will remove more, magnitude unmeasured |
| constrained trajectories frequently hit empty/tight support | **not triggered** — Lane 2: 0/42 mask-empty |
| objective improvement collapses | **not triggered** — Lane 2 G3 PASS, `pathwise_greedy` mean improvement 3.096 vs `unconstrained_greedy` 3.132 |
| only trivial edits remain outside the core | **AT RISK — this is the live one.** Median 5 editable heavy atoms pool-wide |
| applicability is narrow | **TRIGGERED, and must be disclosed: 22.03%** |
| post-hoc filtering performs equally well at much lower cost | **open** — the experiment's actual question |

### The decision

The full BM scaffold fails the reused band on 71.7% of the pool. There are three
responses and only one of them is legitimate:

- ❌ **Widen `CORE_FRACTION_BAND`.** Minting a threshold from the result. Barred.
- ❌ **Hand-pick a smaller sub-scaffold per source.** Selecting a scaffold after
  seeing that the full one is inconvenient. Explicitly barred by the charter.
- ✅ **Keep the object and the bands; draw the panel from the eligible
  subpopulation; report applicability as a result.**

The third is legitimate because the charter *already* directs eligibility to be
frozen from starting-state applicability only, and 21,166 eligible held-in
sources is far more than any panel needs. It is the same construction Lane 2
used (`pathwise_select_*_panel.py`: *"first N eligible sources in shuffled scan
order; no ranking, no preference, no successor enumeration, no arm outcome"*).

**The disclosure it obliges us to make, in the paper, not just here:**

> The hard-scaffold result is scoped to sources whose Bemis–Murcko core occupies
> 20–70% of heavy atoms and leaves at least 8 editable heavy atoms outside it —
> **22.0% of the held-in pool**. On the median drug-like source the Bemis–Murcko
> scaffold is ~79% of the molecule, and protecting all of it leaves too little to
> edit.

That is a real limitation and it is better stated by us than found by a reviewer.

### Recommended next action

**Do not launch.** Return to the lead with:

1. this census;
2. the panel-selection script, written but **not run**, that draws `n` sources
   from the 21,166 eligible held-in keys under the frozen rule;
3. the costed Stage-2 plan in `PROTOCOL.md`;
4. the explicit question of whether Experiment A's overlap with Lane 2's Stage A
   is the intended division of labour.

---

## 6. Reproduction

```bash
python3 scripts/constraints_hard_scaffold_census.py \
    --cohort diagnostics/retarget_calibration_cohort.json \
    --pool   diagnostics/editing_v2_matched_validation_reserve_ids.json.gz \
    --out    diagnostics/constraints_hard_scaffold_census.json
```

RDKit used: **2025.09.6**. The production pin is **2024.03.5**, so the artifact
carries `rdkit_pin_matches_production: false`. Scaffold perception is stable
across these versions for the atom-counting statistics used here, but **any
claim-bearing rerun must use the pinned evaluator**
(`scripts/compose_shared_evaluator_server.py`, Lane 3), which refuses to start
under an unpinned RDKit.
