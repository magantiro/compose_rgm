> **2026-08-13 — the lane was narrowed to an external audit and is now COMPLETE.**
> Three findings were accepted into project record: the
> `LABELED_SUBGRAPH_PRESENCE_INVARIANT` verdict, the GraphXForm hard-constraint
> `N/A`, and the Bemis–Murcko feasibility stop. The internal experiment
> (`PROTOCOL.md`) is **`SUPERSEDED`** — its yield contrast was folded into Lane 2
> on this lane's own recommendation.
>
> **The lane's final product is `EXTERNAL_HARD_CONSTRAINT_AUDIT.md`.** Sections
> below describing Stage-2 arms, panels and costs are **historical**; no internal
> experiment was designed after the reframe and none should be.
>
> **Binding:** no smaller protected core may be substituted for the Bemis–Murcko
> scaffold, and no new constraint may be invented from our own failed scaffold
> result.

# Workstream

- **Name:** Hard structural constraints (Lane 6, `constraints-hard`)
- **Claim ID:** external hard-constraint audit (was: Experiment A — withdrawn)
- **Branch:** `codex/compose-constraints-hard`
- **Base commit:** `f6146d7`
- **HEAD commit:** see `handoff.json.head_commit`
- **Working tree clean:** yes
- **Status:** `DESIGN_ONLY`, except `diagnostics/constraints_hard_scaffold_census.json` which is `SMOKE_HELD_IN`
- **Held-out data opened:** **no**

# One-sentence scientific question

> Is it better to keep a molecular editing process inside the admissible state
> space than to generate trajectories freely and discard unacceptable endpoints
> afterward?

# Claim this work can support

> COMPOSE can impose hard structural requirements by restricting the exact
> canonical successor support at inference time, while keeping the learned
> molecular reference process fixed — `F_C(x) = {y in F(x) : C(y) = 1}`, with no
> objective-specific update to `R_theta`.

# Claims this work cannot support

- **Persistent atom identity.** The provable invariant is
  `LABELED_SUBGRAPH_PRESENCE_INVARIANT`. "Atom-mapped", "atom-tracked", "the same
  atoms", "the scaffold atoms are never touched" are all barred.
- **That COMPOSE invented scaffold preservation.** Prompt-MolOpt^P (NMI 2024) and
  InVirtuoGen (ICLR 2026) do hard user-specified preservation; MolEditRL (ICLR
  2026) does soft; ConStruct (NeurIPS 2024) does hard-constrained graph
  generation.
- **A trajectory-constraint claim for structural cores.** Lane 2 measured
  `recovery_rate = 0.0` over 19 breaking rollouts: destruction is absorbing, so
  endpoint validity implies path validity. Closed; not reopened here.
- **"Without retraining"** — barred project-wide.
- **Anything about the successor fiber.** Stage 1b is blocked; no fiber quantity
  was measured by this lane.

# Frozen inputs

| Object | Path / ID | SHA-256 / identity | Verified? |
|---|---|---|---|
| Process-V2 chemistry | `editing_v2_semantic_rewrite_system` | code identity at `f6146d7` | yes, by read |
| `R_theta` checkpoint | `runs/run_v2_01/R_THETA_CHECKPOINT.pt` | **not loaded by this lane** | n/a — Stage 1b blocked |
| split / panel | `diagnostics/editing_v2_matched_validation_reserve_ids.json.gz`, key `training_source_keys` | `ba9270faf8bea1a67ad103c6da887efeef257a66fcb9ca1c3faf088f770994a1` | yes, recomputed |
| held-in cohort | `diagnostics/retarget_calibration_cohort.json` | file `e1ab574bc175c7a4d5af0e07678cc9d637fc103aba84702efcd6df52e998b5ed`; declared `cohort_sha256 e402e318…` | yes, recomputed |
| sampling law | none — no sampling performed | n/a | n/a |
| goal/oracle | none — no oracle called | n/a | n/a |
| eligibility bands | Lane 2 `pathwise_constraints.py` @ `bcc4a40` | reused verbatim; see `DECISION_LOG.md` D4 | yes, by read |

# Protocol

- **Panel construction:** first `n` eligible sources in seeded shuffled scan order from the 21,166 eligible held-in keys. No ranking, no successor enumeration, no arm outcome. **Not yet drawn.**
- **Arms:** `unconstrained_verified`, `posthoc_scaffold_filter_verified`, `hard_scaffold_mask_verified` (declared `SUPPORT_ABLATION`), optional `hard_scaffold_mask_greedy`.
- **Primary metric:** feasible-output yield at fixed resources, arm 3 vs arm 2.
- **Secondary metrics:** support retention, mask-empty rate, terminal objective improvement, edit count/diversity outside the core, scaffold-relative edit locality, applicability, chemical-envelope fidelity, and all three oracle counters.
- **Independent statistical unit:** the source molecule.
- **Allowed calibration:** none performed.
- **Stop rules:** `PROTOCOL.md` §9. One is **already triggered** — applicability 22.03%.
- **Forbidden adaptations:** `PROTOCOL.md` §10.

# What was implemented

- Stage-0 executor semantics audit, with four reproducible probes.
- Model-free protected-core census over a 30-source cohort and the full 96,094-source held-in pool.
- The hard-mask injection point (`constraints_hard_mask.py`) — filter + renormalize. `DESIGN_ONLY`; no controller.
- 13 tests including the sign-guarantee adversarial fixtures.
- Baseline qualification from primary sources.

**Not implemented, deliberately:** any controller, any Modal app, any panel draw, any fiber measurement.

# Tests and smoke checks

| Test | Result | Artifact |
|---|---|---|
| `tests/test_hard_scaffold_constraint.py` | **13 passed** | — |
| `probes/identity_probe.py` | **PASS** — verdict `LABELED_SUBGRAPH_PRESENCE_INVARIANT` | stdout |
| `probes/gate0_local_authentication_repro.py` | **BLOCKED as expected** | stdout |
| scaffold parent-index cross-check | **0 failures / 30** | census artifact |

# Results

**These are `SMOKE_HELD_IN` (census) and `DESIGN_ONLY` (everything else). None is
claim-bearing. No fiber quantity was measured.**

| Metric | Arm / condition | Value | Uncertainty / denominator |
|---|---|---:|---|
| eligible under frozen bands | held-in pool | **0.2203** | 21,166 / 96,094 |
| BM core fraction | held-in pool | **median 0.7895** | mean 0.7535, n = 96,094 |
| editable atoms outside core | held-in pool | **median 5.0** | mean 6.37, n = 96,094 |
| eligible under frozen bands | 30-source cohort | 0.200 | 6 / 30 |
| BM core minus largest ring system | 30-source cohort | median **+12** atoms | n = 30 |

# Gate verdicts

| Gate | PASS / FAIL / INCONCLUSIVE | Evidence |
|---|---|---|
| Stage 0 — semantics resolvable | **PASS** | 4 probes; `LABELED_SUBGRAPH_PRESENCE_INVARIANT` |
| Stage 1a — protected object usable | **INCONCLUSIVE, leaning FAIL** | usable on 22.03% of sources only; median 5 editable atoms |
| Stage 1b — V3/V4a/V4b/V5a/V5b | **BLOCKED** | Gate-0 local authentication fails |
| Applicability stop rule | **TRIGGERED** | 22.03%; disclosed, not hidden |

# Bugs, invalid instruments, and superseded runs

**Bug 1 — protected-object rule dropped exocyclic double-bonded atoms.**

- **Wrong:** `murcko_atom_indices` pruned terminal non-ring atoms only, dropping ring carbonyl oxygens, so the protected object was not Bemis–Murcko.
- **Detected by:** the census's own `scaffold_parent_index_crosscheck` against `MurckoScaffold.GetScaffoldForMol` — **16 / 30 disagreed**.
- **Did a conclusion depend on it?** Yes, transiently: the uncorrected run reported 8/30 eligible and median core fraction 0.7524. Corrected: 6/30 and 0.8000. **No claim-bearing artifact was produced from the uncorrected rule**, and the first census file was overwritten by the corrected run in the same session.
- **Artifact status:** superseded in place; both values recorded in `SCAFFOLD_FEASIBILITY.md` §2.3.
- **Corrective commit:** `4f314c5`.

**Known-stale artifact belonging to another lane (flagged, not fixed):**
Lane 3's `FAIRNESS_CONTRACT.md:64` still records the GraphXForm heavy-atom
ceiling as "42 = 50 − 8" with `headroom_basis: CHOSEN_SAFETY_MARGIN`, while its
code has `REQUIRED_HEADROOM = 0` / `NONE_NATIVE_CEILING_ONLY`. **The code is
authoritative.** Not this lane's file to change.

# Known limitations

- **The fiber half of Stage 1 was never measured.** Every retention, mask-empty
  and core-violation number in the protocol is a *design target*, not a result.
- **The panel is a biased subpopulation** — small-core, heavily-decorated
  molecules. Must be disclosed in the paper.
- **RDKit 2025.09.6 was used locally; production pins 2024.03.5.** The artifact
  records `rdkit_pin_matches_production: false`. Any claim-bearing rerun must use
  Lane 3's pinned shared evaluator.
- **The predicate implementation is not written.** It must reuse Lane 2's
  `fragment_smarts` / `preserves_motif`, which live on an unmerged branch.
- **Experiment A overlaps Lane 2's Stage A** substantially — see `PROTOCOL.md` §13.

# Exact reproduction commands

```bash
# environment/setup — nothing to install; CPU only
export PYTHONPATH=src

# Stage 0 — executor semantics (no checkpoint, no network, <2s)
python3 docs/workstreams/constraints-hard/probes/identity_probe.py

# Stage 1b blocker — reproduce, do not bypass
python3 docs/workstreams/constraints-hard/probes/gate0_local_authentication_repro.py \
    --local-runtime /path/to/local_runtime

# Stage 1a — model-free census
python3 scripts/constraints_hard_scaffold_census.py \
    --cohort diagnostics/retarget_calibration_cohort.json \
    --pool   diagnostics/editing_v2_matched_validation_reserve_ids.json.gz \
    --out    diagnostics/constraints_hard_scaffold_census.json

# tests
python3 -m pytest tests/test_hard_scaffold_constraint.py -q
```

# Durable artifacts

| Artifact | Path | SHA-256 | Purpose |
|---|---|---|---|
| census | `diagnostics/constraints_hard_scaffold_census.json` | `b9b1412351bd20b470f77737e79e882bde8523d60cf4fcae14ecb32f5284c7e5` | Stage 1a result |
| semantics verdict | `docs/workstreams/constraints-hard/CONSTRAINT_SEMANTICS.md` | `d26b80b12f5d1a2a1a9c05043b6909a3e184d72278d966815977427f0fa3c27e` | Stage 0 |
| feasibility | `docs/workstreams/constraints-hard/SCAFFOLD_FEASIBILITY.md` | `189fe270535a9b9527c3ba6c77ec84ecc88f37e3b54a8d634c963587d18654b1` | Stage 1 |
| protocol | `docs/workstreams/constraints-hard/PROTOCOL.md` | `e952beb5fead7fe4c6f3284b5336a14001bffc8be86f6eb4ca7e578749c5e2b7` | Stage 2 design |
| baselines | `docs/workstreams/constraints-hard/BASELINE_TASK_MATRIX.md` | `ba2eb40e1e42328d0320929606eb9a4336ac83f16322c86ae859c470a337f510` | qualification |

Full list with digests: `handoff.json`.

# Files changed

```text
docs/QUEUED_EXPERIMENTS.md                  (appended, not rewritten)
docs/workstreams/constraints-hard/*.md
docs/workstreams/constraints-hard/handoff.json
docs/workstreams/constraints-hard/probes/*.py
scripts/constraints_hard_scaffold_census.py
src/compose_v4/experiments/constraints_hard_mask.py
tests/test_hard_scaffold_constraint.py
diagnostics/constraints_hard_scaffold_census.json
```

No file belonging to Lane 2, Lane 3, Lane 4 or Lane 5 was modified.

# Recommended next action

> **Check CDD's venue and full threshold set against the proceedings before any
> citation.** The charter described it as NeurIPS 2025; the arXiv comment on
> 2503.09790 is consistent with an ICML 2025 submission, and acceptance could not
> be confirmed. Until that is resolved, cite no venue. Then decide whether the
> related-work paragraph in `EXTERNAL_HARD_CONSTRAINT_AUDIT.md` §5 goes into the
> paper as drafted.

# Actions explicitly not recommended

- Do **not** substitute a smaller protected core — no Bemis–Murcko-lite, no
  pharmacophore, no hand-tuned core. Any smaller core would be chosen *because*
  it leaves room to act.
- Do **not** invent a new constraint from our own failed scaffold result.
- Do **not** widen `CORE_FRACTION_BAND` to raise the 22.03% applicability.
- Do **not** design or build the internal post-hoc / soft-guidance / hard-mask
  contrast here — it belongs to Lane 2.
- Do **not** present any CDD number as a rerun, or quote a CDD SA number without
  its satisfaction rate.
- Do **not** report COMPOSE's 100% constraint satisfaction as an empirical win
  against CDD's 21.3% — that compares a construction guarantee to a measurement.
- Do **not** attempt a de novo head-to-head: there is no de novo `R_theta` in the
  frozen chain and CDD's operative surrogate was never released.
- Do **not** patch the Gate-0 decision to unblock a local fiber census.
- Do **not** revive the withdrawn hand-driven GraphXForm action loop, or add a
  deletion action to GraphXForm.

# Main-session pickup checklist

- [ ] Read `PROTOCOL.md` before any result.
- [ ] Verify frozen-input hashes against `handoff.json`.
- [ ] Confirm held-out-open status: **no**.
- [ ] Reproduce `probes/identity_probe.py` (2 seconds, no checkpoint).
- [ ] Inspect the superseded uncorrected scaffold rule (`DECISION_LOG.md` D5).
- [ ] Decide: MERGE / AUTHORIZE NEXT GATE / STOP-REVISE — and answer §13.
