# Decision log — hard structural constraints (Lane 6)

All decisions 2026-08-13, base commit `f6146d7`, branch
`codex/compose-constraints-hard`. No decision below changes a frozen object
belonging to another lane.

---

## D1 — Claim language frozen to `LABELED_SUBGRAPH_PRESENCE_INVARIANT`

**Evidence available before the decision:** four probes on the executor, run
before any wording was written — micro-operators are slot-preserving; the
production controllers carry the trajectory as canonical SMILES and re-parse at
every step; alias collapse merges successors retaining disjoint atom sets;
canonicalization drops stereochemistry and isotopes.

**Alternatives rejected:** (a) claim `IDENTITY_INVARIANT` and add an
atom-tracking wrapper — rejected because carrying `successor.state` and refusing
alias collapse would change the frozen canonical-successor semantics, i.e. would
study a different process; (b) leave the wording open until results — rejected,
this is exactly the decision the charter required *before* outcomes.

**Changes a frozen object:** no. **Commit:** `4f314c5`.

## D2 — The mask filters the enumerated fiber; the `constraints=` hook is not used

**Evidence:** Probe 4 — `RewriteSystem.apply` raises, and
`canonical_successor_result:557-567` converts any executor exception into a
fatal error. A constraint installed there would crash the kernel rather than
filter it, and would blind a guard that exists to detect genuine model/executor
mask disagreement.

**Alternatives rejected:** (a) install the predicate as a `Constraint` —
rejected on the above; (b) relax the kernel's `try/except` to skip rejected
marks — rejected because it would disable a real defect detector for the
convenience of this lane.

**Changes a frozen object:** no — the kernel is untouched. **Commit:** `4f314c5`.

## D3 — Protected object frozen as the atom/bond-labeled Bemis–Murcko scaffold

Frozen **before** any census number existed, per the charter. Match semantics
(element, formal charge, bond order/aromatic class in; implicit-H,
stereochemistry, isotope out) frozen with written reasons in
`CONSTRAINT_SEMANTICS.md` §6.

**Alternatives rejected:** matching on implicit-H count — would forbid all
decoration and strangle the constraint; fingerprint similarity — explicitly
barred by the workstream doc.

**Changes a frozen object:** no. **Commit:** `4f314c5`.

## D4 — Eligibility bands reused verbatim from Lane 2, not minted

`MIN_CORE_ATOMS=6`, `CORE_FRACTION_BAND=(0.20, 0.70)`, `MIN_FREE_ATOMS=8`,
`HEAVY_ATOM_BAND=(18, 38)`, taken from
`src/compose_v4/experiments/pathwise_constraints.py` at `bcc4a40`.

**Declared change:** Lane 2 applies them to the largest fused ring system; we
apply them to the Bemis–Murcko scaffold, a superset (median +12 atoms here).
Applying an unchanged band to a strictly larger object is the **conservative**
direction. Declared as a reuse under a different protected object, not as the
same rule.

**Changes a frozen object:** no. **Commit:** `4f314c5`.

## D5 — Scaffold rule corrected after the cross-check failed. Recorded, not hidden.

**Evidence before the decision:** the census's own
`scaffold_parent_index_crosscheck` reported **16 of 30** sources disagreeing
with `MurckoScaffold.GetScaffoldForMol`.

**Cause:** Bemis–Murcko keeps exocyclic atoms double-bonded to a scaffold atom
(ring carbonyl oxygens); the initial pruning-only rule dropped them.

**Decision:** add rule 3 to `murcko_atom_indices`; keep the cross-check as a
permanent regression guard; record **both** runs — uncorrected 8/30 eligible and
median core fraction 0.7524, corrected 6/30 and 0.8000.

This decision was made on a **bug detector**, not on an outcome. It is logged
because the correction moved a headline number and must be auditable.

**Changes a frozen object:** yes — the protected-object rule, before any
fiber-level or claim-bearing measurement existed. **Commit:** `4f314c5`.

## D6 — Applicability reported as a result; bands NOT widened

**Evidence:** 22.03% eligible across 96,094 held-in sources; 71.7% fail the core
fraction band; median core fraction 0.7895.

**Alternatives rejected:** (a) widen `CORE_FRACTION_BAND` — minting a threshold
from the result it must clear; barred; (b) hand-pick a smaller sub-scaffold per
source — selecting a scaffold after seeing the full one is inconvenient;
explicitly barred by the charter.

**Decision:** keep object and bands; draw the panel from the 21,166 eligible
held-in sources under the frozen rule; **disclose 22.03% in the paper** and scope
the claim to that subpopulation.

**Changes a frozen object:** no. **Commit:** this one.

## D7 — Gate-0 blocker reported, not bypassed

**Evidence:** `open_process_v2_t1_source` fails locally with
`ProcessV2T1PanelError: the Gate-0 PASS names another Active8/process input`,
matching the diagnosis in `docs/PARETO_PARITY_ENVIRONMENT_STATUS.md`. The model
is built *from* the authenticated source, so no partial local path exists.

**Alternatives rejected:** patching the decision body — the chain is closed by a
canonical-JSON framing guard and a decision self-hash, and the standing
instruction is to report the blocker rather than route around an integrity guard.

**Changes a frozen object:** no. **Commit:** this one.

## D8 — GraphXForm's hard-constraint cell set to `N/A — not natively supported`

**Evidence:** primary sources. `include_structural_constraints` is a hard-coded
solvent-chemistry feasibility filter (paper §3.3.2,
`molecule_evaluator.py::infeasible_by_special_constraints`), gated on
`objective_type in ["IBA","DMBA_TMB"]`, taking no user SMARTS. The action space
is `DontChange`/`AddAtom`/`AddBond` with no deletion (§2.1); §3.3.3 defers
removal to future work.

**This contradicts the charter's stated premise** that GraphXForm supports
"preserving/excluding specified moieties". The primary source wins.

**Consequence:** GraphXForm stays in the table as a source-conditioned editor
(`start_from_smiles` is real and native) but its hard-constraint cell is `N/A`
with the reason stated. The "we did not invent scaffold preservation"
disclaimer now rests on Prompt-MolOpt^P and InVirtuoGen, which genuinely have it.

**Alternatives rejected:** adding a deletion action to GraphXForm — that is
baseline engineering creep and would stop being GraphXForm.

**Changes a frozen object:** no. **Commit:** this one.

## D9 — Prompt-MolOpt^P selected as the one direct structure-preserving editor

**Evidence:** MIT licence; official repo; a **separately released ^P checkpoint**
(`checkpoints/fs_mga_remark_data/model.500000_99.pt`, distinct training script
and dataset), not a paper-only ablation; preservation is **hard by construction**
— the network emits only the replacement fragment and the retained part is
rebuilt from `Chem.RWMol(retain_sub)`.

**Alternatives rejected:** InVirtuoGen — genuine hard clamp but at fragmented-
SMILES *token* positions rather than graph atoms, and CC-BY-NC-SA
(non-commercial); kept `CONDITIONAL` pending coordination. MolEditRL — no
official code found (`OFFICIAL_CODE_UNAVAILABLE`) and its preservation is soft
(KL-to-prior), so it would not be a hard-constraint comparator even if released.

**Changes a frozen object:** no. **Commit:** this one.

## D10 — Recommend folding the yield contrast into Lane 2 rather than duplicating it

**Evidence:** `diagnostics/pathwise_constraints_smoke.json` at
`codex/compose-pathwise-constraints`. Lane 2's Stage A already ran this arm
structure at n = 6, and its `endpoint_only_return_failure` block **is** our
chartered primary contrast, with a correct free-sign argument: `endpoint_only`
returned nothing on 2/6 sources; masked arms returned on 6/6. Their G1 FAIL
closed the *trajectory* framing (destruction is absorbing, `recovery_rate = 0.0`)
without touching the *yield* framing.

**Decision:** report the overlap explicitly and recommend folding, rather than
standing up a parallel harness — the charter warns against creating another
broad optimization lane.

**This is a recommendation, not an action.** No lane's files were modified.

**Changes a frozen object:** no. **Commit:** this one.

---

# Reframe — 2026-08-13: lane narrowed to an external audit

## D11 — Scaffold protocol marked `SUPERSEDED`, not deleted

**Evidence before the decision:** the lead accepted (a) this lane's own §13
recommendation to fold the yield contrast into Lane 2, and (b) the Bemis–Murcko
feasibility stop.

**Decision:** `PROTOCOL.md` carries a `SUPERSEDED` header with both reasons.
`SCAFFOLD_FEASIBILITY.md` keeps `SMOKE_HELD_IN` and gains a header stating that
the measurement stands while the experiment it fed does not.

**Alternatives rejected:** deleting the protocol — the negative cost a
96,094-source census and is a real recorded result.

**Changes a frozen object:** yes — withdraws this lane's experiment design.

## D12 — The scaffold stop binds against substituting a smaller core

**Decision:** no Bemis–Murcko-lite, no reduced ring core, no pharmacophore
substitute, no hand-tuned protected core mined from the held-in corpus. **No new
constraint may be invented from our own failed scaffold result.**

**Reason, written out so it survives a future reader:** we now know the full
scaffold leaves a median of ~5 editable atoms. Any smaller core proposed *after*
that measurement would be chosen precisely because it leaves enough room to act.
That is selection on the outcome and task shopping under another name.

**Changes a frozen object:** no — it forecloses a move that was never frozen.

## D13 — CDD classified contextual, non-head-to-head

**Evidence:** five independent mismatches, in `EXTERNAL_HARD_CONSTRAINT_AUDIT.md`
§4. The decisive ones: CDD's molecular experiment is **de novo, unconditional
QM9 SMILES generation** against COMPOSE's source-conditioned editing; its
optimised predicate is a **GPT-2 (124M) surrogate**, not `sascorer`; and its
reported **0.0% violations** are computed **over valid molecules only**, with
validity collapsing **895 to 353** at tau = 3.0.

**Alternatives rejected:** (a) quoting CDD's QED numbers head-to-head — the task,
the dataset and the predicate all differ; (b) reconstructing CDD to run it — the
official repo is a placeholder README and the surrogate is unreleased, so this is
`BASELINE_IMPLEMENTATION_POLICY` option 3; (c) using either third-party
reimplementation as "the published method"; (d) reporting COMPOSE's 100% against
CDD's rate — comparing a construction guarantee to an empirical measurement,
barred by the sign-guarantee rule.

**Changes a frozen object:** no.

## D14 — The SA predicate is instantiable verbatim; the mechanism is not

**Evidence:** `src/compose_v4/eval/molecular_quality.py:13` already imports
`rdkit.Contrib.SA_Score.sascorer` and exposes `sa_score`. `SA(y) ≤ τ` is a
boolean function of a complete molecule, and `F_C(x)` accepts any decidable
predicate — no differentiability required.

**Decision:** record that the *predicate* transfers with zero new scoring code,
and that this is **not** the same as instantiating CDD's *method*. Saying "we ran
CDD's constraint" while enforcing the true scorer they could not use would be
false.

**Changes a frozen object:** no.

## D15 — Two of this lane's own claims corrected against the camera-ready

A first-pass search produced two errors that a full-PDF reading overturned. Both
had already been reported upward, so both are recorded here rather than quietly
edited.

1. **"CDD's venue is unverified, possibly ICML 2025."** **Wrong.** It is
   **NeurIPS 2025** — camera-ready footer, DOI `10.52202/085713-0415`. My doubt
   came from the **v1** arXiv comment; the paper was also renamed between
   versions (v1: *"Constrained Language Generation with Discrete Diffusion
   Models"*). The charter was right.
2. **"CDD reports 21.3% satisfaction at tau = 3.0 and 63.9% at tau = 4.5."**
   **Wrong — those numbers are not in the paper.** CDD reports **0.0%
   violations at every tau**. The real critique is the denominator: violations
   are computed over valid molecules only, and validity drops 895 to 353.

**Why this is logged rather than fixed silently:** the project rule is that
agents fabricate provenance more readily than numbers, and that a wrong number
gets re-measured while a bad citation gets trusted. A correction that leaves no
trace is the failure mode the rule exists to prevent.

**Consequence:** the audit's substance survives — CDD is still not a structural
guarantee, and it is still not head-to-head comparable — but on different and
better-evidenced grounds.

**Changes a frozen object:** no.
