# STATUS — hard structural constraints (Lane 6)

**Status: COMPLETE.** Scope narrowed 2026-08-13 to an external audit; the audit
is delivered. Nothing internal was designed. Nothing was run.

**Branch:** `codex/compose-constraints-hard` **Base commit:** `f6146d7`

**Held-out data opened:** **NO.** **Modal launched:** no. **GPU:** no.
**Installed into the project environment:** nothing.

---

## Accepted into project record

1. **`LABELED_SUBGRAPH_PRESENCE_INVARIANT`** — the executor gives exact
   labeled-subgraph presence, not atom identity. `CONSTRAINT_SEMANTICS.md`.
2. **GraphXForm hard-constraint `N/A`** — `include_structural_constraints` is a
   hard-coded solvent filter over an add-only action space.
   `BASELINE_TASK_MATRIX.md` §0.
3. **The Bemis–Murcko feasibility stop** — median 78.95% core, ~5 editable
   atoms, 22.03% applicable. `SCAFFOLD_FEASIBILITY.md`.

## Superseded

**`PROTOCOL.md` — the internal scaffold experiment.** Withdrawn on two grounds:
the yield contrast folded into Lane 2 (this lane's own recommendation, accepted),
and the protected object failed its census. Retained, not deleted.

**The scaffold stop is binding and one-directional: no smaller core may be
substituted.** No Bemis–Murcko-lite, no pharmacophore, no hand-tuned core. Any
smaller core would be chosen *because* it leaves room to act — selection on the
outcome. **No new constraint may be invented from our own failed result.**

## Delivered — the audit

**`EXTERNAL_HARD_CONSTRAINT_AUDIT.md`.** CDD / PRODIGY / ConStruct, from primary
sources.

**Verdict:** the constraint *predicate* `SA(y) ≤ τ` is instantiable in COMPOSE
verbatim with zero new scoring code. **No mechanism is, and no published number
is head-to-head comparable.** Classification: **external hard-constraint
competence benchmark — contextual, non-head-to-head.**

Decisive finding: **CDD's constraint is not hard.** It enforces a differentiable
ML surrogate of SA, not RDKit `sascorer`, and reports **21.3% satisfaction at
τ = 3.0**, 63.9% at τ = 4.5.

## Next action

**None from this lane.** Two items belong to whoever picks this up:

1. **Check CDD's venue and full threshold set against the proceedings.** The
   charter said NeurIPS 2025; the arXiv comment suggests an ICML 2025
   submission. **Do not cite a venue until this is resolved.**
2. Decide whether the related-work paragraph in `EXTERNAL_HARD_CONSTRAINT_AUDIT.md`
   §5 goes into the paper as drafted.

## Deliverables

| file | status |
|---|---|
| `EXTERNAL_HARD_CONSTRAINT_AUDIT.md` | **the lane's final product** |
| `CONSTRAINT_SEMANTICS.md` | project record |
| `SCAFFOLD_FEASIBILITY.md` | standing negative result |
| `BASELINE_TASK_MATRIX.md` | verified; comparator placement recorded |
| `PROTOCOL.md` | **`SUPERSEDED`** |
| `SAME_LAB_LINEAGE.md` | framing freeze |
| `DECISION_LOG.md` / `HANDOFF.md` / `handoff.json` | handoff packet |
| `probes/`, `scripts/`, `tests/` | 15 tests passing |

## Unresolved risks

1. **CDD venue and thresholds unverified** — blocks any citation (HIGH).
2. **No CDD code exists** — a rerun is impossible, permanently.
3. **No de novo `R_theta` in the frozen chain** — the kernel dispatches to a de
   novo runtime, but no checkpoint trains it; blocks any de novo head-to-head.
4. The fiber half of the scaffold census was never measured (Gate-0 blocked) and
   now never will be under this lane.
