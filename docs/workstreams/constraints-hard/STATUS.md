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

**Decisive finding — it is about the denominator, not the rate.** CDD reports
**0.0% violations at every τ**, but *over valid molecules only*, and validity
collapses **895 → 353 (≈60%)** at τ = 3.0. Its formal result (Thm 4.1) is a
contraction bound assuming β-prox-regular `C`, with guarantees claimed only for
convex `C` — so the 0% is empirical, not structural. Gradients come from a
**GPT-2 (124M) surrogate**, not `sascorer`. And **no official code exists**: the
repo is a 59-byte placeholder README.

**Two of my own earlier claims were wrong and are corrected in the audit**: CDD
*is* NeurIPS 2025 (I doubted it from a v1 arXiv comment), and the "21.3% / 63.9%
satisfaction" figures I reported **do not appear in the paper**.

## Next action

**None from this lane.** One item belongs to whoever picks this up:

> Decide whether the related-work paragraph in
> `EXTERNAL_HARD_CONSTRAINT_AUDIT.md` §5 goes into the paper as drafted, and
> whether to cite ConStruct's Appendix G.2 as the nearest published analogue.

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

1. **No CDD code exists** — placeholder repo; a rerun is impossible unless the
   authors release it. Two unofficial third-party reimplementations exist and
   **must not** be used as "the published method".
2. **CDD's protocol is unstated, not merely unreleased** — no sample count, no
   seeds, no split, no error bars, no canonicalization, no RDKit version pin. Its
   τ boundaries are therefore not version-portable.
3. **PRODIGY's repo has no licence at all** — same blocker class as DDSBM.
4. **No de novo `R_theta` in the frozen chain** — the kernel dispatches to a de
   novo runtime, but no checkpoint trains it; blocks any de novo head-to-head.
5. The fiber half of the scaffold census was never measured (Gate-0 blocked) and
   now never will be under this lane.
