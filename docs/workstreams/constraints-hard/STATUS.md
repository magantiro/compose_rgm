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

## Delivered — the role relabel and the trilemma

Under `docs/COMPARATOR_ROLES_CANONICAL.md` (comparators have **roles, not
rankings**; selection is **framework-first**):

- **`BASELINE_TASK_MATRIX.md`** relabelled by the four roles, with runnability
  kept **orthogonal** to role — a `FRAMEWORK_NEIGHBOR` can be unrunnable and
  still carry the section.
- **`CONSTRAINTS_SECTION_TRILEMMA.md`** — the section's organizing claim, every
  cell sourced, every satisfaction rate carrying its denominator.
- **`LANE2_FIVE_QUESTIONS.md`** — the five-question block for the internal
  three-arm comparison, recorded before implementation, handed to Lane 2.

**The framework layer carries no numerical rows**, and that is a finding rather
than a gap: the canonical policy makes CDD/PRODIGY/ConStruct numerical *only
under a native common protocol*, and the audit tested that condition and found
none.

Under `docs/AMENDMENT_PUBLISHED_NUMBER_FIRST.md`, each method also carries a
**tier**. The result is stark and worth flagging: **no method in this block is
tier 1 or tier 2.** All three framework neighbours are tier 4 (*cite and discuss,
do not reconstruct*); the two competence comparators are tier 3. **The
constraints section cannot be carried by published numbers**, which is precisely
why the internal matched control is the primary comparison.

**The useful move within tier 4:** CDD's `SA(y) ≤ τ` predicate at
τ ∈ {3.0, 3.5, 4.0, 4.5} is reusable as an **externally defined task**, because
the constraint is then independent of our corpus — structurally unable to repeat
the outcome-selection defect that killed the Bemis–Murcko branch. Three
verbatim-reproduction conditions in `BASELINE_TASK_MATRIX.md` §3a, including
pinning our own RDKit version since CDD pins none. **The task travels; the
numbers do not.**

## Delivered — the SA census design (2026-08-14)

The externally defined constraint `SA(y) ≤ τ` at CDD's published thresholds.

- **`SA_CENSUS_PROTOCOL.md`** — the census design, gates, kill rule, costed plan,
  Gate-0 blocker, and the five-question block for the **two-arm** experiment.
  **The τ rule was committed at `034c294`, before any measurement existed.**
- **`SA_CENSUS_APPLICABILITY.md`** — the model-free half, `SMOKE_HELD_IN`.

**Predicate is instantiable exactly as published**, zero new scoring code.
**Primary τ = 3.0** on CDD's authority (their Figure 4 novelty metric); all four
reported as a predeclared constraint-strength curve, which costs ~nothing since
the fiber is enumerated once and masked four times.

**Applicability: 49.79%** of the held-in pool at the primary τ, against **22.03%**
for the scaffold. **Median slack `τ − SA(x₀)` is 0.1097** — the median eligible
source sits on the boundary, which is what will make the constraint bite *and*
what threatens the retention floor.

**Declared in advance:** it is possible that **no τ passes all gates
simultaneously**. If so the constraint dies — no rescue predicate, no threshold
shopping.

**The fiber half is BLOCKED** on Gate-0, as before. The highest-value quantity it
would return is the **distribution of ΔSA per legal rewrite**, deliberately not
proxied per the MMP rule.

## Next action

**None from this lane.** Three items belong to whoever picks this up:

1. **Confirm the role label for CDD / PRODIGY / ConStruct.** Kept as
   `FRAMEWORK_NEIGHBOR` with `evidence mode: conceptual only`. A strict reading
   of the four definitions would make them `CONCEPTUAL_LINEAGE_ONLY`. Flagged in
   `BASELINE_TASK_MATRIX.md`, not decided.
2. Decide whether the related-work paragraph in
   `EXTERNAL_HARD_CONSTRAINT_AUDIT.md` §5 goes into the paper as drafted, and
   whether to cite ConStruct's Appendix G.2 as the nearest published analogue.
3. **Lane 2** picks up `LANE2_FIVE_QUESTIONS.md` and builds the three-arm
   comparison.

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
