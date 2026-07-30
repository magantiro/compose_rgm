# COMPOSE manuscript completion plan

This is the working contract for turning `main.tex` into a complete,
submission-shaped paper before all experimental numbers exist. The manuscript
may be structurally complete while results are pending, but it may not imply
that a pending experiment succeeded.

## Evidence classes

Every substantive sentence belongs to one of four classes:

1. **Definition or derivation.** Mathematical content proved directly in the
   manuscript or fixed by the executable-process definition.
2. **Frozen implementation fact.** A property of a named, hashed artifact or
   production interface that has been independently checked.
3. **Registered protocol.** A metric, comparator, split, budget, or decision
   rule frozen before the associated result is inspected.
4. **Pending result.** A checkpoint identity, numerical measurement, empirical
   comparison, or conclusion that remains visibly marked with
   `\resultpending{KEY}`.

Historical or superseded measurements may appear only when explicitly labeled
diagnostic. They never fill a production result placeholder.

## Manuscript map

| Section | Content | Current state | What remains |
|---|---|---|---|
| Front matter | title, abstract, one-sentence thesis | written | revise numbers/conclusions after results |
| 1 | motivation, causal story, novelty boundary, related work | written | final citation audit |
| 2 | marked Rewrite Generator Matching and canonical pushforward | written | final notation audit |
| 3 | exact finite-horizon control and dynamic continuation | written | connect to final E5/E6 artifacts |
| 4 | COMPOSE molecular state, operators, network, corpus, and gates | drafted | freeze final architecture, corpus, sampler, objective, and checkpoint fields |
| 5 | experimental protocol, comparators, metrics, and accounting | drafted | freeze final tasks, seeds, and external adapters |
| 6 | results organized by scientific question | scaffolded | replace registered `XXX` keys only from frozen result artifacts |
| 7 | discussion and limitations | scaffolded | update only conclusions licensed by Section 6 |
| 8 | conclusion | scaffolded | one final evidence-bound revision |
| Appendices | algorithms, chemistry contract, data provenance, exact-control details | planned | expand alongside implementation freezes and page-budget review |

## Result-placeholder policy

- Each unresolved value or conclusion uses a stable uppercase key:
  `\resultpending{E2_TRANSPORT_NLL}`, not free-form prose such as “TBD.”
- The same key has one scientific meaning everywhere it appears.
- A result is filled only from a frozen, provenance-validated artifact named in
  `SCIENTIFIC_TRACEABILITY.md`.
- Failed or null results are entered as such; a placeholder is never removed
  merely to make the draft read more positively.
- `make placeholders` prints the live inventory.
- `make release-check` fails if any `\resultpending` or literal `XXX` remains.

## Planned figure and table sequence

1. **Figure 1 — Framework.** Disjoint cardinality strata, executable marks,
   canonical successor quotient, and control interface.
2. **Table 1 — Molecular rewrite basis.** Cardinality effect, semantic role,
   inverse/accelerator status, and final enabled support.
3. **Table 2 — Corpus and training contract.** Data lanes, split units,
   effective coefficients, objective, model, and selected checkpoint.
4. **Figure 2 — Learned transport and structural adaptation.** Learned versus
   uniform transport, size changes, topology changes, and operator ablations.
5. **Table 3 — Editing-prior capability results.** Successor likelihood,
   recovery, path efficiency, and required-family safeguards.
6. **Figure 3 — Exact and dynamic control.** Exact terminal tilt, quotient
   invariance, preference switch, Pareto fan, and pathwise constraints.
7. **Table 4 — Same-base controllers.** Hypervolume and oracle accounting under
   one frozen successor kernel.
8. **Table 5 — External task-level comparators.** Only compatible, independently
   runnable protocols with transparent support and compute differences.
9. **Table 6 — Separate de novo lane.** Unconditional metrics from the separate
   broad-organic timed-CTMC checkpoint.

## Writing order

1. Keep Sections 4 and 5 synchronized with each newly frozen implementation
   contract.
2. Preserve the complete Section 6 layout and fill only registered result
   slots.
3. Keep Sections 7 and 8 evidence-bounded as results arrive.
4. Replace the Figure 1 schematic slot with definition-only artwork.
5. Fill E5 and E6 first because they do not require the selected editing
   checkpoint.
6. Fill E2--E4 after the editing checkpoint freezes.
7. Fill E7 after the frozen-base controller suite.
8. Fill E1 only from its separate de novo checkpoint.
9. Run the claim ledger, citation audit, visual PDF audit, and
   `make release-check`.
