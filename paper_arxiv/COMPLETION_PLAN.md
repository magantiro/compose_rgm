# COMPOSE manuscript completion plan

This is the working contract for turning `main.tex` into a complete,
submission-shaped paper before all experimental numbers exist. The manuscript
may be structurally complete while results are pending, but it may not imply
that a pending experiment succeeded.

## Evidence classes and status

Every material empirical or literature claim belongs to one or more of the
repository evidence classes:

1. **Measured.** Directly observed in an experiment or primary source data.
2. **Computed.** Produced by a versioned, reproducible analysis.
3. **Reported.** Taken from a verified external primary source.
4. **Inferred.** An explicitly labeled interpretation of measured, computed, or
   reported evidence.
5. **Proposed.** A future experiment, design choice, hypothesis, or untested
   mechanism.

Mathematical definitions and proved derivations are tracked separately as
non-empirical claims. A frozen implementation fact or registered protocol is a
provenance/status qualifier, not an evidence class and not a measured result. A
pending checkpoint identity, numerical measurement, empirical comparison, or
conclusion remains visibly marked with `\resultpending{KEY}`.

Historical or superseded measurements may appear only when explicitly labeled
diagnostic. They never fill a production result placeholder.

## Current authority boundary

`paper_arxiv/` is the preferred live long-form package. Scientific prose and
result scaffolding must agree with `docs/PAPER1_FRAMING_AUTHORITATIVE.md`,
`SCIENTIFIC_TRACEABILITY.md`, `docs/CLAIM_LEDGER.md`, the current comparator
registry, and the validated experiment registry.

The completed 16,000-step run is a measured diagnostic with 661,105 admitted
training traces and 32 snapshots. It did not select a checkpoint; ring opening
and Graft remain unresolved, and the inspected final-test aggregates are barred
from selection. Editing-V2 remains `DESIGN_NOT_TRAINING_AUTHORIZED`. Manuscript
completion does not authorize Gate 0, T1, P50, P500, P2000, a long editing run,
or the separate de-novo run.

The five corpus lanes are exactly `observed_local_analogue`,
`operator_aware_real_endpoint`, `linker_positional_topology_analogue`,
`real_endpoint_multistep_path`, and `reversible_synthetic_walk`. Do not add an
"observed series path" lane unless genuine observed action/trajectory
provenance is later acquired and separately contracted.

Candidate routing follows
`configs/editing_v2_candidate_routing_policy_v1.json`; callers do not choose a
lane label. The four-role split follows
`configs/editing_v2_split_assignment_policy_v1.json`, with prospective
85/5/5/5 targets over the census's declared mass unit for training, validation,
controller-validation, and final test. The source envelope must define that
mass unit before the split is interpreted scientifically. The routing and
split policies do not constitute an admitted corpus or training authority.
Keep physical shard, whole-trace Active8, and semantic-sampling blockers
visible until their exact artifacts pass.

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
| Appendices | algorithms, proofs, chemistry/support contract, data provenance, complete protocols, and audit inventory | written and visually audited | keep synchronized with frozen implementation contracts; replace only registered result fields |

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

1. **Figure 1 — Framework (complete).** Definition-only vector artwork showing
   cardinality strata, executable marks, the canonical successor quotient, and
   the control interface.
2. **Table 1 — Molecular rewrite basis.** Cardinality effect, semantic role,
   inverse/accelerator status, and final enabled support.
3. **Table 2 — Corpus and training contract.** Data lanes, split units,
   effective coefficients, objective, model, and selected checkpoint.
4. **Figure 2 — Learned transport and structural adaptation.** Learned,
   uniform-canonical, and state-independent empirical-family transport; size
   changes; topology changes; and registered operator ablations.
5. **Table 3 — Editing-prior capability results.** Successor likelihood,
   recovery, path efficiency, and required-family safeguards.
6. **Figure 3 — Exact and dynamic control.** Exact terminal tilt, quotient
   invariance, preference switch, Pareto fan, and pathwise constraints.
7. **Table 4 — Same-base controllers and search.** Hypervolume, dynamic
   adaptation, completely feasible trajectories, and oracle accounting under
   one frozen successor kernel.
8. **Table 5 — External task-level comparators.** Only compatible, independently
   runnable protocols with transparent support and compute differences.
9. **Table 6 — Separate de novo lane.** Unconditional metrics from the separate
   broad-organic timed-CTMC checkpoint.

## Comparator and result-key synchronization

The following rows are required by `configs/comparator_registry_v1.json`.
`PREIMPLEMENTATION` declares a prospective arm; it does not claim that the arm
is runnable or that a result exists.

| experiment | required rows | placeholder keys to keep stable |
|---|---|---|
| E2 | learned unguided COMPOSE; uniform canonical successor; state-independent empirical-family prior | existing `E2_*_COMPOSE` and `E2_*_UNIFORM`; add `E2_NLL_EMPIRICAL_FAMILY`, `E2_RECOVERY_EMPIRICAL_FAMILY`, `E2_OVERHEAD_EMPIRICAL_FAMILY` |
| E3 | full basis; no insertion; no deletion; fixed cardinality; no `bond_reroute` | existing full/fixed keys; add `E3_{SIZE_SUCCESS,VALID_PATH,COST}_{NOINSERT,NODELETE,NOREROUTE}` |
| E4 | full basis; no cycle operations; no `bond_reroute` | existing full/no-cycle keys; add `E4_{TOPOLOGY_SUCCESS,VALID_PATH,COST}_NOREROUTE` |
| E4 conditional | finite-catalog topology editor | render `PROSPECTIVE_UNAVAILABLE` while its run gate is closed; only after the gate passes add `E4_{TOPOLOGY_SUCCESS,VALID_PATH,COST}_CATALOG` |
| E7 required same-base | unguided; endpoint reranking; greedy one-step; local Boltzmann; static scalarization; MOG-DFM-style; SMC/Feynman--Kac; learned Doob/value; NSGA-II over COMPOSE successors | preserve existing keys and add `E7_{HV,AUC,FEAS,CALLS}_{BOLTZMANN,SCALARIZED,NSGA2}` |
| E7 conditional same-base | MOEA/D over COMPOSE successors; AReUReDi-style annealed locally balanced refinement | do not create result cells until the registered interface/run gate passes; if eligible, use the corresponding `E7_{HV,AUC,FEAS,CALLS}_{MOEAD,AREUREDI}` families |
| E7 dynamic | restart/replan; myopic retargeting; warm-start multiobjective search; reusable-transport dynamic control | add `E7_DYN_RESTART_REPLAN_{REGRET,COST}`, `E7_DYN_MYOPIC_{REGRET,COST}`, `E7_DYN_WARMSTART_{REGRET,COST}`, and `E7_DYN_REUSABLE_{REGRET,COST}` |

The four mechanistic switch controls remain separate from the four dynamic
method classes: continue from the switch state, restart from the original
source, restart from the switch state with a fresh sampler, and use one static
compromise objective. External rows for InVirtuoGen, GenMol, GraphGA, MARS,
RetMol, and HN-GFN appear only after their adapter and task-compatibility gates
are recorded. An unavailable or incompatible row is reported as such, not
silently dropped and not filled with a published number from another task.

The brace notation above denotes a family of explicit keys; the TeX source must
contain concrete `\resultpending{KEY}` calls for each displayed cell. The
placeholder inventory and this table must agree before a result-bearing build.

## Writing order

1. Keep the authority ledger, comparator rows, corpus-lane names, and empirical
   status synchronized across the arXiv and conference-formatted manuscripts.
2. Keep Sections 4 and 5 synchronized with each newly frozen implementation
   contract.
3. Preserve the complete Section 6 layout and fill only registered result
   slots.
4. Keep Sections 7 and 8 evidence-bounded as results arrive.
5. Keep the completed definition-only Figure 1 synchronized with any formal
   notation changes.
6. Fill E5 and E6 first because they do not require the selected editing
   checkpoint.
7. Fill E2--E4 after the editing checkpoint freezes.
8. Fill E7 after the frozen-base controller suite.
9. Fill E1 only from its separate de novo checkpoint.
10. Run the claim ledger, citation audit, visual PDF audit, and
   `make release-check`.

## Appendix inclusion rule

Appendix material must close a reproducibility gap, provide a proof omitted
from the main flow, or answer a likely reviewer question with evidence. It is
not a repository dump. Duplicate main-text exposition, raw hash listings,
superseded implementation archaeology, speculative task families, and tables
without a scientific estimand are removed during the page-budget pass.

The current appendix occupies 16 rendered pages. Algorithms and proofs begin
on page 20, the experimental protocol begins on page 26, and the
reproducibility inventory ends on page 35. This is a working page budget, not a
target to fill; material is added only when it satisfies the inclusion rule
above.
