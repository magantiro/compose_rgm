# Where every external-baseline finding lives

**Consolidated into the main lane.** Before this, the *rules* lived here while
every *finding* they were written from lived on a lane branch the main lane could
not see. Nothing was lost — everything was pushed — but no single authoritative
record existed.

## Policy (main lane)

| doc | holds |
|---|---|
| `docs/COMPARATOR_ROLES_CANONICAL.md` | the four roles; framework-first selection; same-lab policy |
| `docs/BASELINE_IMPLEMENTATION_POLICY.md` | native methods + thin adapters; the creep stop rule |
| `docs/AMENDMENT_PUBLISHED_NUMBER_FIRST.md` | the four tiers |
| `docs/PARETO_COMPARATOR_MATRIX_REQUIREMENT.md` | per-block comparator map; the Pareto gate |
| `docs/BENCHMARK_RULINGS.md` | DDSBM tier-1, DRD2 demoted, T3 feasibility-failed, InVirtuoGen restriction |
| `docs/DDSBM_ENDPOINT_COMPETENCE_PROTOCOL.md` | the frozen tier-1 protocol |

## Evidence (now also here)

| method | verdict | primary-source record |
|---|---|---|
| **DDSBM** | ⬇️ **REMOVED FROM THE MANUSCRIPT** — developmental transport stress test. Executed 5,984/5,984; result preserved unrescued in `DDSBM_FROZEN_RESULT.json`. Native question is distribution transport, not source-conditioned editing | `AMENDMENT_EDITFLOWS_GRIDDD_HEADLINE.md` |
| **HN-GFN** | `TASK_COMPETENCE`; GPU-only, no checkpoint, empty-molecule start, surrogate asymmetry | `multiobjective/BENCHMARK_ALIGNMENT_AUDIT.md` |
| **InversionGNN** | not runnable as shipped; no LICENSE; 70× budget disagreement with HN-GFN | `multiobjective/BENCHMARK_ALIGNMENT_AUDIT.md` |
| **GrIDDD** | ⬆️ **PROMOTED — primary numerical external comparator** on the Jin/ZINC **QED** task. The old `CONCEPTUAL_LINEAGE_ONLY` demotion cited insert/delete being inert on DRD2; that is true for DRD2/logP and **false for QED (45.1 %→33.8 %)**. See `AMENDMENT_EDITFLOWS_GRIDDD_HEADLINE.md` | `multiobjective/FRAMEWORK_NEIGHBOR_GRIDDD.md` |
| **GraphXForm** | `TASK_COMPETENCE`; **capability-disqualified — cannot delete** | `multiobjective/EDITING_COMPETENCE_AUDIT.md` |
| **InVirtuoGen** | selected editor; **output may never become `R_θ` training data** | `multiobjective/EDITING_COMPETENCE_AUDIT.md` |
| **OP-GFN** | excluded — CC BY-NC-**ND** | `multiobjective/BASELINE_TASK_MATRIX.md` |
| **CDD** | tier 4; predicate transfers verbatim, implementation does not | `constraints-hard/EXTERNAL_HARD_CONSTRAINT_AUDIT.md` |
| **PRODIGY** | tier 4 — no LICENSE | `constraints-hard/EXTERNAL_HARD_CONSTRAINT_AUDIT.md` |
| **ConStruct** | lineage; its G.1 planarity negative mirrors our scaffold stop | `constraints-hard/CONSTRAINTS_SECTION_TRILEMMA.md` |
| **GraphGA · MARS · REINVENT** | qualification records, 2026-08-12 | `baseline-qualification/` |
| **MOG-DFM · AReUReDi · pCoMole · PepTune** | same-lab lineage; `pCoMole` still `UNVERIFIED` | `*/SAME_LAB_LINEAGE.md` |

## The registry

`configs/comparator_registry_v3.json` — **20 comparators, 6 with qualification
records.**

The main lane previously carried a **stale** copy: every row `not_yet_qualified`
with `qualification_record: None`, predating the qualification work. The
baseline-qualification version was adopted after verifying **field-by-field that
no main-lane field was absent from it**. It adds `HN_GFN` and the
`external_qualification_registry` section, and updates `adapter_status`,
`qualification_record` and `note` on DDSBM, GraphGA, GraphXForm, MARS, REINVENT.

Per-method evidence detail:
`docs/workstreams/baseline-qualification/comparator_registry_v3.json` (128 KB).

## Still open, and owned by a human

- **`pCoMole`** — OpenReview is not machine-reachable and it is not on arXiv. All
  ten schema fields remain `UNVERIFIED`. Needs a login the lanes do not have.
- **ParetoFlow, A-GPS** — never audited; new work for the Pareto matrix gate.
