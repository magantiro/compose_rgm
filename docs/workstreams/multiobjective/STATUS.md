# STATUS — multiobjective workstream (Lane 5)

**Status:** Stage 0 and Stage 1 complete. **STOPPED at the Stage 2 boundary as
instructed.** Awaiting main.

**Branch:** `codex/compose-multiobjective-package`, based on
`codex/editing-v2-successor-fiber-fastpath` @ `f6146d7`.

**What is running:** nothing. **Compute spent: zero.** No Modal run, no GPU, no
baseline execution, nothing installed into any project environment.

**Held-out data opened:** **no.** This lane opened no data at all — it read
committed artifacts and public repositories.

**Last completed gate:** Stage 1 local qualification. 23 known-answer and
adversarial tests pass (`tests/test_multiobjective_qualification.py`).

**Next action:** main decides two open freezes (below), then authorizes or
declines Stage 2. **Stage 2 requires explicit per-run launch approval, and the
HN-GFN smoke additionally requires GPU authorization.**

---

## The one-line answer per method

| method | runnable | panel | blocker |
|---|---|---|---|
| HN-GFN | yes, on GPU | **B** | no released checkpoint; 10–13 GPU-h per run; BoTorch import broken upstream |
| InversionGNN | **no, as shipped** | **B**, conditional | no checkpoint; call/def arity mismatch; **no licence** |
| OP-GFN | n/a | — | **excluded**: not preference-conditioned; CC BY-NC-**ND** |
| MOG-DFM / AReUReDi / PepTune | n/a | — | related work; sequence state spaces |
| pCoMole | unknown | — | **primary document not read** |

## The recommended minimum defensible baseline set

**HN-GFN in Panel B, plus the internal generate-and-rank P3/P4 in Panel A.**
That is the whole set. Adding InversionGNN is worthwhile only if its two
conditions clear; adding OP-GFN is barred by licence; adding a same-lab method
is neither approved nor useful.

The internal comparator is the more mechanistically important of the two, and it
is Lane 4's to run.

## The three things most likely to be got wrong later

1. **HN-GFN's 1,000-call budget is spent against a surrogate.** Reported on an
   oracle axis alone it would look two orders of magnitude cheaper than COMPOSE
   while measuring amortization, not efficiency. `surrogate_calls` now exists as
   an axis and `oracle_efficiency_verdict` refuses the comparison outright.
2. **HN-GFN's front is global; COMPOSE's are per-source.** Never put them in one
   table as though they were one task. `assert_not_cross_panel` raises.
3. **HV-AUC's 12W/0L is not a cleaner 10W/2L.** It is a mixture whose early
   points are near-definitional; at `k = 5`, the only set-level point, the same
   committed trace is 10W/2L.

## Open freezes — main's decisions, not this lane's

- **Panel B's objective pair.** `{DRD2, QED}` admits HN-GFN but excludes
  InversionGNN; `{GSK3β, JNK3}` admits both externals but is not COMPOSE's
  frozen axis. Recommendation: `{GSK3β, JNK3}`. Must be frozen before any
  outcome exists.
- **Whether COMPOSE can enter Panel B at all**, given that its process is
  defined as editing from a supplied source.

## Escalation

**pCoMole (ICLR 2026 workshop, same lab, "Pareto-Constrained Molecule Editing
with Discrete Flows") has not been read.** OpenReview is not machine-reachable
and the paper is not on arXiv. Two fields — its state space and its guidance
horizon — decide whether it is a lineage citation or a prior-art question. A
human with an OpenReview login or an author contact must obtain it. Do not let a
submission go out with this cell unread.

## Deliverables in this directory

`STATUS.md` · `PROTOCOL.md` · `BASELINE_TASK_MATRIX.md` ·
`SOURCE_CONDITIONING_AUDIT.md` · `SAME_LAB_LINEAGE.md` · `DECISION_LOG.md` ·
`HANDOFF.md` · `handoff.json`

Code: `src/compose_v4/experiments/multiobjective_qualification.py`,
`tests/test_multiobjective_qualification.py`.

Every artifact carries `DESIGN_ONLY`.
