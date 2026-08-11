# Estate Registry

One place to answer "is this current?" without reading the document.

**The rule, and it is the whole point:** exactly three files are current.
Everything else in this repository is history until it is linked from one of
them.

| Current | What |
|---|---|
| `docs/EXPERIMENT_PLAN.md` | the only experiment plan |
| `docs/PROJECT_BOARD.md` | the durable task board |
| `docs/DECISION_LOG.md` | what is established, and what was refuted |

This is a **whitelist, not a blacklist**, and that is deliberate. `docs/` holds
84 markdown files accumulated across several earlier framings of the project;
adjudicating each one is a treadmill, and new ones accrete faster than they can
be triaged. Defaulting everything to history costs nothing and cannot go stale.
**Absence of an ARCHIVED banner does not mean a document is current.**

## Paper directories

All four predate the canonical plan. None is declared the rewrite target; that
decision is open and tracked in `docs/PROJECT_BOARD.md`. Each carries a
`_STATUS.md`. All are preserved unmodified apart from that notice.

| Directory | Size | Last commit | Note |
|---|---|---|---|
| `paper/` | 792K, 48 files | 2026-07-27 | has `tables/`, most built-out table set |
| `paper_arxiv/` | 232K, 16 files | 2026-08-01 | carries `STYLE_SPEC.md` |
| `paper_iclr_control_substrate/` | 1.2M, 24 files | 2026-07-22 | oldest; pairs with the archived `docs/PAPER_REFRAME_CONTROL_SUBSTRATE.md` |
| `paper_iclr_stochastic_rewriting/` | 2.3M, 27 files | 2026-08-01 | designated "Paper 1" by `CLAUDE.md`; largest |

The two most recently committed (`paper_arxiv`, `paper_iclr_stochastic_rewriting`)
share a commit, "paper: define compose naming hierarchy". Filesystem mtimes are
all 2026-08-10 and are **not** evidence of recent editing — they reflect a
checkout, not authorship. Git dates are the reliable signal.

## Documents stamped ARCHIVED

Banner added in place; **not moved**, because ~35 inbound references across the
repo would have broken. References still resolve, and anything following one
meets the banner immediately.

```
paper/PAPER_STATUS.md                     paper_arxiv/COMPLETION_PLAN.md
docs/PAPER_MASTER_PLAN.md                 docs/PAPER1_FRAMING_AUTHORITATIVE.md
docs/PAPER_POSITIONING_EXACT_CONTROL.md   docs/PAPER_REWRITE_BRIEF.md
docs/PAPER_REFRAME_CONTROL_SUBSTRATE.md   docs/DEVELOPMENT_PLAN.md
docs/EXPERIMENT_INFRASTRUCTURE_PLAN.md    docs/BASE_ELEMENT_EXPANSION_PLAN.md
docs/EDITING_V2_CYCLE_OPEN_MIGRATION_PLAN.md
```

`paper/manuscript.pdf` is binary and cannot carry a banner; it is archived by
listing only.

## Known unresolved conflict

`CLAUDE.md` loads every session and its opening paragraph describes the project
as **"Rewrite Generator Matching … Generator Matching learns their contextual
firing rates."** The canonical plan says explicitly not to call the current
system Generator Matching — the present `R_theta` is a legitimate learned
stochastic jump process, but the name reopens an avoidable terminology dispute.

That framing paragraph has **not** been rewritten. It is a substantive claim
about the project's scientific identity, not bookkeeping, and changing it is
the author's call. Flagged in `docs/PROJECT_BOARD.md`.
