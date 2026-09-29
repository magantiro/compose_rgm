# Inputs to the 2026-09-26 submission-state report

**Why these are here: until this commit they existed only in `/tmp`.**

`diagnostics/compose_status_report/report_20260926.md` (commit `1b0f3a6a`, on branch
`pmo-chain-ablation-20260925`) is the results snapshot taken ~4 hours before the
2026-09-26 08:49 EDT deadline. `scripts/reductions_20260927/full_report.py`
generated every table in it, and that script's own README says it consumes
`/tmp/t4_panel4.json`, `/tmp/t4_rev4.json`, `/tmp/armB_final.json` and
`/tmp/armC_metrics.json` — "session scratch names, not a stable interface".

The reduction *scripts* were promoted out of `/tmp` for exactly the right reason
("a report nobody can regenerate is not a result"). Their *inputs* were not.
These four files survived by luck; `/private/tmp` is reaped mid-session, which
has already cost this repo a worktree and nearly cost it a 70,301-entry corpus.

| file | rows | what |
|---|---|---|
| `t4_panel4.json` | 60 | T4 replicate panel, one row per cell (10 complete at snapshot) |
| `t4_rev4.json` | 15 | T4 revival arm (11 complete at snapshot) |
| `armB_final.json` | 18 | PMO arm B, per task/seed: best, top10, auc, state |
| `armC_metrics.json` | 18 | PMO arm C, same shape plus `seconds` |

Row counts match the report's own header exactly: "10 of 60 panel cells
complete, 11 of 15 revival cells complete, PMO A/B 14 of 18 matched pairs
complete, PMO C 17 of 18".

    armB_final.json    3cfe607b6ed1f2e666e56d646282966167e215a4adffd0c9713d3c78b6c69eb6
    armC_metrics.json  379ebf6284b95b17fbc0030160c3caa23552adc4b7b200b0b5a6c7680f4f463b
    t4_panel4.json     244b0022a78d37e30716c5dc8e36e8855320d607fd6eda47d0eca00ae9022893
    t4_rev4.json       18af1290ffe52ba4eaa12ef67b052d989077ed9f0e2fd2b39aa4a615d96cbf12

## What this does and does not buy

It makes the report's **arithmetic** reproducible offline: `full_report.py`
repointed at this directory rebuilds its tables without touching Modal.

It does **not** make the report's *numbers* re-derivable from scratch. Those came
from live Modal volumes across three profiles (`nitya`, `rahul`,
`rahul-94866`), and `MODAL_PROFILE` is not inferable from a volume name. If
those volumes are gone, these files are the only record.

## Read the report's own status line before quoting anything from it

> Nothing here is a finished panel; every unfinished value is best-so-far.

At the snapshot the T4 replicate panel was 10/60 complete with a ~37 h ETA. The
per-cell `state` column (`D` = that replicate finished, `NNr` = rounds so far)
is what says which values are final. A mean taken over a column containing
in-flight cells is a lower bound on COMPOSE, not a result.

## Not the submitted paper's table

The workshop package (`paper_gem_neurips2026`) typesets
`tables/t4_summary_row.tex` from `diagnostics/t4_combined_table.json` — GenMol /
RetMol / GraphGA at 500 calls per cell. This report is a later, broader snapshot
that adds InVirtuoGen and replicates 2 and 3. See `experiments/t4/README.md` for
the three distinct T4 result sets and which one is which.
