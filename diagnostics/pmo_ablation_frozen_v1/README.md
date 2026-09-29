# PMO ablation frozen inputs and reports — brought under version control

This is a verbatim copy of `~/compose_pmo_ablation`, which was an **unversioned plain
directory** outside any repository. It held the only local copy of arm A's per-seed metrics
(`inputs/pmo_1k_final.json`), which every A/B and A/B/C table in this project reads.

The home-directory original is deliberately left in place, because
`scripts/reductions_20260927/full_report.py` and the other reduction scripts reference
`~/compose_pmo_ablation/inputs/pmo_1k_final.json` by that path. **This copy is the durable
record; that path is the live one.** If you repoint the scripts, repoint them here.

Contents:

* `frozen_config_v1.json` — the frozen A/B protocol: tasks and seeds, the arm-B uniformity
  definition, `arm_a_seed_verification` (18/18 seeds verified by counting oracle receipts
  directly, with a withdrawn substitution recorded), and `charged_call_accounting` (the
  authoritative charged count is resolved `oracle/query_*/result.json`, NOT
  `progress.json:charged_oracle_calls`, which lags).
* `inputs/pmo_1k_final.json` — arm A per-task per-seed `best` / `top10` / `auc`. **Load-bearing.**
* `inputs/pmo_1k_{auc,clean,prov}.json` — earlier reductions of the same 1k runs.
* `MECHANISM_REPORT_PART{1..5}.md` — the code-grounded mechanism report.
* `CORRECTIONS_AND_BINDINGS.md` — corrections made during that report.
* `armC/FROZEN_ARM_C_V1.md` — the frozen arm-C specification.
* `armC/dev_panel.py`, `armC/dev_panel_v1.json` — the arm-C pre-launch development panel
  and its result.
