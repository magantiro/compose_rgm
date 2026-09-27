# Reduction scripts for the 2026-09-26/27 T4 and PMO reports

These produced every table in `diagnostics/compose_status_report/report_20260926.md` and
`diagnostics/pmo_abc_ablation_v1/reduction_v1.json`. They lived in `/tmp` during the session
and are promoted here because `/tmp` is reaped and a report nobody can regenerate is not a
result.

All read live Modal volumes. **The profile matters and is not inferable from the volume name**
(profile `rahul` addresses the `kosha-labs` workspace). Invoke with the right one:

    # PMO arm B  (workspace nitya)
    MODAL_PROFILE=nitya   python arm_metrics.py scored_chain_   > armB.json
    # PMO arm C  (workspace kosha-labs)
    MODAL_PROFILE=rahul   python arm_metrics.py scored_rebind_  > armC.json
    # T4 main replicate panel
    MODAL_PROFILE=rahul-94866 python t4_full.py v1     panel    > t4_panel.json
    # T4 revival arm
    MODAL_PROFILE=nitya       python t4_full.py rev-v1 revival  > t4_rev.json
    # T4 remaining-time estimate (writes /tmp/t4_eta_<suffix>.json as a side effect)
    MODAL_PROFILE=rahul-94866 python t4_eta.py v1
    # PMO arm C liveness, gentle on the volume rate limit
    MODAL_PROFILE=rahul python armc_health.py

`full_report.py` consumes `/tmp/t4_panel4.json`, `/tmp/t4_rev4.json`, `/tmp/armB_final.json`,
`/tmp/armC_metrics.json`, `~/compose_pmo_ablation/inputs/pmo_1k_final.json` and
`compose_t4_nitya/diagnostics/T4_FROZEN_RESULT_v1.md`. **Repoint those paths** before reuse;
they are session scratch names, not a stable interface.

## Environments, not interchangeable

    ~/compose_pmo_oracle_env      python 3.11 + rdkit 2023.9.6   PMO scripts
    ~/compose_region_pinned_env   python 3.11 + rdkit 2024.3.5   T4 scripts

## Two rate-limit lessons paid for in this session

* `Volume.listdir` on a campaign's `oracle/` directory per campaign trips
  `ResourceExhaustedError: VolumeListFiles rate limit exceeded`. Read `canary_v1.json` or
  `progress.json` instead — one read per campaign — and sleep ~0.25 s between campaigns.
* `Volume.listdir("/", recursive=True)` once per volume is far cheaper than walking it, which
  is why `t4_full.py` does that and then only opens `result.json`/`checkpoint.json`.
