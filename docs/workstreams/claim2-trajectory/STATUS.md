# Claim 2 — trajectory characterization — STATUS

**Status:** `DESIGN_ONLY` — built, tested locally, frozen, **not run**.

**Branch:** `codex/compose-claim2-trajectory`
**Base commit:** `04f1c46` (`Add parallel workstream plan and agent handoff template`)

**Held-out data opened:** **NO.** The matched reserve has not been read, sampled
or listed. The confirmatory panel exists as a committed selection rule and an
executable command, not as a list of molecules.

**What is running:** nothing. No Modal job has been launched by this lane.

**Last completed gate:**
- transport laws, metric suite, rollout app, panel selection and analysis
  implemented; **117 local tests pass**, no Modal, no checkpoint;
- descriptor envelope calibrated on all 96,094 held-in molecules **before any
  trajectory exists** — `diagnostics/claim2_descriptor_envelope.json`,
  envelope sha256 `fb369161b303a49c…`, held-in self-retention **0.9639**;
- held-in development panel frozen — 36 sources, 4 support bands × 3 size
  bands × 3 sources, panel sha256 `a4282229742eff2e…`;
- protocol and the mobility–fidelity decision rule frozen **before any result**
  — `configs/claim2_trajectory_protocol_v1.json`.

**AUTHORIZED 2026-08-12 by the main lane — the 8-source smoke only.**
Scope frozen before the run in `PROTOCOL.md` §"What the 8-source smoke is
allowed to answer": it may report the six instrument/cost questions and
**may not answer Claim 2**. No automatic promotion to the 36-source run.

**Authorized command:**

```bash
modal run --detach modal_apps/claim2_trajectory_characterization_app.py \
  --sources 8 --seeds 2 --horizon 6 --kernel-budget 40
# then confirm: modal app list  ->  ephemeral (detached)
```

8 held-in sources × 3 arms × 2 seeds × H=6, CPU only, 2 CPU per container.
**Expected ~1.5 container-hours, worst case ~3.8**, wall ~11–30 min.
See `PROTOCOL.md` §Cost for the measured basis of that estimate.

Shards commit per source as they finish and a relaunch skips any that already
match, so an outage costs only the sources that had not completed.

**Blocked on:** the lead's explicit go-ahead. The brief said build, cost, and
stop.

**Nothing downstream of the smoke is authorized**, including the full held-in
development run (~10 container-hours) and the confirmatory matched-reserve run
(~27 container-hours).
