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
  implemented; **89 local tests pass**, no Modal, no checkpoint;
- descriptor envelope calibrated on all 96,094 held-in molecules **before any
  trajectory exists** — `diagnostics/claim2_descriptor_envelope.json`,
  envelope sha256 `fb369161b303a49c…`, held-in self-retention **0.9639**;
- held-in development panel frozen — 36 sources, 4 support bands × 3 size
  bands × 3 sources, panel sha256 `a4282229742eff2e…`;
- protocol and the mobility–fidelity decision rule frozen **before any result**
  — `configs/claim2_trajectory_protocol_v1.json`.

**Next action (needs authorization — one bounded step):**

```bash
modal run modal_apps/claim2_trajectory_characterization_app.py \
  --sources 8 --seeds 2 --horizon 6 --kernel-budget 40
```

8 held-in sources × 3 arms × 2 seeds × H=6, CPU only, 2 CPU per container.
**Expected ~1.5 container-hours, worst case ~3.8**, wall ~11–30 min.
See `PROTOCOL.md` §Cost for the measured basis of that estimate.

**Blocked on:** the lead's explicit go-ahead. The brief said build, cost, and
stop.

**Nothing downstream of the smoke is authorized**, including the full held-in
development run (~10 container-hours) and the confirmatory matched-reserve run
(~27 container-hours).
