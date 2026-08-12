# Workstream

- **Name:** Claim 2 — What molecular transport did `R_theta` learn?
- **Claim ID:** 2 (Workstream B)
- **Branch:** `codex/compose-claim2-trajectory`
- **Base commit:** `04f1c46` — *Add parallel workstream plan and agent handoff template*
- **HEAD commit:** `e563603` (implementation); this handoff is the commit on top of it
- **Working tree clean:** yes
- **Status:** `DESIGN_ONLY`
- **Held-out data opened:** **no**

# One-sentence scientific question

> Iterated under the frozen `R_theta`, do executable rewrite trajectories move
> structurally while avoiding the cycling, descriptor drift and operator
> collapse that unlearned laws on the identical support induce?

# Claim this work can support

> Nothing yet. The instrument is built, frozen and locally tested; no
> trajectory exists. What it can support **once the smoke runs** is whether the
> three arms are genuinely different transport laws and what the run costs.

# Claims this work cannot support

- Any statement about `R_theta`'s trajectories — none have been generated.
- Any held-out or confirmatory statement. The matched reserve is unopened.
- Physical dynamics or synthetic route plausibility.
- Medicinal plausibility. The envelope is a distributional sanity check.
- Registry capability-cell coverage. Shards record `family:table` pairs; mapping
  to the 22 registry cells needs the semantic classifier and is not implemented.

# Frozen inputs

| Object | Path / ID | SHA-256 / identity | Verified? |
|---|---|---|---|
| Process-V2 chemistry | `configs/editing_v2_process_v2_capability_cells.json` | `069e0f5e528ebccc…`; process identity `0c938177a34819e6…`; 8 active families; max 40 active atoms | yes, read locally |
| `R_theta` checkpoint | `/artifacts/editing_v2/r_theta_run/runs/run_v2_01/R_THETA_CHECKPOINT.pt` | decision record `diagnostics/editing_v2_r_theta_FROZEN.json` = `30a468b947877 7b9…`; run `run_v2_01`, step 12,500; identity `426ac15e7f4cc68c…`; state `c977ee3fe0cfdcaf…` | decision record verified locally; **checkpoint bytes not verified — it lives on the Modal volume and this lane launched nothing** |
| split / panel | `diagnostics/editing_v2_matched_validation_reserve_ids.json.gz` | `ba9270faf8bea1a6…`; 96,094 held-in / 10,653 reserve sources | yes |
| sampling law | `diagnostics/editing_v2_sampling_law_v2.json` | file `d6e0f3294ecd4e35…`; declared `frozen_sha256 b0cc66f168f1cf3e…` — matches the `law_frozen_sha256` in the Experiment 1 result | yes |
| goal/oracle | NONE — this lane has no objective, oracle or controller | — | n/a |
| baseline definition | `configs/comparator_registry_v3.json` | `2281d19ad6f997fa…`, `frozen_definitions.state_independent_empirical_family_prior` | yes |
| descriptor envelope | `diagnostics/claim2_descriptor_envelope.json` | file `7290b0cf7f24630a…`; envelope `fb369161b303a49c…` | created here |
| development panel | `diagnostics/claim2_trajectory_development_panel.json` | file `3842d769ab13f73a…`; panel `a4282229742eff2e…` | created here |

# Protocol

- **Panel construction:** held-in development panel of 36 sources, 4 scaffold-support bands × 3 heavy-atom size bands × 3 sources; heavy atoms restricted to [12, 34] so the frozen 40-atom process ceiling cannot masquerade as a learned preference against growth. Confirmatory matched-reserve panel: **definition and command only, not materialized**.
- **Arms:** `r_theta` · `uniform_canonical` (1/|N+(x)| over distinct successors after alias collapse) · `empirical_family` (frozen realized family law, renormalized over families legal at x, uniform within family, **summed** across cross-family aliases). Identical executable support at every state; `law_only` comparison.
- **Primary metric:** the mobility–fidelity frontier. Mobility = median over sources of per-source mean endpoint ECFP4 Tanimoto distance from source. Fidelity = mean over sources of per-source fraction of committed intermediate states inside the held-in envelope. Verdict ∈ {`dominates`, `dominated_by`, `incomparable`, `unresolved`}, using only axes whose paired bootstrap 95% interval over sources excludes zero.
- **Secondary metrics:** heavy-atom change; ring-**system** change; unique-state fraction; immediate-reversal, two-cycle and any-state-revisit rates; early dead-end rate; operator-family entropy and coverage; multi-family trajectory fraction; source-conditioned endpoint uniqueness; standardized descriptor drift; arm-divergence TV. All stratified by support band.
- **Independent statistical unit:** the source molecule. Seeds are repeated measures, averaged within source before any interval.
- **Allowed calibration:** the envelope (held-in, before any trajectory — done); the development panel (mechanical, held-in — done); one held-in H=6 vs H=8 comparison **only if** H=6 fails to expose movement.
- **Stop rules:** arms nearly identical after the single horizon check; family collapse or pathological cycling; broad descriptor drift; mobility indistinguishable from trivial growth/shrinkage; kernel cross-check disagreement → `INVALID_INSTRUMENT`.
- **Forbidden adaptations:** retraining or reselecting `R_theta`; changing objective, architecture, corpus, operators or process identity; refitting the envelope after seeing trajectories; adding/removing/reweighting an arm after results; scalarizing the two axes after seeing estimates; opening the reserve without written authorization; any use of the global final test.

# What was implemented

- `src/compose_v4/experiments/claim2_transport_laws.py` — `SuccessorRow` (one enumeration carrying keys, learned probabilities, alias counts **and** per-successor family attribution); the three laws as pure functions; inverse-CDF sampling taking the variate as an argument so arms share it; `arm_divergence`; `enumerate_successor_row` (the Modal-side bridge); `cross_check_against_production_kernel`.
- `src/compose_v4/experiments/claim2_trajectory_metrics.py` — the nine frozen descriptors, envelope calibration and membership, ECFP4 distance, fused-ring-**system** count, `trajectory_health`, fractional-attribution family entropy, `FrontierPoint` / `frontier_verdict`, `METRIC_REGISTRY` + `validate_metric_registry`, `BANNED_STATISTICS`, paired bootstrap over sources.
- `modal_apps/claim2_trajectory_characterization_app.py` — CPU-only rollout; one durable shard per source committed to the volume; `drive()` fan-out on Modal so a client disconnect cannot stall the run; per-state enumeration cache shared across arms and seeds; per-task kernel budget with a launcher-side worst-case check.
- `scripts/claim2_select_trajectory_panel.py` — outcome-independent panel selection for both pools, with the reserve gated behind a required written authorization.
- `scripts/claim2_calibrate_descriptor_envelope.py`, `scripts/analyse_claim2_trajectories.py`.
- `configs/claim2_trajectory_protocol_v1.json` — the machine-readable frozen protocol including the cost model.

**Not implemented, deliberately:** the optional fourth arm (empirical family + learned identity); registry capability-cell classification.

# Tests and smoke checks

| Test | Result | Artifact |
|---|---|---|
| `tests/test_claim2_transport_laws.py` (24) | PASS | law arithmetic reproduced by hand from the frozen `q`; normalization; empirical ≠ uniform; CRN coupling; degenerate-support detection |
| `tests/test_claim2_trajectory_metrics.py` (39) | PASS | descriptors, envelope, ECFP4, ring systems, health, entropy, all four frontier verdicts, the falsifiability gate, banned statistics, paired bootstrap |
| `tests/test_claim2_trajectory_analysis.py` (15) | PASS | full shard → frontier pipeline on synthetic shards of real molecules; instrument checks; power floor; **verdict comes out against `R_theta`** when the reference arm cycles |
| `tests/test_claim2_trajectory_panel.py` (13) | PASS | held-out gate fires in both directions; band definitions match the carve; committed panel is held-in, hash-bound, fully stratified |
| `tests/test_claim2_handoff_manifest.py` (13) | PASS | every digest in `handoff.json` recomputed from the file it describes; held-out status self-consistent; smoke command matches the app's real entrypoint and its budget covers the declared worst case |
| `tests/test_claim2_rollout_resumability.py` (13) | PASS | all 8 shard-reuse rejection cases; entrypoint and module scope proven free of `rdkit`/`compose_v4` for Modal's launch interpreter |
| **Total** | **117 passed** | `PYTHONPATH=src python3 -m pytest tests/test_claim2_*.py -q` |
| Held-in smoke on Modal | **NOT RUN** | awaiting authorization |

# Results

**None.** No trajectory exists. The only measured quantities are calibration
facts, and they are `DESIGN_ONLY`:

| Metric | Arm / condition | Value | Uncertainty / denominator |
|---|---|---:|---|
| envelope held-in self-retention | held-in calibration sample | 0.9639 | 92,627 / 96,094 molecules |
| distinct Murcko scaffolds | held-in universe | 37,446 | over 96,094 molecules |
| development panel cells filled | 4 bands × 3 size bands | 12 / 12 | 3 sources each |
| ring-containing development sources | held-in panel | 35 / 36 | — |
| fresh reserve pool after exclusions | matched reserve | 10,487 | of 10,653; 166 burned |

# Gate verdicts

| Gate | PASS / FAIL / INCONCLUSIVE | Evidence |
|---|---|---|
| Metric suite frozen before any result | PASS | `configs/claim2_trajectory_protocol_v1.json` and both modules committed at `e563603`, before any rollout |
| Falsifiability gate executable | PASS | `validate_metric_registry` refuses an empty `falsifying_observation`; called by the analysis script before reading a shard |
| Frontier verdict can contradict the claim | PASS | `test_pipeline_detects_a_cycling_arm_through_the_health_metrics` returns `dominated_by`/`incomparable` against `R_theta` |
| Envelope calibrated before trajectories | PASS | envelope committed at `e563603`; no rollout exists |
| Held-out boundary intact | PASS | reserve gate unit-tested; no reserve source list materialized anywhere |
| Baselines reuse the frozen definitions | PASS | empirical-family law reproduced by hand from `comparator_registry_v3.json`'s five steps |
| Arms genuinely differ **on real molecules** | INCONCLUSIVE | proven on synthetic rows; **this is what the smoke is for** |
| Cost per enumeration for this code path | INCONCLUSIVE | quoted 1.2–1.5× over a measured 14 s/call baseline; the smoke measures it |
| Kernel cross-check agrees on real states | INCONCLUSIVE | logic implemented and gated; needs the runtime |

# Bugs, invalid instruments, and superseded runs

No invalid runs — nothing has been run. Three defects were found and fixed
during development. None affected a conclusion, because there are none.

- **Panel digest hashed presentation order, not the source set.** Caught by
  `test_committed_panel_sources_are_distinct_and_hash_bound`. A cosmetic
  reordering would have looked like a different panel. Fixed and the panel
  regenerated; final panel sha256 `a4282229742eff2e…`. The superseded value
  never left the branch.

- **A fabricated SHA-256 tail in `handoff.json`.** The envelope digest was
  recorded with a real 16-character prefix and an **invented** remainder, and
  caught on a spot check against the artifact. A wrong digest is worse than a
  missing one because it looks verified, and acceptance gates 2 and 3 depend on
  these values. Now permanently guarded:
  `tests/test_claim2_handoff_manifest.py` recomputes every digest in the
  manifest from the file it describes.

- **Enumeration failures were retried and mislabelled.** A state whose
  enumeration raised was retried by every one of the `3·seeds` trajectories
  reaching it — each retry paying the full enumeration cost while consuming no
  budget, because a failed call is not a call — and the resulting truncation
  was labelled `budget_exhausted`. Two consequences if shipped: unbounded time
  on a pathological source, and enumeration failures silently inflating the
  early-dead-end rate of whichever arm walked into the bad state. Failures are
  now cached alongside successes and reported separately in every shard and in
  the analysis instrument checks.

# Known limitations

1. **Nothing has been executed against the real kernel.** Every guarantee here
   is local. The three arms are proven distinct on synthetic rows; whether they
   diverge on real molecular supports is the smoke's first question.
2. **The cost estimate carries an unmeasured 1.2–1.5× factor** for this lane's
   extra model forward. The smoke records `seconds_per_kernel_call` per shard.
3. **The checkpoint is development-only** — it predates the code-commit binding
   required for paper-bearing runs. Every result from this lane will be
   development evidence until the three frozen seeds exist.
4. **Acyclic sources concentrate in the densest support band** because an empty
   Murcko scaffold is one class. Recorded per source, not corrected.
5. **`family:table` is not the registry capability cell.** No registry-cell
   coverage claim is made.
6. **Fractional family attribution is a choice**, fixed before results. Raw
   family sets are in every shard so an alternative needs no rerun.
7. **The envelope's `formal_charge` coordinate is near-degenerate** — the
   process is charge-preserving, so [0, 1] is effectively constant and it can
   only fail for a source that starts outside.
8. **Resumability is proven only on synthetic shards.** The reuse predicate is
   pure and all eight rejection cases are tested, but no shard has ever been
   written by a real run.

# Exact reproduction commands

```bash
# environment/setup
cd <repo>
git checkout codex/compose-claim2-trajectory
export PYTHONPATH=src

# tests (local, no Modal, no checkpoint, ~5s)
python3 -m pytest tests/test_claim2_transport_laws.py \
                 tests/test_claim2_trajectory_metrics.py \
                 tests/test_claim2_trajectory_analysis.py \
                 tests/test_claim2_trajectory_panel.py -q

# rebuild the frozen calibration artifacts (deterministic; ~2 min)
python3 scripts/claim2_calibrate_descriptor_envelope.py \
  --out diagnostics/claim2_descriptor_envelope.json
python3 scripts/claim2_select_trajectory_panel.py --pool training --count 36 \
  --out diagnostics/claim2_trajectory_development_panel.json

# smoke  -- NOT YET AUTHORIZED. CPU only. ~1.5 container-hours expected.
modal run --detach modal_apps/claim2_trajectory_characterization_app.py \
  --sources 8 --seeds 2 --horizon 6 --kernel-budget 40
# confirm detached:  modal app list   ->   ephemeral (detached)
# a relaunch after an outage skips every source whose shard already matches

# analysis (local, on the shards the smoke commits to the volume)
python3 scripts/analyse_claim2_trajectories.py \
  --shards <volume>/editing_v2/r_theta_run/claim2_trajectory_smoke \
  --envelope diagnostics/claim2_descriptor_envelope.json \
  --out diagnostics/claim2_trajectory_smoke.json --status SMOKE_HELD_IN
```

# Durable artifacts

| Artifact | Path | SHA-256 | Purpose |
|---|---|---|---|
| frozen protocol | `configs/claim2_trajectory_protocol_v1.json` | `ec0e9541941abf68…` | machine-readable contract + cost model |
| descriptor envelope | `diagnostics/claim2_descriptor_envelope.json` | `7290b0cf7f24630a…` | fidelity axis, calibrated pre-trajectory |
| development panel | `diagnostics/claim2_trajectory_development_panel.json` | `3842d769ab13f73a…` | 36 held-in sources |
| transport laws | `src/compose_v4/experiments/claim2_transport_laws.py` | `af64a6ca9981566b…` | three laws, one support |
| metric suite | `src/compose_v4/experiments/claim2_trajectory_metrics.py` | `ca6dc267e07cdabc…` | frontier + falsifiability gate |
| rollout app | `modal_apps/claim2_trajectory_characterization_app.py` | `9bb4943fa373bb98…` | CPU-only shard-durable rollout |
| panel selection | `scripts/claim2_select_trajectory_panel.py` | `d418aa1f5e86b43d…` | both pools; reserve gated |
| envelope calibration | `scripts/claim2_calibrate_descriptor_envelope.py` | `eb6d2913d4d253cd…` | reproduces the envelope |
| analysis | `scripts/analyse_claim2_trajectories.py` | `7753e8b340c8fa58…` | shards → frontier |
| manifest refresh | `scripts/claim2_refresh_handoff_manifest.py` | `265538cd5f87b55e…` | recomputes every digest here; `--check` gates a handoff |

Digests above are the values in `handoff.json`, which
`tests/test_claim2_handoff_manifest.py` recomputes from the tree on every run.
`python3 scripts/claim2_refresh_handoff_manifest.py --check` is the one-command
version of gate 2.

No load-bearing artifact exists only in `/private/tmp` or a scratchpad.

# Files changed

```text
configs/claim2_trajectory_protocol_v1.json
diagnostics/claim2_descriptor_envelope.json
diagnostics/claim2_trajectory_development_panel.json
docs/workstreams/claim2-trajectory/DECISION_LOG.md
docs/workstreams/claim2-trajectory/HANDOFF.md
docs/workstreams/claim2-trajectory/PROTOCOL.md
docs/workstreams/claim2-trajectory/STATUS.md
docs/workstreams/claim2-trajectory/handoff.json
modal_apps/claim2_trajectory_characterization_app.py
scripts/analyse_claim2_trajectories.py
scripts/claim2_calibrate_descriptor_envelope.py
scripts/claim2_refresh_handoff_manifest.py
scripts/claim2_select_trajectory_panel.py
src/compose_v4/experiments/claim2_trajectory_metrics.py
src/compose_v4/experiments/claim2_transport_laws.py
tests/test_claim2_handoff_manifest.py
tests/test_claim2_rollout_resumability.py
tests/test_claim2_trajectory_analysis.py
tests/test_claim2_trajectory_metrics.py
tests/test_claim2_trajectory_panel.py
tests/test_claim2_transport_laws.py
```

**No existing file was modified.** Everything is additive; `R_theta`, the
kernel, the corpus and every other lane's code are untouched.

# Recommended next action

One bounded action:

> Authorize the 8-source held-in smoke —
> `modal run --detach modal_apps/claim2_trajectory_characterization_app.py --sources 8
> --seeds 2 --horizon 6 --kernel-budget 40` — at an expected **1.5
> container-hours** (worst case 3.8), CPU only. It resolves the three
> `INCONCLUSIVE` gates above: whether the three arms genuinely diverge on real
> molecular supports, whether this lane's kernel agrees with
> `canonical_successor_result`, and the true per-enumeration cost that every
> later estimate depends on.

# Actions explicitly not recommended

- The full held-in development run (36 sources × 3 seeds, ~10 container-hours)
  before the smoke reports its measured cost per call.
- The confirmatory matched-reserve run (~27 container-hours) — and it must not
  be scheduled at all until the smoke's arm-divergence and kernel-agreement
  gates pass.
- Opening or materializing the matched-reserve panel.
- Adding the optional fourth arm.
- Any H=8 comparison before H=6 has been shown insufficient on held-in data.
- Reading any point estimate from the 8-source smoke as a direction. The
  analysis script refuses to emit a verdict below 20 sources; do not work around
  that by reading the frontier table.

# Main-session pickup checklist

- [ ] Read protocol before results. (There are no results.)
- [ ] Verify all frozen-input hashes — table above; the `R_theta` checkpoint
      bytes are unverified because nothing was launched.
- [ ] Confirm held-out-open status: **no**, and gated in code.
- [ ] Reproduce one smoke: run the local test suite; the Modal smoke is the
      action being requested.
- [ ] Inspect known-invalid runs: none exist.
- [ ] Decide explicitly whether to authorize the 8-source held-in smoke.
