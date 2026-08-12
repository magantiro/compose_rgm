"""Claim 2: what molecular transport did the frozen ``R_theta`` learn?

Three transport laws, ONE executable canonical-successor support, one horizon,
one set of seeds.  This app does exactly one thing: emit trajectories.  Every
metric -- mobility, cycling, envelope retention, the frontier -- is computed
locally from the committed shards by ``scripts/analyse_claim2_trajectories.py``.
That split is deliberate.  Enumeration is the only expensive step here, so a
metric bug must cost a local rerun rather than a container-hour.

    r_theta            the frozen learned reference process
    uniform_canonical  1 / |N+(x)|
    empirical_family   frozen realized family frequencies, renormalized over
                       families legal at x, uniform over distinct successors
                       within family, SUMMED across cross-family aliases

CPU ONLY. No GPU is requested anywhere in this file.

WHAT MAKES THIS A MEASUREMENT
-----------------------------
Three guards, because this project has repeatedly been bitten by statistics
whose sign was fixed before any data existed:

1.  **Arm divergence.** At every state the pairwise total variation between the
    three laws is recorded.  Where |N+(x)| == 1 the arms coincide by
    construction and the state carries no information about which law produced
    it.  A run whose arms never diverge is an ``INVALID_INSTRUMENT``, not a
    null result, and ``--smoke`` refuses to pass without measurable divergence.

2.  **Kernel cross-check.** ``enumerate_successor_row`` reaches the learned law
    by a different production route than ``canonical_successor_result``, which
    every other COMPOSE experiment consumes.  Each source verifies the two
    against each other on its source state.  If they disagreed, every number in
    this workstream would describe a private kernel.

3.  **Common random numbers.** All three arms consume the identical uniform
    variate at the identical step, over the identical sorted successor order.
    Where two laws agree they take the same action, so a divergence in realized
    trajectories is attributable to the laws rather than to the sampler.

WHAT THIS APP DOES NOT DO
-------------------------
No controller, no objective, no oracle, no goal.  Every accepted canonical
successor is committed as sampled.  Nothing here selects an action by a
quantity it will later be scored on.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from modal_apps.run_process_v2_p50_app import (
    ARTIFACT_ROOT,
    REMOTE_ROOT,
    ROOT,
    artifact_volume,
)
from modal_apps.run_process_v2_p50_app import image as _base_image

PANEL_FILE = "diagnostics/claim2_trajectory_development_panel.json"
SAMPLING_LAW_FILE = "diagnostics/editing_v2_sampling_law_v2.json"

image = (
    _base_image
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}"})
    .add_local_file(ROOT / PANEL_FILE, str(REMOTE_ROOT / PANEL_FILE), copy=True)
    .add_local_file(ROOT / SAMPLING_LAW_FILE, str(REMOTE_ROOT / SAMPLING_LAW_FILE), copy=True)
)
app = modal.App("compose-v4-claim2-trajectory")

RUN_ROOT = "/artifacts/editing_v2/r_theta_run"
TIME_POINT = 0.5
CANONICAL_SLOTS = 48

#: Productive-edit horizon. The workstream contract starts at 6 and permits a
#: single held-in comparison at 8 ONLY if 6 proves too short to expose
#: structural movement. It is not a tuning knob.
HORIZON = 6

#: Fixed rollout seeds per source. Repeated measures from one source, NOT
#: independent examples: every statistic averages within source before any
#: interval is computed over sources.
SEEDS = (0, 1, 2)

#: Hard per-task enumeration budget. A pathological source cannot spend the
#: whole run; it records ``budget_exhausted`` and returns what it has.
#:
#: Worst case per source is ``1 + 3*seeds*(horizon-1) + 1``: the shared source
#: state, then at most one new state per step for each of the 3*seeds
#: trajectories, plus one cross-check enumeration. For the 2-seed / H=6 smoke
#: that is 32, so 40 leaves margin without letting a single source run away.
DEFAULT_KERNEL_BUDGET = 40


def _runtime():
    """Frozen model + one cached canonical-successor row builder.

    The cache is keyed by canonical state key and shared across arms and seeds.
    That is the single largest cost saving in the design: three arms coupled by
    common random numbers revisit the same states constantly, and each state is
    enumerated once no matter how many trajectories pass through it.
    """
    import sys

    import torch

    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from rdkit import RDLogger

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.claim2_transport_laws import (
        SuccessorRow,
        cross_check_against_production_kernel,
        enumerate_successor_row,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_panel import (
        open_process_v2_t1_source,
    )
    from compose_v4.experiments.editing_v2_process_v2_t1_runtime import (
        build_process_v2_score_revised_scratch_runtime,
        load_materialized_scorer_state,
    )
    from compose_v4.experiments.editing_v2_r_theta_corpus_training import (
        CHECKPOINT_FILENAME,
    )

    RDLogger.DisableLog("rdApp.*")
    artifact_volume.reload()
    paths = json.loads((Path(RUN_ROOT) / "run_inputs" / "RUN_PATHS.json").read_text())
    source = open_process_v2_t1_source(
        Path(paths["active8_root"]),
        gate_zero_decision_path=Path(paths["gate_zero"]),
        artifact_root=Path(paths["artifact_root"]),
        repo_root=REMOTE_ROOT,
    )
    bundle = load_materialized_scorer_state(Path(paths["materialized_scorer"]))
    runtime, _b, _c = build_process_v2_score_revised_scratch_runtime(
        source, materialized_state=bundle
    )
    model = runtime.model
    checkpoint = torch.load(
        Path(RUN_ROOT) / "runs" / "run_v2_01" / CHECKPOINT_FILENAME,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()

    cache: dict[str, SuccessorRow] = {}
    failed: dict[str, str] = {}
    counter = {"calls": 0, "seconds": 0.0, "cross_check_calls": 0}

    def state_of(key: str):
        return pad_molecular_graph(smiles_to_molecular_graph(key), CANONICAL_SLOTS)

    def row_for(key: str, budget: int) -> SuccessorRow | None:
        """Enumerated row for ``key``; ``None`` on budget exhaustion or failure.

        Failures are cached alongside successes. Without that, a state whose
        enumeration raises would be retried by every one of the 3*seeds
        trajectories that reaches it -- each retry paying the full enumeration
        cost while consuming no budget, because a failed call is not a call.
        The caller distinguishes the two ``None`` cases through ``failed``.
        """
        if key in cache:
            return cache[key]
        if key in failed:
            return None
        if counter["calls"] >= budget:
            return None
        started = time.perf_counter()
        try:
            row = enumerate_successor_row(model, state_of(key), TIME_POINT)
        except Exception as error:  # noqa: BLE001 - one bad state must not lose the source
            failed[key] = f"{type(error).__name__}: {error}"[:220]
            counter["seconds"] += time.perf_counter() - started
            print(f"    enumeration failed at {key}: {failed[key]}", flush=True)
            return None
        counter["calls"] += 1
        counter["seconds"] += time.perf_counter() - started
        cache[key] = row
        return row

    def cross_check(key: str, row: SuccessorRow) -> dict[str, Any]:
        counter["cross_check_calls"] += 1
        return cross_check_against_production_kernel(
            model, state_of(key), TIME_POINT, row
        )

    return row_for, cross_check, counter, failed, checkpoint.get("selected_step")


@app.function(
    image=image,
    cpu=2.0,
    memory=12 * 1024,
    timeout=5 * 60 * 60,
    max_containers=40,
    retries=1,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def roll_source(task: dict[str, Any]) -> dict[str, Any]:
    """Every arm and seed from one source molecule. One durable shard."""
    import random

    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph
    from compose_v4.experiments.claim2_transport_laws import (
        ARMS,
        arm_divergence,
        law_for_arm,
        sample_index,
    )
    from compose_v4.rewrite.kernel import canonical_state_key

    started = time.perf_counter()
    row_for, cross_check, counter, failed, selected_step = _runtime()
    horizon = int(task.get("horizon", HORIZON))
    seeds = tuple(task.get("seeds", SEEDS))
    budget = int(task.get("kernel_budget", DEFAULT_KERNEL_BUDGET))
    frequencies = dict(task["family_frequencies"])

    start_key = canonical_state_key(
        pad_molecular_graph(smiles_to_molecular_graph(task["source"]), CANONICAL_SLOTS)
    )

    # Every arm and seed starts from the same enumerated state, so the source
    # row is also the cheapest place to prove this kernel agrees with the one
    # every other COMPOSE experiment uses.
    source_row = row_for(start_key, budget)
    kernel_agreement: dict[str, Any] | None = None
    if source_row is not None and not source_row.is_terminal:
        kernel_agreement = cross_check(start_key, source_row)

    divergence_by_state: dict[str, dict[str, Any]] = {}

    def record_divergence(row) -> None:
        if row.source_key in divergence_by_state:
            return
        measured = arm_divergence(row, frequencies)
        divergence_by_state[row.source_key] = {
            "support_size": measured.support_size,
            "pairwise": measured.pairwise,
            "minimum": measured.minimum,
            "degenerate": measured.degenerate,
        }

    trajectories: list[dict[str, Any]] = []
    budget_exhausted = False
    for seed in seeds:
        # ONE variate stream per seed, consumed identically by all three arms.
        variates = [random.Random(seed * 7919 + step).random() for step in range(horizon)]
        for arm in ARMS:
            states = [start_key]
            families: list[list[str]] = []
            cells: list[list[str]] = []
            supports: list[int] = []
            stop = "horizon"
            for step in range(horizon):
                row = row_for(states[-1], budget)
                if row is None:
                    # An enumeration FAILURE is not a chemical dead end and must
                    # not be counted as one; the early-dead-end rate is a
                    # reported metric.
                    if states[-1] in failed:
                        stop = "enumeration_failed"
                    else:
                        stop = "budget_exhausted"
                        budget_exhausted = True
                    break
                if row.is_terminal:
                    stop = "terminal_state"
                    break
                record_divergence(row)
                supports.append(row.support_size)
                law = law_for_arm(arm, row, frequencies)
                index = sample_index(law, variates[step])
                states.append(row.successor_keys[index])
                families.append(list(row.families[index]))
                cells.append(list(row.cells[index]) if row.cells else [])
            trajectories.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "states": states,
                    "families": families,
                    "cells": cells,
                    "support_sizes": supports,
                    "committed_edits": len(states) - 1,
                    "stop_reason": stop,
                }
            )

    payload = {
        "schema": "compose.claim2.trajectory_shard",
        "status": task.get("status", "SMOKE_HELD_IN"),
        "index": int(task["index"]),
        "source": task["source"],
        "start_key": start_key,
        "support_band": task.get("support_band"),
        "size_band": task.get("size_band"),
        "heavy_atoms": task.get("heavy_atoms"),
        "has_ring": task.get("has_ring"),
        "horizon": horizon,
        "seeds": list(seeds),
        "arms": list(ARMS),
        "time_point": TIME_POINT,
        "canonical_slots": CANONICAL_SLOTS,
        "family_frequencies": frequencies,
        "family_law_sha256": task.get("family_law_sha256"),
        "panel_sha256": task.get("panel_sha256"),
        "code_commit": task.get("code_commit"),
        "checkpoint_selected_step": selected_step,
        "source_support_size": 0 if source_row is None else source_row.support_size,
        "kernel_agreement": kernel_agreement,
        "state_divergence": divergence_by_state,
        "trajectories": trajectories,
        "kernel_calls": counter["calls"],
        "cross_check_calls": counter["cross_check_calls"],
        "kernel_seconds": round(counter["seconds"], 2),
        "seconds_per_kernel_call": (
            round(counter["seconds"] / counter["calls"], 2) if counter["calls"] else None
        ),
        "kernel_budget": budget,
        "budget_exhausted": budget_exhausted,
        "enumeration_failures": failed,
        "seconds": round(time.perf_counter() - started, 1),
    }

    out = Path(RUN_ROOT) / task.get("out_dir", "claim2_trajectory_smoke")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"source-{int(task['index']):04d}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    artifact_volume.commit()

    minimum = min(
        (row["minimum"] for row in divergence_by_state.values()), default=0.0
    )
    landings = {
        (t["arm"], t["seed"]): t["states"][-1] for t in trajectories
    }
    print(
        f"[{int(task['index']):03d}] {counter['calls']} calls "
        f"{payload['seconds']:.0f}s ({payload['seconds_per_kernel_call']}s/call)  "
        f"|N+|={payload['source_support_size']}  minTV={minimum:.3f}  "
        f"distinct landings {len(set(landings.values()))}/{len(landings)}"
        + ("  BUDGET EXHAUSTED" if budget_exhausted else ""),
        flush=True,
    )
    return {
        "index": int(task["index"]),
        "kernel_calls": counter["calls"],
        "seconds": payload["seconds"],
        "seconds_per_kernel_call": payload["seconds_per_kernel_call"],
        "minimum_divergence": minimum,
        "budget_exhausted": budget_exhausted,
        "enumeration_failures": len(failed),
        "kernel_agrees": None if kernel_agreement is None else kernel_agreement["agrees"],
    }


#: Fields that must match for an already-committed shard to be reused. A shard
#: produced under a different panel, horizon, seed set or budget is NOT the same
#: measurement, and silently reusing it would be worse than recomputing it.
SHARD_IDENTITY_FIELDS = ("source", "horizon", "panel_sha256", "family_law_sha256")


def _reusable_shard(path: Path, task: dict[str, Any]) -> bool:
    """Whether a committed shard already answers exactly this task."""
    try:
        payload = json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - a truncated shard is simply redone
        return False
    if payload.get("schema") != "compose.claim2.trajectory_shard":
        return False
    for field in SHARD_IDENTITY_FIELDS:
        if payload.get(field) != task.get(field):
            return False
    if list(payload.get("seeds", ())) != list(task.get("seeds", ())):
        return False
    if payload.get("kernel_budget") != task.get("kernel_budget"):
        return False
    # A shard truncated by an exhausted budget is not a completed measurement.
    return not payload.get("budget_exhausted", False)


@app.function(
    image=image,
    cpu=0.25,
    memory=2048,
    timeout=10 * 60 * 60,
    volumes={str(ARTIFACT_ROOT): artifact_volume},
)
def drive(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Fan out ON MODAL so a client disconnect cannot stall or lose the run.

    Server-side fan-out stops ``.map()`` stalling when the client goes away, but
    it does NOT keep the app alive through a client death -- launch with
    ``modal run --detach`` for that, and verify with ``modal app list`` showing
    ``ephemeral (detached)``.

    Neither of those survives a DNS failure at launch, so durability is the
    third layer: every task commits its own shard as it completes, and this
    driver skips sources whose shard is already committed and matches the
    current task identity. A relaunch after an outage therefore costs only the
    sources that had not finished.
    """
    artifact_volume.reload()
    pending, skipped = [], []
    for task in tasks:
        shard = (
            Path(RUN_ROOT)
            / task.get("out_dir", "claim2_trajectory_smoke")
            / f"source-{int(task['index']):04d}.json"
        )
        if shard.exists() and _reusable_shard(shard, task):
            skipped.append(int(task["index"]))
        else:
            pending.append(task)
    if skipped:
        print(
            f"resuming: {len(skipped)} shard(s) already committed and matching "
            f"({sorted(skipped)}); {len(pending)} to run",
            flush=True,
        )

    done, failed = [], 0
    for result in roll_source.map(
        pending, order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False
    ):
        if isinstance(result, Exception):
            failed += 1
            print(f"  task failed: {type(result).__name__}: {result}"[:200], flush=True)
            continue
        done.append(result)
    rates = [row["seconds_per_kernel_call"] for row in done if row["seconds_per_kernel_call"]]
    return {
        "completed": len(done),
        "skipped_already_committed": len(skipped),
        "requested": len(tasks),
        "failed": failed,
        "kernel_calls": sum(row["kernel_calls"] for row in done),
        "budget_exhausted": sum(bool(row["budget_exhausted"]) for row in done),
        "enumeration_failures": sum(row["enumeration_failures"] for row in done),
        "kernel_disagreements": sum(row["kernel_agrees"] is False for row in done),
        "minimum_divergence": min((row["minimum_divergence"] for row in done), default=0.0),
        # The number every later cost estimate depends on, measured rather than
        # assumed. This is one of the three things the smoke exists to resolve.
        "median_seconds_per_kernel_call": (
            sorted(rates)[len(rates) // 2] if rates else None
        ),
    }


@app.local_entrypoint()
def main(
    sources: int = 8,
    seeds: int = 2,
    horizon: int = HORIZON,
    kernel_budget: int = DEFAULT_KERNEL_BUDGET,
    out_dir: str = "claim2_trajectory_smoke",
    status: str = "SMOKE_HELD_IN",
) -> None:
    """Launch the held-in characterization. Defaults are the 8-source smoke."""
    import subprocess

    panel = json.loads((ROOT / PANEL_FILE).read_text())
    law = json.loads((ROOT / SAMPLING_LAW_FILE).read_text())
    frequencies = dict(law["realized_coefficients"]["by_family"])

    if panel["held_out_opened"]:
        raise SystemExit(
            f"{PANEL_FILE} is a HELD-OUT panel. This entrypoint runs held-in "
            "characterization only; opening the reserve is a main-workstream decision."
        )

    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()

    rows = panel["sources"][:sources]
    tasks = [
        {
            **row,
            "horizon": horizon,
            "seeds": list(SEEDS[:seeds]),
            "kernel_budget": kernel_budget,
            "family_frequencies": frequencies,
            "family_law_sha256": law["frozen_sha256"],
            "panel_sha256": panel["panel_sha256"],
            "code_commit": commit,
            "out_dir": out_dir,
            "status": status,
        }
        for row in rows
    ]

    # The source state is enumerated once and shared by every arm and seed; each
    # trajectory can then contribute at most one new state per step. Plus one
    # cross-check enumeration against canonical_successor_result.
    per_source = 1 + 3 * seeds * (horizon - 1) + 1
    if per_source > kernel_budget:
        raise SystemExit(
            f"worst case is {per_source} enumerations per source but the budget is "
            f"{kernel_budget}; raise --kernel-budget or the run truncates trajectories "
            "silently on the sources that need the support most."
        )
    print(
        f"CLAIM 2 -- {len(tasks)} held-in sources x 3 arms x {seeds} seeds, "
        f"horizon {horizon}\n"
        f"  panel {panel['panel_sha256'][:16]}  family law {law['frozen_sha256'][:16]}\n"
        f"  commit {commit[:12]}  CPU only, 2 CPU per container\n"
        f"  worst-case enumerations per source: {per_source} "
        f"(cache shared across arms and seeds; budget {kernel_budget})\n"
        f"  worst case total: {per_source * len(tasks)} enumerations\n"
        f"  LAUNCH WITH --detach, and confirm 'ephemeral (detached)' in `modal app list`.\n"
        f"  Shards commit per source as they finish and a relaunch skips them, so an\n"
        f"  outage costs only the sources that had not completed."
    )

    summary = drive.remote(tasks)
    print(
        f"\ncompleted {summary['completed']}/{len(tasks)}, "
        f"skipped {summary['skipped_already_committed']} already committed, "
        f"failed {summary['failed']}"
    )
    print(f"  kernel calls {summary['kernel_calls']}")
    print(f"  MEASURED median seconds per enumeration: "
          f"{summary['median_seconds_per_kernel_call']}  "
          f"(protocol assumed 14 s/call x 1.2-1.5 overhead)")
    print(f"  budget exhausted on {summary['budget_exhausted']} sources")
    print(f"  enumeration failures {summary['enumeration_failures']}")
    print(f"  minimum arm divergence across all states: {summary['minimum_divergence']:.4f}")
    if summary["kernel_disagreements"]:
        raise SystemExit(
            f"{summary['kernel_disagreements']} sources disagree with "
            "canonical_successor_result. Every downstream number would describe a "
            "private kernel. Mark the run INVALID_INSTRUMENT and fix before analysing."
        )
    if summary["minimum_divergence"] <= 1e-6 and summary["completed"]:
        print(
            "\nWARNING: some state had all three arms numerically identical. That is "
            "expected where |N+(x)| == 1 and is why per-state divergence is recorded; "
            "check the shard breakdown before reading any arm comparison."
        )
