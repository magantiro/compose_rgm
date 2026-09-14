#!/usr/bin/env python3
"""Prepare, preflight, launch, monitor and collect Dynamic COMPOSE v2."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
from compose_v4.control.dynamic_program_synthesis_v2 import (
    CHANNELS,
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
    DynamicV2ProgramOptimizer,
    gate_state_summary,
    initial_dynamic_program_batch_v2,
    synthesize_structured_program,
    validate_gate_state,
)
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.experiments import t4_frozen_program_benchmark as benchmark
from compose_v4.experiments.continuation_profile import (
    publish_json,
    sha256_file,
)
from compose_v4.experiments.t4_dynamic_v2 import (
    APP,
    APP_NAME,
    CONTRACT,
    EMPTY_LIBRARY,
    KIND,
    PREFLIGHT,
    SELECTED_UNITS,
    SOURCE_CONTRACT,
    load_contract,
    v2_contract_payload,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "diagnostics/t4_dynamic_v2"
REFERENCE = ATTEMPT / "comparison_reference.json"
RECEIPT = ATTEMPT / "launch.json"
RESULT = ATTEMPT / "result.json"


def _read_sealed(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if set(value) != {"payload", "payload_sha256"}:
        raise ValueError(f"{path}: expected one sealed JSON envelope")
    from compose_v4.control.docking_value import identity

    if identity(value["payload"]) != value["payload_sha256"]:
        raise ValueError(f"{path}: sealed payload identity changed")
    return value["payload"], {
        "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_payload_sha256": value["payload_sha256"],
    }


def _result_summary(path: Path, *, role: str) -> dict:
    payload, provenance = _read_sealed(path)
    if payload.get("status") != "complete" or payload.get("champion") is None:
        raise ValueError(f"{path}: comparison result is not durably complete")
    return {
        **provenance,
        "role": role,
        "unit_id": payload["unit"]["unit_id"],
        "oracle_calls": payload["oracle_calls"],
        "successful_oracle_calls": payload["successful_oracle_calls"],
        "failed_oracle_calls": payload["failed_oracle_calls"],
        "termination": payload["termination"],
        "best_score": payload["champion"]["score"],
        "best_query_index": payload["champion"]["query_index"],
        "endpoint_sha256": __import__(
            "compose_v4.control.docking_value", fromlist=["identity"]
        ).identity(payload["champion"]["endpoint"]),
        "curve": payload["curve"],
    }


def _sealed_artifact_identity(path: Path, *, role: str) -> dict:
    payload, provenance = _read_sealed(path)
    return {
        **provenance,
        "role": role,
        "schema_version": payload.get("schema_version"),
        "run_id": payload.get("task", {}).get("run_id"),
        "calls": payload.get("calls"),
        "new_call_ceiling": payload.get("new_call_ceiling"),
    }


def prepare(args) -> None:
    if any(path.exists() for path in (ROOT / CONTRACT, REFERENCE, ROOT / PREFLIGHT)):
        raise ValueError(
            "Dynamic-v2 development artifacts already exist; do not relock"
        )
    source = unseal(ROOT / SOURCE_CONTRACT)
    full_paths = {
        "5ht1b_0_r0": args.full_5ht1b,
        "braf_1_r0": args.full_braf,
        "jak2_1_r0": args.full_jak2,
    }
    v0_paths = {
        "5ht1b_0_r0": args.v0_5ht1b,
        "braf_1_r0": args.v0_braf,
        "jak2_1_r0": args.v0_jak2,
    }
    reference = {
        "schema_version": "t4_dynamic_v2_comparison_reference_v1",
        "role": (
            "read-only Full-146 and Dynamic-v0 outcomes plus immutable "
            "Dynamic-v1 launch identity"
        ),
        "full_146": {
            unit: _result_summary(path, role="Full-146")
            for unit, path in full_paths.items()
        },
        "dynamic_v0": {
            unit: _result_summary(path, role="Dynamic-v0")
            for unit, path in v0_paths.items()
        },
        "dynamic_v1_launch": _sealed_artifact_identity(
            ROOT / "diagnostics/t4_dynamic_v1/launch.json",
            role="Dynamic-v1 immutable concurrent launch",
        ),
        "new_oracle_calls": 0,
    }
    if set(reference["full_146"]) != set(SELECTED_UNITS) or set(
        reference["dynamic_v0"]
    ) != set(SELECTED_UNITS):
        raise ValueError("comparison reference unit census changed")
    seal(REFERENCE, reference)

    contract = deepcopy(source)
    contract["library_path"] = EMPTY_LIBRARY
    contract["library_programs"] = 0
    contract["dynamic_v2_offline_comparison_reference"] = {
        "path": str(REFERENCE.relative_to(ROOT)),
        "sha256": sha256_file(REFERENCE),
        "available_to_runtime": False,
    }
    material = (
        SOURCE_CONTRACT,
        EMPTY_LIBRARY,
        "docs/GENMOL_T4_SEEDS.json",
        "docs/T4_DYNAMIC_V2.md",
        "modal_apps/genmol_t4_opt_app.py",
        APP,
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/dynamic_program_synthesis.py",
        "src/compose_v4/control/dynamic_program_synthesis_v1.py",
        "src/compose_v4/control/dynamic_program_synthesis_v2.py",
        "src/compose_v4/control/program_transfer.py",
        "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
        "src/compose_v4/experiments/t4_dynamic_v2.py",
        "tools/t4_dynamic_v2_benchmark.py",
    )
    contract["inputs"] = {path: sha256_file(ROOT / path) for path in material}
    contract["information_regime"] = {
        **contract["information_regime"],
        "dynamic_v2": (
            "zero initial routes; generic shallow/structured modules and one "
            "parent-specific gate; no Dynamic-v1 outcome is consumed"
        ),
        "comparison": (
            "Full-146 and Dynamic-v0 outcomes reused read-only; Dynamic-v1 is "
            "referenced only by its immutable launch receipt"
        ),
    }
    contract["limitations"] = [
        *contract["limitations"],
        "Dynamic-v2 was designed on three answer-known T4 development cells.",
        "One search replicate and stochastic docking limit general conclusions.",
        "The remaining twelve T4 cells are outside this development milestone.",
    ]
    contract["dynamic_v2_development"] = v2_contract_payload()
    seal(ROOT / CONTRACT, contract)
    load_contract(ROOT)
    print(
        json.dumps(
            {
                "contract": CONTRACT,
                "contract_sha256": sha256_file(ROOT / CONTRACT),
                "reference": str(REFERENCE.relative_to(ROOT)),
                "reference_sha256": sha256_file(REFERENCE),
                "selected_units": SELECTED_UNITS,
                "new_call_ceiling": 3000,
                "new_oracle_calls": 0,
            },
            indent=2,
        )
    )


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def build_preflight(*, code_revision: str | None = None) -> dict:
    if rdBase.rdkitVersion != "2024.03.5":
        raise ValueError("Dynamic-v2 structural preflight requires RDKit 2024.03.5")
    contract = load_contract(ROOT)
    began, rows = perf_counter(), []
    for unit_id in SELECTED_UNITS:
        unit = next(row for row in contract["units"] if row["unit_id"] == unit_id)
        cell = contract["cells"][unit["cell"]]
        source = decode_state(cell["source_state"])
        config = replace(
            benchmark.configured(contract, unit),
            seed=contract["cold_start_seed"],
            candidates_per_batch=4,
            attempts_per_batch=32,
            wall_seconds=15,
        )
        kwargs = {
            "source_group": benchmark.identity(
                {
                    "target": unit["target"],
                    "source_idx": unit["source_idx"],
                    "seed": unit["original_seed"],
                }
            ),
            "oracle_protocol": unit["oracle_protocol"],
            "eligibility": benchmark.strict_endpoint_scorer(
                unit["original_seed"], delta=contract["delta"]
            ),
        }
        first = initial_dynamic_program_batch_v2(source, (), config, **kwargs)
        second = initial_dynamic_program_batch_v2(source, (), config, **kwargs)
        if first["batch_id"] != second["batch_id"]:
            raise ValueError(f"Dynamic-v2 initial batch is nondeterministic: {unit_id}")
        if not first["candidates"]:
            raise ValueError(
                f"Dynamic-v2 representative preflight yielded no candidate: {unit_id}"
            )
        validate_gate_state(first["gate_state_after_preparation"])
        if {
            row["planner_channel"]
            for row in first["attempts"]
            if "planner_channel" in row
        } - set(CHANNELS):
            raise ValueError(f"Dynamic-v2 emitted an unknown channel: {unit_id}")
        for candidate in first["candidates"]:
            program = EditProgram.from_payload(candidate["program"])
            _, replay = execute_program_graph(
                decode_state(candidate["source_state"]),
                compile_program_graph(program),
                tuple(candidate["assignment"]),
                max_primitives=config.max_primitives,
                max_blocks=config.max_blocks,
            )
            if (
                replay != candidate["trace"]
                or replay["endpoint"] != candidate["endpoint"]
            ):
                raise ValueError(f"Dynamic-v2 candidate changed on replay: {unit_id}")

        # Exercise each proposal channel explicitly in the pinned chemistry
        # environment. This is a structural gate only and makes no oracle call.
        eligible_by_channel = {}
        for channel_index, (channel, synthesizer) in enumerate(
            (
                (SHALLOW_CHANNEL, synthesize_dynamic_program),
                (STRUCTURED_CHANNEL, synthesize_structured_program),
            )
        ):
            channel_candidates = []
            channel_attempts = 0
            import numpy as np

            rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, 83, channel_index])
            )
            for _ in range(64):
                channel_attempts += 1
                try:
                    _, _, _, trace, _ = synthesizer(
                        source,
                        rng,
                        max_modules=3,
                        max_primitives=config.max_primitives,
                        max_blocks=config.max_blocks,
                    )
                except ValueError:
                    continue
                if kwargs["eligibility"]({"smiles": trace["endpoint"]}).get(
                    "oracle_eligible"
                ):
                    channel_candidates.append(trace["endpoint"])
                    break
            if not channel_candidates:
                raise ValueError(
                    f"Dynamic-v2 {channel} produced no eligible pinned candidate: "
                    f"{unit_id}"
                )
            eligible_by_channel[channel] = {
                "attempts": channel_attempts,
                "endpoint": channel_candidates[0],
            }

        optimizer = DynamicV2ProgramOptimizer(
            benchmark.configured(contract, unit),
            source_group=kwargs["source_group"],
            oracle_protocol=kwargs["oracle_protocol"],
            hierarchy=None,
        )
        rows.append(
            {
                "unit_id": unit_id,
                "candidates": len(first["candidates"]),
                "attempts": len(first["attempts"]),
                "batch_id": first["batch_id"],
                "proposal_seconds": first["proposal_seconds"],
                "optimizer_class": type(optimizer).__name__,
                "eligible_by_channel": eligible_by_channel,
            }
        )
    return {
        "schema_version": "t4_dynamic_v2_preflight_v1",
        "passed": len(rows) == 3,
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "representative_candidates": rows,
        "preflight_batch_size": 4,
        "seconds": perf_counter() - began,
        "rdkit": rdBase.rdkitVersion,
        "new_oracle_calls": 0,
        "code_revision": _revision() if code_revision is None else code_revision,
    }


def preflight() -> None:
    output = ROOT / PREFLIGHT
    if output.exists():
        raise ValueError("Dynamic-v2 preflight exists; do not overwrite")
    body = build_preflight()
    publish_json(output, body)
    print(json.dumps(body, indent=2))


def refresh_prequery_contract() -> None:
    """Reseal only pre-oracle implementation identities before the first commit."""
    contract_path = ROOT / CONTRACT
    if (ROOT / PREFLIGHT).exists() or RECEIPT.exists():
        raise ValueError("cannot refresh Dynamic-v2 after preflight or launch locking")
    contract = unseal(contract_path)
    prior_policy = contract.get("dynamic_v2_development", {})
    expected_policy = v2_contract_payload()
    comparison_only_prior = dict(prior_policy)
    comparison_only_prior.pop("full_146_dynamic_v0_dynamic_v1_reused_read_only", None)
    comparison_only_prior.update(
        {
            "offline_comparators": ["Dynamic-v0", "Dynamic-v1", "Full-146"],
            "comparison_outcomes_available_to_runtime": False,
        }
    )
    if prior_policy != expected_policy and comparison_only_prior != expected_policy:
        raise ValueError("Dynamic-v2 scientific policy changed before refresh")
    contract["dynamic_v2_development"] = expected_policy
    prior_sha256 = sha256_file(contract_path)
    contract["dynamic_v2_offline_comparison_reference"] = {
        "path": str(REFERENCE.relative_to(ROOT)),
        "sha256": sha256_file(REFERENCE),
        "available_to_runtime": False,
    }
    material = (
        SOURCE_CONTRACT,
        EMPTY_LIBRARY,
        "docs/GENMOL_T4_SEEDS.json",
        "docs/T4_DYNAMIC_V2.md",
        "modal_apps/genmol_t4_opt_app.py",
        APP,
        "src/compose_v4/control/adaptive_program_optimizer.py",
        "src/compose_v4/control/dynamic_program_synthesis.py",
        "src/compose_v4/control/dynamic_program_synthesis_v1.py",
        "src/compose_v4/control/dynamic_program_synthesis_v2.py",
        "src/compose_v4/control/program_transfer.py",
        "src/compose_v4/experiments/t4_frozen_program_benchmark.py",
        "src/compose_v4/experiments/t4_dynamic_v2.py",
        "tools/t4_dynamic_v2_benchmark.py",
    )
    contract["inputs"] = {path: sha256_file(ROOT / path) for path in material}
    contract.setdefault("dynamic_v2_prequery_repairs", []).append(
        {
            "reason": (
                "reseal pre-oracle implementation identities and keep all comparator "
                "outcomes outside the v2 runtime image"
            ),
            "prior_contract_sha256": prior_sha256,
            "oracle_calls": 0,
            "scientific_policy_changed": False,
        }
    )
    seal(contract_path, contract)
    load_contract(ROOT)
    print(
        json.dumps(
            {
                "status": "prequery_contract_refreshed",
                "prior_contract_sha256": prior_sha256,
                "contract_sha256": sha256_file(contract_path),
                "oracle_calls": 0,
                "scientific_policy_changed": False,
            },
            indent=2,
        )
    )


def _task(*, include_preflight: bool = True):
    from modal_apps.run_process_v2_p50_app import local_image_revision
    from tools.preflight import assert_synced

    revision = local_image_revision(
        expected_commit=assert_synced(strict=True)["commit"]
    )
    paths = [CONTRACT, APP, "modal_apps/genmol_t4_opt_app.py"]
    if include_preflight:
        paths.append(PREFLIGHT)
    body = {
        "image_revision": revision,
        "files_sha256": {path: sha256_file(ROOT / path) for path in paths},
    }
    return {**body, "run_id": benchmark.identity(body)}


def remote_preflight() -> None:
    import modal

    output = ROOT / PREFLIGHT
    if output.exists():
        raise ValueError("Dynamic-v2 preflight exists; do not overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("remote preflight requires a clean committed worktree")
    task = _task(include_preflight=False)
    body = modal.Function.from_name(APP_NAME, "structural_preflight").remote(task)
    if body.get("passed") is not True or body.get("new_oracle_calls") != 0:
        raise ValueError("remote Dynamic-v2 structural preflight did not pass")
    publish_json(output, body)
    print(json.dumps(body, indent=2))


def launch() -> None:
    import modal

    if RECEIPT.exists():
        raise ValueError(
            "Dynamic-v2 launch receipt exists; monitor rather than relaunch"
        )
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Dynamic-v2 launch requires a clean committed worktree")
    contract = load_contract(ROOT)
    reference = unseal(REFERENCE)
    if (
        sha256_file(REFERENCE)
        != contract["dynamic_v2_offline_comparison_reference"]["sha256"]
    ):
        raise ValueError("Dynamic-v2 offline comparison reference changed")
    if any(
        reference["dynamic_v0"][unit]["unit_id"] != unit
        or reference["dynamic_v0"][unit]["oracle_calls"] < 1
        for unit in SELECTED_UNITS
    ):
        raise ValueError("Dynamic-v0 comparison units are not durably complete")
    task = _task()
    remote = modal.Function.from_name(APP_NAME, "preflight").remote(task)
    if remote.get("passed") is not True or remote.get("new_oracle_calls") != 0:
        raise ValueError("Dynamic-v2 remote zero-oracle preflight failed")
    worker = modal.Function.from_name(APP_NAME, "worker")
    calls = {
        unit: worker.spawn({**task, "unit_id": unit}).object_id
        for unit in SELECTED_UNITS
    }
    saved = {
        "schema_version": "t4_dynamic_v2_launch_v1",
        "task": task,
        "calls": calls,
        "volume": "compose-v4-artifacts",
        "volume_path": f"{KIND}/{task['run_id']}",
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "preflight_sha256": sha256_file(ROOT / PREFLIGHT),
        "new_call_ceiling": contract["dynamic_v2_development"]["new_call_ceiling"],
        "automatic_retries": 0,
    }
    seal(RECEIPT, saved)
    print(
        json.dumps(
            {key: value for key, value in saved.items() if key != "task"}, indent=2
        )
    )


def _remote_json(volume, path):
    raw = b"".join(volume.read_file(path))
    value = json.loads(raw)
    if set(value) == {"payload", "payload_sha256"}:
        from compose_v4.control.docking_value import identity

        if identity(value["payload"]) != value["payload_sha256"]:
            raise ValueError(f"corrupt sealed remote artifact: {path}")
        return value["payload"]
    return value


def _remote_gzip_json(volume, path):
    return json.loads(gzip.decompress(b"".join(volume.read_file(path))))


def _unit_state(volume, saved, unit):
    folder = f"{saved['volume_path']}/units/{unit}"
    try:
        names = {row.path.rsplit("/", 1)[-1] for row in volume.listdir(folder)}
    except __import__("modal").exception.NotFoundError:
        names = set()
    for name in ("result.json", "failure.json"):
        if name in names:
            return _remote_json(volume, f"{folder}/{name}")
    if "checkpoint.json.gz" in names:
        checkpoint = _remote_gzip_json(volume, f"{folder}/checkpoint.json.gz")
        champion = checkpoint.get("champion")
        return {
            "status": "running",
            "oracle_calls": checkpoint.get("query_count", 0),
            "best_score": None if champion is None else champion["score"],
            "curve": checkpoint.get("curve", []),
            "updated_at_utc": checkpoint.get("updated_at_utc"),
        }
    if "progress.json" in names:
        return _remote_json(volume, f"{folder}/progress.json")
    return {}


def _best_at(curve, calls):
    if not calls or len(curve) < calls:
        return None
    return curve[calls - 1]["best_score"]


def _unit_gate_summary(volume, saved, unit):
    path = f"{saved['volume_path']}/units/{unit}/checkpoint.json.gz"
    checkpoint = _remote_gzip_json(volume, path)
    state = checkpoint.get("search", {}).get("dynamic_v2_state")
    if state is None:
        raise ValueError(f"Dynamic-v2 checkpoint lacks gate state: {unit}")
    validate_gate_state(state)
    return gate_state_summary(state)


def status() -> None:
    import modal

    saved = unseal(RECEIPT)
    reference = unseal(REFERENCE)
    volume = modal.Volume.from_name(saved["volume"])
    v1_saved = unseal(ROOT / "diagnostics/t4_dynamic_v1/launch.json")
    if (
        sha256_file(ROOT / "diagnostics/t4_dynamic_v1/launch.json")
        != reference["dynamic_v1_launch"]["source_sha256"]
    ):
        raise ValueError("Dynamic-v1 launch identity changed after the v2 lock")
    rows = []
    for unit in SELECTED_UNITS:
        current = _unit_state(volume, saved, unit)
        v1_current = _unit_state(volume, v1_saved, unit)
        calls = current.get(
            "oracle_calls", current.get("queries_total", current.get("queries", 0))
        )
        best = current.get("best_score")
        if best is None and current.get("champion") is not None:
            best = current["champion"]["score"]
        full = reference["full_146"][unit]
        v0 = reference["dynamic_v0"][unit]
        v1_curve = v1_current.get("curve", [])
        v1_best = v1_current.get("best_score")
        if v1_best is None and v1_current.get("champion") is not None:
            v1_best = v1_current["champion"]["score"]
        full_same = _best_at(full["curve"], calls)
        v0_same = _best_at(v0["curve"], calls)
        v1_same = _best_at(v1_curve, calls)
        rows.append(
            {
                "unit": unit,
                "dynamic_v2_status": current.get("status", "pending"),
                "dynamic_v2_calls": calls,
                "dynamic_v2_best": best,
                "dynamic_v1_status": v1_current.get("status", "pending"),
                "dynamic_v1_calls_now": v1_current.get(
                    "oracle_calls",
                    v1_current.get("queries_total", v1_current.get("queries", 0)),
                ),
                "dynamic_v1_best_now": v1_best,
                "dynamic_v0_at_v2_calls": v0_same,
                "dynamic_v1_at_v2_calls": v1_same,
                "full_146_at_v2_calls": full_same,
                "gap_v2_vs_full_at_same_calls": (
                    None if best is None or full_same is None else best - full_same
                ),
                "full_146_final": full["best_score"],
                "dynamic_v0_final": v0["best_score"],
                "updated_at_utc": current.get("at", current.get("completed_at_utc")),
            }
        )
    print(
        json.dumps(
            {
                "units": rows,
                "dynamic_v2_total_calls": sum(row["dynamic_v2_calls"] for row in rows),
            },
            indent=2,
        )
    )


def collect() -> None:
    import modal

    if RESULT.exists():
        raise ValueError("Dynamic-v2 result exists; do not overwrite")
    saved = unseal(RECEIPT)
    reference = unseal(REFERENCE)
    volume = modal.Volume.from_name(saved["volume"])
    v1_saved = unseal(ROOT / "diagnostics/t4_dynamic_v1/launch.json")
    rows, calls = [], 0
    for unit in SELECTED_UNITS:
        path = f"{saved['volume_path']}/units/{unit}/result.json"
        current = _remote_json(volume, path)
        if current.get("status") != "complete":
            raise ValueError(f"Dynamic-v2 unit is not complete: {unit}")
        v1_current = _unit_state(volume, v1_saved, unit)
        if v1_current.get("status") != "complete":
            raise ValueError(f"Dynamic-v1 comparator is not complete: {unit}")
        calls += current["oracle_calls"]
        rows.append(
            {
                "unit": unit,
                "dynamic_v2": {
                    "best_score": current["champion"]["score"],
                    "oracle_calls": current["oracle_calls"],
                    "termination": current["termination"],
                    "curve": current["curve"],
                    "gate": _unit_gate_summary(volume, saved, unit),
                },
                "dynamic_v1": {
                    "best_score": v1_current["champion"]["score"],
                    "oracle_calls": v1_current["oracle_calls"],
                    "termination": v1_current["termination"],
                    "curve": v1_current["curve"],
                },
                "dynamic_v0": reference["dynamic_v0"][unit],
                "full_146": reference["full_146"][unit],
            }
        )
    body = {
        "schema_version": "t4_dynamic_v2_result_v1",
        "contract": CONTRACT,
        "contract_sha256": sha256_file(ROOT / CONTRACT),
        "launch_receipt_sha256": sha256_file(RECEIPT),
        "new_oracle_calls": calls,
        "new_call_ceiling": 3000,
        "rows": rows,
        "evidence": "winner-informed three-cell Dynamic COMPOSE v2 development diagnostic",
        "limitations": [
            "one search replicate per development cell",
            "stochastic docking",
            "answer-known Full-146 transformations informed the generic v1/v2 capabilities",
            "not held-out benchmark evidence",
        ],
    }
    seal(RESULT, body)
    print(json.dumps({"new_oracle_calls": calls, "rows": rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "refresh-prequery-contract",
            "preflight",
            "remote-preflight",
            "launch",
            "status",
            "collect",
        ),
    )
    parser.add_argument("--full-5ht1b", type=Path)
    parser.add_argument("--full-braf", type=Path)
    parser.add_argument("--full-jak2", type=Path)
    parser.add_argument("--v0-5ht1b", type=Path)
    parser.add_argument("--v0-braf", type=Path)
    parser.add_argument("--v0-jak2", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        required = (
            args.full_5ht1b,
            args.full_braf,
            args.full_jak2,
            args.v0_5ht1b,
            args.v0_braf,
            args.v0_jak2,
        )
        if any(path is None for path in required):
            parser.error("prepare requires all six Full-146/Dynamic-v0 result paths")
        prepare(args)
    elif args.mode == "refresh-prequery-contract":
        refresh_prequery_contract()
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "remote-preflight":
        remote_preflight()
    elif args.mode == "launch":
        launch()
    elif args.mode == "status":
        status()
    else:
        collect()


if __name__ == "__main__":
    main()
