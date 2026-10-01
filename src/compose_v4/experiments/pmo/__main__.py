"""Validate or run one local PMO campaign with a pinned frozen reference."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from compose_v4.control.adaptive_program_optimizer import ProgramSearchConfig
from compose_v4.control.docking_value import identity
from compose_v4.control.pmo_reference_controller import (
    PmoProposalConfig,
    PmoReferenceController,
    initial_pmo_program_batch,
)
from compose_v4.control.program_campaign import ProgramQueryLedger, run_program_campaign
from compose_v4.control.program_task import ProgramTask, archive_top_k, pmo_top_ten_auc
from compose_v4.control.reference_guidance import GuidanceConfig
from compose_v4.experiments.continuation_profile import publish_json, verify_file
from compose_v4.experiments.core_runtime import core_runtime_identity
from compose_v4.experiments.pmo_oracle_worker import PmoOracleClient
from compose_v4.model.reference_checkpoint import load_frozen_reference

ROOT = Path(__file__).resolve().parents[4]
ARMS = {"structured", "uniform_chain", "created_atom_rebinding"}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _resolve_input(config_path: Path, field: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{config_path}: {field} must be a nonempty path")
    resolved = (config_path.parent / value).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{config_path}: {field} is missing: {resolved}")
    return str(resolved)


def read_config(path: Path) -> dict:
    path = path.resolve(strict=True)
    raw = json.loads(path.read_text())
    required = {
        "schema_version",
        "task",
        "budget",
        "metrics_calls",
        "rounds",
        "queries_per_round",
        "seed",
        "arm",
        "proposal",
        "guidance",
        "reference",
        "jump_plans",
        "initialization",
        "oracle",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or raw["schema_version"] != "compose.pmo.config.v1"
    ):
        raise ValueError(f"invalid PMO campaign configuration: {path}")
    audit_manifest = json.loads((ROOT / "experiments/pmo/assets.json").read_text())
    audit = audit_manifest["oracle_audit"]
    audit_path = (ROOT / "experiments/pmo" / audit["path"]).resolve()
    verify_file(audit_path, audit["sha256"])
    tasks = json.loads(audit_path.read_text())["tasks"]
    if raw["task"] not in tasks or raw["arm"] not in ARMS:
        raise ValueError(f"PMO task or proposal arm is outside the declared support: {path}")
    for key in ("budget", "metrics_calls", "rounds", "queries_per_round"):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f"{path}: {key} must be a positive integer")
    if raw["metrics_calls"] > raw["budget"] or raw["budget"] < 16:
        raise ValueError(f"{path}: PMO metric window or initialization exceeds the budget")
    if type(raw["seed"]) is not int or raw["seed"] < 0:
        raise ValueError(f"{path}: seed must be a nonnegative integer")
    guidance = GuidanceConfig(**raw["guidance"])
    if not isinstance(raw["proposal"], dict) or set(raw["proposal"]) != {
        "candidate_pool_limit",
        "replacement_option_rate",
        "realization_seconds_cap",
    }:
        raise ValueError(f"{path}: PMO proposal policy needs three explicit settings")
    PmoProposalConfig(**raw["proposal"])
    reference = raw["reference"]
    if (guidance.mode != "off") != (reference is not None):
        raise ValueError(f"{path}: reference asset is required exactly in shadow or active mode")
    for name in ("jump_plans", "initialization", "reference"):
        asset = raw[name]
        if asset is None:
            if name != "reference":
                raise ValueError(f"{path}: {name} is required")
            continue
        fields = {"path", "sha256"} | (
            {"catalog_fingerprint", "catalog_path", "catalog_sha256"}
            if name == "reference"
            else set()
        )
        if not isinstance(asset, dict) or set(asset) != fields:
            raise ValueError(f"{path}: {name} needs exactly {sorted(fields)}")
        asset["path"] = _resolve_input(path, f"{name}.path", asset["path"])
        verify_file(Path(asset["path"]), asset["sha256"])
        if name == "reference":
            asset["catalog_path"] = _resolve_input(
                path, "reference.catalog_path", asset["catalog_path"]
            )
            verify_file(Path(asset["catalog_path"]), asset["catalog_sha256"])
    oracle = raw["oracle"]
    if not isinstance(oracle, dict) or set(oracle) != {"python", "asset_root"}:
        raise ValueError(f"{path}: oracle needs python and asset_root fields")
    oracle["python"] = _resolve_input(path, "oracle.python", oracle["python"])
    if oracle["asset_root"] is not None:
        oracle["asset_root"] = _resolve_input(path, "oracle.asset_root", oracle["asset_root"])
    if (
        tasks[raw["task"]]["classification"].startswith("ASSET_BACKED")
        and oracle["asset_root"] is None
    ):
        raise ValueError(f"{path}: {raw['task']} requires a pinned PyTDC asset root")
    return raw


def _initialization(asset: dict) -> dict:
    value = json.loads(Path(asset["path"]).read_text())
    body = {key: row for key, row in value.items() if key != "lock_sha256"}
    if identity(body) != value.get("lock_sha256"):
        raise ValueError("PMO initialization lock changed")
    if value.get("count") != 16 or len(value.get("candidates", ())) != 16:
        raise ValueError("PMO initialization must contain 16 task-independent molecules")
    if any("score" in row or "task" in row for row in value["candidates"]):
        raise ValueError("PMO initialization cannot contain objective information")
    return value


def _jump_checkpoint(asset: dict) -> dict:
    path = Path(asset["path"])
    if path.suffix != ".gz":
        raise ValueError("PMO jump plans must use the versioned gzip asset")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        envelope = json.load(stream)
    if identity(envelope["payload"]) != envelope.get("payload_sha256"):
        raise ValueError("PMO jump-plan payload identity changed")
    value = envelope["payload"]["checkpoints"]["shared_all_routes"]
    if value.get("fit_scope") != "shared_all_routes" or value.get("maximum_primitives") != 32:
        raise ValueError("PMO jump-plan support changed")
    return value


def _search_config(seed: int, queries: int) -> ProgramSearchConfig:
    return replace(
        ProgramSearchConfig.program_only_recipe(seed=seed, score_direction="maximize"),
        parent_allocation="niche_score",
        attempts_per_batch=128,
        candidates_per_batch=queries,
        wall_seconds=1e9,
        proposal_cache_entries=128,
    )


def _set_arm(arm: str) -> None:
    os.environ["PMO_UNIFORM_CHAIN"] = "1" if arm == "uniform_chain" else "0"
    os.environ["PMO_BINDING_REBIND"] = "1" if arm == "created_atom_rebinding" else "0"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run", "resume"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command != "validate" and args.output is None:
        parser.error("run and resume require --output")
    raw = read_config(args.config)
    core = core_runtime_identity()
    initial = _initialization(raw["initialization"])
    checkpoint = _jump_checkpoint(raw["jump_plans"])
    if raw["reference"] is not None:
        reference = raw["reference"]
        load_frozen_reference(
            Path(reference["path"]),
            expected_sha256=reference["sha256"],
            expected_catalog_fingerprint=reference["catalog_fingerprint"],
            catalog_path=Path(reference["catalog_path"]),
            expected_catalog_sha256=reference["catalog_sha256"],
        )
    if args.command == "validate":
        print(json.dumps({"status": "files_verified", "charged_calls": 0, "task": raw["task"]}))
        return
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if args.command == "run" and output.exists():
        raise FileExistsError(f"refusing to overwrite an existing PMO run: {output}")
    if args.command == "resume" and not manifest_path.exists():
        raise FileNotFoundError(f"PMO run manifest is missing: {manifest_path}")
    _set_arm(raw["arm"])
    reference_spec = {"guidance": raw["guidance"], "asset": raw["reference"]}
    with PmoOracleClient(
        python=Path(raw["oracle"]["python"]),
        task=raw["task"],
        asset_root=None
        if raw["oracle"]["asset_root"] is None
        else Path(raw["oracle"]["asset_root"]),
        source_root=ROOT,
        stderr_path=output / "oracle_worker.stderr.log",
    ) as oracle:
        campaign_manifest = {
            "schema_version": "compose.pmo.run.v1",
            "configuration_sha256": _sha256(args.config.resolve()),
            "configuration": raw,
            "core": core,
            "oracle": oracle.identity,
            "initialization_lock_sha256": initial["lock_sha256"],
            "jump_checkpoint_sha256": raw["jump_plans"]["sha256"],
            "reference": reference_spec,
        }
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != campaign_manifest:
                raise ValueError("PMO run manifest changed on resume")
        else:
            publish_json(manifest_path, campaign_manifest)
        task = ProgramTask(
            raw["task"],
            identity({"task": raw["task"], "oracle": oracle.identity}),
            "pmo",
        )
        ledger = ProgramQueryLedger(output / "oracle", task, oracle, budget=raw["budget"])
        result = run_program_campaign(
            output=output / "campaign",
            task=task,
            config=_search_config(raw["seed"], raw["queries_per_round"]),
            initialization=initial,
            library=(),
            ledger=ledger,
            rounds=raw["rounds"],
            queries_per_round=raw["queries_per_round"],
            hierarchy=None,
            fit_model=None,
            stagnation_rounds=None,
            bootstrap_rounds=1,
            initialization_mode="all_scored_pool",
            initial_parent_fraction=0.2,
            optimizer_type=PmoReferenceController,
            optimizer_kwargs={
                "jump_checkpoint": checkpoint,
                "reference_spec": reference_spec,
                "proposal_spec": raw["proposal"],
            },
            initial_batch_fn=initial_pmo_program_batch,
            initial_batch_kwargs={"proposal_spec": raw["proposal"]},
        )
        values = [row["score"] for row in ledger.rows]
        completed = len(values) >= raw["metrics_calls"]
        report = {
            "schema_version": "compose.pmo.result.v1",
            "status": "complete" if completed else "incomplete",
            "charged_calls": len(values),
            "reporting_calls": raw["metrics_calls"] if completed else None,
            "best_score": max(values[: raw["metrics_calls"]]) if completed else None,
            "final_top10": archive_top_k(
                [(row["endpoint"], row["score"]) for row in ledger.rows[: raw["metrics_calls"]]],
                k=10,
            )
            if completed
            else None,
            "auc_top10": pmo_top_ten_auc(
                values[: raw["metrics_calls"]], budget=raw["metrics_calls"]
            )
            if completed
            else None,
            "campaign_status": {
                "remaining_queries": result["remaining_queries"],
                "completed_rounds": len(result["history"]),
            },
        }
        publish_json(output / "result.json", report)
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
