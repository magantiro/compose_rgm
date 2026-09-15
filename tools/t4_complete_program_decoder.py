"""Fit, generate and evaluate the grouped T4 complete-program decoder."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

from compose_v4.control.autoregressive_program_decoder import (
    LEARNED,
    MARGINAL,
    AutoregressivePolicy,
    DecoderConfig,
    TrainingConfig,
    decode_programs,
    fit_autoregressive_policy,
    fit_legal_where_how_ranker,
    legal_where_how_metrics,
    marginal_rule_control_probabilities,
    teacher_forced_metrics,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_legal_action_policy import RankerConfig
from compose_v4.experiments.route_proposal_quality import SCHEMA as QUALITY_SCHEMA
from compose_v4.experiments.route_proposal_quality import evaluate
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_route_policy_comparison import (
    predeclared_source_folds,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.t4_program_vocabulary_audit import source_group_map
from tools.t4_route_distillation import _teacher_traces

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_complete_program_decoder_v1.json"
T4_CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
LIBRARY = ROOT / "diagnostics/t4_shared_program_controller/attempt_2/shared_library.json"
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
RUNTIME_SCHEMA = "t4_complete_program_decoder_runtime_v1"
SOURCE_SCHEMA = "t4_complete_program_decoder_source_manifest_v1"
EVALUATION_SCHEMA = "t4_complete_program_decoder_evaluation_manifest_v1"
LOCK_SCHEMA = "t4_complete_program_decoder_candidate_lock_v1"
REPORT_SCHEMA = "t4_complete_program_decoder_report_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sealed(path: Path) -> dict:
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    envelope = json.loads(raw)
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256", envelope.get("contract_sha256"))
    if not isinstance(payload, dict) or claimed != identity(payload):
        raise ValueError(f"artifact is not self-hashed: {path}")
    return payload


def publish(path: Path, payload: dict, *, compressed: bool = False) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite decoder artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    encoded = (json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n").encode()
    target = path.with_suffix(path.suffix + ".tmp")
    target.write_bytes(gzip.compress(encoded, mtime=0) if compressed else encoded)
    target.replace(path)


def git_revision() -> tuple[str, bool]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    return revision, dirty


def load_inputs() -> tuple[dict, list[dict], dict[str, dict]]:
    contract = load_sealed(CONTRACT)
    if contract["oracle"]["calls_authorized"] != 0:
        raise ValueError("decoder contract unexpectedly authorizes oracle calls")
    for name, path in (
        ("teacher_library", LIBRARY),
        ("source_registry", SEEDS),
    ):
        if sha256(path) != contract["inputs"][name]["sha256"]:
            raise ValueError(f"decoder input hash changed: {path}")
    library = json.loads(LIBRARY.read_text())
    traces, exclusions = _teacher_traces(library)
    if exclusions or len(traces) != 77:
        raise RuntimeError(
            f"decoder teacher census changed: {len(traces)} admitted, {exclusions}"
        )
    t4_contract = unseal(T4_CONTRACT)
    metadata = source_group_map(t4_contract, json.loads(SEEDS.read_text()))
    return contract, traces, metadata


def _fold(traces: list[dict], metadata: dict[str, dict], fold: int):
    folds = predeclared_source_folds(metadata)
    selected = next((row for row in folds if row["fold"] == fold), None)
    if selected is None:
        raise ValueError(f"unknown grouped fold: {fold}")
    train = [row for row in traces if row["source_group"] in selected["train_sources"]]
    test = [row for row in traces if row["source_group"] in selected["test_sources"]]
    if not train or not test:
        raise RuntimeError("grouped decoder split is empty")
    return selected, train, test


def fit_fold(fold: int, output: Path) -> None:
    contract, traces, metadata = load_inputs()
    split, train, test = _fold(traces, metadata, fold)
    training = TrainingConfig(seed=20260914 + fold)
    ranker = RankerConfig(negatives_per_teacher=32)
    began = perf_counter()
    policy, history = fit_autoregressive_policy(
        train, maximum=contract["support"]["maximum_primitives"], config=training
    )
    marginal_rule, marginal_control = marginal_rule_control_probabilities(
        train,
        maximum=contract["support"]["maximum_primitives"],
        exploration_floor=training.exploration_floor,
    )
    legal, legal_fit = fit_legal_where_how_ranker(
        train,
        maximum=contract["support"]["maximum_primitives"],
        config=ranker,
    )
    teacher_metrics = {
        "train": teacher_forced_metrics(
            policy, train, maximum=contract["support"]["maximum_primitives"]
        ),
        "test": teacher_forced_metrics(
            policy, test, maximum=contract["support"]["maximum_primitives"]
        ),
    }
    legal_metrics = {
        "train": legal_where_how_metrics(train, legal),
        "test": legal_where_how_metrics(test, legal),
    }
    revision, dirty = git_revision()
    runtime = {
        "schema_version": RUNTIME_SCHEMA,
        "fold": fold,
        "autoregressive_policy": policy.to_checkpoint(),
        "legal_where_how_policy": legal,
        "marginal_rule_probabilities": list(marginal_rule),
        "marginal_control_probabilities": list(marginal_control),
        "training_identity": identity(
            {
                "schema_version": "t4_complete_decoder_fold_training_v1",
                "fold": fold,
                "contract": identity(contract),
                "autoregressive": policy.training_identity,
                "legal": legal["training_identity"],
            }
        ),
        "runtime_teacher_rows": 0,
        "runtime_task_fields": 0,
        "new_oracle_calls": 0,
    }
    sources = []
    evaluation = []
    by_source = {}
    for teacher in test:
        by_source.setdefault(teacher["source_group"], []).append(teacher)
    for source_group, teachers in sorted(by_source.items()):
        state = encode_state(teachers[0]["source"])
        source_case_id = identity(
            {
                "schema_version": "t4_complete_decoder_source_case_v1",
                "fold": fold,
                "state": state,
            }
        )
        sources.append(
            {
                "source_case_id": source_case_id,
                "fold": fold,
                "source_state": state,
            }
        )
        evaluation.append(
            {
                "source_case_id": source_case_id,
                "source_group": source_group,
                "source_state": state,
                "teachers": [
                    {
                        "teacher_id": f"teacher-{index}",
                        "endpoint_state": row["trace"]["states"][-1],
                    }
                    for index, row in enumerate(teachers)
                ],
            }
        )
    source_manifest = {
        "schema_version": SOURCE_SCHEMA,
        "fold": fold,
        "source_cases": sources,
        "teacher_fields_present": False,
        "task_identity_present": False,
        "new_oracle_calls": 0,
    }
    evaluation_manifest = {
        "schema_version": EVALUATION_SCHEMA,
        "fold": fold,
        "train_sources": list(split["train_sources"]),
        "test_sources": list(split["test_sources"]),
        "cases": evaluation,
        "new_oracle_calls": 0,
    }
    report = {
        "schema_version": "t4_complete_program_decoder_fit_report_v1",
        "fold": fold,
        "training_routes": len(train),
        "held_routes": len(test),
        "history": history,
        "teacher_forced": teacher_metrics,
        "legal_where_how": legal_metrics,
        "legal_fit": legal_fit,
        "elapsed_seconds": perf_counter() - began,
        "implementation": {"revision": revision, "working_tree_dirty": dirty},
        "new_oracle_calls": 0,
    }
    publish(output / "runtime_checkpoint.json.gz", runtime, compressed=True)
    publish(output / "source_manifest.json", source_manifest)
    publish(output / "evaluation_manifest.json.gz", evaluation_manifest, compressed=True)
    publish(output / "fit_report.json", report)


def generate_fold(checkpoint_path: Path, source_path: Path, output: Path) -> None:
    runtime = load_sealed(checkpoint_path)
    sources = load_sealed(source_path)
    if runtime.get("schema_version") != RUNTIME_SCHEMA:
        raise ValueError("decoder runtime schema mismatch")
    if sources.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("decoder source manifest schema mismatch")
    if runtime["fold"] != sources["fold"]:
        raise ValueError("decoder runtime/source fold mismatch")
    if sources.get("teacher_fields_present") is not False:
        raise ValueError("candidate generation received teacher fields")
    policy = AutoregressivePolicy.from_checkpoint(runtime["autoregressive_policy"])
    decoder_config = DecoderConfig()
    cases = []
    for row in sources["source_cases"]:
        source = decode_state(row["source_state"])
        decoded = []
        for decoder in (MARGINAL, LEARNED):
            began = perf_counter()
            result = decode_programs(
                source,
                decoder=decoder,
                config=decoder_config,
                marginal_rule_probabilities=tuple(
                    runtime["marginal_rule_probabilities"]
                ),
                marginal_control_probabilities=tuple(
                    runtime["marginal_control_probabilities"]
                ),
                policy=policy if decoder == LEARNED else None,
                legal_checkpoint=(
                    runtime["legal_where_how_policy"] if decoder == LEARNED else None
                ),
            )
            result["proposal_seconds"] = perf_counter() - began
            decoded.append(result)
        cases.append(
            {
                "source_case_id": row["source_case_id"],
                "fold": row["fold"],
                "source_state": row["source_state"],
                "decoders": decoded,
            }
        )
    lock = {
        "schema_version": LOCK_SCHEMA,
        "fold": runtime["fold"],
        "training_identity": runtime["training_identity"],
        "source_cases": cases,
        "teacher_fields_present": False,
        "task_identity_present": False,
        "new_oracle_calls": 0,
    }
    publish(output, lock, compressed=output.suffix == ".gz")


def evaluate_fold(lock_path: Path, evaluation_path: Path, output: Path) -> None:
    lock = load_sealed(lock_path)
    evaluation_manifest = load_sealed(evaluation_path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("candidate lock schema mismatch")
    if evaluation_manifest.get("schema_version") != EVALUATION_SCHEMA:
        raise ValueError("evaluation manifest schema mismatch")
    locked = {row["source_case_id"]: row for row in lock["source_cases"]}
    cases = []
    for reference in evaluation_manifest["cases"]:
        generated = locked.get(reference["source_case_id"])
        if generated is None:
            raise ValueError("candidate lock is missing a held source")
        pools = []
        for decoded in generated["decoders"]:
            rows = decoded["snapshots"]["32"]
            pools.append(
                {
                    "policy_id": decoded["decoder"],
                    "proposal_seconds": decoded["proposal_seconds"],
                    "attempts": [
                        {
                            "attempt_id": row["candidate_id"],
                            "rank": rank,
                            "status": "complete",
                            "endpoint_state": row["endpoint_state"],
                        }
                        for rank, row in enumerate(rows, 1)
                    ],
                }
            )
        cases.append(
            {
                "source_id": reference["source_group"],
                "source_state": reference["source_state"],
                "teachers": reference["teachers"],
                "policy_pools": pools,
            }
        )
    quality_input = {
        "schema_version": QUALITY_SCHEMA,
        "oracle_calls": 0,
        "split": {
            "evaluation_role": "test",
            "train_sources": evaluation_manifest["train_sources"],
            "calibration_sources": [],
            "test_sources": evaluation_manifest["test_sources"],
        },
        "cases": cases,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "fold": lock["fold"],
        "quality": evaluate(quality_input, cutoffs=(1, 5, 10, 32, 128)),
        "telemetry": {
            row["source_case_id"]: {
                decoded["decoder"]: decoded["telemetry"]
                for decoded in row["decoders"]
            }
            for row in lock["source_cases"]
        },
        "new_oracle_calls": 0,
    }
    publish(output, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    fit.add_argument("--output", type=Path, required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--sources", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--lock", type=Path, required=True)
    evaluator.add_argument("--evaluation", type=Path, required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit_fold(args.fold, args.output)
    elif args.command == "generate":
        generate_fold(args.checkpoint, args.sources, args.output)
    else:
        evaluate_fold(args.lock, args.evaluation, args.output)


if __name__ == "__main__":
    main()
