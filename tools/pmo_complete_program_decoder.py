"""Prepare, generate, and evaluate the sealed PMO complete-program decoder gate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import resource
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.experiments.pmo_complete_program_decoder import (
    DecoderConfig,
    build_candidate_lock,
    evaluate_candidate_lock,
    seal_candidate_lock,
    source_manifest_from_panels,
    validate_candidate_lock,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = Path("configs/pmo_complete_program_decoder_v1.json")
DEFAULT_SOURCE_MANIFEST = Path(
    "diagnostics/pmo_complete_program_decoder/implementation/source_manifest.json"
)
DEFAULT_GENERATION = Path("diagnostics/pmo_complete_program_decoder/attempt_1")
IMPLEMENTATION_PATHS = (
    CONTRACT,
    Path("docs/PMO_COMPLETE_PROGRAM_DECODER.md"),
    Path("modal_apps/pmo_complete_program_decoder_app.py"),
    Path("src/compose_v4/experiments/pmo_complete_program_decoder.py"),
    Path("tests/test_pmo_complete_program_decoder.py"),
    Path("tools/pmo_complete_program_decoder.py"),
    Path("tools/pmo_complete_program_decoder_parallel.py"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    return value


def _load_contract(root: Path) -> tuple[dict, str]:
    path = root / CONTRACT
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "pmo_complete_program_decoder_contract_v1"
        or envelope.get("contract_sha256") != identity(payload)
    ):
        raise ValueError(f"invalid PMO complete-program decoder contract: {path}")
    return payload, envelope["contract_sha256"]


def _read_json(path: Path) -> dict:
    raw = (
        gzip.decompress(path.read_bytes()).decode()
        if path.suffix == ".gz"
        else path.read_text()
    )
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _load_pinned(root: Path, specification: dict) -> dict:
    path = root / specification["path"]
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"sealed physical hash mismatch: {path}")
    envelope = _read_json(path)
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or envelope.get("payload_sha256") != identity(payload)
        or envelope["payload_sha256"] != specification["payload_sha256"]
    ):
        raise ValueError(f"sealed payload hash mismatch: {path}")
    return payload


def _load_envelope(path: Path) -> tuple[dict, str]:
    envelope = _read_json(path)
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(
        payload
    ):
        raise ValueError(f"invalid deterministic envelope: {path}")
    return payload, envelope["payload_sha256"]


def _publish_file(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ready = _json_ready(value)
    raw = json.dumps(ready, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_envelope(path: Path, payload: dict) -> None:
    ready = _json_ready(payload)
    _publish_file(path, {"payload": ready, "payload_sha256": identity(ready)})


def _code_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_committed_implementation(root: Path) -> None:
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            *(str(path) for path in IMPLEMENTATION_PATHS),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "authoritative decoding requires the decoder implementation, contract, "
            "documentation and tests to be committed and unchanged"
        )


def _decoder_config(contract: dict) -> DecoderConfig:
    row = contract["decoder"]
    return DecoderConfig(
        beam_widths=tuple(row["beam_widths"]),
        snapshot_depths=tuple(row["snapshot_depths"]),
        output_cutoffs=tuple(row["output_cutoffs"]),
        successors_per_rule=int(row["successors_per_rule"]),
        maximum_primitives=int(row["maximum_primitives"]),
        maximum_components=int(row["maximum_components"]),
        maximum_active_atoms=int(row["maximum_active_atoms"]),
    )


def _require_authoritative_execution(contract: dict) -> dict:
    legal = contract["inputs"].get("legal_action_authoritative")
    if contract.get("authoritative_execution_enabled") is not True or not isinstance(
        legal, dict
    ):
        raise RuntimeError(
            "authoritative decoder execution is disabled until the legal-action "
            "checkpoint is rerun from clean committed source and pinned in the contract"
        )
    return legal


def prepare_sources(root: Path, contract: dict, output: Path) -> None:
    panels = _load_pinned(root, contract["inputs"]["panel_corpus"])
    manifest = source_manifest_from_panels(
        panels, split_identity=contract["split_identity"]
    )
    manifest["input"] = {
        "panel_corpus_sha256": contract["inputs"]["panel_corpus"]["sha256"],
        "panel_corpus_payload_sha256": contract["inputs"]["panel_corpus"][
            "payload_sha256"
        ],
    }
    _publish_envelope(root / output, manifest)


def generate(
    root: Path,
    contract: dict,
    contract_sha256: str,
    source_manifest_path: Path,
    output: Path,
) -> None:
    legal_specification = _require_authoritative_execution(contract)
    if contract.get("execution", {}).get("mode") == "durable_source_case_shards":
        raise RuntimeError(
            "authoritative generation requires the durable source-case shard launcher"
        )
    _require_committed_implementation(root)
    source_manifest, source_manifest_payload_sha256 = _load_envelope(
        root / source_manifest_path
    )
    dependency = _load_pinned(root, contract["inputs"]["dependency_policy_runtime"])
    legal = _load_pinned(root, legal_specification)
    started = perf_counter()
    payload = build_candidate_lock(
        source_manifest,
        dependency,
        legal,
        _decoder_config(contract),
    )
    payload["provenance"] = {
        "contract_sha256": contract_sha256,
        "code_revision": _code_revision(root),
        "source_manifest_payload_sha256": source_manifest_payload_sha256,
        "dependency_policy_runtime_sha256": contract["inputs"][
            "dependency_policy_runtime"
        ]["sha256"],
        "dependency_policy_runtime_payload_sha256": contract["inputs"][
            "dependency_policy_runtime"
        ]["payload_sha256"],
        "legal_action_runtime_sha256": legal_specification["sha256"],
        "legal_action_runtime_payload_sha256": legal_specification["payload_sha256"],
        "implementation_file_sha256": {
            str(path): sha256_file(root / path) for path in IMPLEMENTATION_PATHS
        },
    }
    sealed = seal_candidate_lock(payload)
    output_path = root / output
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_path}")
    output_path.mkdir(parents=True)
    try:
        lock_path = output_path / "candidate_lock.json"
        _publish_file(lock_path, sealed)
        receipt = {
            "schema_version": "pmo_complete_program_decoder_generation_receipt_v1",
            "candidate_lock_path": lock_path.name,
            "candidate_lock_sha256": sha256_file(lock_path),
            "candidate_lock_payload_sha256": sealed["payload_sha256"],
            "wall_seconds": perf_counter() - started,
            "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "new_oracle_calls": 0,
        }
        _publish_envelope(output_path / "generation_receipt.json", receipt)
    except BaseException:
        # Preserve no partial authoritative directory. Both files are newly created here.
        for child in output_path.iterdir():
            child.unlink()
        output_path.rmdir()
        raise


def evaluate(
    root: Path,
    contract: dict,
    candidate_lock_path: Path,
    generation_receipt_path: Path,
    output: Path,
) -> None:
    _require_authoritative_execution(contract)
    _require_committed_implementation(root)
    lock_path = root / candidate_lock_path
    receipt, _ = _load_envelope(root / generation_receipt_path)
    sealed = _read_json(lock_path)
    payload = validate_candidate_lock(sealed)
    if payload.get("provenance", {}).get("contract_sha256") != identity(contract):
        raise ValueError("candidate lock was generated under a different contract")
    if receipt.get("candidate_lock_sha256") != sha256_file(lock_path):
        raise ValueError("generation receipt does not identify the candidate lock file")
    if receipt.get("candidate_lock_payload_sha256") != sealed["payload_sha256"]:
        raise ValueError("generation receipt candidate payload identity mismatch")

    # Teacher artifacts are intentionally loaded only after the candidate lock is sealed
    # and matched to its independent generation receipt.
    panels = _load_pinned(root, contract["inputs"]["panel_corpus"])
    dependency = _load_pinned(root, contract["inputs"]["dependency_region_corpus"])
    report = evaluate_candidate_lock(
        sealed,
        panels,
        dependency,
        _decoder_config(contract),
    )
    report["generation_compute"] = {
        key: receipt[key]
        for key in ("wall_seconds", "peak_rss_raw", "platform", "python")
    }
    report["provenance"] = sealed["payload"].get("provenance", {})
    report["evaluation_inputs"] = {
        name: {
            "path": specification["path"],
            "sha256": specification["sha256"],
            "payload_sha256": specification["payload_sha256"],
        }
        for name, specification in (
            ("panel_corpus", contract["inputs"]["panel_corpus"]),
            (
                "dependency_region_corpus",
                contract["inputs"]["dependency_region_corpus"],
            ),
        )
    }
    _publish_envelope(root / output, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-sources")
    prepare.add_argument("--output", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    generation = commands.add_parser("generate")
    generation.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    generation.add_argument("--output", type=Path, default=DEFAULT_GENERATION)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument(
        "--candidate-lock",
        type=Path,
        default=DEFAULT_GENERATION / "candidate_lock.json",
    )
    evaluation.add_argument(
        "--generation-receipt",
        type=Path,
        default=DEFAULT_GENERATION / "generation_receipt.json",
    )
    evaluation.add_argument(
        "--output", type=Path, default=DEFAULT_GENERATION / "result.json"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    contract, contract_sha256 = _load_contract(root)
    if args.command == "prepare-sources":
        prepare_sources(root, contract, args.output)
    elif args.command == "generate":
        generate(
            root,
            contract,
            contract_sha256,
            args.source_manifest,
            args.output,
        )
    else:
        evaluate(
            root,
            contract,
            args.candidate_lock,
            args.generation_receipt,
            args.output,
        )


if __name__ == "__main__":
    main()
