"""Run every QED test source with bounded workers and checked resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from compose_v4.experiments.qed_shared_smc import QEDSMCConfig
from compose_v4.experiments.qed_shared_sources import load_qed_source_roles
from tools.verify_qed_assets import verify_assets

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_command(
    index: int,
    output: Path,
    *,
    value: Path,
    checkpoint: Path,
    time: float,
    config: QEDSMCConfig,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "tools/run_qed_shared_source.py"),
        "--index",
        str(index),
        "--value",
        str(value),
        "--checkpoint",
        str(checkpoint),
        "--time",
        str(time),
        "--horizon",
        str(config.horizon),
        "--particles",
        str(config.particles),
        "--candidates",
        str(config.candidates),
        "--output",
        str(output),
    ]


def _expected_code_hashes() -> dict[str, str]:
    return {
        "runner": sha256(ROOT / "tools/run_qed_shared_source.py"),
        "smc": sha256(ROOT / "src/compose_v4/experiments/qed_shared_smc.py"),
        "reference": sha256(ROOT / "src/compose_v4/experiments/qed_shared_reference.py"),
        "value": sha256(ROOT / "src/compose_v4/experiments/qed_shared_value.py"),
    }


def validate_existing(
    path: Path,
    index: int,
    source: str,
    *,
    split_sha256: str,
    reference_manifest_sha256: str,
    value_assets_manifest_sha256: str,
    value_metadata_sha256: str,
    checkpoint_sha256: str,
    time: float,
    code_sha256: dict[str, str],
    config: QEDSMCConfig,
) -> None:
    try:
        record = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError(f"QED resume result cannot be read: {path}") from error
    if not isinstance(record, dict):
        raise TypeError(f"QED resume result must be an object: {path}")
    expected = {
        "schema_version": "compose.qed.shared_result.v1",
        "test_index": index,
        "source_split_sha256": split_sha256,
        "reference_manifest_sha256": reference_manifest_sha256,
        "value_assets_manifest_sha256": value_assets_manifest_sha256,
        "value_metadata_sha256": value_metadata_sha256,
        "code_sha256": code_sha256,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"QED resume result {path} has incompatible {field}")
    result = record.get("result")
    if not isinstance(result, dict):
        raise TypeError(f"QED resume result lacks a result object: {path}")
    if result.get("source_original") != source:
        raise ValueError(f"QED resume result {path} has incompatible source")
    if result.get("value_head_source_split_sha256") != split_sha256:
        raise ValueError(f"QED resume result {path} has incompatible value-head split")
    reference = result.get("reference")
    if (
        not isinstance(reference, dict)
        or reference.get("checkpoint_sha256") != checkpoint_sha256
        or reference.get("time") != time
    ):
        raise ValueError(f"QED resume result {path} has incompatible reference")
    expected_config = {
        "horizon": config.horizon,
        "particles": config.particles,
        "candidates": config.candidates,
        "qed_minimum": config.qed_minimum,
        "similarity_minimum": config.similarity_minimum,
    }
    if result.get("configuration") != expected_config:
        raise ValueError(f"QED resume result {path} has incompatible configuration")
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != config.candidates:
        raise ValueError(f"QED resume result {path} has incompatible candidate count")


def _run(command: list[str], output: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    result = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"QED source failed for {output} with exit {result.returncode}: "
            f"{result.stdout}\n{result.stderr}"
        )


def run_panel(
    *,
    output: Path,
    value: Path,
    checkpoint: Path,
    time: float,
    config: QEDSMCConfig,
    workers: int,
    resume: bool,
) -> tuple[int, int]:
    if type(workers) is not int or workers < 1:
        raise ValueError("QED panel workers must be a positive integer")
    if not math.isfinite(time) or not 0.0 <= time <= 1.0:
        raise ValueError("QED panel reference time must be finite and in [0, 1]")
    output, value, checkpoint = output.resolve(), value.resolve(), checkpoint.resolve()
    checked = verify_assets(ROOT, checkpoint, value)
    metadata_path = value / "metadata.json"
    metadata = json.loads(metadata_path.read_bytes())
    if metadata.get("reference", {}).get("time") != time:
        raise ValueError(f"QED value head at {metadata_path} uses another reference time")
    if config.horizon > metadata["budget_max"]:
        raise ValueError(f"QED panel horizon exceeds the fitted value-head budget: {metadata_path}")
    roles = load_qed_source_roles(ROOT, ROOT / "experiments/qed/shared_sources.json")
    expected_names = {f"source_{index:04d}.json" for index in range(len(roles.test))}
    if output.exists():
        extras = {path.name for path in output.glob("*.json")} - expected_names
        if extras:
            raise ValueError(f"QED panel contains unexpected JSON outputs: {sorted(extras)[:5]}")
    planned = tuple(
        (index, output / f"source_{index:04d}.json") for index in range(len(roles.test))
    )
    existing = tuple((index, path) for index, path in planned if path.exists())
    if existing and not resume:
        raise FileExistsError(f"QED panel output already exists: {existing[0][1]}; use --resume")
    for index, path in existing:
        validate_existing(
            path,
            index,
            roles.test[index],
            split_sha256=roles.manifest_sha256,
            reference_manifest_sha256=checked["reference_manifest_sha256"],
            value_assets_manifest_sha256=checked["value_assets"]["manifest_sha256"],
            value_metadata_sha256=sha256(metadata_path),
            checkpoint_sha256=checked["shared_reference_checkpoint"]["sha256"],
            time=time,
            code_sha256=_expected_code_hashes(),
            config=config,
        )
    pending = tuple((index, path) for index, path in planned if not path.exists())
    if pending:
        output.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tuple(
                executor.map(
                    lambda item: _run(
                        source_command(
                            item[0],
                            item[1],
                            value=value,
                            checkpoint=checkpoint,
                            time=time,
                            config=config,
                        ),
                        item[1],
                    ),
                    pending,
                )
            )
    return len(pending), len(existing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--value", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "local_assets/fragments/r_theta_nll.pt"
    )
    parser.add_argument("--time", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = QEDSMCConfig(
        horizon=args.horizon,
        particles=args.particles,
        candidates=args.candidates,
    )
    started, kept = run_panel(
        output=args.output,
        value=args.value,
        checkpoint=args.checkpoint,
        time=args.time,
        config=config,
        workers=args.workers,
        resume=args.resume,
    )
    print(f"QED panel complete: new={started}, verified_existing={kept}")


if __name__ == "__main__":
    main()
