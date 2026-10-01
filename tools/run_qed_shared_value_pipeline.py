"""Generate the source-disjoint QED value corpus and fit its separate head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from compose_v4.experiments.qed_shared_sources import QEDSourceRoles, load_qed_source_roles

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceJob:
    role: str
    index: int

    @property
    def stem(self) -> str:
        return f"{self.role}_{self.index:04d}"


def source_jobs() -> tuple[SourceJob, ...]:
    roles = load_qed_source_roles(ROOT, ROOT / "experiments/qed/shared_sources.json")
    return tuple(SourceJob("train", index) for index in range(len(roles.train))) + tuple(
        SourceJob("validation", index) for index in range(len(roles.validation))
    )


def stage_command(
    stage: str,
    job: SourceJob,
    workspace: Path,
    *,
    horizon: int,
    replicates: int,
    time: float,
    checkpoint: Path,
) -> tuple[list[str], Path]:
    rollout = workspace / "rollouts" / f"{job.stem}.json"
    feature = workspace / "features" / f"{job.stem}.npz"
    if stage == "rollouts":
        return (
            [
                sys.executable,
                str(ROOT / "tools/run_qed_shared_rollout.py"),
                "--role",
                job.role,
                "--index",
                str(job.index),
                "--horizon",
                str(horizon),
                "--replicates",
                str(replicates),
                "--time",
                str(time),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(rollout),
            ],
            rollout,
        )
    if stage == "features":
        if not rollout.is_file():
            raise FileNotFoundError(f"QED rollout is absent: {rollout}")
        return (
            [
                sys.executable,
                str(ROOT / "tools/build_qed_shared_features.py"),
                "--rollout",
                str(rollout),
                "--role",
                job.role,
                "--index",
                str(job.index),
                "--budget-max",
                str(horizon),
                "--time",
                str(time),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(feature),
            ],
            feature,
        )
    raise ValueError(f"unsupported QED value-pipeline stage: {stage}")


def _run(command: list[str], output: Path) -> None:
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"QED job failed for {output} with exit {result.returncode}: "
            f"{result.stdout}\n{result.stderr}"
        )


def _reference_fields(time: float) -> dict[str, object]:
    manifest = json.loads((ROOT / "experiments/fragments/assets.json").read_text())
    return {
        "checkpoint_sha256": manifest["assets"]["checkpoint"]["sha256"],
        "catalog_fingerprint": manifest["catalog_fingerprint"],
        "catalog_sha256": manifest["assets"]["catalog"]["sha256"],
        "time": time,
        "persistent_slots": 48,
        "active_atom_boundary": "condition_on_supported_successors",
        "transition_law": "canonical_successor_embedded_jump",
    }


def _check_fields(actual: object, expected: dict, path: Path) -> None:
    if not isinstance(actual, Mapping):
        raise TypeError(f"QED resume output must contain an object: {path}")
    for field, value in expected.items():
        if actual.get(field) != value:
            raise ValueError(
                f"QED resume output {path} has incompatible {field}: "
                f"{actual.get(field)!r}, expected {value!r}"
            )


def _validate_rollout(
    path: Path,
    job: SourceJob,
    *,
    horizon: int,
    replicates: int,
    reference_fields: dict[str, object],
    roles: QEDSourceRoles,
) -> None:
    try:
        rollout = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError(f"QED resume rollout cannot be read: {path}") from error
    sources = roles.train if job.role == "train" else roles.validation
    input_index = roles.train_input_indices[job.index] if job.role == "train" else job.index
    _check_fields(
        rollout,
        {
            "schema_version": "compose.qed.shared_rollout.v1",
            "source_role": job.role,
            "source_index": job.index,
            "source_original": sources[job.index],
            "source_input_row_index": input_index,
            "source_split_sha256": roles.manifest_sha256,
            "configuration": {"horizon": horizon, "replicates": replicates},
        },
        path,
    )
    reference = rollout.get("reference")
    if not isinstance(reference, dict):
        raise TypeError(f"QED resume rollout lacks reference identity: {path}")
    _check_fields(reference, reference_fields, path)
    trajectories = rollout.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != replicates:
        raise ValueError(f"QED resume rollout has the wrong trajectory count: {path}")


def _validate_feature(
    path: Path,
    rollout: Path,
    job: SourceJob,
    *,
    horizon: int,
    reference_fields: dict[str, object],
    roles: QEDSourceRoles,
) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        raise ValueError(f"QED resume feature metadata cannot be read: {path}") from error
    input_index = roles.train_input_indices[job.index] if job.role == "train" else job.index
    _check_fields(
        metadata,
        {
            "schema_version": "compose.qed.shared_features.v4",
            "role": job.role,
            "source_index": job.index,
            "source_input_row_index": input_index,
            "source_split_sha256": roles.manifest_sha256,
            "budget_max": horizon,
            "target_semantics": "terminal_region",
            "rollout_sha256": hashlib.sha256(rollout.read_bytes()).hexdigest(),
            "builder_sha256": hashlib.sha256(
                (ROOT / "src/compose_v4/experiments/qed_shared_training.py").read_bytes()
            ).hexdigest(),
        },
        path,
    )
    reference = metadata.get("reference")
    if not isinstance(reference, dict):
        raise TypeError(f"QED resume feature lacks reference identity: {path}")
    _check_fields(reference, reference_fields, path)


def run_stage(
    stage: str,
    jobs: tuple[SourceJob, ...],
    workspace: Path,
    *,
    workers: int,
    resume: bool,
    horizon: int,
    replicates: int,
    time: float,
    checkpoint: Path,
) -> tuple[int, int]:
    planned = tuple(
        stage_command(
            stage,
            job,
            workspace,
            horizon=horizon,
            replicates=replicates,
            time=time,
            checkpoint=checkpoint,
        )
        for job in jobs
    )
    existing = tuple(output for _, output in planned if output.exists())
    if existing and not resume:
        raise FileExistsError(
            f"QED {stage} output already exists: {existing[0]}; use --resume to keep it"
        )
    if existing:
        roles = load_qed_source_roles(ROOT, ROOT / "experiments/qed/shared_sources.json")
        reference_fields = _reference_fields(time)
        for job, (_, output) in zip(jobs, planned, strict=True):
            if not output.exists():
                continue
            rollout = workspace / "rollouts" / f"{job.stem}.json"
            if stage == "rollouts":
                _validate_rollout(
                    output,
                    job,
                    horizon=horizon,
                    replicates=replicates,
                    reference_fields=reference_fields,
                    roles=roles,
                )
            else:
                _validate_rollout(
                    rollout,
                    job,
                    horizon=horizon,
                    replicates=replicates,
                    reference_fields=reference_fields,
                    roles=roles,
                )
                _validate_feature(
                    output,
                    rollout,
                    job,
                    horizon=horizon,
                    reference_fields=reference_fields,
                    roles=roles,
                )
    pending = tuple((command, output) for command, output in planned if not output.exists())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tuple(executor.map(lambda item: _run(*item), pending))
    return len(pending), len(existing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--stage", choices=("rollouts", "features", "fit", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--time", type=float, default=0.5)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "local_assets/fragments/r_theta_nll.pt"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.horizon < 1 or args.replicates < 1:
        parser.error("workers, horizon and replicates must be positive")
    if not math.isfinite(args.time) or not 0 < args.time < 1:
        parser.error("reference time must be finite and strictly between zero and one")
    workspace = args.workspace.resolve()
    jobs = source_jobs()
    stages = ("rollouts", "features") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "fit":
            continue
        created, existing = run_stage(
            stage,
            jobs,
            workspace,
            workers=args.workers,
            resume=args.resume,
            horizon=args.horizon,
            replicates=args.replicates,
            time=args.time,
            checkpoint=args.checkpoint.resolve(),
        )
        print(f"{stage}: {created} created, {existing} retained", flush=True)
    if args.stage in ("fit", "all"):
        command = [
            sys.executable,
            str(ROOT / "tools/fit_qed_shared_value.py"),
            "--shards",
            str(workspace / "features"),
            "--time",
            str(args.time),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--budget-max",
            str(args.horizon),
            "--output",
            str(workspace / "value"),
        ]
        if (workspace / "value").exists():
            raise FileExistsError(f"QED value output already exists: {workspace / 'value'}")
        _run(command, workspace / "value")
        print(f"value: {workspace / 'value'}", flush=True)


if __name__ == "__main__":
    main()
