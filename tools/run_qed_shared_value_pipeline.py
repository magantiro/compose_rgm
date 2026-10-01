"""Generate the source-disjoint QED value corpus and fit its separate head."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

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
