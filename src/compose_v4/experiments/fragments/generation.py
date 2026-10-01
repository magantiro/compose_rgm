"""Local fragment generation with explicit assets and atomic output publication."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .assets import child_path, load_registry, sha256, verify_assets
from .errors import EvidenceError


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_revision(root: Path) -> str | None:
    # Git otherwise walks upward and attributes a plain source export to an
    # unrelated enclosing checkout. A .git file also supports linked worktrees.
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_settings(root: Path, task: str) -> dict:
    path = root / "experiments/fragments/generation.json"
    settings = json.loads(path.read_text())
    if settings.get("schema") != "compose_fragment_generation_config_v1":
        raise EvidenceError(f"invalid fragment generation schema: {path}")
    if task not in settings.get("tasks", {}):
        raise EvidenceError(f"unknown task {task!r} in {path}")
    maximum = settings.get("max_attempts_per_prompt_seed")
    if type(maximum) is not int or maximum < 1:
        raise EvidenceError(f"invalid max_attempts_per_prompt_seed in {path}: {maximum!r}")
    prompts = settings.get("prompts")
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(prompt, str) or not prompt for prompt in prompts)
        or len(set(prompts)) != len(prompts)
    ):
        raise EvidenceError(f"invalid prompt panel in {path}")
    return settings


def run_generation(
    *,
    root: Path,
    assets: Path,
    task: str,
    output: Path,
    python: Path,
    attempts: int,
    seeds: list[int] | None = None,
    prompts: list[str] | None = None,
    evaluate: bool = True,
) -> Path:
    root, output = root.resolve(), output.absolute()
    registry = load_registry(root)
    settings = load_settings(root, task)
    task_settings = settings["tasks"][task]
    seeds = seeds if seeds is not None else task_settings["seeds"]
    prompts = prompts if prompts is not None else settings["prompts"]
    maximum = settings["max_attempts_per_prompt_seed"]
    if type(attempts) is not int or not 1 <= attempts <= maximum:
        raise EvidenceError(f"attempts must be within 1..{maximum} per prompt/seed")
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise EvidenceError("seeds must be distinct nonnegative integers")
    if (
        not prompts
        or len(prompts) != len(set(prompts))
        or not set(prompts) <= set(settings["prompts"])
    ):
        raise EvidenceError("prompts must be distinct members of the frozen prompt panel")
    paths = verify_assets(registry, assets, task, include_evaluator=evaluate)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 5_000_000_000:
        raise EvidenceError("less than the frozen 5 GB disk-free safety floor")
    lock = output.with_name(f".{output.name}.lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.pending-", dir=output.parent))
    config = {
        "task": task,
        "seeds": seeds,
        "prompts": prompts,
        "attempts": attempts,
        "evaluate": evaluate,
        "environment": registry["environment"],
        "assets": {name: str(path) for name, path in paths.items()},
        "asset_sha256": {name: registry["assets"][name]["sha256"] for name in paths},
    }
    if task.startswith("superstructure_"):
        config["sampler_config"] = task_settings["sampler_config"]
        config["attachment_control"] = task_settings["attachment_control"]
    code_dir = Path(__file__).resolve().parent
    revision = source_revision(root)
    provenance = {
        "schema": "compose_fragment_generation_provenance_v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "code_revision": revision,
        "adapter_sha256": {p.name: sha256(p) for p in sorted(code_dir.glob("*.py"))},
        "runtime": "library",
        "configuration": config,
        "asset_registry_sha256": sha256(root / "experiments/fragments/assets.json"),
        "generation_config_sha256": sha256(root / "experiments/fragments/generation.json"),
        "asset_sha256": {name: registry["assets"][name]["sha256"] for name in paths},
        "mode": "local_generation",
        "status": "running",
        "original_launch_authorization_reused": False,
    }
    try:
        write_json(stage / "provenance.json", provenance)
        invocation = {
            **config,
            "runtime": str(root),
            "result_path": str(stage / "result.json"),
        }
        write_json(stage / "request.json", invocation)
        env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
        env.update(
            PYTHONHASHSEED="0",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        with (stage / "worker.log").open("w") as log:
            # -I ignores PYTHONHASHSEED. -s -P retains the declared seed and
            # excludes user-site modules and the working directory from imports.
            process = subprocess.run(
                [
                    str(python.absolute()),
                    "-s",
                    "-P",
                    str(code_dir / "generation_worker.py"),
                    str(stage / "request.json"),
                ],
                cwd=stage,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode:
            raise EvidenceError(
                f"fragment worker exited {process.returncode}, preserved log: {stage / 'worker.log'}"
            )
        result = json.loads((stage / "result.json").read_text())
        for relative, expected in result["source_sha256"].items():
            if sha256(child_path(root, relative)) != expected:
                raise EvidenceError(f"library source changed during generation: {relative}")
        if result.get("schema") != "compose_fragment_generation_v1" or len(result["cells"]) != len(
            seeds
        ) * len(prompts):
            raise EvidenceError("worker returned an incomplete population")
        if any(len(cell["attempts"]) != attempts for cell in result["cells"]):
            raise EvidenceError("worker returned incomplete output slots")
        identities = [(cell["seed"], cell["prompt"]) for cell in result["cells"]]
        if set(identities) != {(seed, prompt) for seed in seeds for prompt in prompts}:
            raise EvidenceError("worker returned wrong or duplicate prompt/seed cells")
        verify_assets(registry, assets, task, include_evaluator=evaluate)
        provenance.update(status="complete", result_sha256=sha256(stage / "result.json"))
        provenance["source_sha256"] = result["source_sha256"]
        write_json(stage / "provenance.json", provenance)
        (stage / "README.md").write_text(
            "# Local fragment execution\n\n"
            "Exact configuration, source and asset hashes are in provenance.json. "
            "Every attempted slot is retained.\n"
        )
        if output.exists():
            raise FileExistsError(f"output appeared during generation: {output}")
        stage.rename(output)
    except (OSError, ValueError, KeyError, TypeError, EvidenceError) as error:
        provenance.update(status="failed", error=str(error))
        write_json(stage / "provenance.json", provenance)
        raise
    finally:
        lock.unlink(missing_ok=True)
    return output
