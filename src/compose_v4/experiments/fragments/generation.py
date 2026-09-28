"""Portable orchestration for isolated, frozen fragment generation runtimes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .assets import child_path, load_registry, sha256, verify_assets
from .evidence import EvidenceError


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


def verify_runtime(root: Path, task: str) -> tuple[Path, dict]:
    directory = child_path(root / "experiments/fragments/runtime", task)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema") != "compose_fragment_runtime_source_v1"
        or manifest.get("task") != task
    ):
        raise EvidenceError(f"wrong runtime manifest: {directory}")
    archive = directory / "source.zip"
    if sha256(archive) != manifest["archive_sha256"]:
        raise EvidenceError(f"runtime archive hash mismatch: {archive}")
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(set(bundle.namelist())) or set(bundle.namelist()) != set(
            manifest["files"]
        ):
            raise EvidenceError(f"runtime archive membership mismatch: {archive}")
        import hashlib

        for name, expected in manifest["files"].items():
            child_path(directory, name)
            info = bundle.getinfo(name)
            if info.external_attr >> 16 & 0o170000 not in (0, 0o100000):
                raise EvidenceError(f"runtime contains a non-regular file: {name}")
            if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise EvidenceError(f"runtime member hash mismatch: {name}")
    return archive, manifest


def fixtures(root: Path, task: str) -> list[dict]:
    directory = root / "experiments/fragments/fixtures"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "compose_fragment_parity_fixtures_v1":
        raise EvidenceError("wrong parity fixture schema")
    rows = manifest["tasks"][task]
    for row in rows:
        path = child_path(directory, row["path"])
        if sha256(path) != row["sha256"]:
            raise EvidenceError(f"fixture hash mismatch: {path}")
    return rows


def check_parity(root: Path, task: str, result: dict, records: list[dict]) -> dict:
    """Compare full panels, or exact superstructure actions and endpoints."""
    directory = root / "experiments/fragments/fixtures"
    expected = [
        json.loads((directory / row["path"]).read_text())
        for row in records
        if not row["source_path"].startswith("configs/")
    ]
    actual = result["cells"][0]["attempts"]
    if task.startswith("superstructure_"):
        report = expected[0]["results"]["superstructure_generation"]
        row = report["per_drug"]["BARICITINIB"][0]
        keys = ("emitted_smiles", "committed_smiles", "accepted_actions", "events")
        pairs = [({k: row["attempt_records"][0][k] for k in keys}, {k: actual[0][k] for k in keys})]
    else:
        pairs = [
            (saved["panel"], new["panel"]) for saved, new in zip(expected, actual, strict=True)
        ]
    matches = [saved == new for saved, new in pairs]
    differences = []
    for index, (saved, new) in enumerate(pairs):
        if saved != new:
            differences.append(
                {
                    "attempt": index,
                    "different_top_level_fields": sorted(
                        key for key in set(saved) | set(new) if saved.get(key) != new.get(key)
                    ),
                }
            )
    return {
        "schema": "compose_fragment_runtime_parity_v1",
        "task": task,
        "passed": all(matches),
        "attempts_checked": len(pairs),
        "matches": matches,
        "comparison": "exact JSON values; complete program panels including RNG/probabilities, or superstructure accepted actions/events/endpoints",
        "selection": "first BARICITINIB attempt at first declared seed; first two linker attempts",
        "differences": differences,
        "inputs": records,
        "limit": "bounded implementation parity, not a full benchmark rerun or a statistical result",
    }


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
    parity: bool = False,
    evaluate: bool = True,
) -> Path:
    root, output = root.resolve(), output.absolute()
    registry = load_registry(root)
    paths = verify_assets(registry, assets, task)
    archive, snapshot = verify_runtime(root, task)
    evidence = json.loads((root / "experiments/fragments/manifest.json").read_text())
    prompt_task = "superstructure_generation" if task == "superstructure_uniform" else task
    seeds = seeds if seeds is not None else evidence["generation_seeds"][prompt_task]
    prompts = prompts if prompts is not None else evidence["prompts"]
    if not 1 <= attempts <= evidence["attempts_per_prompt_seed"]:
        raise EvidenceError("attempts must be within 1..100 per prompt/seed")
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise EvidenceError("seeds must be distinct nonnegative integers")
    if (
        not prompts
        or len(prompts) != len(set(prompts))
        or not set(prompts) <= set(evidence["prompts"])
    ):
        raise EvidenceError("prompts must be distinct members of the frozen prompt panel")
    records = fixtures(root, task)
    contract_file = root / "experiments/fragments/fixtures" / records[0]["path"]
    historical_contract = json.loads(contract_file.read_text())["payload"]
    if parity:
        seeds, prompts, attempts = (
            [evidence["generation_seeds"][prompt_task][0]],
            ["BARICITINIB"],
            2 if task == "linker_design" else 1,
        )
        evaluate = False
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
    }
    if task.startswith("superstructure_"):
        config["sampler_config"] = historical_contract["sampler_config"]
        config["attachment_control"] = historical_contract["attachment_control"]
    code_dir = Path(__file__).resolve().parent
    revision = source_revision(root)
    provenance = {
        "schema": "compose_fragment_generation_provenance_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": revision,
        "adapter_sha256": {p.name: sha256(p) for p in sorted(code_dir.glob("*.py"))},
        "source_snapshot": snapshot,
        "configuration": config,
        "asset_registry_sha256": sha256(root / "experiments/fragments/assets.json"),
        "asset_sha256": {name: registry["assets"][name]["sha256"] for name in paths},
        "mode": "bounded_historical_parity" if parity else "local_generation",
        "status": "running",
        "original_launch_authorization_reused": False,
        "historical_protocol_inputs": records,
    }
    try:
        write_json(stage / "provenance.json", provenance)
        with tempfile.TemporaryDirectory(prefix="fragment-runtime-") as scratch:
            runtime = Path(scratch)
            with zipfile.ZipFile(archive) as bundle:
                # Verification above rejects traversal, duplicates and non-regular files.
                bundle.extractall(runtime)
            for name, path in paths.items():
                if name.startswith("evaluator_"):
                    relative = Path(registry["assets"][name]["path"]).relative_to("evaluator")
                    target = runtime / ".official_eval_cache" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            invocation = {
                **config,
                "runtime": str(runtime),
                "result_path": str(stage / "result.json"),
            }
            write_json(runtime / "request.json", invocation)
            env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
            env.update(
                PYTHONHASHSEED="0",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            with (stage / "worker.log").open("w") as log:
                # -I would ignore PYTHONHASHSEED. -s -P isolates imports while
                # retaining the explicitly pinned hash seed and the selected venv.
                process = subprocess.run(
                    [
                        str(python.absolute()),
                        "-s",
                        "-P",
                        str(code_dir / "generation_worker.py"),
                        str(runtime / "request.json"),
                    ],
                    cwd=runtime,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if process.returncode:
                raise EvidenceError(
                    f"fragment worker exited {process.returncode}; preserved log: {stage / 'worker.log'}"
                )
        result = json.loads((stage / "result.json").read_text())
        if result.get("schema") != "compose_fragment_generation_v1" or len(result["cells"]) != len(
            seeds
        ) * len(prompts):
            raise EvidenceError("worker returned an incomplete population")
        if any(len(cell["attempts"]) != attempts for cell in result["cells"]):
            raise EvidenceError("worker returned incomplete output slots")
        identities = [(cell["seed"], cell["prompt"]) for cell in result["cells"]]
        if set(identities) != {(seed, prompt) for seed in seeds for prompt in prompts}:
            raise EvidenceError("worker returned wrong or duplicate prompt/seed cells")
        verify_assets(registry, assets, task)
        if parity:
            comparison = check_parity(root, task, result, records)
            write_json(stage / "parity.json", comparison)
            if not comparison["passed"]:
                raise EvidenceError(
                    f"historical parity failed; details preserved: {stage / 'parity.json'}"
                )
        provenance.update(status="complete", result_sha256=sha256(stage / "result.json"))
        write_json(stage / "provenance.json", provenance)
        (stage / "README.md").write_text(
            "# Local fragment execution\n\n"
            + (
                "Bounded parity check against saved attempts. Not a new benchmark result.\n"
                if parity
                else "New local run of the frozen sampler. This does not replace the original paper results.\n"
            )
            + "\nExact configuration, source and asset hashes are in provenance.json. Every attempted slot is retained.\n"
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
