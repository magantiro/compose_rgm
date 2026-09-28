"""Preserve a historical fragment source dependency closure without editing it.

Maintainer tool, not a benchmark launcher. Reads the explicitly selected source
worktree and emits a deterministic ZIP plus a manifest in a NEW directory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOTS = {
    "motif_extension": [
        "compose_v4.benchmark.fragment_motif_focused_programs",
        "compose_v4.benchmark.joint_completion_prior",
    ],
    "scaffold_decoration": [
        "compose_v4.benchmark.fragment_pendant_programs",
        "compose_v4.benchmark.joint_mass_pendant_policy",
    ],
    "linker_design": ["compose_v4.benchmark.fragment_linker_sampler"],
    "superstructure_generation": ["compose_v4.benchmark.fragment_conditioned_sampler"],
    "superstructure_uniform": [
        "compose_v4.benchmark.fragment_conditioned_sampler",
        "compose_v4.benchmark.uniform_native_mark_law",
    ],
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def closure(source: Path, roots: list[str]) -> dict[str, bytes]:
    """Resolve local imports, including function-local imports and package inits."""
    locations = [source / "src", source / "scripts", source / "tools", source]
    pending, seen, files = list(roots), set(), {}
    while pending:
        module = pending.pop()
        if not module or module in seen:
            continue
        seen.add(module)
        path = None
        for base in locations:
            for candidate in (
                base / (module.replace(".", "/") + ".py"),
                base / module.replace(".", "/") / "__init__.py",
            ):
                if candidate.is_file():
                    path = candidate
                    break
            if path is not None:
                break
        if path is None:
            namespace = any((base / module.replace(".", "/")).is_dir() for base in locations)
            if module.startswith("compose_v4") and not namespace:
                raise ValueError(f"unresolved project module: {module}")
            continue
        data = path.read_bytes()
        files[path.relative_to(source).as_posix()] = data
        parts = module.split(".")
        pending.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        package = parts if path.name == "__init__.py" else parts[:-1]
        for node in ast.walk(ast.parse(data, filename=str(path))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[: len(package) - node.level + 1] if node.level else []
                name = ".".join(prefix + ([node.module] if node.module else []))
                if name:
                    pending.append(name)
                # from package import submodule, as opposed to a symbol.
                for alias in node.names:
                    sub = f"{name}.{alias.name}" if name else alias.name
                    if any(
                        (base / (sub.replace(".", "/") + ".py")).is_file()
                        or (base / sub.replace(".", "/") / "__init__.py").is_file()
                        for base in locations
                    ):
                        pending.append(sub)
    return files


def write_archive(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task", choices=ROOTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--replace-generated",
        action="store_true",
        help="replace this tool's verified generated archive, not source files",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    for flags in ([], ["--cached"]):
        subprocess.run(["git", "-C", str(source), "diff", *flags, "--exit-code"], check=True)
    files = closure(
        source,
        ROOTS[args.task]
        + [
            "evaluate_tracelet_rollouts",
            "fetch_official_fragment_evaluator",
            "run_fragment_constrained_suite",
        ],
    )
    # Capture only tracked source, never an uncommitted implementation.
    for name, data in files.items():
        original = subprocess.check_output(["git", "-C", str(source), "show", f"HEAD:{name}"])
        if original != data:
            raise ValueError(f"source changed during capture: {name}")
    if args.output.exists():
        if not args.replace_generated or {p.name for p in args.output.iterdir()} != {
            "source.zip",
            "manifest.json",
        }:
            raise FileExistsError(args.output)
        old = json.loads((args.output / "manifest.json").read_text())
        if (
            old.get("task") != args.task
            or digest((args.output / "source.zip").read_bytes()) != old["archive_sha256"]
        ):
            raise ValueError("refusing to replace an unverified generated archive")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix=".capture-") as temporary:
        archive = Path(temporary) / "source.zip"
        write_archive(archive, files)
        manifest = {
            "schema": "compose_fragment_runtime_source_v1",
            "task": args.task,
            "source_revision": revision,
            "archive_sha256": digest(archive.read_bytes()),
            "files": {name: digest(data) for name, data in sorted(files.items())},
            "scope": "historical Python dependency closure, unchanged bytes; not launch authorization",
        }
        saved = Path(temporary) / "manifest.json"
        saved.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        size = archive.stat().st_size
        os.replace(archive, args.output / "source.zip")
        os.replace(saved, args.output / "manifest.json")
    print(
        json.dumps(
            {
                "task": args.task,
                "files": len(files),
                "bytes": size,
                "sha256": manifest["archive_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
