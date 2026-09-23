"""Build the set of repository files that are the subject of a sha256 pin.

This repository is content-addressed. Source files are hashed into launch
contracts (``runtime_inputs_sha256``, ``implementation_sha256`` and roughly
twenty other container names) and into the Process-V2 identity. Changing one
byte of such a file moves its hash and every contract pinning it then refuses to
load; MOVING or RENAMING one is worse, because the pin then addresses nothing and
an unresolvable pin reads as verified rather than failing loudly. The repository
has three recorded instances of exactly that.

The scan is deliberately STRUCTURAL rather than key-name driven. Pins appear
under at least twenty-two different container names, several of them generic
(``inputs``, ``files``, ``paths``, ``added``), so an allowlist of key names would
miss pins silently. Instead a pin is recognised by shape:

``key_is_path``
    A mapping entry whose KEY is a repo-relative path and whose VALUE is a
    64-hex digest. This is the dominant shape.

``sibling``
    A mapping that carries a 64-hex digest under some key and a repo-relative
    path as a string value under another. This over-collects: ``query_id`` and
    ``run_id`` are also 64-hex, so a mapping pairing a path with a query id is
    reported as a pin.

``python_constant``
    The same two shapes in a module-level literal in Python source, plus the
    explicitly declared Process-V2 implementation list.

``image_mount``
    A file named in a Modal ``add_local_file``/``add_local_dir`` call. These are
    not hashed, but they are mounted into a launch image by path, so moving one
    breaks the launch exactly as a moved pin does. ``README.md`` states the case:
    a checkpoint under ``docs/`` is load-bearing when an app mounts it, though
    nothing imports it.

OVER-COLLECTION IS THE SAFE DIRECTION AND IS INTENTIONAL. A false positive costs
one file left alone. A false negative costs a refused contract, or worse a
silently de-authorized run. Every entry carries the artifacts that pin it, so a
reader can judge any individual case rather than trusting the classifier.

Only paths that resolve to a file tracked in this repository are recorded; the
contracts also pin container-absolute paths such as ``/opt/dock/qvina02``, which
are not this repository's files to move.

Usage::

    python3 tools/repo_pinned_file_set.py --out diagnostics/repo_hygiene/pinned.json
    python3 tools/repo_pinned_file_set.py --check src/compose_v4/chem/state.py
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Where pins are written. ``recipes`` and ``docs`` are included because launch
# receipts and sealed capsules are stored there too.
SCAN_DIRECTORIES = ("configs", "diagnostics", "modal_apps", "tools", "scripts", "recipes", "docs")
PYTHON_SCAN_DIRECTORIES = ("src", "tools", "scripts", "modal_apps")

PATH_SUFFIXES = (".py", ".json", ".md", ".txt", ".yaml", ".yml", ".jsonl", ".pt", ".csv", ".smiles")

MOUNT_CALLS = ("add_local_file", "add_local_dir", "copy_local_file", "copy_local_dir")


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def tracked_files(root: pathlib.Path) -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=str(root), capture_output=True, text=True, check=True
    )
    return set(out.stdout.split("\n")) - {""}


def looks_like_path(value: str) -> bool:
    return value.endswith(PATH_SUFFIXES) and len(value) < 400


# ---- JSON scanning ----


def _iter_mappings(node: Any) -> Iterable[dict]:
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def scan_json(path: pathlib.Path, tracked: set[str], sink: dict[str, list[dict]]) -> None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return
    artifact = str(path)
    for mapping in _iter_mappings(document):
        digest_keys = [k for k, v in mapping.items() if isinstance(v, str) and HEX64.match(v)]
        for key, value in mapping.items():
            if (
                isinstance(value, str)
                and HEX64.match(value)
                and looks_like_path(str(key))
                and str(key) in tracked
            ):
                sink[str(key)].append(
                    {"artifact": artifact, "shape": "key_is_path", "container_key": str(key)}
                )
        if not digest_keys:
            continue
        for key, value in mapping.items():
            if isinstance(value, str) and looks_like_path(value) and value in tracked:
                sink[value].append(
                    {
                        "artifact": artifact,
                        "shape": "sibling",
                        "container_key": str(key),
                        "digest_keys": sorted(digest_keys)[:4],
                    }
                )


# ---- Python scanning ----


def scan_python(path: pathlib.Path, tracked: set[str], sink: dict[str, list[dict]]) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return
    artifact = str(path)

    for node in ast.walk(tree):
        # A literal mapping of path -> digest, wherever it appears.
        if isinstance(node, ast.Dict):
            for key_node, value_node in zip(node.keys, node.values):
                key = key_node.value if isinstance(key_node, ast.Constant) else None
                value = value_node.value if isinstance(value_node, ast.Constant) else None
                if (
                    isinstance(key, str)
                    and isinstance(value, str)
                    and HEX64.match(value)
                    and key in tracked
                ):
                    sink[key].append(
                        {"artifact": artifact, "shape": "python_constant", "container_key": key}
                    )
        # A module-level constant naming a set of implementation files.
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if not name.isupper() and not name.lstrip("_").isupper():
                    continue
                tokens = name.upper()
                if not any(
                    marker in tokens
                    for marker in ("PATH", "FILE", "SOURCE", "IMPLEMENTATION", "MODULE")
                ):
                    continue
                for literal in ast.walk(node.value):
                    if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                        candidate = literal.value
                        if candidate in tracked:
                            sink[candidate].append(
                                {
                                    "artifact": artifact,
                                    "shape": "python_constant",
                                    "container_key": name,
                                }
                            )
        # Modal image mounts: not hashed, but addressed by path at launch.
        if isinstance(node, ast.Call):
            func = node.func
            attribute = func.attr if isinstance(func, ast.Attribute) else None
            if attribute in MOUNT_CALLS:
                for literal in ast.walk(node):
                    if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                        candidate = literal.value.lstrip("./")
                        if candidate in tracked:
                            sink[candidate].append(
                                {
                                    "artifact": artifact,
                                    "shape": "image_mount",
                                    "container_key": attribute,
                                }
                            )


def process_v2_paths(root: pathlib.Path) -> list[str]:
    """The explicitly declared Process-V2 implementation boundary.

    Resolved by importing the module rather than by re-reading the literal,
    because the tuple is built from another constant and a re-derived copy could
    silently disagree with the value the identity is actually computed over.
    """
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from compose_v4.rewrite import editing_v2_process_identity as identity

    return sorted(identity._PROCESS_V2_IMPLEMENTATION_RELATIVE_PATHS)


def build(root: pathlib.Path) -> dict[str, Any]:
    tracked = tracked_files(root)
    sink: dict[str, list[dict]] = defaultdict(list)

    for directory in SCAN_DIRECTORIES:
        base = root / directory
        if base.is_dir():
            for path in sorted(base.rglob("*.json")):
                scan_json(path, tracked, sink)
    for directory in PYTHON_SCAN_DIRECTORIES:
        base = root / directory
        if base.is_dir():
            for path in sorted(base.rglob("*.py")):
                scan_python(path, tracked, sink)

    for relative in process_v2_paths(root):
        if relative in tracked:
            sink[relative].append(
                {
                    "artifact": "src/compose_v4/rewrite/editing_v2_process_identity.py",
                    "shape": "process_v2_identity",
                    "container_key": "_PROCESS_V2_IMPLEMENTATION_RELATIVE_PATHS",
                }
            )

    entries: dict[str, Any] = {}
    for relative in sorted(sink):
        records = sink[relative]
        artifacts = sorted({r["artifact"] for r in records})
        shapes = sorted({r["shape"] for r in records})
        entries[relative] = {
            "pin_count": len(records),
            "shapes": shapes,
            "pinning_artifact_count": len(artifacts),
            # Bounded so the report stays readable; the count above is exact.
            "pinning_artifacts": artifacts[:25],
            "pinning_artifacts_truncated": len(artifacts) > 25,
            "process_v2_identity": "process_v2_identity" in shapes,
            "hash_pinned": bool({"key_is_path", "sibling", "python_constant", "process_v2_identity"}
                               & set(shapes)),
            "image_mounted_only": shapes == ["image_mount"],
        }

    by_top = defaultdict(int)
    for relative in entries:
        by_top[relative.split("/")[0]] += 1

    return {
        "schema_version": 1,
        "baseline_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, check=True
        ).stdout.strip(),
        "policy": (
            "READ-ONLY. Do not edit, move, rename, split or reformat any path listed here. "
            "Collection is deliberately generous: a false positive costs one untouched file, "
            "a false negative costs a refused contract or a silently de-authorized run."
        ),
        "scan_directories": list(SCAN_DIRECTORIES),
        "python_scan_directories": list(PYTHON_SCAN_DIRECTORIES),
        "tracked_file_count": len(tracked),
        "pinned_file_count": len(entries),
        "hash_pinned_count": sum(1 for e in entries.values() if e["hash_pinned"]),
        "image_mounted_only_count": sum(1 for e in entries.values() if e["image_mounted_only"]),
        "process_v2_identity_count": sum(1 for e in entries.values() if e["process_v2_identity"]),
        "pinned_by_top_directory": dict(sorted(by_top.items(), key=lambda kv: -kv[1])),
        "files": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--check", help="report whether one path is pinned, and by what")
    args = parser.parse_args()

    root = repo_root()
    report = build(root)

    if args.check:
        entry = report["files"].get(args.check)
        if entry is None:
            print(f"NOT PINNED: {args.check}")
            print("  (still run tools/repo_audit.py --check before moving it)")
            return 0
        print(f"PINNED: {args.check}")
        print(json.dumps(entry, indent=2))
        return 1

    if not args.out:
        parser.error("--out is required unless --check is given")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"tracked={report['tracked_file_count']} pinned={report['pinned_file_count']} "
        f"hash_pinned={report['hash_pinned_count']} "
        f"mount_only={report['image_mounted_only_count']} -> {out}"
    )
    print(json.dumps(report["pinned_by_top_directory"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
