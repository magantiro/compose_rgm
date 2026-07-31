#!/usr/bin/env python3
"""Build a structural-only editing-V2 split-component census from normalized JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from compose_v4.data import editing_corpus_contract as editing_corpus_contract_module
from compose_v4.data import editing_v2_split_census as split_census_module
from compose_v4.data.editing_corpus_contract import load_editing_corpus_contract
from compose_v4.data.editing_v2_split_census import (
    EditingV2SplitCensusError,
    IMPLEMENTATION_PROVENANCE_SCHEMA,
    IMPLEMENTATION_PROVENANCE_SCHEMA_VERSION,
    build_split_component_census,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_jsonl(path: Path) -> Iterator[object]:
    """Yield normalized source rows without materializing the JSONL input."""

    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise EditingV2SplitCensusError(
                    f"{path} line {line_number} is not valid JSON: {exc}"
                ) from exc


def _count_nonempty_lines(path: Path) -> int:
    with path.open() as handle:
        return sum(bool(line.strip()) for line in handle)


def _file_identity(path: Path, *, repository_root: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(repository_root))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": _file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _git_output(repository_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise EditingV2SplitCensusError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _implementation_provenance(
    *,
    repository_root: Path,
    policy_path: Path,
    editing_corpus_contract_path: Path,
) -> dict[str, object]:
    source_paths = (
        Path(__file__),
        Path(split_census_module.__file__),
        Path(editing_corpus_contract_module.__file__),
    )
    return {
        "schema": IMPLEMENTATION_PROVENANCE_SCHEMA,
        "schema_version": IMPLEMENTATION_PROVENANCE_SCHEMA_VERSION,
        "code_revision": {
            "commit_sha": _git_output(repository_root, "rev-parse", "HEAD"),
            "dirty": bool(
                _git_output(
                    repository_root,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=normal",
                )
            ),
        },
        "python_runtime": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "source_files": [
            _file_identity(path, repository_root=repository_root) for path in source_paths
        ],
        "policy_file": _file_identity(
            policy_path,
            repository_root=repository_root,
        ),
        "editing_corpus_contract_file": _file_identity(
            editing_corpus_contract_path,
            repository_root=repository_root,
        ),
    }


def _publish(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise EditingV2SplitCensusError(
                f"refusing to overwrite a different census artifact: {path}"
            )
        return
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--policy-json",
        type=Path,
        required=True,
        help="explicit hash-bound split-census policy JSON",
    )
    parser.add_argument(
        "--editing-corpus-contract",
        type=Path,
        required=True,
        help="validated editing-corpus contract defining the ordered lane contract",
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    nonempty_jsonl_rows = _count_nonempty_lines(args.input_jsonl)
    policy = json.loads(args.policy_json.read_text())
    editing_corpus_contract = load_editing_corpus_contract(args.editing_corpus_contract)
    census = build_split_component_census(
        _iter_jsonl(args.input_jsonl),
        policy=policy,
        editing_corpus_contract=editing_corpus_contract,
        implementation_provenance=_implementation_provenance(
            repository_root=repository_root,
            policy_path=args.policy_json,
            editing_corpus_contract_path=args.editing_corpus_contract,
        ),
        source_stream={
            "path": str(args.input_jsonl),
            "sha256": _file_sha256(args.input_jsonl),
            "bytes": args.input_jsonl.stat().st_size,
            "nonempty_jsonl_rows": nonempty_jsonl_rows,
        },
    )
    content = (json.dumps(census, indent=2, sort_keys=True) + "\n").encode()
    _publish(args.output_json, content)
    print(
        json.dumps(
            {
                "status": census["status"],
                "status_scope": census["status_scope"],
                "census_sha256": census["census_sha256"],
                "valid_vertices": census["input_summary"]["valid_vertices"],
                "invalid_rows": census["input_summary"]["invalid_rows"],
                "hard_components": census["hard_component_census"]["component_count"],
                "split_authority": census["authority"]["split"],
                "training_authority": census["authority"]["training"],
                "output": str(args.output_json),
            },
            sort_keys=True,
        )
    )
    return 0 if census["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
