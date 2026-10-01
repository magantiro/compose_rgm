"""Check or fetch the pinned fragment evaluator and benchmark prompts.

The six upstream files come from an immutable InVirtuoGen commit. The prompt
table adds one terminal newline to its pinned reference file. This command
checks identities but never imports or executes downloaded code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/fragments/assets.json"
DEFAULT_DESTINATION = ROOT / "local_assets/fragments"
SOURCE_REVISION = "b50bb3ae2bdcb9df581f0b219d79cf14b05d0fbb"
SOURCE_IDENTITY = f"InVirtuoGen_results@{SOURCE_REVISION}:"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/invirtuolabs/InVirtuoGen_results/{SOURCE_REVISION}/"
)
USER_AGENT = "COMPOSE-research-assets/1.0"
EVALUATOR_KEYS = frozenset(
    {
        "evaluator_metrics",
        "evaluator_mol",
        "evaluator_downstream",
        "evaluator_frags",
        "evaluator_reference_metrics",
        "evaluator_fragments",
    }
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_path(entry: dict) -> str:
    origin = entry["origin"]
    relative = entry["path"]
    if not origin.startswith(SOURCE_IDENTITY):
        raise ValueError(f"unexpected evaluator origin: {origin}")
    upstream = origin.removeprefix(SOURCE_IDENTITY)
    source = Path(upstream)
    if (
        not upstream
        or source.is_absolute()
        or ".." in source.parts
        or relative != f"evaluator/pkg/{upstream}"
    ):
        raise ValueError(f"evaluator path does not match origin: {relative}")
    return upstream


def ensure_asset(destination: Path, entry: dict, *, download: bool) -> dict:
    upstream = source_path(entry)
    target = destination / entry["path"]
    expected = entry["sha256"]
    if target.is_file():
        actual = file_sha256(target)
        if actual != expected:
            raise ValueError(f"fragment evaluator identity mismatch: {target}: {actual}")
        return {"path": str(target), "sha256": actual, "status": "ready"}
    if target.exists():
        raise ValueError(f"fragment evaluator path is not a file: {target}")
    if not download:
        return {"path": str(target), "sha256": expected, "status": "missing"}

    target.parent.mkdir(parents=True, exist_ok=True)
    url = SOURCE_URL + upstream
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(request, timeout=60) as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual = file_sha256(temporary)
        if actual != expected:
            raise ValueError(f"fragment evaluator download identity mismatch: {url}: {actual}")
        os.chmod(temporary, 0o644)
        os.link(temporary, target)
        return {"path": str(target), "sha256": actual, "status": "downloaded"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_prompts(destination: Path, prompts: dict, source_entry: dict, *, download: bool) -> dict:
    if prompts.get("source_sha256") != source_entry["sha256"]:
        raise ValueError("fragment prompt source identity does not match evaluator manifest")
    target = destination / prompts["path"]
    expected = prompts["sha256"]
    if target.is_file():
        actual = file_sha256(target)
        if actual != expected:
            raise ValueError(f"fragment prompt identity mismatch: {target}: {actual}")
        return {"path": str(target), "sha256": actual, "status": "ready"}
    if target.exists():
        raise ValueError(f"fragment prompt path is not a file: {target}")
    if not download:
        return {"path": str(target), "sha256": expected, "status": "missing"}

    source = destination / source_entry["path"]
    if not source.is_file() or file_sha256(source) != source_entry["sha256"]:
        raise ValueError(f"verified upstream fragment table is missing: {source}")
    content = source.read_bytes()
    if content.endswith(b"\n"):
        raise ValueError(f"upstream fragment table unexpectedly has a terminal newline: {source}")
    content += b"\n"
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError(f"normalized fragment prompt identity mismatch: {actual} != {expected}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.link(temporary, target)
        return {"path": str(target), "sha256": actual, "status": "derived"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_assets(destination: Path, *, download: bool) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema") != "compose_fragment_assets_v1":
        raise ValueError(f"{MANIFEST}: unexpected fragment asset schema")
    assets = manifest.get("assets")
    if (
        not isinstance(assets, dict)
        or {k for k in assets if k.startswith("evaluator_")} != EVALUATOR_KEYS
    ):
        raise ValueError(f"{MANIFEST}: expected exactly six evaluator assets")
    rows = {
        key: ensure_asset(destination, assets[key], download=download)
        for key in sorted(EVALUATOR_KEYS)
    }
    rows["prompts"] = ensure_prompts(
        destination, assets["prompts"], assets["evaluator_fragments"], download=download
    )
    return {
        "schema_version": "compose.fragment.upstream_assets.v1",
        "source_revision": SOURCE_REVISION,
        "license": manifest["evaluator_license"],
        "assets": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--download", action="store_true", help="fetch missing upstream files")
    args = parser.parse_args()
    result = fetch_assets(args.destination, download=args.download)
    print(json.dumps(result, sort_keys=True))
    return 1 if any(row["status"] == "missing" for row in result["assets"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
