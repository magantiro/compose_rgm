"""Check or fetch the exact external PyTDC PMO oracle assets.

The Dataverse file IDs come from PyTDC 1.1.15 ``metadata.oracle2id``.
Downloaded files are published only after SHA-256 verification. This tool does
not import PyTDC, deserialize a pickle or call an oracle.
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
MANIFEST = ROOT / "experiments/pmo/assets.json"
DEFAULT_DESTINATION = ROOT / "local_assets/pmo/oracle-assets"
DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/"
USER_AGENT = "COMPOSE-research-assets/1.0"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def asset_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"invalid oracle asset path: {relative}")
    return root / path


def ensure_asset(root: Path, entry: dict, *, download: bool) -> dict:
    relative = entry["relative_path"]
    target = asset_path(root, relative)
    expected = entry["sha256"]
    if target.exists():
        actual = file_sha256(target)
        if actual != expected:
            raise ValueError(f"oracle asset identity mismatch: {target}: {actual}")
        return {"path": str(target), "sha256": actual, "status": "ready"}
    if not download:
        return {"path": str(target), "sha256": expected, "status": "missing"}

    target.parent.mkdir(parents=True, exist_ok=True)
    url = DATAVERSE_URL + str(entry["dataverse_file_id"])
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
            raise ValueError(f"oracle download identity mismatch: {url}: {actual} != {expected}")
        os.link(temporary, target)
        return {"path": str(target), "sha256": actual, "status": "downloaded"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--download", action="store_true", help="fetch missing assets from Dataverse"
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    rows = {
        task: ensure_asset(args.destination, entry, download=args.download)
        for task, entry in sorted(manifest["oracle_assets"].items())
    }
    print(
        json.dumps(
            {"schema_version": "compose.pmo.oracle_assets.v1", "assets": rows}, sort_keys=True
        )
    )
    return 1 if any(row["status"] == "missing" for row in rows.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
