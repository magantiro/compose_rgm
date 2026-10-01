"""Check or fetch the hash-pinned MOOD docking files used by T4.

MOOD's QuickVina2 executable targets Linux x86-64. Open Babel is installed
separately. This command never executes either program or performs docking.
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
MANIFEST = ROOT / "experiments/t4/targets.json"
DEFAULT_DESTINATION = ROOT / "local_assets/t4"
SOURCE_REVISION = "ac9730818325a72f68d748802721762ab6134552"
SOURCE_URL = f"https://raw.githubusercontent.com/SeulLee05/MOOD/{SOURCE_REVISION}/scorer/"
USER_AGENT = "COMPOSE-research-assets/1.0"
TARGETS = frozenset({"5ht1b", "braf", "fa7", "jak2", "parp1"})


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ensure_asset(
    destination: Path,
    *,
    relative_path: str,
    source_path: str,
    expected_sha256: str,
    download: bool,
    executable: bool = False,
) -> dict:
    target = destination / relative_path
    if target.is_file():
        actual = file_sha256(target)
        if actual != expected_sha256:
            raise ValueError(f"T4 docking asset identity mismatch: {target}: {actual}")
        if executable and not os.access(target, os.X_OK):
            raise ValueError(f"T4 docking executable lacks execute permission: {target}")
        return {"path": str(target), "sha256": actual, "status": "ready"}
    if target.exists():
        raise ValueError(f"T4 docking asset path is not a file: {target}")
    if not download:
        return {"path": str(target), "sha256": expected_sha256, "status": "missing"}

    target.parent.mkdir(parents=True, exist_ok=True)
    url = SOURCE_URL + source_path
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
        if actual != expected_sha256:
            raise ValueError(f"T4 docking asset download identity mismatch: {url}: {actual}")
        os.chmod(temporary, 0o755 if executable else 0o644)
        os.link(temporary, target)
        return {"path": str(target), "sha256": actual, "status": "downloaded"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_assets(destination: Path, *, download: bool) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema_version") != "compose.t4.targets.v1":
        raise ValueError(f"{MANIFEST}: unexpected T4 target schema")
    targets = manifest.get("targets")
    if not isinstance(targets, dict) or set(targets) != TARGETS:
        raise ValueError(f"{MANIFEST}: expected exactly five T4 targets")
    assets = {
        "qvina02": ensure_asset(
            destination,
            relative_path="bin/qvina02",
            source_path="qvina02",
            expected_sha256=manifest["qvina_sha256"],
            download=download,
            executable=True,
        )
    }
    for name, target in sorted(targets.items()):
        assets[f"receptor_{name}"] = ensure_asset(
            destination,
            relative_path=f"receptors/{name}.pdbqt",
            source_path=f"receptors/{name}.pdbqt",
            expected_sha256=target["receptor_sha256"],
            download=download,
        )
    return {
        "schema_version": "compose.t4.docking_assets.v1",
        "source_revision": SOURCE_REVISION,
        "qvina_platform": "linux_x86_64",
        "assets": assets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--download", action="store_true", help="fetch missing MOOD assets")
    args = parser.parse_args()
    result = fetch_assets(args.destination, download=args.download)
    print(json.dumps(result, sort_keys=True))
    return 1 if any(row["status"] == "missing" for row in result["assets"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
