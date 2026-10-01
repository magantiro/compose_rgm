"""Reduce a complete QED test panel after verifying all candidate receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from compose_v4.experiments.qed_shared_reduction import reduce_qed_sources
from compose_v4.experiments.qed_shared_smc import QEDSMCConfig
from compose_v4.experiments.qed_shared_sources import load_qed_source_roles

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=8)
    args = parser.parse_args()

    roles = load_qed_source_roles(ROOT, ROOT / "experiments/qed/shared_sources.json")
    fragment_manifest = ROOT / "experiments/fragments/assets.json"
    checkpoint_sha256 = json.loads(fragment_manifest.read_text())["assets"]["checkpoint"]["sha256"]
    config = QEDSMCConfig(
        horizon=args.horizon,
        particles=args.particles,
        candidates=args.candidates,
    )
    reduced = reduce_qed_sources(
        args.results.resolve(),
        roles.test,
        source_split_sha256=roles.manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
        config=config,
    )
    reduced.update(
        {
            "reference_manifest_sha256": sha256(fragment_manifest),
            "code_revision": revision(),
            "reducer_sha256": sha256(Path(__file__)),
            "reduction_sha256": sha256(ROOT / "src/compose_v4/experiments/qed_shared_reduction.py"),
        }
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite QED reduction: {output}")
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(reduced, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite QED reduction: {output}")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"wrote {output} ({sha256(output)}), "
        f"success={reduced['successful_sources']}/{reduced['sources']}"
    )


if __name__ == "__main__":
    main()
