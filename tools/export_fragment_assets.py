"""Export path-neutral fragment assets without changing their sampling data.

The input identities come from the fragment asset manifest. This command never
updates those inputs. It changes only local filesystem paths in provenance
fields, then records the original and exported hashes separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/fragments/assets.json"
ASSET_KEYS = ("region_catalog", "pendant_catalog", "joint_prior", "mass_prior")
SCHEMAS = {
    "region_catalog": "split_first_training_region_catalog_v1",
    "pendant_catalog": "split_first_training_pendant_catalog_v1",
    "joint_prior": "joint_completion_structural_prior_v1",
    "mass_prior": "split_first_training_decoration_mass_prior_v1",
}
ANCHORS = ("guacamol", "data", "diagnostics", "src", "tools")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def portable_label(raw: str) -> str:
    """Keep the source suffix and accept an already portable label."""
    path = PurePosixPath(raw)
    if not path.is_absolute():
        if len(path.parts) >= 2 and path.parts[0] in ANCHORS and ".." not in path.parts:
            return path.as_posix()
        raise ValueError(f"unrecognized portable source label: {raw}")
    parts = path.parts
    first_anchor = min((parts.index(anchor) for anchor in ANCHORS if anchor in parts), default=None)
    if first_anchor is not None:
        suffix = parts[first_anchor:]
        if len(suffix) >= 2:
            return PurePosixPath(*suffix).as_posix()
    raise ValueError(f"no portable source label for: {raw}")


def normalize(asset: dict, key: str) -> dict:
    if key not in SCHEMAS or asset.get("schema") != SCHEMAS[key]:
        raise ValueError(f"unexpected {key} schema")
    result = dict(asset)
    if "source" in result:
        result["source"] = portable_label(result["source"])
    if "input_sha256" in result:
        inputs = result["input_sha256"]
        if not isinstance(inputs, dict):
            raise ValueError(f"{key}: input_sha256 is not a mapping")
        renamed = {portable_label(path): digest for path, digest in inputs.items()}
        if len(renamed) != len(inputs):
            raise ValueError(f"{key}: source labels collide")
        result["input_sha256"] = renamed
    return result


def scientific_payload(asset: dict) -> dict:
    """Exclude only the provenance paths rewritten by this export."""
    return {key: value for key, value in asset.items() if key not in {"source", "input_sha256"}}


def stable_bytes(asset: dict) -> bytes:
    return (json.dumps(asset, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def publish(target: Path, encoded: bytes) -> None:
    expected = hashlib.sha256(encoded).hexdigest()
    if target.is_file():
        if sha256(target) != expected:
            raise FileExistsError(f"refusing to replace different output: {target}")
        return
    if target.exists():
        raise FileExistsError(f"output path is not a file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_one(source: Path, target: Path, *, key: str, expected: str) -> dict:
    original_hash = sha256(source)
    if original_hash != expected:
        raise ValueError(f"{key}: source identity mismatch: {original_hash} != {expected}")
    original = json.loads(source.read_text())
    clean = normalize(original, key)
    if scientific_payload(clean) != scientific_payload(original):
        raise AssertionError(f"{key}: scientific fields changed")
    encoded = stable_bytes(clean)
    if b"/Users/" in encoded or b"/home/" in encoded:
        raise ValueError(f"{key}: private path remains in export")
    exported_hash = hashlib.sha256(encoded).hexdigest()
    publish(target, encoded)
    return {
        "asset": key,
        "original_sha256": original_hash,
        "export_sha256": exported_hash,
        "path": target.name,
        "scientific_payload_equal": True,
    }


def export_all(source_dir: Path, output_dir: Path) -> dict:
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("fragment asset export must not overwrite its source")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema") != "compose_fragment_assets_v1":
        raise ValueError(f"{MANIFEST}: unexpected schema")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=".fragment-export-"))
    try:
        results = []
        for key in ASSET_KEYS:
            entry = manifest["assets"][key]
            results.append(
                export_one(
                    source_dir / entry["path"],
                    staging / entry["path"],
                    key=key,
                    expected=entry["sha256"],
                )
            )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        receipt = {
            "schema_version": "compose.fragment.path_neutral_export.v1",
            "source_manifest_sha256": sha256(MANIFEST),
            "implementation_sha256": sha256(Path(__file__)),
            "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "assets": results,
        }
        publish(staging / "export_manifest.json", stable_bytes(receipt))
        if output_dir.exists():
            if not output_dir.is_dir():
                raise FileExistsError(f"export path is not a directory: {output_dir}")
            expected_names = {path.name for path in staging.iterdir()}
            actual_names = {path.name for path in output_dir.iterdir()}
            if actual_names != expected_names or any(
                not (output_dir / name).is_file()
                or sha256(output_dir / name) != sha256(staging / name)
                for name in expected_names
            ):
                raise FileExistsError(f"refusing to replace different export: {output_dir}")
        else:
            staging.rename(output_dir)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_all(args.source_dir, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
