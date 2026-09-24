"""Build a split-clean rooted attachment catalog as one immutable artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from rdkit import Chem

from compose_v4.benchmark.fragment_constrained import load_genmol_prompts
from compose_v4.benchmark.training_attachment_fragments import (
    build_attachment_catalog,
    catalog_bytes,
    physical_sha256,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--expected-prompt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    destination = args.output_dir.resolve()
    if destination.exists():
        raise FileExistsError(f"catalog output already exists: {destination}")
    source = args.source.resolve(strict=True)
    prompt_manifest = args.prompt_manifest.resolve(strict=True)
    if physical_sha256(prompt_manifest) != args.expected_prompt_sha256:
        raise ValueError(f"prompt manifest SHA-256 mismatch: {prompt_manifest}")
    prompts = load_genmol_prompts(prompt_manifest)
    excluded_canonical = frozenset(
        Chem.MolToSmiles(Chem.MolFromSmiles(prompt.original_smiles), canonical=True)
        for prompt in prompts
    )
    if len(excluded_canonical) != 10:
        raise ValueError("expected exactly ten distinct benchmark reference molecules")
    root = Path(__file__).resolve().parents[1]
    code_path = root / "src/compose_v4/benchmark/training_attachment_fragments.py"
    code_sha256_at_start = physical_sha256(code_path)
    builder_sha256_at_start = physical_sha256(Path(__file__).resolve())
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        catalog = build_attachment_catalog(
            source,
            expected_sha256=args.expected_sha256,
            excluded_canonical=excluded_canonical,
        )
        payload = catalog_bytes(catalog)
        catalog_path = stage / "catalog.json"
        catalog_path.write_bytes(payload)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if physical_sha256(code_path) != code_sha256_at_start:
            raise RuntimeError("catalog implementation changed while extraction was running")
        if physical_sha256(Path(__file__).resolve()) != builder_sha256_at_start:
            raise RuntimeError("catalog builder changed while extraction was running")
        manifest = {
            "schema": "training_attachment_catalog_build_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_origin": "historical GuacaMol 500k training subset already used by COMPOSE",
            "source_access_basis": "existing local research asset; redistribution license not asserted",
            "source_path": str(source),
            "source_sha256": catalog.source_sha256,
            "prompt_manifest": str(prompt_manifest),
            "prompt_manifest_sha256": args.expected_prompt_sha256,
            "excluded_reference_identity_count": len(excluded_canonical),
            "excluded_reference_rows": list(catalog.excluded_reference_rows),
            "source_rows": catalog.source_rows,
            "unique_source_molecules": catalog.unique_source_molecules,
            "duplicate_source_rows": catalog.duplicate_source_rows,
            "catalog_entry_count": len(catalog.entries),
            "catalog_occurrence_count": sum(entry.occurrences for entry in catalog.entries),
            "catalog_sha256": hashlib.sha256(payload).hexdigest(),
            "implementation_sha256": code_sha256_at_start,
            "builder_sha256": builder_sha256_at_start,
            "git_commit": commit,
            "python_version": platform.python_version(),
            "rdkit_version": catalog.rdkit_version,
            "max_fragment_atoms": catalog.max_fragment_atoms,
            "excluded": catalog.excluded,
            "sample_seed": None,
            "split_identity": "GuacaMol frozen training subset only; no benchmark prompts or held-out rows",
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if physical_sha256(catalog_path) != manifest["catalog_sha256"]:
            raise RuntimeError("catalog publication hash changed during build")
        os.rename(stage, destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
