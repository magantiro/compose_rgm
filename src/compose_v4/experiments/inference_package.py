"""Authenticated inference-only serialization of an already selected CPU model.

This module does not select/train weights or reopen training data. A caller must
first authenticate the original runtime. Explicit source changes are probe-only
inputs until an independently pinned parity receipt admits them for search.
"""

from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.editing_p50_gate import state_dict_semantic_sha256
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.model.factorized_tracelet_rate_model import FactorizedTraceletRateModel

CACHE_SOURCES = (
    "src/compose_v4/chem/molecular_graph.py",
    "src/compose_v4/experiments/production_successor_kernel.py",
)


def software() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "rdkit": rdBase.rdkitVersion,
    }


def dependency_sources(sources: dict[str, str]) -> dict[str, str]:
    """Retain molecular/model dependencies, not task policy or launcher files."""
    directories = tuple(f"src/compose_v4/{d}/" for d in ("chem", "model", "rewrite", "data"))
    explicit = {
        "src/compose_v4/experiments/production_successor_kernel.py",
        "src/compose_v4/experiments/factorized_mark_conditional.py",
        "src/compose_v4/experiments/tracelet_conditional.py",
        "src/compose_v4/experiments/successor_kernel.py",
    }
    return {p: h for p, h in sorted(sources.items()) if p.startswith(directories) or p in explicit}


def write_package(directory: Path, model, *, provenance: dict) -> dict:
    if not isinstance(model, FactorizedTraceletRateModel) or model.training:
        raise ValueError("export requires the selected eval-mode production model")
    if any(t.device.type != "cpu" for t in model.state_dict().values()):
        raise ValueError("inference export requires CPU tensors")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise ValueError("complete inference package already exists; validate and reuse it")
    temporary = directory / "model.partial.pt"
    torch.save(model, temporary)
    model_path = directory / "model.pt"
    temporary.replace(model_path)
    manifest = {
        "schema_version": "compose_inference_package_v1",
        "software": software(),
        "tensor_sha256": state_dict_semantic_sha256(model.state_dict()),
        "model_sha256": sha256_file(model_path),
        "provenance": provenance,
    }
    seal(directory / "manifest.json", manifest)
    return manifest


def load_package(
    directory: Path,
    *,
    manifest_sha256: str,
    repo_root: Path,
    probe_source_changes: dict[str, str] | None = None,
    qualified_source_receipt: tuple[Path, str] | None = None,
):
    """Verify before unpickling a trusted, locally produced inference artifact.

    ``probe_source_changes`` does not authorize search. It permits only the two
    serialization-cache files for a source-bound, zero-oracle equivalence probe.
    Ordinary inference either uses identical dependencies, or supplies a pinned
    passing paired receipt. Cross-worker numerical admission belongs to its
    downstream run contract and is not implied by this source-compatibility check.
    """
    verify_file(directory / "manifest.json", manifest_sha256)
    manifest = unseal(directory / "manifest.json")
    if (
        manifest["schema_version"] != "compose_inference_package_v1"
        or manifest["software"] != software()
    ):
        raise ValueError("inference package schema/software mismatch")
    sources = manifest["provenance"]["dependency_sources"]
    changes = probe_source_changes or {}
    if qualified_source_receipt is not None:
        if probe_source_changes is not None:
            raise ValueError("cannot mix probe overrides with qualified source receipt")
        path, digest = qualified_source_receipt
        verify_file(path, digest)
        receipt = unseal(path)
        if (
            receipt.get("status") != "pass"
            or receipt.get("oracle_calls") != 0
            or receipt["export"]["manifest_sha256"] != manifest_sha256
            or receipt["package_tensor_sha256"] != manifest["tensor_sha256"]
            or len(receipt["rows"]) != 3
            or not all(r["parity"] for r in receipt["rows"])
        ):
            raise ValueError("source qualification does not bind passing model/cache evidence")
        changes = receipt["cache_sources"]
    if not sources or not set(changes).issubset(CACHE_SOURCES):
        raise ValueError("unauthorized inference dependency override")
    for path, digest in sources.items():
        verify_file(repo_root / path, changes.get(path, digest))
    if not set(changes).issubset(sources):
        raise ValueError("override is absent from the authenticated source inventory")
    verify_file(directory / "model.pt", manifest["model_sha256"])
    model = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    if not isinstance(model, FactorizedTraceletRateModel) or model.training:
        raise ValueError("inference package contains the wrong model or training mode")
    if state_dict_semantic_sha256(model.state_dict()) != manifest["tensor_sha256"]:
        raise ValueError("inference package selected tensor state changed")
    torch.set_num_threads(1)
    model.requires_grad_(False)
    return model, manifest
