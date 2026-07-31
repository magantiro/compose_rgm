#!/usr/bin/env python3
"""Benchmark-lead acceptance under the three corpus scopes.

The primary application edits real lead-optimization benchmark molecules. This reports how many of them are
even REPRESENTABLE under (1) the production broad-organic charge-preserving scope, (2) a broad-organic
neutral-only ablation, and (3) the old CNOF-neutral filter -- and, for every excluded lead, the exact
unsupported feature. Silently dropping S/P/Cl/Br/I- or charge-bearing leads would bias the benchmark, so
this coverage census is a prerequisite for the headline experiments.

Run: PYTHONPATH=src python scripts/benchmark_lead_scope_coverage.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import rdBase

from compose_v4.chem.molecular_graph import (
    IDX_TO_ELEMENT,
    MolecularGraphError,
    is_element,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import is_connected_or_null
from compose_v4.data.organic_corpus import (
    BROAD_ORGANIC_NEUTRAL_V1,
    BROAD_ORGANIC_V1,
    classify_smiles,
)

REPO = Path(__file__).resolve().parent.parent
_JIN = "configs/benchmarks/jin_iclr19_qed_test_exact_v1.csv"
_SMILES_COL = "canonical_nonisomeric_smiles"  # our representation drops stereo/isotopes
_OUTPUT = "diagnostics/composition/benchmark_lead_scope_coverage.json"
_MATERIAL_IMPLEMENTATION_PATHS = (
    "scripts/benchmark_lead_scope_coverage.py",
    "src/compose_v4/data/organic_corpus.py",
    "src/compose_v4/chem/molecular_graph.py",
    "src/compose_v4/chem/state.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _git_revision_and_material_hashes() -> tuple[str, list[dict[str, str]]]:
    """Bind the result to a clean revision of every material implementation file."""

    revision = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("repository revision is not a full lowercase Git commit")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *_MATERIAL_IMPLEMENTATION_PATHS,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if dirty:
        raise RuntimeError(
            f"refusing to publish scope coverage from dirty material implementation files: {dirty}"
        )
    identities = [
        {
            "path": relative_path,
            "sha256": _sha256(REPO / relative_path),
        }
        for relative_path in _MATERIAL_IMPLEMENTATION_PATHS
    ]
    return revision, identities


def _write_json_atomically(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
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
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _cnof_neutral_ok(smi: str, *, max_atoms: int = 48) -> bool:
    """The retired CNOF-neutral filter (C/N/O/F elements, neutral, connected, size-bounded)."""
    if "." in smi:
        return False
    try:
        graph = smiles_to_molecular_graph(smi)
    except (MolecularGraphError, ValueError):
        return False
    if graph.n_real_atoms == 0 or graph.n_real_atoms > max_atoms or not is_connected_or_null(graph):
        return False
    real = is_element(graph.atom_types)
    if np.any(graph.formal_charges[real] != 0):
        return False
    return all(IDX_TO_ELEMENT[int(t)] in {"C", "N", "O", "F"} for t in graph.atom_types[real])


def coverage(csv_path: Path) -> dict:
    rows = list(csv.DictReader(csv_path.open()))
    n = len(rows)
    accepted = {"broad_organic_v1": 0, "broad_organic_neutral_v1": 0, "cnof_neutral": 0}
    excluded_broad: Counter = Counter()
    excluded_neutral: Counter = Counter()
    charged = 0
    for row in rows:
        smi = row[_SMILES_COL]
        ok_broad, reason_broad = classify_smiles(smi, BROAD_ORGANIC_V1)
        ok_neutral, reason_neutral = classify_smiles(smi, BROAD_ORGANIC_NEUTRAL_V1)
        ok_cnof = _cnof_neutral_ok(smi)
        accepted["broad_organic_v1"] += int(ok_broad)
        accepted["broad_organic_neutral_v1"] += int(ok_neutral)
        accepted["cnof_neutral"] += int(ok_cnof)
        if not ok_broad:
            excluded_broad[reason_broad] += 1
        if not ok_neutral:
            excluded_neutral[reason_neutral] += 1
        if int(row.get("formal_charge", "0")) != 0:
            charged += 1
    return {
        "benchmark": "jin_iclr19_qed_test_exact_v1",
        "smiles_column": _SMILES_COL,
        "n_leads": n,
        "accepted": accepted,
        "accepted_fraction": {k: round(v / n, 4) for k, v in accepted.items()},
        "broad_recovers_over_cnof": accepted["broad_organic_v1"] - accepted["cnof_neutral"],
        "excluded_features_broad_scope": dict(excluded_broad),
        "excluded_features_neutral_scope": dict(excluded_neutral),
        "leads_with_nonzero_net_charge": charged,
        "scope_hashes": {
            "broad_organic_v1": BROAD_ORGANIC_V1.scope_hash(),
            "broad_organic_neutral_v1": BROAD_ORGANIC_NEUTRAL_V1.scope_hash(),
        },
    }


def build_artifact(
    csv_path: Path,
    *,
    code_revision: str,
    implementation_files: list[dict[str, str]],
) -> dict:
    """Return the computed census with a complete, deterministic evidence ledger."""

    result = coverage(csv_path)
    return {
        "schema": "compose.benchmark_lead_scope_coverage",
        "schema_version": 2,
        "evidence_class": "computed",
        "provenance": {
            "command": "PYTHONPATH=src python scripts/benchmark_lead_scope_coverage.py",
            "code_revision": code_revision,
            "implementation_files": implementation_files,
            "inputs": [
                {
                    "path": csv_path.relative_to(REPO).as_posix(),
                    "sha256": _sha256(csv_path),
                    "role": "fixed_benchmark_support_census",
                }
            ],
            "configuration": {
                "smiles_column": _SMILES_COL,
                "broad_organic_scope": BROAD_ORGANIC_V1.descriptor(),
                "broad_organic_neutral_scope": BROAD_ORGANIC_NEUTRAL_V1.descriptor(),
                "cnof_neutral_scope": {
                    "allowed_elements": ["C", "N", "O", "F"],
                    "allow_charges": False,
                    "connectedness": "connected_or_null_with_nonempty_required",
                    "max_atoms": 48,
                },
            },
            "determinism": {
                "seed": None,
                "seed_derivation": "not_applicable_exact_rowwise_census",
                "hardware_relevance": "none_discrete_deterministic_classification",
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
            },
            "sample_count": result["n_leads"],
            "exclusions": {
                "broad_organic_v1": result["excluded_features_broad_scope"],
                "broad_organic_neutral_v1": result["excluded_features_neutral_scope"],
                "cnof_neutral_rejected_count": (
                    result["n_leads"] - result["accepted"]["cnof_neutral"]
                ),
            },
            "split_identity": {
                "benchmark": result["benchmark"],
                "role": "fixed_external_support_census_not_model_selection",
            },
        },
        "result": result,
    }


def main() -> int:
    benchmark_path = REPO / _JIN
    revision, implementation_files = _git_revision_and_material_hashes()
    artifact = build_artifact(
        benchmark_path,
        code_revision=revision,
        implementation_files=implementation_files,
    )
    result = artifact["result"]
    n = result["n_leads"]
    print(f"{result['benchmark']}: {n} leads")
    for scope, count in result["accepted"].items():
        print(f"  {scope:28s}: {count}/{n} = {count / n:.1%}")
    print(f"  broad recovers over CNOF-neutral : {result['broad_recovers_over_cnof']} leads")
    print(f"  charged leads (kept by broad only): {result['leads_with_nonzero_net_charge']}/{n}")
    print(
        f"  excluded under broad             : {result['excluded_features_broad_scope'] or 'none'}"
    )
    out = REPO / _OUTPUT
    _write_json_atomically(out, artifact)
    print(f"-> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
