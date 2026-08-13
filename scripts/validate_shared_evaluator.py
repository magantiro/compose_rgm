"""Validate the shared evaluator: same panel, two RDKits, disagreements recorded.

Pushes a fixed molecule panel through the caller's local RDKit and through the
production-pinned shared evaluator (2024.3.5), and asserts canonical-key
agreement. Disagreements are **recorded, not smoothed** — a canonical-key
difference between a baseline's RDKit and the pinned one is precisely what the
shared evaluator exists to make visible.

    python scripts/validate_shared_evaluator.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import rdkit
from rdkit import Chem, RDLogger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from compose_v4.experiments.shared_evaluator import (  # noqa: E402
    DEFAULT_PINNED_PYTHON,
    SharedEvaluator,
    compare_canonicalization,
)

RDLogger.DisableLog("rdApp.*")

COHORT = REPO / "diagnostics" / "retarget_calibration_cohort.json"
OUTPUT = REPO / "diagnostics" / "baselines" / "shared_evaluator_validation.json"

#: Adversarial cases on top of the held-in panel. Aromaticity perception,
#: tautomer-ish spellings, charges, stereo and explicit hydrogens are where
#: RDKit releases historically move, and CLAUDE.md records a real 50-state
#: parity failure caused by an aromaticity-perception change.
EDGE_CASES = [
    "c1ccccc1", "C1=CC=CC=C1",
    "c1ccc2ccccc2c1", "C1=CC2=CC=CC=C2C=C1",
    "c1cc[nH]c1", "C1=CNC=C1",
    "O=c1cccc[nH]1", "Oc1ccccn1",
    "CC(=O)[O-]", "CC(=O)O",
    "C[N+](C)(C)C",
    "N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O",
    "[H]OC", "OC",
    "c1ccc(-c2ccccc2)cc1",
    "C1=CC2=C(C=C1)N=CN2", "c1ccc2[nH]cnc2c1",
    "[O-][N+](=O)c1ccccc1",
    "S(=O)(=O)(O)c1ccccc1",
    "C1CC1", "C1CCCCC1",
    "not_a_molecule", "C1CC", "",
]


def local_canonical(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(molecule)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned-python", type=Path, default=DEFAULT_PINNED_PYTHON)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    cohort = json.loads(COHORT.read_text())
    panel = [row["source"] for row in cohort["sources"]] + EDGE_CASES

    with SharedEvaluator(objective="developability", python=args.pinned_python) as ev:
        comparison = compare_canonicalization(panel, ev, local_canonical)
        # A score must also survive the boundary, not merely a key.
        probe = ev.evaluate(cohort["sources"][0]["source"])

    checks = {
        "evaluator_is_pinned_to_production_rdkit": (
            comparison["pinned_rdkit"] == "2024.03.5"
        ),
        "no_canonical_key_disagreements": comparison["n_disagreements"] == 0,
        "no_validity_disagreements": (
            not comparison["valid_in_local_only"]
            and not comparison["valid_in_pinned_only"]
        ),
        "evaluator_returns_a_score": probe.get("score") is not None,
    }

    report = {
        "schema": "compose.baselines.shared_evaluator_validation",
        "title": "SHARED EVALUATOR VALIDATION — pinned RDKit 2024.3.5",
        "artifact_status": "SMOKE_HELD_IN",
        "architecture": (
            "baseline-native environment -> raw SMILES -> shared evaluator pinned "
            "to production RDKit 2024.3.5 -> canonical key + validity + oracle "
            "score. Baselines are NOT forced into one environment; that breaks the "
            "old implementations. Canonicalization and scoring happen exactly once, "
            "under the pin, so unique_valid_canonical_evaluations is comparable "
            "across methods."
        ),
        "held_out_data_opened": False,
        "panel": {
            "held_in_sources": len(cohort["sources"]),
            "cohort_sha256": cohort["cohort_sha256"],
            "edge_cases": len(EDGE_CASES),
            "total": len(panel),
        },
        "local_rdkit": rdkit.__version__,
        "local_python": platform.python_version(),
        "comparison": comparison,
        "score_probe": probe,
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    print(f"local rdkit {rdkit.__version__} vs pinned {comparison['pinned_rdkit']}")
    print(f"panel {comparison['n']}, agreements {comparison['agreements']}, "
          f"disagreements {comparison['n_disagreements']}")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"verdict: {report['verdict']}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
