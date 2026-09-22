"""Why does the planned arm lose QED?  Partial out size and ring count.

Arm C1 reproduces the corpus ring-size law and still scores lower mean QED than
the unplanned arm A.  Three explanations are separable from the endpoints alone
and only one of them would make the loss an artifact:

  size        C1 molecules are smaller, and QED falls with heavy atoms
  ring count  C1 installs fewer ring SYSTEMS than A, and rings carry QED
  the plan    something about pinning the skeleton costs QED on its own

The repo has already been burned by reading a raw contrast as a mechanism when
size carried it, so this fits ``QED ~ arm + heavy + ring_systems`` by ordinary
least squares and reports the ARM coefficient with its standard error.  A
negative control is included: shuffling the arm label must drive that
coefficient to zero, because an instrument that cannot return "it was size"
cannot be trusted when it returns "it was the plan".

Usage::

    python3 scripts/denovo_ring_marginal_qed_attribution.py <shard-dir> [--arms A C1]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, RDConfig

from compose_v4.eval.denovo_ring_marginal import (
    SMALL_RING_MAXIMUM,
    ring_system_signature,
)

RDLogger.DisableLog("rdApp.*")

sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer


def _rows(shard_dir: Path, arms: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    for arm in arms:
        for shard in sorted((shard_dir / arm).glob("*.json")):
            payload = json.loads(shard.read_text())
            for record in payload["records"]:
                smiles = record.get("smiles")
                if not smiles:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                signature = ring_system_signature(mol)
                sizes = [size for system in signature for size in system]
                rows.append(
                    {
                        "arm": arm,
                        "smiles": smiles,
                        "qed": float(QED.qed(mol)),
                        "sa": float(sascorer.calculateScore(mol)),
                        "heavy": float(mol.GetNumHeavyAtoms()),
                        "systems": float(len(signature)),
                        "rings": float(len(sizes)),
                        "strained": float(
                            any(size <= SMALL_RING_MAXIMUM for size in sizes)
                        ),
                    }
                )
    return rows


def _fit(
    rows: list[dict],
    treated: str,
    columns: tuple[str, ...],
    response_name: str = "qed",
) -> dict:
    """OLS of ``response_name`` on an arm indicator plus ``columns``."""
    indicator = np.array([1.0 if row["arm"] == treated else 0.0 for row in rows])
    covariates = [np.array([row[column] for row in rows]) for column in columns]
    design = np.column_stack([np.ones(len(rows)), indicator, *covariates])
    response = np.array([row[response_name] for row in rows])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    dof = len(rows) - design.shape[1]
    sigma2 = float(residual @ residual) / dof
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    return {
        "response": response_name,
        "terms": ("intercept", f"arm=={treated}", *columns),
        "coefficients": [float(value) for value in coefficients],
        "stderr": [float(math.sqrt(covariance[i, i])) for i in range(len(coefficients))],
        "n": len(rows),
        "dof": dof,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir")
    parser.add_argument("--arms", nargs=2, default=["A", "C1"])
    parser.add_argument("--seed", type=int, default=20260922)
    arguments = parser.parse_args()

    control, treated = arguments.arms
    rows = _rows(Path(arguments.shard_dir), (control, treated))
    report: dict = {"arms": {control: 0, treated: 0}}
    for row in rows:
        report["arms"][row["arm"]] += 1

    for arm in (control, treated):
        subset = [row for row in rows if row["arm"] == arm]
        report[arm] = {
            "n": len(subset),
            "mean_qed": float(np.mean([row["qed"] for row in subset])),
            "mean_sa": float(np.mean([row["sa"] for row in subset])),
            "mean_heavy": float(np.mean([row["heavy"] for row in subset])),
            "mean_systems": float(np.mean([row["systems"] for row in subset])),
            "mean_rings": float(np.mean([row["rings"] for row in subset])),
            "fraction_strained": float(np.mean([row["strained"] for row in subset])),
        }

    for response in ("qed", "sa"):
        report[f"{response}_raw"] = _fit(rows, treated, (), response)
        report[f"{response}_size_adjusted"] = _fit(rows, treated, ("heavy",), response)
        report[f"{response}_size_and_ring_adjusted"] = _fit(
            rows, treated, ("heavy", "systems"), response
        )
        report[f"{response}_full"] = _fit(
            rows, treated, ("heavy", "systems", "rings", "strained"), response
        )

    # NEGATIVE CONTROL: with the arm label shuffled the arm term must collapse.
    rng = np.random.default_rng(arguments.seed)
    labels = [row["arm"] for row in rows]
    shuffled = list(labels)
    rng.shuffle(shuffled)
    permuted = [dict(row, arm=label) for row, label in zip(rows, shuffled)]
    for response in ("qed", "sa"):
        report[f"negative_control_shuffled_arm_{response}"] = _fit(
            permuted, treated, ("heavy", "systems"), response
        )

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
