"""Audit free endpoint margins in a sealed retained-interface candidate lock."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_fiber_campaign import Fiber, sascorer
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid

SCHEMA_VERSION = "t4_nodistill_retained_interface_failure_margin_audit_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_envelope(path: Path) -> tuple[dict[str, Any], str]:
    encoded = (
        gzip.decompress(path.read_bytes())
        if path.suffix == ".gz"
        else path.read_bytes()
    )
    envelope = json.loads(encoded)
    payload = envelope.get("payload")
    payload_sha256 = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or identity(payload) != payload_sha256:
        raise ValueError(f"invalid self-hashed envelope: {path}")
    return payload, str(payload_sha256)


def _publish_once(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sealed audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_sha256 = identity(payload)
    path.write_text(
        json.dumps(
            {"payload": payload, "payload_sha256": payload_sha256},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    return payload_sha256


def run_audit(
    *, repository_root: Path, lock_path: Path, output_path: Path
) -> dict[str, Any]:
    lock, lock_identity = _read_envelope(lock_path)
    if (
        lock.get("schema_version")
        != "t4_nodistill_retained_interface_candidate_lock_v3"
    ):
        raise ValueError("failure-margin audit requires one sealed v3 candidate lock")
    source_state = lock["source_state"]
    # The lock deliberately stores only a source hash/encoded graph. The source SMILES
    # comes from the same hash-bound frozen registry used by its contract.
    contract, contract_identity = _read_envelope(
        repository_root / "configs/t4_nodistill_retained_interface_gate_v3.json"
    )
    cell = next(row for row in contract["cells"] if row["cell_key"] == lock["cell_key"])
    registry_path = repository_root / contract["inputs"]["source_registry"]["path"]
    if _sha256(registry_path) != contract["inputs"]["source_registry"]["sha256"]:
        raise ValueError("source registry changed before failure-margin audit")
    registry = json.loads(registry_path.read_text())
    source_smiles = str(registry[int(cell["source_index"])]["smiles"])
    if identity(source_smiles) != lock["source_smiles_sha256"]:
        raise ValueError("source SMILES no longer matches the sealed lock")
    delta = float(lock["delta"])
    fiber = Fiber(source_smiles, delta)
    rows = []
    failure_combinations: Counter[str] = Counter()
    pass_counts: Counter[str] = Counter()
    for candidate in lock["candidates"]:
        smiles = str(candidate["canonical_smiles"])
        molecule = Chem.MolFromSmiles(smiles)
        parseable = molecule is not None
        connected = parseable and "." not in smiles
        if not parseable:
            row = {
                "endpoint_key_sha256": candidate["endpoint_key_sha256"],
                "parseable": False,
            }
            rows.append(row)
            failure_combinations["parse"] += 1
            continue
        heavy = int(molecule.GetNumHeavyAtoms())
        similarity = float(
            DataStructs.TanimotoSimilarity(
                fiber.seed, fiber.generator.GetFingerprint(molecule)
            )
        )
        qed = float(QED.qed(molecule))
        sa = float(sascorer.calculateScore(molecule))
        valid = bool(structurally_valid(smiles))
        passes = {
            "connected": connected,
            "heavy_atom_capacity": heavy <= 40,
            "similarity": similarity >= delta,
            "qed": qed >= 0.6,
            "sa": sa <= 4.0,
            "structural_validity": valid,
        }
        for name, passed in passes.items():
            pass_counts[name] += int(passed)
        failures = sorted(name for name, passed in passes.items() if not passed)
        failure_combinations["+".join(failures) if failures else "none"] += 1
        margins = {
            "similarity": similarity - delta,
            "qed": qed - 0.6,
            "sa": 4.0 - sa,
            "heavy_atom_capacity": 40 - heavy,
        }
        row = {
            "endpoint_key_sha256": candidate["endpoint_key_sha256"],
            "macro_plan_identity": candidate["macro_plan_identity"],
            "macro_plan_name": candidate["macro_plan"]["name"],
            "delta_heavy_atoms": candidate["delta_heavy_atoms"],
            "delta_cycle_rank": candidate["delta_cycle_rank"],
            "retained_fraction": candidate["retained_fraction"],
            "created_attachment_interface_count": candidate[
                "created_attachment_interface_count"
            ],
            "parseable": True,
            "connected": connected,
            "structurally_valid": valid,
            "heavy_atoms": heavy,
            "similarity": similarity,
            "qed": qed,
            "sa": sa,
            "margins": margins,
            "failed_free_constraints": failures,
            "all_free_constraints_pass": not failures,
            "continuous_deficit": sum(max(0.0, -value) for value in margins.values()),
        }
        rows.append(row)

    ranked = sorted(
        (row for row in rows if row.get("parseable")),
        key=lambda row: (
            row["continuous_deficit"],
            -row["margins"]["similarity"],
            row["endpoint_key_sha256"],
        ),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence": "computed zero-oracle free endpoint margin audit",
        "cell_key": lock["cell_key"],
        "thresholds": {
            "similarity_minimum": delta,
            "qed_minimum": 0.6,
            "sa_maximum": 4.0,
            "heavy_atom_maximum": 40,
            "structural_validity_required": True,
        },
        "input": {
            "candidate_lock": {
                "path": str(lock_path.relative_to(repository_root)),
                "sha256": _sha256(lock_path),
                "payload_sha256": lock_identity,
            },
            "contract": {
                "path": "configs/t4_nodistill_retained_interface_gate_v3.json",
                "sha256": _sha256(
                    repository_root
                    / "configs/t4_nodistill_retained_interface_gate_v3.json"
                ),
                "payload_sha256": contract_identity,
            },
            "source_registry": {
                "path": str(registry_path.relative_to(repository_root)),
                "sha256": _sha256(registry_path),
            },
            "source_state_sha256": identity(source_state),
        },
        "counts": {
            "candidate_count": len(rows),
            "pass_by_constraint": dict(sorted(pass_counts.items())),
            "failure_combinations": dict(sorted(failure_combinations.items())),
            "all_free_constraints_pass": sum(
                bool(row.get("all_free_constraints_pass")) for row in rows
            ),
        },
        "nearest_candidates": ranked[:10],
        "all_candidate_margins": rows,
        "interpretation": (
            "The failure stage is free endpoint feasibility after exact generation. "
            "Counts identify which continuous, task-independent feasibility headrooms "
            "a later generic allocator may use; no teacher or objective enters this audit."
        ),
        "implementation": {
            "code_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
            ).strip(),
            "script_sha256": _sha256(Path(__file__)),
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
    }
    payload_sha256 = _publish_once(output_path, payload)
    return {
        "payload_sha256": payload_sha256,
        "counts": payload["counts"],
        "nearest_candidate": ranked[0] if ranked else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(
            "diagnostics/t4_nodistill_retained_interface_gate_v3/attempt_1/locks/"
            "jak2_0_d06.json.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "diagnostics/t4_nodistill_retained_interface_gate_v3/attempt_1/"
            "jak2_failure_margin_audit.json"
        ),
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_audit(
        repository_root=root,
        lock_path=(root / arguments.lock).resolve(),
        output_path=(root / arguments.output).resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
