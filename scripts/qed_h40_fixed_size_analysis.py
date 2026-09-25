"""Validate and compare the frozen H40 QED full and fixed-size source records.

Missing source records remain missing. No paired estimate is emitted until both
arms have eight valid terminal returns for all 800 declared sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, rdFingerprintGenerator

N_SOURCES = 800
N_RETURNS = 8
QED_MIN = 0.90
SIM_MIN = 0.40
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_REPS = 20_000
FIXED_FAMILIES = {
    "atom_restate",
    "bond_reorder",
    "bond_reroute",
    "cycle_insert",
    "cycle_attach",
    "ring_system_restate",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_counts(full: list[bool], fixed: list[bool]) -> dict[str, int]:
    if len(full) != len(fixed):
        raise ValueError("paired source lists have different lengths")
    return {
        "both": sum(a and b for a, b in zip(full, fixed, strict=True)),
        "full_only": sum(a and not b for a, b in zip(full, fixed, strict=True)),
        "fixed_only": sum(not a and b for a, b in zip(full, fixed, strict=True)),
        "neither": sum(not a and not b for a, b in zip(full, fixed, strict=True)),
    }


def _record_path(primary: Path, recovery: Path | None, name: str) -> Path | None:
    first = primary / name
    second = recovery / name if recovery is not None else None
    found = [path for path in (first, second) if path is not None and path.is_file()]
    if len(found) == 2 and _sha256(found[0]) != _sha256(found[1]):
        raise ValueError(f"conflicting primary and recovered records: {name}")
    return found[0] if found else None


def _validate_record(
    path: Path, *, index: int, source: str, fixed: bool,
    fingerprint_generator: Any,
) -> tuple[bool, dict[str, Any]]:
    record = json.loads(path.read_text())
    if record.get("index") != index or record.get("source") != source:
        raise ValueError(f"index/source mismatch in {path}")
    if record.get("horizon") != 40 or record.get("budget_max") != 24:
        raise ValueError(f"H40/budget mismatch in {path}")
    if record.get("head_dir") != "hphi_v2":
        raise ValueError(f"head mismatch in {path}")
    if fixed:
        if record.get("size_fixed") is not True:
            raise ValueError(f"missing fixed-size flag in {path}")
        if set(record.get("allowed_families", ())) != FIXED_FAMILIES:
            raise ValueError(f"restricted family set mismatch in {path}")
        if record.get("empty_restricted_fiber_rule") != "kill_particle_no_retry":
            raise ValueError(f"empty-support rule mismatch in {path}")
    elif record.get("size_fixed", False):
        raise ValueError(f"full-arm record is fixed-size in {path}")

    arm = record.get("arms", {}).get("restart")
    if not isinstance(arm, dict):
        raise TypeError(f"missing restart arm in {path}")
    candidates = arm.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != N_RETURNS:
        raise ValueError(f"expected eight terminal returns in {path}")
    if [candidate.get("k") for candidate in candidates] != list(range(N_RETURNS)):
        raise ValueError(f"noncontiguous candidate indices in {path}")

    source_mol = Chem.MolFromSmiles(source)
    if source_mol is None:
        raise ValueError(f"unparseable declared source {index}: {source}")
    source_fp = fingerprint_generator.GetFingerprint(source_mol)
    slots: list[dict[str, Any]] = []
    for candidate in candidates:
        returned = candidate.get("returned")
        mol = Chem.MolFromSmiles(returned) if isinstance(returned, str) else None
        if mol is None:
            raise ValueError(f"unparseable returned molecule in {path}, k={candidate['k']}")
        if fixed and mol.GetNumHeavyAtoms() != source_mol.GetNumHeavyAtoms():
            raise ValueError(f"fixed-size endpoint changed atom count in {path}")
        qed = float(QED.qed(mol))
        similarity = float(DataStructs.TanimotoSimilarity(
            source_fp, fingerprint_generator.GetFingerprint(mol)
        ))
        if abs(qed - float(candidate["terminal_qed"])) > 1e-10:
            raise ValueError(f"QED mismatch in {path}, k={candidate['k']}")
        if abs(similarity - float(candidate["terminal_sim"])) > 1e-10:
            raise ValueError(f"source similarity mismatch in {path}, k={candidate['k']}")
        success = qed >= QED_MIN and similarity >= SIM_MIN
        if success != candidate.get("success"):
            raise ValueError(f"candidate success mismatch in {path}, k={candidate['k']}")
        if candidate.get("extinct") and returned != source:
            raise ValueError(f"extinct slot did not return source in {path}")
        slots.append({
            "k": candidate["k"],
            "returned": returned,
            "qed": qed,
            "similarity": similarity,
            "success": success,
            "extinct": candidate["extinct"],
        })
    solved = any(slot["success"] for slot in slots)
    if solved != arm.get("success"):
        raise ValueError(f"source success mismatch in {path}")
    return solved, {"path": str(path), "sha256": _sha256(path), "slots": slots}


def analyze(
    *, source_path: Path, full_dir: Path, fixed_dir: Path,
    full_recovery_dir: Path | None = None,
) -> dict[str, Any]:
    sources = [line.strip() for line in source_path.read_text().splitlines() if line.strip()]
    if len(sources) != N_SOURCES or len(set(sources)) != N_SOURCES:
        raise ValueError("QED source panel must contain 800 distinct nonempty sources")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rows: list[dict[str, Any]] = []
    missing: dict[str, list[int]] = {"full": [], "fixed": []}
    for index, source in enumerate(sources):
        row: dict[str, Any] = {"index": index, "source": source}
        for arm, primary, recovery, suffix in (
            ("full", full_dir, full_recovery_dir, ""),
            ("fixed", fixed_dir, None, "_fixed-size"),
        ):
            name = f"{index:03d}_H40_hphi_v2{suffix}_k0-8.json"
            path = _record_path(primary, recovery, name)
            if path is None:
                missing[arm].append(index)
                continue
            solved, evidence = _validate_record(
                path, index=index, source=source, fixed=arm == "fixed",
                fingerprint_generator=generator,
            )
            row[arm] = {"success": solved, **evidence}
        rows.append(row)

    result: dict[str, Any] = {
        "schema": "compose.qed_h40_fixed_size_paired_analysis",
        "schema_version": 1,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1], text=True,
            ).strip(),
            "analysis_script_sha256": _sha256(Path(__file__)),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "rdkit_version": rdBase.rdkitVersion,
            "platform": platform.platform(),
            "precision": "float64",
        },
        "source_panel": {"path": str(source_path), "sha256": _sha256(source_path)},
        "region": {"qed_min": QED_MIN, "morgan_tanimoto_min": SIM_MIN},
        "returns_per_source": N_RETURNS,
        "source_count": N_SOURCES,
        "missing": missing,
        "rows": rows,
        "status": "INCOMPLETE" if any(missing.values()) else "COMPLETE",
    }
    if result["status"] == "INCOMPLETE":
        return result
    full = [bool(row["full"]["success"]) for row in rows]
    fixed = [bool(row["fixed"]["success"]) for row in rows]
    paired = paired_counts(full, fixed)
    delta = np.asarray(full, dtype=np.int8) - np.asarray(fixed, dtype=np.int8)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for j in range(BOOTSTRAP_REPS):
        samples[j] = float(delta[rng.integers(0, N_SOURCES, N_SOURCES)].mean())
    result["summary"] = {
        "full_success": sum(full),
        "fixed_success": sum(fixed),
        "full_rate": sum(full) / N_SOURCES,
        "fixed_rate": sum(fixed) / N_SOURCES,
        "full_minus_fixed_rate": float(delta.mean()),
        "paired_categories": paired,
        "source_bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPS,
            "interval": list(np.quantile(samples, [0.025, 0.975])),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--full-recovery-dir", type=Path)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        source_path=args.sources, full_dir=args.full_dir,
        full_recovery_dir=args.full_recovery_dir, fixed_dir=args.fixed_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "missing_counts": {arm: len(indices) for arm, indices in result["missing"].items()},
        "missing_full": result["missing"]["full"],
    }, sort_keys=True))
    if "summary" in result:
        print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
