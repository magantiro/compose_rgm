"""Read-only quality-availability analysis of locked linker development panels."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "fragment-linker-novelty-v1"
PILOT = SOURCE / "diagnostics/fragment_linker_novelty_pilot_v1"
OUTPUT = ROOT / "diagnostics/fragment_linker_quality_headroom_v1/result.json"
SUMMARY_SHA256 = "33458214e6073a4f799780313ec874a495a04190662de3f3a1116ed88a60eb1a"
EVALUATOR_SHA256 = "3c4bb7c6727cbeaf02d3d5eebf1deac27f77bab68d929b61dbe4154911e2b099"
PROPERTY_MODULE_SHA256 = "ed5609c72a3effe2547f4b264e35be8134affccc206978605307d1365a028145"


def physical_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quality_flags(sa: float, qed: float) -> tuple[bool, bool, bool]:
    """Return QED failure, SA failure, and joint quality admission."""
    qed_failed = not np.isfinite(qed) or qed < 0.6
    sa_failed = not np.isfinite(sa) or sa > 4.0
    return qed_failed, sa_failed, not (qed_failed or sa_failed)


def immutable_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"quality headroom result already exists: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".result-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"quality headroom result already exists: {OUTPUT}")
    summary_path = PILOT / "summary.json"
    if physical_sha256(summary_path) != SUMMARY_SHA256:
        raise ValueError("locked novelty-4 development summary changed")
    summary = json.loads(summary_path.read_text())
    manifest_path = PILOT / "manifest.json"
    if physical_sha256(manifest_path) != summary["manifest_sha256"]:
        raise ValueError("locked novelty-4 development manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if summary["seed"] != 5 or summary["attempts_per_arm"] != 1000:
        raise ValueError("unexpected linker development panel")
    drugs = tuple(manifest["contract"]["drugs"])
    if len(drugs) != 10 or len(set(drugs)) != 10:
        raise ValueError("linker prompt population changed")
    package = SOURCE / ".official_eval_cache/pkg"
    evaluator = package / "in_virtuo_gen/train_utils/metrics.py"
    property_module = package / "in_virtuo_gen/utils/mol.py"
    if physical_sha256(evaluator) != EVALUATOR_SHA256:
        raise ValueError("pinned official evaluator changed")
    if physical_sha256(property_module) != PROPERTY_MODULE_SHA256:
        raise ValueError("pinned official property computation changed")
    sys.path.insert(0, str(package))
    from in_virtuo_gen.utils.mol import compute_single_property

    sascorer = sys.modules["sascorer"]

    property_cache: dict[str, tuple[float, float]] = {}

    def properties(smiles: str) -> tuple[float, float]:
        if smiles not in property_cache:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or Chem.MolToSmiles(mol, canonical=True) != smiles:
                raise ValueError(f"noncanonical or invalid supported endpoint: {smiles}")
            values = compute_single_property(smiles)
            if values is None or len(values) != 2:
                raise ValueError(f"upstream property calculation failed: {smiles}")
            property_cache[smiles] = (float(values[0]), float(values[1]))
        return property_cache[smiles]

    per_prompt = {}
    input_hashes = {}
    for drug in drugs:
        counts = {
            "attempts": 0,
            "selected_quality_attempts": 0,
            "selected_quality_unique": 0,
            "panels_with_quality_offer": 0,
            "panels_with_unseen_quality_offer": 0,
            "selected_failure_with_quality_offer": 0,
            "selected_failure_with_unseen_quality_offer": 0,
            "selected_duplicate_quality_with_unseen_quality_offer": 0,
            "selected_qed_failure_only": 0,
            "selected_sa_failure_only": 0,
            "selected_joint_failure": 0,
            "model_supported_offers": 0,
            "quality_qualified_offers": 0,
        }
        emitted: set[str] = set()
        quality_selected: set[str] = set()
        quality_offered: set[str] = set()
        missed_unseen_indices = []
        for index in range(100):
            rel = f"attempts/novelty4/{drug}_{index:03d}.json"
            path = PILOT / rel
            digest = physical_sha256(path)
            if summary["artifact_hashes"].get(rel) != digest:
                raise ValueError(f"locked linker attempt changed: {path}")
            data = json.loads(path.read_text())
            panel = data["panel"]
            if (
                data["arm"] != "novelty4"
                or data["drug"] != drug
                or data["attempt_index"] != index
                or panel["offered_count"] != 8
                or len(panel["offered"]) != 8
                or panel["output_count"] != 1
                or not data["selected_valid_connected"]
                or not data["selected_exact_core_path_fidelity"]
            ):
                raise ValueError(f"malformed or unfaithful linker attempt: {path}")
            supported = [
                row for row in panel["offered"] if row["status"] == "model_supported"
            ]
            if not supported or len({row["endpoint"] for row in supported}) != len(supported):
                raise ValueError(f"unsupported or duplicate model panel: {path}")
            for row in supported:
                provenance = row["provenance"]
                if not (
                    provenance["fidelity"]["satisfied"]
                    and provenance["exact_mapped_core_identity_checked"]
                    and provenance["source_core_locked_all_states"]
                ):
                    raise ValueError(f"model-supported offer lacks prompt fidelity: {path}")
            selected = panel["selected_smiles"]
            if selected not in {row["endpoint"] for row in supported}:
                raise ValueError(f"selected endpoint not in model-supported panel: {path}")
            qualified = {
                row["endpoint"]
                for row in supported
                if quality_flags(*properties(row["endpoint"]))[2]
            }
            selected_qed_fail, selected_sa_fail, selected_quality = quality_flags(
                *properties(selected)
            )
            unseen_quality = qualified - emitted
            counts["attempts"] += 1
            counts["model_supported_offers"] += len(supported)
            counts["quality_qualified_offers"] += len(qualified)
            counts["selected_quality_attempts"] += int(selected_quality)
            counts["panels_with_quality_offer"] += int(bool(qualified))
            counts["panels_with_unseen_quality_offer"] += int(bool(unseen_quality))
            counts["selected_failure_with_quality_offer"] += int(
                not selected_quality and bool(qualified)
            )
            missed_unseen = not selected_quality and bool(unseen_quality)
            counts["selected_failure_with_unseen_quality_offer"] += int(missed_unseen)
            if missed_unseen:
                missed_unseen_indices.append(index)
            counts["selected_duplicate_quality_with_unseen_quality_offer"] += int(
                selected_quality and selected in emitted and bool(unseen_quality)
            )
            counts["selected_qed_failure_only"] += int(selected_qed_fail and not selected_sa_fail)
            counts["selected_sa_failure_only"] += int(selected_sa_fail and not selected_qed_fail)
            counts["selected_joint_failure"] += int(selected_qed_fail and selected_sa_fail)
            quality_offered.update(qualified)
            if selected_quality:
                quality_selected.add(selected)
            emitted.add(selected)
            input_hashes[rel] = digest
        counts["selected_quality_unique"] = len(quality_selected)
        row_path = PILOT / "rows/novelty4" / f"{drug}.json"
        row_rel = str(row_path.relative_to(PILOT))
        if physical_sha256(row_path) != summary["artifact_hashes"][row_rel]:
            raise ValueError(f"locked official metric row changed: {row_path}")
        recorded_quality = float(json.loads(row_path.read_text())["metrics"]["quality"])
        if abs(recorded_quality - len(quality_selected)) > 1e-9:
            raise ValueError(f"upstream property predicate does not reproduce quality: {drug}")
        per_prompt[drug] = {
            "counts": counts,
            "distinct_quality_endpoints_offered": len(quality_offered),
            "distinct_quality_endpoints_selected": len(quality_selected),
            "missed_unseen_quality_attempt_indices": missed_unseen_indices,
            "recorded_official_quality_percent": recorded_quality,
        }
        print(json.dumps({"drug": drug, "counts": counts}), flush=True)

    totals = {
        key: sum(row["counts"][key] for row in per_prompt.values())
        for key in next(iter(per_prompt.values()))["counts"]
    }
    result = {
        "schema": "fragment_linker_quality_headroom_v1",
        "role": "post-hoc fixed-panel quality diagnostic; not a fresh benchmark",
        "decision_doc": "docs/FRAGMENT_LINKER_QUALITY_HEADROOM_2026-09-25.md",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "script_sha256": physical_sha256(Path(__file__)),
        "summary_sha256": SUMMARY_SHA256,
        "manifest_sha256": physical_sha256(manifest_path),
        "evaluator_sha256": EVALUATOR_SHA256,
        "property_module_sha256": PROPERTY_MODULE_SHA256,
        "sascorer_sha256": physical_sha256(Path(sascorer.__file__)),
        "input_attempt_sha256": input_hashes,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "per_prompt": per_prompt,
        "totals": totals,
        "distinct_property_endpoints_computed": len(property_cache),
    }
    immutable_json(OUTPUT, result)
    print(json.dumps({"result": str(OUTPUT), "totals": totals}), flush=True)


if __name__ == "__main__":
    main()
