"""Descriptor-only audit of saved warm-continuation parent units, no generation."""

from __future__ import annotations

import argparse
import ast
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer

from compose_v4.chem.molecular_graph import is_element
from compose_v4.control.fused_option import FusedProgress, completed_fused_cycle
from compose_v4.control.macro_engine import slot_ring_systems
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_warm_continuation import endpoint, payload_hash
from compose_v4.gates.med_chem_gate import validity_reasons
from compose_v4.rewrite.trace_shard import decode_state
from tools.t4_warm_audit import candidate_summary, counts

ROOT = Path(__file__).resolve().parents[1]


def benchmark_limits() -> dict:
    """Read the existing app's literal threshold registry, without importing Modal."""
    tree = ast.parse((ROOT / "modal_apps/genmol_t4_opt_app.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [item.id for item in node.targets[0].elts]
            if "QED_MIN" in names and "SA_MAX" in names:
                values = dict(zip(names, ast.literal_eval(node.value), strict=True))
                return {"qed_min": values["QED_MIN"], "sa_max": values["SA_MAX"]}
    raise ValueError("T4 app threshold registry not found; do not substitute defaults")


def ring_descriptors(smiles: str) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"unparseable saved molecule: {smiles}")
    return {
        "rdkit_bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(molecule),
        "rdkit_spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(molecule),
        "sssr_ring_sizes": sorted(map(len, molecule.GetRingInfo().AtomRings())),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(molecule),
        "qed_structural_alert_count": int(QED.properties(molecule).ALERTS),
    }


def report(root: Path) -> dict:
    start = perf_counter()
    inventory_path = root / "remote_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    for name, identity in inventory["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"inventory path escapes run directory: {name}")
        verify_file(path, identity["sha256"])
    warm = unseal(root / "warm_start.json")
    paths = sorted((root / "round_2/parents").glob("*.json"))
    units = [unseal(path) for path in paths]
    if not units or [u["parent_index"] for u in units] != list(range(len(units))):
        raise ValueError("audit requires a nonempty contiguous completed parent prefix")
    task = units[0]["task"]
    if any(u["task"] != task for u in units) or task["warm_start_sha256"] != payload_hash(warm):
        raise ValueError("parent units disagree on task or exact warm archive")
    if (root / "round_2/docking_started.json").exists():
        raise ValueError("this pre-oracle audit must not conceal an attempted docking batch")
    ledger = json.loads((root / "round_2/executor_attempts_0.json").read_text())
    parents = {c["smiles"]: c for c in warm["archive"]}
    seed = warm["archive"][0]
    seed_counts = counts(seed["state"])
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed_fp = generator.GetFingerprint(Chem.MolFromSmiles(seed["smiles"]))
    limits = {**benchmark_limits(), "similarity_min": task["delta"]}
    records = []
    for unit in units:
        # This is a trace lookup only, not a newly manufactured candidate lock.
        trace_lookup = {"work": [unit["work"]]}
        for candidate in unit["candidates"]:
            row = candidate_summary(candidate, trace_lookup, parents, seed_counts)
            molecule = Chem.MolFromSmiles(candidate["smiles"])
            properties = {
                "qed": float(QED.qed(molecule)),
                "sa": float(sascorer.calculateScore(molecule)),
                "sim": float(
                    DataStructs.TanimotoSimilarity(seed_fp, generator.GetFingerprint(molecule))
                ),
            }
            failures = []
            for field, bound, fail in (
                ("qed", "qed_min", properties["qed"] < limits["qed_min"]),
                ("sa", "sa_max", properties["sa"] > limits["sa_max"]),
                ("sim", "similarity_min", properties["sim"] < limits["similarity_min"]),
            ):
                if fail:
                    failures.append(
                        {"property": field, "value": properties[field], "limit": limits[bound]}
                    )
            before = ring_descriptors(candidate["parent"])
            after = ring_descriptors(candidate["smiles"])
            deltas = {
                key: after[key] - before[key]
                for key in (
                    "rdkit_bridgehead_atoms",
                    "rdkit_spiro_atoms",
                    "aromatic_rings",
                )
            }
            origin = decode_state(parents[candidate["parent"]]["state"])
            product = decode_state(endpoint(trace_lookup, candidate))
            old_slots = frozenset(int(i) for i in np.flatnonzero(is_element(origin.atom_types)))
            new_systems = [sorted(s) for s in slot_ring_systems(product) if s.isdisjoint(old_slots)]
            fused_witness = None
            if candidate["option"] == "build_fused_ring":
                transitions = [
                    r
                    for r in unit["work"]["sampled_transitions"]
                    if r["bundle_id"] == candidate["bundle_id"] and r["step"] == candidate["step"]
                ]
                fused_witness = all(
                    completed_fused_cycle(
                        origin, product, FusedProgress.from_payload(r["fused_progress"])
                    )
                    for r in transitions
                ) and bool(transitions)
                if not fused_witness:
                    raise ValueError("saved fused program lacks its exact-slot aromatic C6 witness")
            records.append(
                {
                    **row,
                    "parent_index": unit["parent_index"],
                    "properties": properties,
                    "t4_constraints_pass_local": not failures,
                    "constraint_failures": failures,
                    "ring_descriptors": after,
                    "ring_descriptor_delta": deltas,
                    "new_only_ring_system_slots": new_systems,
                    "aromatic_c6_fused_program_verified": fused_witness,
                    "new_bridge_or_spiro_warning": any(
                        deltas[k] > 0 for k in ("rdkit_bridgehead_atoms", "rdkit_spiro_atoms")
                    ),
                    "legacy_gate_reasons_diagnostic_only": validity_reasons(candidate["smiles"]),
                }
            )
    unique = {r["smiles"]: r for r in records}
    ring_gain = [r for r in unique.values() if r["d_cycle_rank"] > 0]
    dependencies = [
        Path(__file__),
        ROOT / "tools/t4_warm_audit.py",
        ROOT / "modal_apps/genmol_t4_opt_app.py",
        ROOT / "src/compose_v4/experiments/t4_warm_continuation.py",
        ROOT / "src/compose_v4/control/fused_option.py",
        ROOT / "src/compose_v4/control/macro_engine.py",
        ROOT / "src/compose_v4/control/graph_geometry.py",
        ROOT / "src/compose_v4/gates/med_chem_gate.py",
        Path(QED.__file__),
        Path(sascorer.__file__),
        Path(sascorer.__file__).with_name("fpscores.pkl.gz"),
    ]
    return {
        "schema_version": "t4_saved_ring_quality_audit_v1",
        "evidence_type": "computed_local_descriptors_of_partial_preparation",
        "source_revision": task["code_revision"],
        "audit_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_inventory_sha256": sha256_file(inventory_path),
        "input_inventory": inventory,
        "configuration": task,
        "limits": limits,
        "dependency_sha256": {str(p): sha256_file(p) for p in dependencies},
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "execution": "local CPU, serial descriptors, float64",
        },
        "randomness": "no sampling or learned fitting in audit; source RNG is in hashed receipts",
        "split": "previously inspected development cell; no held-out inference",
        "status": "partial_preparation_not_an_oracle_pool",
        "summary": {
            "completed_parents": len(units),
            "expected_parents": task["lineages"],
            "completed_bundles": sum(len(u["bundles"]) for u in units),
            "candidate_records": len(records),
            "unique_canonical_candidates": len(unique),
            "t4_constraints_pass_local": sum(
                r["t4_constraints_pass_local"] for r in unique.values()
            ),
            "ring_gain_candidates": len(ring_gain),
            "ring_gain_t4_constraints_pass_local": sum(
                r["t4_constraints_pass_local"] for r in ring_gain
            ),
            "bridge_or_spiro_warning_candidates": sum(
                r["new_bridge_or_spiro_warning"] for r in unique.values()
            ),
            "bridge_or_spiro_warning_t4_constraints_pass_local": sum(
                r["new_bridge_or_spiro_warning"] and r["t4_constraints_pass_local"]
                for r in unique.values()
            ),
            "options": dict(
                sorted(Counter(b["option"] for u in units for b in u["bundles"]).items())
            ),
            "checkpointed_executor_calls": units[-1]["cumulative_executor_calls"],
            "total_attempted_executor_calls": ledger["prior_calls"] + len(ledger["attempts"]),
            "new_oracle_calls": 0,
            "cumulative_source_oracle_calls": warm["oracle_attempts"],
        },
        "candidates": records,
        "audit_seconds": perf_counter() - start,
        "limitations": [
            "Saved parent prefix only, not a completed round or selected/docked batch.",
            "Local RDKit differs from remote pinned runtime; descriptors must be recomputed there before locking.",
            "T4 constraints, SA, QED and topology warnings do not establish synthesis, stability, safety or affinity.",
            "SSSR ring sizes are basis-dependent descriptors, not graph cycle rank or macro proof.",
            "Legacy med-chem gate was partly calibrated on inspected IVG winners; diagnostic only, not independent validation.",
            "New-only ring systems use saved slot set difference; only constructive no-deletion programs support birth provenance.",
            "No topology ban, winner-derived reward, new threshold, molecular generation, or docking was introduced by this audit.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root)
    print(publish_json(args.output, result))
    print(json.dumps(result["summary"], sort_keys=True, indent=2))
