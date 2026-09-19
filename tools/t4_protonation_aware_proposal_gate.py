#!/usr/bin/env python3
"""Run the deterministic zero-oracle protonation-aware proposal gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import QED, DataStructs, RDConfig, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.protonation_aware_proposal import (
    ProtonationAwareProposalConfig,
    propose_protonation_aware_candidates,
)
from compose_v4.control.route_distilled_goal_expert import (
    RouteDistilledGoalExpert,
)
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid

SCHEMA_VERSION = "t4_protonation_aware_proposal_gate_result_v1"
DEFAULT_CONTRACT = Path("configs/t4_protonation_aware_proposal_gate_v1.json")
DEFAULT_OUTPUT = Path("diagnostics/t4_protonation_aware_proposal_gate_v1/attempt_1/result.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_envelope(path: Path) -> tuple[dict, str]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(payload):
        raise ValueError(f"input is not a valid self-hashed envelope: {path}")
    return payload, str(envelope["payload_sha256"])


def _atomic_write(path: Path, payload: dict) -> None:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(envelope, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _distribution(values: list[float | int]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _endpoint_metrics(
    seed_smiles: str, delta: float, candidates: list[dict]
) -> tuple[list[dict], dict]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    seed = Chem.MolFromSmiles(seed_smiles)
    if seed is None:
        raise ValueError("source registry contains an invalid SMILES")
    seed_fp = generator.GetFingerprint(seed)
    evaluated = []
    funnel: Counter[str] = Counter()
    for candidate in candidates:
        molecule = Chem.MolFromSmiles(candidate["smiles"])
        if molecule is None or "." in candidate["smiles"]:
            metrics = {
                "parseable": False,
                "structural_valid": False,
                "similarity": None,
                "qed": None,
                "sa": None,
                "heavy": None,
                "similarity_pass": False,
                "qed_pass": False,
                "sa_pass": False,
                "fully_eligible": False,
                "murcko_scaffold": None,
            }
        else:
            canonical = Chem.MolToSmiles(molecule)
            similarity = float(
                DataStructs.TanimotoSimilarity(seed_fp, generator.GetFingerprint(molecule))
            )
            qed = float(QED.qed(molecule))
            sa = float(sascorer.calculateScore(molecule))
            structural = bool(structurally_valid(canonical))
            sim_pass, qed_pass, sa_pass = similarity >= delta, qed >= 0.6, sa <= 4.0
            metrics = {
                "parseable": True,
                "structural_valid": structural,
                "similarity": similarity,
                "qed": qed,
                "sa": sa,
                "heavy": int(molecule.GetNumHeavyAtoms()),
                "similarity_pass": sim_pass,
                "qed_pass": qed_pass,
                "sa_pass": sa_pass,
                "fully_eligible": structural and sim_pass and qed_pass and sa_pass,
                "murcko_scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=molecule),
            }
        flags = {
            "parseable": metrics["parseable"],
            "structural_valid": metrics["structural_valid"],
            "similarity": metrics["similarity_pass"],
            "qed": metrics["qed_pass"],
            "sa": metrics["sa_pass"],
            "all": metrics["fully_eligible"],
        }
        for name, passed in flags.items():
            funnel[name] += int(passed)
        for left in ("structural_valid", "similarity", "qed", "sa"):
            for right in ("structural_valid", "similarity", "qed", "sa"):
                if left < right and flags[left] and flags[right]:
                    funnel[f"{left}&{right}"] += 1
        evaluated.append({**candidate, "endpoint_metrics": metrics})

    eligible = [row for row in evaluated if row["endpoint_metrics"]["fully_eligible"]]
    eligible_scaffolds = {str(row["endpoint_metrics"]["murcko_scaffold"]) for row in eligible}
    charge_transitions = Counter(
        (
            int(row["charge_transition"]["source_total_formal_charge"]),
            int(row["charge_transition"]["after_first_primitive_total_formal_charge"]),
            int(row["charge_transition"]["endpoint_total_formal_charge"]),
        )
        for row in evaluated
    )
    summary = {
        "funnel": dict(sorted(funnel.items())),
        "fully_eligible_count": len(eligible),
        "fully_eligible_unique_endpoints": len({row["endpoint_key"] for row in eligible}),
        "fully_eligible_unique_murcko_scaffolds": len(eligible_scaffolds),
        "fully_eligible_murcko_scaffolds": sorted(eligible_scaffolds),
        "program_kind_counts": dict(Counter(row["program_kind"] for row in evaluated)),
        "structural_lane_counts": dict(Counter(row["structural_lane"] for row in evaluated)),
        "eligible_structural_lane_counts": dict(
            Counter(row["structural_lane"] for row in eligible)
        ),
        "program_length_distribution": _distribution(
            [int(row["primitive_edits"]) for row in evaluated]
        ),
        "eligible_program_length_distribution": _distribution(
            [int(row["primitive_edits"]) for row in eligible]
        ),
        "charge_transition_counts": {
            f"{source}->{first}->{endpoint}": count
            for (source, first, endpoint), count in sorted(charge_transitions.items())
        },
    }
    return evaluated, summary


def run_gate(contract_path: Path) -> dict:
    contract, contract_payload_sha256 = _read_envelope(contract_path)
    registry_path = Path(contract["inputs"]["source_registry"]["path"])
    checkpoint_path = Path(contract["inputs"]["shared_route_checkpoint"]["path"])
    support_contract_path = Path(contract["inputs"]["governing_support_contract"]["path"])
    if _sha256_file(registry_path) != contract["inputs"]["source_registry"]["sha256"]:
        raise ValueError("source-registry physical hash disagrees with the contract")
    if _sha256_file(checkpoint_path) != contract["inputs"]["shared_route_checkpoint"]["sha256"]:
        raise ValueError("route-checkpoint physical hash disagrees with the contract")
    checkpoint, checkpoint_payload_sha256 = _read_envelope(checkpoint_path)
    if checkpoint_payload_sha256 != contract["inputs"]["shared_route_checkpoint"]["payload_sha256"]:
        raise ValueError("route-checkpoint payload hash disagrees with the contract")
    _, support_payload_sha256 = _read_envelope(support_contract_path)
    if support_payload_sha256 != contract["inputs"]["governing_support_contract"]["payload_sha256"]:
        raise ValueError("governing support contract identity disagrees")

    registry = {int(row["idx"]): row for row in json.loads(registry_path.read_text())}
    route_expert = RouteDistilledGoalExpert.from_checkpoint(checkpoint["expert"])
    proposal = contract["proposal"]
    config = ProtonationAwareProposalConfig(
        seed=int(proposal["seed"]),
        shallow_draws=int(proposal["shallow_draws_per_protonation_action"]),
        maximum_primitives=int(proposal["maximum_primitives"]),
        maximum_active_atoms=int(proposal["maximum_active_atoms"]),
        persistent_slots=int(proposal["persistent_slots"]),
        retained_maximum_fragment_atoms=int(proposal["retained_core"]["maximum_fragment_atoms"]),
        retained_maximum_stages=int(proposal["retained_core"]["maximum_stages"]),
        retained_maximum_prefixes=int(proposal["retained_core"]["maximum_prefixes"]),
        route_pool_size=int(proposal["route_complete_region"]["pool_size"]),
        route_realization_limit=int(proposal["route_complete_region"]["realization_limit"]),
        route_beam_width=int(proposal["route_complete_region"]["beam_width"]),
        route_expansion_width=int(proposal["route_complete_region"]["expansion_width"]),
        route_max_bindings_per_template=int(
            proposal["route_complete_region"]["max_bindings_per_template"]
        ),
        route_maximum_expansions=int(proposal["route_complete_region"]["maximum_expansions"]),
        route_candidate_timeout_seconds=float(
            proposal["route_complete_region"]["per_candidate_timeout_seconds"]
        ),
    )

    root_results = []
    for root_spec in contract["roots"]:
        registry_index = int(root_spec["registry_index"])
        row = registry[registry_index]
        source = pad_molecular_graph(
            smiles_to_molecular_graph(str(row["smiles"])),
            config.persistent_slots,
        )
        started = time.perf_counter()
        candidates, telemetry = propose_protonation_aware_candidates(
            source,
            route_expert,
            config=config,
        )
        elapsed = time.perf_counter() - started
        evaluated, summary = _endpoint_metrics(
            str(row["smiles"]), float(root_spec["delta"]), candidates
        )
        root_results.append(
            {
                "label": str(root_spec["label"]),
                "registry_index": registry_index,
                "registry_target": str(row["target"]),
                "source": str(row["smiles"]),
                "delta": float(root_spec["delta"]),
                "selection_reason": str(root_spec["selection_reason"]),
                "elapsed_seconds": elapsed,
                "telemetry": telemetry,
                "summary": summary,
                "candidates": evaluated,
            }
        )

    primary = next(row for row in root_results if row["label"] == "primary_strict_support_failure")
    abstention = next(
        row for row in root_results if row["label"] == "no_admitted_site_abstention_contrast"
    )
    aggregate = {
        "primary_fully_eligible_unique_endpoints": primary["summary"][
            "fully_eligible_unique_endpoints"
        ],
        "primary_fully_eligible_unique_murcko_scaffolds": primary["summary"][
            "fully_eligible_unique_murcko_scaffolds"
        ],
        "primary_diverse_support_pass": primary["summary"]["fully_eligible_unique_endpoints"] >= 2,
        "all_published_candidates_begin_with_one_protonation_action": all(
            sum(
                action["executor_rule"] == "atom_protonation_restate"
                for action in candidate["actions"]
            )
            == 1
            and candidate["actions"][0]["executor_rule"] == "atom_protonation_restate"
            for root in root_results
            for candidate in root["candidates"]
        ),
        "exact_execution_precision_numerator": sum(
            int(root["telemetry"]["exact_execution_precision_numerator"]) for root in root_results
        ),
        "exact_execution_precision_denominator": sum(
            int(root["telemetry"]["exact_execution_precision_denominator"]) for root in root_results
        ),
        "no_site_abstention_pass": bool(
            abstention["telemetry"]["protonation_abstention"] and not abstention["candidates"]
        ),
        "docking_calls": 0,
        "oracle_calls": 0,
        "modal_launches": 0,
    }
    denominator = int(aggregate["exact_execution_precision_denominator"])
    aggregate["exact_execution_precision"] = (
        aggregate["exact_execution_precision_numerator"] / denominator if denominator else None
    )
    aggregate["gate_pass"] = bool(
        aggregate["primary_diverse_support_pass"]
        and aggregate["all_published_candidates_begin_with_one_protonation_action"]
        and aggregate["exact_execution_precision"] == 1.0
        and aggregate["no_site_abstention_pass"]
    )

    source_files = (
        "src/compose_v4/control/protonation_aware_proposal.py",
        "src/compose_v4/control/protonation_restate_program.py",
        "src/compose_v4/control/route_distilled_goal_expert.py",
        "src/compose_v4/rewrite/action_codec_v5.py",
        "src/compose_v4/rewrite/kernel.py",
        "src/compose_v4/rewrite/operators.py",
        "tools/t4_protonation_aware_proposal_gate.py",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence": "computed deterministic zero-oracle autonomous proposal support",
        "claim_boundary": contract["claim_boundary"],
        "inputs": {
            "contract": {
                "path": str(contract_path),
                "sha256": _sha256_file(contract_path),
                "payload_sha256": contract_payload_sha256,
            },
            "source_registry": {
                "path": str(registry_path),
                "sha256": _sha256_file(registry_path),
            },
            "route_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256_file(checkpoint_path),
                "payload_sha256": checkpoint_payload_sha256,
            },
            "governing_support_contract": {
                "path": str(support_contract_path),
                "sha256": _sha256_file(support_contract_path),
                "payload_sha256": support_payload_sha256,
            },
        },
        "implementation": {
            "code_revision": _git_revision(),
            "source_files_sha256": {path: _sha256_file(Path(path)) for path in source_files},
            "config": vars(config),
        },
        "software": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "platform": platform.platform(),
        },
        "roots": root_results,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run_gate(args.contract)
    _atomic_write(args.output, payload)
    print(json.dumps(payload["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
