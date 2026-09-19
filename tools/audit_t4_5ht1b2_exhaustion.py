"""Reconstruct the zero-query 5HT1B-2 delta=0.6 proposal funnel.

This diagnostic deliberately runs the frozen production proposal functions with the
frozen root and random seeds, but replaces only the terminal endpoint gate with an
observational gate that records every exact constructed endpoint.  It never calls a
docking or task oracle.  Expert shards can run independently, then ``merge`` produces
a self-hashed evidence artifact and a concise interpretation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDConfig, rdBase
from rdkit.Chem import QED, rdFingerprintGenerator

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from compose_v4.control.docking_value import identity
from compose_v4.gates.med_chem_gate import is_valid as structurally_valid

SCHEMA = "t4_5ht1b2_delta06_exhaustion_audit_v1"
SHARD_SCHEMA = "t4_5ht1b2_delta06_exhaustion_expert_shard_v1"
CELL = "5ht1b_2"
STRICT_CELL = "docking_5ht1b_idx2_thr6"
EXPERTS = ("shallow", "anchored_replacement", "route_complete_region")
GATES = ("structural_valid", "similarity", "qed", "sa")
QED_MIN = 0.6
SA_MAX = 4.0
HEAVY_MAX = 40


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _read_payload(path: Path) -> dict:
    envelope = _read_json(path)
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
        raise TypeError(f"{path}: expected a payload envelope")
    if envelope.get("payload_sha256") != identity(envelope["payload"]):
        raise ValueError(f"{path}: payload identity mismatch")
    return envelope["payload"]


def _write_envelope(path: Path, payload: dict) -> None:
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid endpoint SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule)


class ObservationalFiber:
    """Production-compatible endpoint gate that observes rather than filters.

    ``expand`` invokes this method only after a complete structural goal has been
    instantiated.  Returning every connected, representable RDKit molecule lets the
    production generation path run unchanged while retaining endpoints that the real
    task gate would reject.
    """

    def __init__(self, seed_smiles: str, delta: float):
        self.delta = float(delta)
        self.support = "observational_audit_only"
        self.root_smiles = _canonical(seed_smiles)
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048
        )
        seed = Chem.MolFromSmiles(seed_smiles)
        if seed is None:
            raise ValueError("audit seed is invalid")
        self.seed = self.generator.GetFingerprint(seed)

    def check(self, smiles: str) -> dict | None:
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None or "." in smiles:
            return None
        heavy = molecule.GetNumHeavyAtoms()
        if heavy > HEAVY_MAX:
            return None
        canonical = Chem.MolToSmiles(molecule)
        return {
            "smiles": canonical,
            "similarity": float(
                DataStructs.TanimotoSimilarity(
                    self.seed, self.generator.GetFingerprint(molecule)
                )
            ),
            "qed": float(QED.qed(molecule)),
            "sa": float(sascorer.calculateScore(molecule)),
            "heavy": int(heavy),
            "structural_valid": bool(structurally_valid(canonical)),
        }


def _properties(smiles: str, fiber: ObservationalFiber) -> dict:
    row = fiber.check(smiles)
    if row is None:
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        return {
            "smiles": smiles,
            "parseable": molecule is not None,
            "connected": bool(molecule is not None and "." not in smiles),
            "representable": bool(
                molecule is not None and molecule.GetNumHeavyAtoms() <= HEAVY_MAX
            ),
        }
    return {**row, "parseable": True, "connected": True, "representable": True}


def _gate_flags(row: dict, delta: float) -> dict[str, bool]:
    return {
        "structural_valid": bool(row["structural_valid"]),
        "similarity": float(row["similarity"]) >= delta,
        "qed": float(row["qed"]) >= QED_MIN,
        "sa": float(row["sa"]) <= SA_MAX,
    }


def _feasibility_distance(row: dict, delta: float) -> float:
    flags = _gate_flags(row, delta)
    return float(
        (0.0 if flags["structural_valid"] else 1.0)
        + max(0.0, delta - float(row["similarity"]))
        + max(0.0, QED_MIN - float(row["qed"]))
        + max(0.0, float(row["sa"]) - SA_MAX)
    )


def _quantiles(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {name: None for name in ("min", "q25", "median", "q75", "max")}
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def classify_candidates(rows: Iterable[dict], *, delta: float) -> dict:
    """Summarize all endpoint gates and every non-empty gate intersection."""

    canonical_rows: dict[str, dict] = {}
    for source in rows:
        row = dict(source)
        canonical_rows.setdefault(str(row["smiles"]), row)
    ordered = [canonical_rows[key] for key in sorted(canonical_rows)]
    flags = {row["smiles"]: _gate_flags(row, delta) for row in ordered}
    intersections = {}
    for size in range(1, len(GATES) + 1):
        for subset in itertools.combinations(GATES, size):
            intersections["&".join(subset)] = sum(
                all(flags[row["smiles"]][gate] for gate in subset) for row in ordered
            )
    closest = sorted(
        ordered,
        key=lambda row: (
            _feasibility_distance(row, delta),
            -float(row["similarity"]),
            -float(row["qed"]),
            float(row["sa"]),
            row["smiles"],
        ),
    )[:10]
    closest_rows = []
    for row in closest:
        closest_rows.append(
            {
                "smiles": row["smiles"],
                "structural_valid": bool(row["structural_valid"]),
                "heavy": int(row["heavy"]),
                "similarity": float(row["similarity"]),
                "qed": float(row["qed"]),
                "sa": float(row["sa"]),
                "similarity_margin": float(row["similarity"]) - delta,
                "qed_margin": float(row["qed"]) - QED_MIN,
                "sa_margin": SA_MAX - float(row["sa"]),
                "feasibility_distance": _feasibility_distance(row, delta),
            }
        )
    return {
        "unique_endpoints": len(ordered),
        "gate_intersections": intersections,
        "all_eligible": intersections["&".join(GATES)],
        "distributions": {
            "similarity": _quantiles(row["similarity"] for row in ordered),
            "qed": _quantiles(row["qed"] for row in ordered),
            "sa": _quantiles(row["sa"] for row in ordered),
            "heavy": _quantiles(row["heavy"] for row in ordered),
        },
        "closest_to_feasible": closest_rows,
    }


def _verify_production_module_hash(module_name: str, expected: str) -> str:
    module = importlib.import_module(module_name)
    path = Path(module.__file__).resolve()
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"production source mismatch for {module_name}: {actual} != {expected}; "
            "run with PYTHONPATH pointing at the frozen launch revision"
        )
    return str(path)


def _generate_expert_shard(args: argparse.Namespace) -> dict:
    contract = _read_payload(args.contract)
    if contract["schema_version"] != "t4_shared_retained_fiber_scored_contract_v2":
        raise ValueError("unexpected frozen contract schema")
    cell = next(row for row in contract["cells"] if row["cell"] == CELL)
    root = str(cell["smiles"])
    delta = float(contract["delta"])
    manifest = _read_payload(args.proposal_manifest)
    request = next(
        row
        for row in manifest["requests"]
        if row["cell"] == CELL and row["expert"] == args.expert
    )
    expected_sources = contract["runtime_inputs_sha256"]
    loaded_sources = {
        "t4_fiber_campaign": _verify_production_module_hash(
            "compose_v4.experiments.t4_fiber_campaign",
            expected_sources["src/compose_v4/experiments/t4_fiber_campaign.py"],
        )
    }
    if args.expert == "route_complete_region":
        loaded_sources["route_distilled_goal_expert"] = _verify_production_module_hash(
            "compose_v4.control.route_distilled_goal_expert",
            expected_sources["src/compose_v4/control/route_distilled_goal_expert.py"],
        )
    fiber = ObservationalFiber(root, delta)
    started = time.time()
    if args.expert in {"shallow", "anchored_replacement"}:
        from compose_v4.experiments.t4_fiber_campaign import expand

        proposal = contract["proposal"][args.expert]
        rows = expand(
            root,
            float(request["parent_score"]),
            fiber,
            np.random.default_rng(int(request["proposal_seed"])),
            draws=int(proposal["draws"]),
            multi_region=True,
            horizon=int(proposal["horizon"]),
            proposal_lane=args.expert,
        )
        telemetry = {
            "raw_draws": int(proposal["draws"]),
            "unique_exact_constructed_endpoints": len(rows),
        }
        proposed_unique = len(rows)
        exact_unique = len(rows)
    else:
        from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
        from compose_v4.chem.state import pad_molecular_graph
        from compose_v4.control.route_distilled_goal_expert import (
            RouteDistilledGoalExpert,
            propose_route_expert_candidates,
        )

        checkpoint = _read_payload(args.checkpoint)
        model = RouteDistilledGoalExpert.from_checkpoint(checkpoint["expert"])
        proposal = contract["proposal"][args.expert]
        rows, telemetry = propose_route_expert_candidates(
            pad_molecular_graph(smiles_to_molecular_graph(root), 48),
            model,
            pool_size=int(proposal["pool_size"]),
            realization_limit=int(proposal["realization_limit"]),
            beam_width=int(proposal["beam_width"]),
            expansion_width=int(proposal["expansion_width"]),
            max_bindings_per_template=int(proposal["max_bindings_per_template"]),
            maximum_expansions=int(proposal["maximum_expansions"]),
            scale_balanced=bool(proposal["scale_balanced"]),
        )
        rows = [{**row, **_properties(row["smiles"], fiber)} for row in rows]
        proposed_unique = int(telemetry["unique_ranked_endpoints"])
        exact_unique = len(rows)
    summary = classify_candidates(rows, delta=delta)
    return {
        "schema_version": SHARD_SCHEMA,
        "evidence": "computed zero-oracle replay of frozen production proposal path",
        "cell": CELL,
        "expert": args.expert,
        "root": root,
        "delta": delta,
        "proposal_seed": int(request["proposal_seed"]),
        "production_code_revision": str(request["code_revision"]),
        "audit_code_revision": _git_revision(),
        "proposed_unique_endpoints": proposed_unique,
        "exact_executed_unique_endpoints": exact_unique,
        "summary": summary,
        "telemetry": telemetry,
        "elapsed_seconds": time.time() - started,
        "inputs": {
            "contract": {
                "path": str(args.contract),
                "sha256": _sha256_file(args.contract),
                "payload_sha256": identity(contract),
            },
            "proposal_manifest": {
                "path": str(args.proposal_manifest),
                "sha256": _sha256_file(args.proposal_manifest),
                "payload_sha256": identity(manifest),
            },
            "checkpoint": (
                {
                    "path": str(args.checkpoint),
                    "sha256": _sha256_file(args.checkpoint),
                    "payload_sha256": identity(_read_payload(args.checkpoint)),
                }
                if args.expert == "route_complete_region"
                else None
            ),
            "loaded_production_sources": loaded_sources,
            "expected_runtime_source_hashes": expected_sources,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "costs": {"docking_calls": 0, "oracle_calls": 0, "modal_launches": 0},
        "endpoints": sorted(rows, key=lambda row: row["smiles"]),
    }


def _known_answer_rows(
    strict_corpus_path: Path,
    full146_forensics_path: Path,
    fiber: ObservationalFiber,
    generated_smiles: set[str],
) -> list[dict]:
    known = []
    strict = _read_payload(strict_corpus_path)
    for row in strict["endpoint_references"]:
        if row.get("cell") != STRICT_CELL:
            continue
        endpoint = _canonical(row["endpoint_smiles"])
        properties = _properties(endpoint, fiber)
        known.append(
            {
                "evidence_role": "reported_delta06_IVG_endpoint_known_answer_only",
                "endpoint": endpoint,
                "route_status": row["status"],
                "reported_scores": sorted(
                    {
                        float(value["reported_docking_score"])
                        for value in row["delta06_run_references"]
                    }
                ),
                "generated_by_frozen_pool": endpoint in generated_smiles,
                "source_formal_charge": int(
                    Chem.GetFormalCharge(Chem.MolFromSmiles(fiber.root_smiles))
                ),
                "endpoint_formal_charge": int(
                    Chem.GetFormalCharge(Chem.MolFromSmiles(endpoint))
                ),
                **properties,
            }
        )
    forensics = _read_payload(full146_forensics_path)
    for route in forensics["routes"]:
        if route.get("cell") != CELL:
            continue
        receipt_path = Path(route["teacher_receipt"])
        receipt = _read_json(receipt_path)
        endpoint = _canonical(receipt["pair"]["target"])
        properties = _properties(endpoint, fiber)
        known.append(
            {
                "evidence_role": "Full146_delta04_teacher_endpoint_known_answer_only",
                "endpoint": endpoint,
                "route_status": route["representation"]["status"],
                "reported_scores": sorted(
                    {
                        float(value["reported_external_docking_score"])
                        for value in route.get("known_labels", [])
                    }
                ),
                "generated_by_frozen_pool": endpoint in generated_smiles,
                "source_formal_charge": int(
                    Chem.GetFormalCharge(Chem.MolFromSmiles(fiber.root_smiles))
                ),
                "endpoint_formal_charge": int(
                    Chem.GetFormalCharge(Chem.MolFromSmiles(endpoint))
                ),
                **properties,
            }
        )
    return sorted(known, key=lambda row: (row["evidence_role"], row["endpoint"]))


def _merge(args: argparse.Namespace) -> dict:
    contract = _read_payload(args.contract)
    cell = next(row for row in contract["cells"] if row["cell"] == CELL)
    fiber = ObservationalFiber(str(cell["smiles"]), float(contract["delta"]))
    shards = [_read_payload(path) for path in args.shards]
    by_expert = {row["expert"]: row for row in shards}
    if set(by_expert) != set(EXPERTS):
        raise ValueError(
            f"expected exactly one shard for every expert, got {sorted(by_expert)}"
        )
    generated_smiles = {
        row["smiles"] for shard in shards for row in shard.get("endpoints", [])
    }
    known_answers = _known_answer_rows(
        args.strict_corpus,
        args.full146_forensics,
        fiber,
        generated_smiles,
    )
    run_receipts = {}
    for name, path in (
        ("final_result", args.final_result),
        ("proposal_manifest", args.proposal_manifest),
        ("shallow_receipt", args.shallow_receipt),
        ("anchored_receipt", args.anchored_receipt),
        ("route_receipt", args.route_receipt),
    ):
        payload = _read_payload(path)
        run_receipts[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "payload_sha256": identity(payload),
            "payload": payload,
        }
    experts = {}
    for expert in EXPERTS:
        shard = by_expert[expert]
        experts[expert] = {
            key: shard[key]
            for key in (
                "proposal_seed",
                "proposed_unique_endpoints",
                "exact_executed_unique_endpoints",
                "summary",
                "telemetry",
                "elapsed_seconds",
                "software",
            )
        }
    route_remote = run_receipts["route_receipt"]["payload"]["answer"]
    if int(route_remote["telemetry"]["complete_programs_committed"]) != 96:
        raise ValueError("durable production route receipt no longer proves 96 commits")
    payload = {
        "schema_version": SCHEMA,
        "evidence": "computed zero-oracle frozen-production forensic",
        "generated_at": datetime.now(UTC).isoformat(),
        "scientific_problem": (
            "identify the exact endpoint gate responsible for 5HT1B-2 delta=0.6 "
            "candidate exhaustion"
        ),
        "primary_output": "per-expert complete-endpoint constraint funnel",
        "cell": CELL,
        "root": cell["smiles"],
        "delta": float(contract["delta"]),
        "production_code_revision": run_receipts["proposal_manifest"]["payload"][
            "requests"
        ][0]["code_revision"],
        "audit_code_revision": _git_revision(),
        "observed_run_outcome": {
            "status": run_receipts["final_result"]["payload"]["status"],
            "charged_calls": int(
                run_receipts["final_result"]["payload"]["charged_calls"]
            ),
            "final_best": float(run_receipts["final_result"]["payload"]["final_best"]),
        },
        "experts": experts,
        "known_answer_diagnostics": known_answers,
        "decisive_mechanism": (
            "filled after the complete measured funnel is inspected; this field must "
            "not infer a cause from the zero-candidate symptom alone"
        ),
        "smallest_general_fix": (
            "filled after the complete measured funnel is inspected; no controller "
            "change is made by this audit"
        ),
        "costs": {"docking_calls": 0, "oracle_calls": 0, "modal_launches": 0},
        "inputs": {
            "contract": {
                "path": str(args.contract),
                "sha256": _sha256_file(args.contract),
                "payload_sha256": identity(contract),
            },
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": _sha256_file(args.checkpoint),
                "payload_sha256": identity(_read_payload(args.checkpoint)),
            },
            "strict_delta06_corpus": {
                "path": str(args.strict_corpus),
                "sha256": _sha256_file(args.strict_corpus),
                "payload_sha256": identity(_read_payload(args.strict_corpus)),
            },
            "full146_forensics": {
                "path": str(args.full146_forensics),
                "sha256": _sha256_file(args.full146_forensics),
                "payload_sha256": identity(_read_payload(args.full146_forensics)),
            },
            "expert_shards": [
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "payload_sha256": identity(_read_payload(path)),
                }
                for path in args.shards
            ],
            "durable_run_receipts": run_receipts,
        },
        "limitations": [
            "This is zero-oracle answer-known forensics, not autonomous recovery evidence.",
            (
                "The local audit environment is recorded per expert; the frozen "
                "production source hashes and random seeds are verified before replay."
            ),
            (
                "The strict delta=0.6 route corpus records no exact executable witness "
                "for this cell when the endpoint changes formal charge."
            ),
        ],
    }
    return payload


def _interpret(payload: dict) -> str:
    lines = [
        "# 5HT1B-2 delta=0.6 candidate-exhaustion forensic",
        "",
        (
            f"The frozen run stopped after the root call with status "
            f"`{payload['observed_run_outcome']['status']}`. This audit made zero "
            "oracle or docking calls."
        ),
        "",
        "## Complete endpoint funnel",
        "",
        "| Expert | Proposed unique | Exact endpoints | Structural | Sim | QED | SA | All eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for expert in EXPERTS:
        row = payload["experts"][expert]
        gates = row["summary"]["gate_intersections"]
        lines.append(
            f"| {expert} | {row['proposed_unique_endpoints']} | "
            f"{row['exact_executed_unique_endpoints']} | {gates['structural_valid']} | "
            f"{gates['similarity']} | {gates['qed']} | {gates['sa']} | "
            f"{row['summary']['all_eligible']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["decisive_mechanism"],
            "",
            "Smallest general fix: " + payload["smallest_general_fix"],
            "",
            (
                "Known endpoints are included only as labelled answer-known "
                "diagnostics. They were never injected into a proposal pool."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("expert")
    shard.add_argument("--expert", choices=EXPERTS, required=True)
    shard.add_argument("--contract", type=Path, required=True)
    shard.add_argument("--checkpoint", type=Path, required=True)
    shard.add_argument("--proposal-manifest", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--contract", type=Path, required=True)
    merge.add_argument("--checkpoint", type=Path, required=True)
    merge.add_argument("--strict-corpus", type=Path, required=True)
    merge.add_argument("--full146-forensics", type=Path, required=True)
    merge.add_argument("--proposal-manifest", type=Path, required=True)
    merge.add_argument("--final-result", type=Path, required=True)
    merge.add_argument("--shallow-receipt", type=Path, required=True)
    merge.add_argument("--anchored-receipt", type=Path, required=True)
    merge.add_argument("--route-receipt", type=Path, required=True)
    merge.add_argument("--shards", type=Path, nargs=3, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--interpretation", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "expert":
        _write_envelope(args.output, _generate_expert_shard(args))
        return
    payload = _merge(args)
    _write_envelope(args.output, payload)
    args.interpretation.write_text(_interpret(payload))


if __name__ == "__main__":
    main()
