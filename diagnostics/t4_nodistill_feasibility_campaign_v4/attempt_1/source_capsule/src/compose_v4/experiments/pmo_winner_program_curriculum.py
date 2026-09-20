"""Winner-informed Perindopril programs with exact replay and counted scoring."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.control.edit_program import EditProgram, execute_bound_program, extract_program
from compose_v4.control.program_task import pmo_top_ten_auc
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.parent_edit_cycles import PMO_ORACLE_SHA256, pmo_oracle
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.winner_paths import PathConfig, find_path_from_state, replay
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

KIND = "pmo_winner_program_curriculum"
CONTRACT = f"configs/{KIND}.json"
OUTPUT = f"diagnostics/{KIND}"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def implementation_sha256(root: Path) -> dict[str, str]:
    paths = (
        "src/compose_v4/experiments/pmo_winner_program_curriculum.py",
        "src/compose_v4/control/edit_program.py",
        "src/compose_v4/control/program_task.py",
        "src/compose_v4/experiments/winner_paths.py",
        "src/compose_v4/experiments/parent_edit_cycles.py",
        "src/compose_v4/rewrite/kernel.py",
        "src/compose_v4/rewrite/action_codec_v4.py",
        "src/compose_v4/rewrite/trace_shard.py",
        "src/compose_v4/chem/molecular_graph.py",
    )
    return {path: sha256_file(root / path) for path in paths}


def load_contract(root: Path) -> dict:
    path = root / CONTRACT
    contract = _read(path)
    if contract.get("schema_version") != "pmo_winner_program_curriculum_contract_v1":
        raise ValueError("unexpected winner-program curriculum contract schema")
    if contract.get("task") != "perindopril_mpo":
        raise ValueError("winner-program curriculum is locked to Perindopril MPO")
    support = contract.get("support", {})
    if support != {
        "active_atoms_max": 40,
        "persistent_slots": 48,
        "formal_charge_changes": False,
        "stereochemistry": "out_of_scope",
        "max_primitives_per_complete_program": 96,
        "max_blocks_per_complete_program": 2,
    }:
        raise ValueError("winner-program support or work limits changed")
    oracle = contract.get("oracle", {})
    if (
        oracle.get("budget_ceiling"),
        oracle.get("expected_queries"),
        oracle.get("metric_budget"),
        oracle.get("metric_frequency"),
        oracle.get("finish"),
        oracle.get("pytdc_version"),
        oracle.get("rdkit_version"),
        oracle.get("oracle_source_sha256"),
    ) != (16, 15, 10000, 100, True, "0.3.6", "2024.03.5", PMO_ORACLE_SHA256):
        raise ValueError("winner-program oracle or metric lock changed")
    comparator = contract.get("comparators", {})
    verify_file(root / comparator["source_path"], comparator["source_sha256"])
    reported = _read(root / comparator["source_path"])
    if (
        reported["targets"]["perindopril_mpo"],
        reported["no_prescreen_targets_partial"]["perindopril_mpo"],
        comparator["prescreen_auc_top10"],
        comparator["no_prescreen_auc_top10"],
    ) != (0.753, 0.645, 0.753, 0.645):
        raise ValueError("IVG Perindopril comparator differs from the frozen source")
    for record in (*contract["route_inputs"].values(), *contract["other_inputs"].values()):
        verify_file(root / record["path"], record["sha256"])
    recovery = _read(root / contract["other_inputs"]["recovery_config"]["path"])
    if recovery["target"]["smiles"] != contract["public_endpoint"]["smiles"]:
        raise ValueError("public endpoint differs from the verified recovery target")
    if tuple(contract["root_order"]) != tuple(f"original_root_{i}" for i in range(4)):
        raise ValueError("scored root order changed")
    if tuple(contract["variants"]["distal_elements"]) != (
        "B",
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
    ):
        raise ValueError("retrospectively selected endpoint family changed")
    return contract


def _stage(name: str, result: dict) -> dict:
    endpoint = canonical_state_key(decode_state(result["states"][-1]))
    if endpoint != result["target_2d"]:
        raise ValueError(f"{name} target differs from its saved exact final state")
    return {
        "name": name,
        "states": result["states"],
        "actions": result["actions"],
        "endpoint": endpoint,
    }


def _variant_smiles(base: str, element: str) -> str:
    token = "(Br)"
    if base.count(token) != 1:
        raise ValueError("public endpoint no longer has one declared distal bromine")
    return base.replace(token, f"({element})", 1)


def _molecule_record(smiles: str) -> dict:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"unparseable curriculum endpoint: {smiles}")
    canonical = Chem.MolToSmiles(molecule, isomericSmiles=False)
    graph = smiles_to_molecular_graph(canonical)
    return {
        "canonical_smiles": canonical,
        "heavy_atoms": graph.n_real_atoms,
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(molecule),
    }


def build_curriculum(root: Path, contract: dict) -> dict:
    route_rows = {
        name: _read(root / record["path"]) for name, record in contract["route_inputs"].items()
    }
    base = _molecule_record(contract["public_endpoint"]["smiles"])["canonical_smiles"]
    route_audit = []
    for name, row in sorted(route_rows.items()):
        result = row["result"]
        states = replay(result["source_state"], result["actions"], result["target_2d"])
        if result["status"] != "witness_found" or states != result["states"]:
            raise ValueError(f"saved public-target witness does not replay: {name}")
        if result["target_2d"] != base:
            raise ValueError(f"saved witness has a different public endpoint: {name}")
        route_audit.append(
            {
                "source_id": name,
                "coverage": "replayed",
                "primitive_steps": len(result["actions"]),
                "source_endpoint": canonical_state_key(decode_state(result["source_state"])),
            }
        )

    root_row = route_rows[contract["root_order"][0]]["result"]
    source = decode_state(root_row["source_state"])
    base_state = root_row["states"][-1]
    base_stage = _stage("public_endpoint_recovery", root_row)
    path_config = PathConfig(**contract["compilation"])
    targets = [
        (f"distal_{element.lower()}", _variant_smiles(base, element), "computed_plateau_variant")
        for element in contract["variants"]["distal_elements"]
    ]
    targets.append(
        (
            "shorter_segment",
            contract["variants"]["shorter_segment_smiles"],
            "computed_local_improvement_variant",
        )
    )
    programs = []
    for variant_id, target_smiles, evidence_role in targets:
        target = _molecule_record(target_smiles)
        if target["heavy_atoms"] > contract["support"]["active_atoms_max"]:
            raise ValueError(f"curriculum target exceeds active-atom support: {variant_id}")
        if target["canonical_smiles"] == base:
            suffix = None
            stages = [base_stage]
        else:
            suffix = find_path_from_state(
                base_state,
                target["canonical_smiles"],
                path_config,
                base,
            )
            if suffix["status"] != "witness_found":
                raise ValueError(
                    f"bounded suffix compilation failed: {variant_id}: {suffix['status']}"
                )
            if replay(base_state, suffix["actions"], suffix["target_2d"]) != suffix["states"]:
                raise ValueError(f"suffix exact replay failed: {variant_id}")
            stages = [base_stage, _stage(f"{variant_id}_suffix", suffix)]
        program, assignment = extract_program(source, stages)
        endpoint, receipt = execute_bound_program(
            source,
            program,
            assignment,
            max_primitives=contract["support"]["max_primitives_per_complete_program"],
            max_blocks=contract["support"]["max_blocks_per_complete_program"],
        )
        if receipt["endpoint"] != target["canonical_smiles"]:
            raise ValueError(f"extracted program misses its target: {variant_id}")
        if endpoint.n_real_atoms != target["heavy_atoms"]:
            raise ValueError(f"endpoint atom count differs after replay: {variant_id}")
        programs.append(
            {
                "variant_id": variant_id,
                "evidence_role": evidence_role,
                "program": program.payload(),
                "assignment": list(assignment),
                "target": target,
                "suffix": None
                if suffix is None
                else {
                    "primitive_steps": len(suffix["actions"]),
                    "actions": suffix["actions"],
                    "states": suffix["states"],
                },
                "receipt": receipt,
            }
        )

    endpoints = [row["receipt"]["endpoint"] for row in programs]
    program_ids = [
        row["program"]["schema_version"] + ":" + EditProgram.from_payload(row["program"]).program_id
        for row in programs
    ]
    structural_gate = {
        "expected_programs": 11,
        "replayed_programs": len(programs),
        "unique_endpoints": len(set(endpoints)),
        "unique_programs": len(set(program_ids)),
        "all_within_support": all(row["target"]["heavy_atoms"] <= 40 for row in programs),
    }
    structural_gate["passed"] = structural_gate == {
        **structural_gate,
        "replayed_programs": 11,
        "unique_endpoints": 11,
        "unique_programs": 11,
        "all_within_support": True,
    }
    if not structural_gate["passed"]:
        raise ValueError(f"curriculum structural gate failed: {structural_gate}")

    roots = []
    for root_id in contract["root_order"]:
        result = route_rows[root_id]["result"]
        roots.append(
            {
                "root_id": root_id,
                "state": result["source_state"],
                "endpoint": canonical_state_key(decode_state(result["source_state"])),
            }
        )
    return {
        "schema_version": "pmo_winner_program_curriculum_v1",
        "contract_path": CONTRACT,
        "contract_sha256": sha256_file(root / CONTRACT),
        "input_sha256": {
            record["path"]: record["sha256"]
            for record in (*contract["route_inputs"].values(), *contract["other_inputs"].values())
        }
        | {contract["comparators"]["source_path"]: contract["comparators"]["source_sha256"]},
        "implementation_sha256": implementation_sha256(root),
        "code_revision": _git(root, "rev-parse", "HEAD"),
        "code_worktree_status": _git(root, "status", "--porcelain"),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": importlib.metadata.version("numpy"),
        },
        "information_regime": contract["information_regime"],
        "route_audit": route_audit,
        "roots": roots,
        "programs": sorted(programs, key=lambda row: row["variant_id"]),
        "structural_gate": structural_gate,
        "new_oracle_calls": 0,
    }


def locked_queries(curriculum: dict, contract: dict) -> list[dict]:
    if not curriculum["structural_gate"]["passed"]:
        raise ValueError("cannot score a curriculum that failed its structural gate")
    roots = {row["root_id"]: row for row in curriculum["roots"]}
    queries = [
        {"role": "charged_root", "source_id": root_id, "smiles": roots[root_id]["endpoint"]}
        for root_id in contract["root_order"]
    ]
    source = decode_state(roots[contract["root_order"][0]]["state"])
    for row in curriculum["programs"]:
        program = EditProgram.from_payload(row["program"])
        _, receipt = execute_bound_program(
            source,
            program,
            tuple(row["assignment"]),
            max_primitives=contract["support"]["max_primitives_per_complete_program"],
            max_blocks=contract["support"]["max_blocks_per_complete_program"],
        )
        if receipt != row["receipt"]:
            raise ValueError(f"locked program replay changed: {row['variant_id']}")
        queries.append(
            {
                "role": "complete_program_output",
                "source_id": contract["root_order"][0],
                "variant_id": row["variant_id"],
                "program_id": program.program_id,
                "smiles": receipt["endpoint"],
            }
        )
    if len(queries) != contract["oracle"]["expected_queries"]:
        raise ValueError("locked query count differs from the contract")
    if len({row["smiles"] for row in queries}) != len(queries):
        raise ValueError("locked query list contains duplicate canonical molecules")
    return queries


def score_queries(queries: list[dict], evaluate, *, budget: int, frequency: int) -> dict:
    rows = []
    for index, query in enumerate(queries):
        began = perf_counter()
        score = float(evaluate(query["smiles"]))
        rows.append(
            {**query, "query_index": index, "score": score, "seconds": perf_counter() - began}
        )
    rewards = [row["score"] for row in rows]
    curve = [
        {
            "queries": index + 1,
            "best": max(rewards[: index + 1]),
            "top10": sum(sorted(rewards[: index + 1], reverse=True)[:10]) / min(index + 1, 10),
        }
        for index in range(len(rows))
    ]
    return {
        "rows": rows,
        "query_curve": curve,
        "oracle_calls": len(rows),
        "best_score": max(rewards),
        "final_top10": curve[-1]["top10"],
        "auc_top10_official_10k": pmo_top_ten_auc(
            rewards, budget=budget, frequency=frequency, finish=True
        ),
    }


def prepare(root: Path, output: Path) -> dict:
    contract = load_contract(root)
    destination = output / "curriculum.json"
    if destination.exists():
        raise ValueError(f"curriculum output already exists: {destination}")
    payload = build_curriculum(root, contract)
    seal(destination, payload)
    return payload


def run(root: Path, output: Path) -> dict:
    contract = load_contract(root)
    curriculum_path = output / "curriculum.json"
    curriculum = unseal(curriculum_path)
    if curriculum["contract_sha256"] != sha256_file(root / CONTRACT):
        raise ValueError("curriculum was prepared under a different contract")
    if curriculum["implementation_sha256"] != implementation_sha256(root):
        raise ValueError("curriculum implementation differs from the running source")
    if _git(root, "status", "--porcelain"):
        raise ValueError("authoritative scoring requires a clean committed source tree")
    if rdBase.rdkitVersion != contract["oracle"]["rdkit_version"]:
        raise ValueError("RDKit version differs from the frozen oracle environment")
    if importlib.metadata.version("PyTDC") != contract["oracle"]["pytdc_version"]:
        raise ValueError("PyTDC version differs from the frozen oracle environment")
    # Constructing the adapter installs the repository's audited ``rdkit.six``
    # compatibility shim before PyTDC 0.3.6 imports its oracle module. Modern
    # RDKit no longer ships that module. Adapter construction performs no score.
    evaluate = pmo_oracle(contract["task"], {contract["task"]: "pinned_pmo_perindopril_v1"})
    import tdc.chem_utils.oracle.oracle as oracle_module

    verify_file(Path(oracle_module.__file__), contract["oracle"]["oracle_source_sha256"])
    queries = locked_queries(curriculum, contract)
    result_path = output / "result.json"
    if result_path.exists():
        return unseal(result_path)
    oracle_root = output / "oracle"
    oracle_root.mkdir(parents=True, exist_ok=True)
    scored = []
    for index, query in enumerate(queries):
        folder = oracle_root / f"query_{index:06d}"
        started, completed = folder / "started.json", folder / "result.json"
        if completed.exists():
            row = unseal(completed)
            if row["query"] != query or row["query_index"] != index:
                raise ValueError(f"completed query identity differs at index {index}")
            scored.append(row)
            continue
        if started.exists():
            raise RuntimeError(f"ambiguous started oracle query is not retried: {started}")
        seal(started, {"query_index": index, "query": query, "automatic_retry": False})
        began = perf_counter()
        value = float(evaluate(query["smiles"]))
        row = {
            "query_index": index,
            "query": query,
            "score": value,
            "seconds": perf_counter() - began,
            "status": "complete",
        }
        seal(completed, row)
        scored.append(row)
    summary = score_queries(
        [row["query"] for row in scored],
        lambda smiles: next(row["score"] for row in scored if row["query"]["smiles"] == smiles),
        budget=contract["oracle"]["metric_budget"],
        frequency=contract["oracle"]["metric_frequency"],
    )
    # Replace synthetic timing from deterministic replay with measured query timing.
    summary["rows"] = [
        {
            **row["query"],
            "query_index": row["query_index"],
            "score": row["score"],
            "seconds": row["seconds"],
        }
        for row in scored
    ]
    no_prescreen = contract["comparators"]["no_prescreen_auc_top10"]
    prescreen = contract["comparators"]["prescreen_auc_top10"]
    result = {
        "schema_version": "pmo_winner_program_curriculum_result_v1",
        "contract": contract,
        "contract_sha256": sha256_file(root / CONTRACT),
        "curriculum_path": str(curriculum_path.relative_to(root)),
        "curriculum_sha256": sha256_file(curriculum_path),
        "implementation_sha256": implementation_sha256(root),
        "code_revision": _git(root, "rev-parse", "HEAD"),
        "code_worktree_status_at_start": "",
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "pytdc": importlib.metadata.version("PyTDC"),
            "numpy": importlib.metadata.version("numpy"),
            "machine": platform.machine(),
        },
        **summary,
        "oracle_seconds": sum(row["seconds"] for row in scored),
        "comparisons": {
            "ivg_no_prescreen_auc_top10": no_prescreen,
            "ivg_prescreen_auc_top10": prescreen,
            "beats_ivg_no_prescreen": summary["auc_top10_official_10k"] > no_prescreen,
            "beats_ivg_prescreen": summary["auc_top10_official_10k"] > prescreen,
        },
        "interpretation": "winner-informed answer-known Perindopril development; not held-out or general PMO superiority",
        "automatic_retries": 0,
    }
    seal(result_path, result)
    return result
