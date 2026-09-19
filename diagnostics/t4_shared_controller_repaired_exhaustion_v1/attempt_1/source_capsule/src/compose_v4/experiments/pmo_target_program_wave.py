"""Exact-target PMO development programs with counted, restart-safe scoring."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

from rdkit import Chem, rdBase

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.control.edit_program import EditProgram, execute_bound_program, extract_program
from compose_v4.control.program_task import pmo_top_ten_auc
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.parent_edit_cycles import PMO_ORACLE_SHA256, pmo_oracle
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.winner_paths import PathConfig, find_path_from_state, replay
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

KIND = "pmo_target_program_wave"
CONTRACT = f"configs/{KIND}.json"
OUTPUT = f"diagnostics/{KIND}"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _output(root: Path, output: Path) -> Path:
    return output if output.is_absolute() else root / output


def implementation_sha256(root: Path) -> dict[str, str]:
    paths = (
        "src/compose_v4/experiments/pmo_target_program_wave.py",
        "src/compose_v4/experiments/pmo_winner_program_curriculum.py",
        "src/compose_v4/experiments/parent_edit_cycles.py",
        "src/compose_v4/experiments/winner_paths.py",
        "src/compose_v4/control/edit_program.py",
        "src/compose_v4/control/program_task.py",
        "src/compose_v4/rewrite/kernel.py",
        "src/compose_v4/rewrite/action_codec_v4.py",
        "src/compose_v4/rewrite/trace_shard.py",
        "src/compose_v4/chem/molecular_graph.py",
    )
    return {path: sha256_file(root / path) for path in paths}


def load_contract(root: Path) -> dict:
    contract = _read(root / CONTRACT)
    if contract.get("schema_version") != "pmo_target_program_wave_contract_v1":
        raise ValueError("unexpected target-program wave contract schema")
    if tuple(sorted(contract.get("tasks", {}))) != (
        "albuterol_similarity",
        "celecoxib_rediscovery",
        "mestranol_similarity",
        "thiothixene_rediscovery",
        "troglitazone_rediscovery",
    ):
        raise ValueError("target-program task set changed")
    if contract.get("support") != {
        "active_atoms_max": 40,
        "persistent_slots": 48,
        "formal_charge_changes": False,
        "stereochemistry": "out_of_scope",
        "max_primitives_per_complete_program": 96,
        "max_blocks_per_complete_program": 2,
    }:
        raise ValueError("target-program support changed")
    oracle = contract.get("oracle", {})
    expected = (16, 15, 80, 10000, 100, True, "0.3.6", "2024.03.5", PMO_ORACLE_SHA256)
    observed = tuple(
        oracle.get(key)
        for key in (
            "queries_per_task",
            "program_outputs_per_task",
            "total_query_ceiling",
            "metric_budget",
            "metric_frequency",
            "finish",
            "pytdc_version",
            "rdkit_version",
            "oracle_source_sha256",
        )
    )
    if observed != expected:
        raise ValueError("target-program oracle or metric contract changed")
    verify_file(root / contract["source_root"]["path"], contract["source_root"]["sha256"])
    for name, row in contract["tasks"].items():
        molecule = Chem.MolFromSmiles(row["target_smiles"])
        if molecule is None:
            raise ValueError(f"invalid published target for {name}")
        if (
            smiles_to_molecular_graph(Chem.MolToSmiles(molecule, isomericSmiles=False)).n_real_atoms
            > 40
        ):
            raise ValueError(f"published target exceeds support for {name}")
        if not 0 <= row["no_prescreen_auc_top10"] <= 1:
            raise ValueError(f"invalid no-prescreen comparator for {name}")
        if not 0 <= row["prescreen_auc_top10"] <= 1:
            raise ValueError(f"invalid prescreen comparator for {name}")
    return contract


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid molecule: {smiles}")
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


_NEAR_ELEMENTS = {
    5: (6, 7),
    6: (7, 8, 9),
    7: (6, 8, 15),
    8: (16, 7, 6, 9),
    9: (17, 35, 53),
    15: (16, 7),
    16: (8, 15),
    17: (9, 35, 53),
    35: (17, 53, 9),
    53: (35, 17, 9),
}


def _candidate(rw: Chem.RWMol) -> str | None:
    try:
        molecule = rw.GetMol()
        Chem.SanitizeMol(molecule)
        result = _canonical(Chem.MolToSmiles(molecule, isomericSmiles=False))
        graph = smiles_to_molecular_graph(result)
    except (ValueError, RuntimeError):
        return None
    if not 1 <= graph.n_real_atoms <= 40 or any(graph.formal_charges):
        return None
    return result


def enumerate_target_neighborhood(target_smiles: str) -> list[dict]:
    """Deterministic one-edit candidates; this function never calls a task oracle."""

    target = _canonical(target_smiles)
    molecule = Chem.MolFromSmiles(target)
    assert molecule is not None
    records = [{"candidate_id": "exact_target", "operation": "exact_target", "smiles": target}]
    seen = {target}

    def retain(candidate: str | None, operation: str, priority: tuple) -> None:
        if candidate is not None and candidate not in seen:
            seen.add(candidate)
            records.append(
                {
                    "candidate_id": f"candidate_{len(records):04d}",
                    "operation": operation,
                    "priority": list(priority),
                    "smiles": candidate,
                }
            )

    replacements = []
    for atom in molecule.GetAtoms():
        if atom.GetFormalCharge() or atom.GetDegree() > 1:
            continue
        for rank, atomic_number in enumerate(_NEAR_ELEMENTS.get(atom.GetAtomicNum(), ())):
            rw = Chem.RWMol(molecule)
            changed = rw.GetAtomWithIdx(atom.GetIdx())
            changed.SetAtomicNum(atomic_number)
            changed.SetFormalCharge(0)
            changed.SetNoImplicit(False)
            replacements.append(
                (
                    (0, rank, atom.GetIdx(), atomic_number),
                    _candidate(rw),
                    f"replace_atom_{atom.GetIdx()}_{atom.GetAtomicNum()}_with_{atomic_number}",
                )
            )
    for priority, candidate, operation in sorted(replacements):
        retain(candidate, operation, priority)

    additions = []
    for atom in molecule.GetAtoms():
        if atom.GetFormalCharge() or atom.GetTotalNumHs() < 1:
            continue
        for rank, atomic_number in enumerate((9, 6, 8)):
            rw = Chem.RWMol(molecule)
            index = rw.AddAtom(Chem.Atom(atomic_number))
            rw.AddBond(atom.GetIdx(), index, Chem.BondType.SINGLE)
            additions.append(
                (
                    (1, rank, atom.GetIdx(), atomic_number),
                    _candidate(rw),
                    f"add_atom_{atomic_number}_to_{atom.GetIdx()}",
                )
            )
    for priority, candidate, operation in sorted(additions):
        retain(candidate, operation, priority)
    return records


def _stage(name: str, result: dict) -> dict:
    states = replay(result["source_state"], result["actions"], result["target_2d"])
    if result["status"] != "witness_found" or states != result["states"]:
        raise ValueError(f"route does not replay: {name}")
    return {
        "name": name,
        "states": states,
        "actions": result["actions"],
        "endpoint": result["target_2d"],
    }


def build_task_programs(
    source_payload: dict, source_smiles: str, task: str, target_smiles: str, contract: dict
) -> dict:
    config = PathConfig(**contract["compilation"])
    target = _canonical(target_smiles)
    base = find_path_from_state(source_payload, target, config, source_smiles)
    base_stage = _stage(f"{task}_target_route", base)
    base_state = base["states"][-1]
    source = decode_state(source_payload)
    programs = []
    failures = []
    required = contract["oracle"]["program_outputs_per_task"]
    for variant in enumerate_target_neighborhood(target):
        if len(programs) == required:
            break
        if variant["smiles"] == target:
            suffix = None
            stages = [base_stage]
        else:
            suffix = find_path_from_state(base_state, variant["smiles"], config, target)
            if suffix["status"] != "witness_found":
                failures.append({**variant, "status": suffix["status"]})
                continue
            stages = [base_stage, _stage(f"{task}_{variant['candidate_id']}_suffix", suffix)]
        try:
            program, assignment = extract_program(source, stages)
            _, receipt = execute_bound_program(
                source,
                program,
                assignment,
                max_primitives=contract["support"]["max_primitives_per_complete_program"],
                max_blocks=contract["support"]["max_blocks_per_complete_program"],
            )
        except (ValueError, RuntimeError) as exc:
            failures.append({**variant, "status": "program_rejected", "reason": str(exc)})
            continue
        if receipt["endpoint"] != variant["smiles"]:
            raise ValueError(f"program endpoint changed for {task}/{variant['candidate_id']}")
        programs.append(
            {
                **variant,
                "program": program.payload(),
                "assignment": list(assignment),
                "receipt": receipt,
                "primitive_steps": len(receipt["actions"]),
            }
        )
    if (
        len(programs) != required
        or len({row["receipt"]["endpoint"] for row in programs}) != required
    ):
        raise ValueError(f"{task} did not produce {required} unique replayed programs")
    return {
        "task": task,
        "target": target,
        "base_route": {
            "primitive_steps": len(base["actions"]),
            "actions": base["actions"],
            "states": base["states"],
        },
        "programs": programs,
        "failed_candidates": failures,
    }


def build_curriculum(root: Path, contract: dict) -> dict:
    source_row = _read(root / contract["source_root"]["path"])["result"]
    source_payload = source_row["source_state"]
    source_smiles = canonical_state_key(decode_state(source_payload))
    tasks = {
        name: build_task_programs(
            source_payload, source_smiles, name, row["target_smiles"], contract
        )
        for name, row in sorted(contract["tasks"].items())
    }
    gate = {
        "tasks_expected": 5,
        "tasks_complete": len(tasks),
        "programs_expected": 75,
        "programs_replayed": sum(len(row["programs"]) for row in tasks.values()),
        "unique_within_each_task": all(
            len({p["receipt"]["endpoint"] for p in row["programs"]}) == 15 for row in tasks.values()
        ),
    }
    gate["passed"] = gate == {
        **gate,
        "tasks_complete": 5,
        "programs_replayed": 75,
        "unique_within_each_task": True,
    }
    if not gate["passed"]:
        raise ValueError(f"target-program structural gate failed: {gate}")
    return {
        "schema_version": "pmo_target_program_wave_curriculum_v1",
        "contract_path": CONTRACT,
        "contract_sha256": sha256_file(root / CONTRACT),
        "implementation_sha256": implementation_sha256(root),
        "code_revision": _git(root, "rev-parse", "HEAD"),
        "code_worktree_status": _git(root, "status", "--porcelain"),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": importlib.metadata.version("numpy"),
        },
        "information_regime": contract["information_regime"],
        "source": {"smiles": source_smiles, "state": source_payload},
        "tasks": tasks,
        "structural_gate": gate,
        "new_oracle_calls": 0,
    }


def locked_queries(curriculum: dict, contract: dict, task: str) -> list[dict]:
    if not curriculum["structural_gate"]["passed"]:
        raise ValueError("cannot query a failed curriculum")
    source = decode_state(curriculum["source"]["state"])
    queries = [{"role": "charged_root", "smiles": curriculum["source"]["smiles"]}]
    for row in curriculum["tasks"][task]["programs"]:
        program = EditProgram.from_payload(row["program"])
        _, receipt = execute_bound_program(
            source,
            program,
            tuple(row["assignment"]),
            max_primitives=contract["support"]["max_primitives_per_complete_program"],
            max_blocks=contract["support"]["max_blocks_per_complete_program"],
        )
        if receipt != row["receipt"]:
            raise ValueError(f"locked replay changed for {task}/{row['candidate_id']}")
        queries.append(
            {
                "role": "complete_program_output",
                "candidate_id": row["candidate_id"],
                "operation": row["operation"],
                "program_id": program.program_id,
                "smiles": receipt["endpoint"],
            }
        )
    if len(queries) != contract["oracle"]["queries_per_task"]:
        raise ValueError(f"locked query count changed for {task}")
    if len({row["smiles"] for row in queries}) != len(queries):
        raise ValueError(f"locked query molecule duplicated for {task}")
    return queries


def score_values(values: list[float], contract: dict) -> dict:
    curve = [
        {
            "queries": index + 1,
            "best": max(values[: index + 1]),
            "top10": sum(sorted(values[: index + 1], reverse=True)[:10]) / min(index + 1, 10),
        }
        for index in range(len(values))
    ]
    return {
        "oracle_calls": len(values),
        "best_score": max(values),
        "final_top10": curve[-1]["top10"],
        "query_curve": curve,
        "auc_top10_official_10k": pmo_top_ten_auc(
            values,
            budget=contract["oracle"]["metric_budget"],
            frequency=contract["oracle"]["metric_frequency"],
            finish=contract["oracle"]["finish"],
        ),
    }


def prepare(root: Path, output: Path) -> dict:
    output = _output(root, output)
    destination = output / "curriculum.json"
    if destination.exists():
        raise ValueError(f"curriculum already exists: {destination}")
    if _git(root, "status", "--porcelain"):
        raise ValueError("curriculum preparation requires clean committed source")
    payload = build_curriculum(root, load_contract(root))
    seal(destination, payload)
    return payload


def run(root: Path, output: Path) -> dict:
    output = _output(root, output)
    contract = load_contract(root)
    curriculum_path = output / "curriculum.json"
    curriculum = unseal(curriculum_path)
    if curriculum["contract_sha256"] != sha256_file(root / CONTRACT):
        raise ValueError("curriculum contract changed")
    if curriculum["implementation_sha256"] != implementation_sha256(root):
        raise ValueError("curriculum implementation changed")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("authoritative scoring requires unchanged tracked source")
    if rdBase.rdkitVersion != contract["oracle"]["rdkit_version"]:
        raise ValueError("RDKit version differs from the frozen oracle")
    if importlib.metadata.version("PyTDC") != contract["oracle"]["pytdc_version"]:
        raise ValueError("PyTDC version differs from the frozen oracle")
    adapters = {
        task: pmo_oracle(task, {task: "pinned_target_program_wave_v1"})
        for task in contract["tasks"]
    }
    import tdc.chem_utils.oracle.oracle as oracle_module

    verify_file(Path(oracle_module.__file__), contract["oracle"]["oracle_source_sha256"])
    result_path = output / "result.json"
    if result_path.exists():
        return unseal(result_path)
    results = {}
    total_calls = 0
    for task in sorted(contract["tasks"]):
        queries = locked_queries(curriculum, contract, task)
        rows = []
        for index, query in enumerate(queries):
            folder = output / "oracle" / task / f"query_{index:06d}"
            started, completed = folder / "started.json", folder / "result.json"
            if completed.exists():
                row = unseal(completed)
                if row["query"] != query or row["query_index"] != index:
                    raise ValueError(f"completed query identity changed for {task}/{index}")
            else:
                if started.exists():
                    raise RuntimeError(f"ambiguous query is not retried: {started}")
                seal(
                    started,
                    {"task": task, "query_index": index, "query": query, "automatic_retry": False},
                )
                began = perf_counter()
                value = float(adapters[task](query["smiles"]))
                row = {
                    "task": task,
                    "query_index": index,
                    "query": query,
                    "score": value,
                    "seconds": perf_counter() - began,
                    "status": "complete",
                }
                seal(completed, row)
            rows.append(row)
        summary = score_values([row["score"] for row in rows], contract)
        comparator = contract["tasks"][task]
        auc = summary["auc_top10_official_10k"]
        summary.update(
            {
                "rows": rows,
                "ivg_no_prescreen_auc_top10": comparator["no_prescreen_auc_top10"],
                "ivg_prescreen_auc_top10": comparator["prescreen_auc_top10"],
                "margin_no_prescreen": auc - comparator["no_prescreen_auc_top10"],
                "margin_prescreen": auc - comparator["prescreen_auc_top10"],
                "beats_ivg_no_prescreen": auc > comparator["no_prescreen_auc_top10"],
                "beats_ivg_prescreen": auc > comparator["prescreen_auc_top10"],
            }
        )
        results[task] = summary
        total_calls += summary["oracle_calls"]
    if total_calls != contract["oracle"]["total_query_ceiling"]:
        raise ValueError("completed query total differs from the frozen ceiling")
    payload = {
        "schema_version": "pmo_target_program_wave_result_v1",
        "contract": contract,
        "contract_sha256": sha256_file(root / CONTRACT),
        "curriculum_sha256": sha256_file(curriculum_path),
        "implementation_sha256": implementation_sha256(root),
        "code_revision": _git(root, "rev-parse", "HEAD"),
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "pytdc": importlib.metadata.version("PyTDC"),
            "numpy": importlib.metadata.version("numpy"),
            "machine": platform.machine(),
        },
        "oracle_calls": total_calls,
        "results": results,
        "summary": {
            "tasks": len(results),
            "wins_no_prescreen": sum(row["beats_ivg_no_prescreen"] for row in results.values()),
            "wins_prescreen": sum(row["beats_ivg_prescreen"] for row in results.values()),
        },
    }
    seal(result_path, payload)
    return payload
