"""Formula- and panel-informed PMO programs with exact counted scoring."""

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
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.edit_program import EditProgram, execute_bound_program
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.pmo_property_panel_refinement import (
    build_task_programs,
)
from compose_v4.experiments.pmo_property_program_wave import verified_adapters
from compose_v4.experiments.pmo_target_program_wave import _canonical, score_values
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state, encode_state

KIND = "pmo_formula_median_panel_wave"
CONTRACT = f"configs/{KIND}.json"
OUTPUT = f"diagnostics/{KIND}"
CURRICULUM_NAME = "curriculum.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _output(root: Path, output: Path) -> Path:
    return output if output.is_absolute() else root / output


def implementation_sha256(root: Path) -> dict[str, str]:
    paths = (
        "src/compose_v4/experiments/pmo_formula_median_panel_wave.py",
        "src/compose_v4/experiments/pmo_property_panel_refinement.py",
        "src/compose_v4/experiments/pmo_property_program_wave.py",
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
        "src/compose_v4/chem/state.py",
    )
    return {path: sha256_file(root / path) for path in paths}


def load_contract(root: Path) -> dict:
    contract = _read(root / CONTRACT)
    if contract.get("schema_version") != "pmo_formula_median_panel_wave_contract_v1":
        raise ValueError("unexpected formula/median panel-wave contract schema")
    if tuple(sorted(contract.get("tasks", {}))) != ("isomers_c7h8n2o2", "median1"):
        raise ValueError("formula/median task set changed")
    if contract.get("support") != {
        "active_atoms_max": 40,
        "persistent_slots": 48,
        "formal_charge_changes": False,
        "stereochemistry": "out_of_scope",
        "max_primitives_per_complete_program": 96,
        "max_blocks_per_complete_program": 2,
    }:
        raise ValueError("formula/median support changed")
    oracle = contract.get("oracle", {})
    expected = (11, 10, 22, 10000, 100, True, "0.3.6", "2024.03.5")
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
        )
    )
    if observed != expected:
        raise ValueError("formula/median oracle or metric contract changed")
    if tuple(sorted(contract.get("source_roots", {}))) != tuple(
        sorted(contract["tasks"])
    ):
        raise ValueError("task-specific source-root set changed")
    for name, row in sorted(contract["source_roots"].items()):
        smiles = _canonical(row["smiles"])
        graph = smiles_to_molecular_graph(smiles)
        if graph.n_real_atoms > 40 or any(graph.formal_charges):
            raise ValueError(f"unsupported source root for {name}")
        if row.get("source_role") != "predeclared_task_specific_generic_root":
            raise ValueError(f"source-root role changed for {name}")
    allowed_roles = {
        "panel_caption_transcription",
        "formula_matched_positional_variant",
    }
    for name, row in sorted(contract["tasks"].items()):
        candidates = row.get("candidates", [])
        if len(candidates) != oracle["program_outputs_per_task"]:
            raise ValueError(f"candidate count changed for {name}")
        canonical = []
        for index, candidate in enumerate(candidates):
            smiles = _canonical(candidate["smiles"])
            graph = smiles_to_molecular_graph(smiles)
            if graph.n_real_atoms > 40 or any(graph.formal_charges):
                raise ValueError(f"unsupported candidate for {name}/{index}")
            if candidate.get("source_role") not in allowed_roles:
                raise ValueError(f"unknown evidence role for {name}/{index}")
            if name == "isomers_c7h8n2o2":
                molecule = Chem.MolFromSmiles(smiles)
                assert molecule is not None
                if rdMolDescriptors.CalcMolFormula(molecule) != row["target_formula"]:
                    raise ValueError(f"molecular formula mismatch for {name}/{index}")
            canonical.append(smiles)
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"canonical candidate duplication for {name}")
        if _canonical(contract["source_roots"][name]["smiles"]) in canonical:
            raise ValueError(f"source root duplicates a candidate for {name}")
        for key in ("no_prescreen_auc_top10", "prescreen_auc_top10"):
            if not 0 <= row[key] <= 1:
                raise ValueError(f"invalid {key} comparator for {name}")
    return contract


def build_curriculum(root: Path, contract: dict) -> dict:
    sources = {}
    tasks = {}
    for name, row in sorted(contract["tasks"].items()):
        source_row = contract["source_roots"][name]
        source_state = pad_molecular_graph(
            smiles_to_molecular_graph(_canonical(source_row["smiles"])), 48
        )
        source_payload = encode_state(source_state)
        source_smiles = canonical_state_key(source_state)
        sources[name] = {
            "smiles": source_smiles,
            "state": source_payload,
            "source_role": source_row["source_role"],
        }
        tasks[name] = build_task_programs(
            source_payload, source_smiles, name, row["candidates"], contract
        )
    gate = {
        "tasks_expected": 2,
        "tasks_complete": len(tasks),
        "programs_expected": 20,
        "programs_replayed": sum(len(row["programs"]) for row in tasks.values()),
        "unique_within_each_task": all(
            len({p["receipt"]["endpoint"] for p in row["programs"]}) == 10
            for row in tasks.values()
        ),
    }
    gate["passed"] = gate == {
        **gate,
        "tasks_complete": 2,
        "programs_replayed": 20,
        "unique_within_each_task": True,
    }
    if not gate["passed"]:
        raise ValueError(f"formula/median structural gate failed: {gate}")
    return {
        "schema_version": "pmo_formula_median_panel_wave_curriculum_v1",
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
        "sources": sources,
        "tasks": tasks,
        "structural_gate": gate,
        "new_oracle_calls": 0,
    }


def locked_queries(curriculum: dict, contract: dict, task: str) -> list[dict]:
    if not curriculum["structural_gate"]["passed"]:
        raise ValueError("cannot query a failed curriculum")
    source_row = curriculum["sources"][task]
    source = decode_state(source_row["state"])
    queries = [
        {
            "role": "charged_task_root",
            "source_role": source_row["source_role"],
            "smiles": source_row["smiles"],
        }
    ]
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
                "source_role": row["source_role"],
                "program_id": program.program_id,
                "compilation_mode": row["compilation_mode"],
                "smiles": receipt["endpoint"],
            }
        )
    if len(queries) != contract["oracle"]["queries_per_task"]:
        raise ValueError(f"locked query count changed for {task}")
    if len({row["smiles"] for row in queries}) != len(queries):
        raise ValueError(f"locked query molecule duplicated for {task}")
    return queries


def prepare(root: Path, output: Path) -> dict:
    output = _output(root, output)
    destination = output / CURRICULUM_NAME
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
    curriculum_path = output / CURRICULUM_NAME
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
    adapters = verified_adapters(root, contract)
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
                    raise ValueError(
                        f"completed query identity changed for {task}/{index}"
                    )
            else:
                if started.exists():
                    raise RuntimeError(f"ambiguous query is not retried: {started}")
                seal(
                    started,
                    {
                        "task": task,
                        "query_index": index,
                        "query": query,
                        "automatic_retry": False,
                    },
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
        "schema_version": "pmo_formula_median_panel_wave_result_v1",
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
            "wins_no_prescreen": sum(
                row["beats_ivg_no_prescreen"] for row in results.values()
            ),
            "wins_prescreen": sum(
                row["beats_ivg_prescreen"] for row in results.values()
            ),
        },
    }
    seal(result_path, payload)
    return payload
