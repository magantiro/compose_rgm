"""Winner-informed learned-property PMO programs with exact counted scoring."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

from rdkit import Chem, rdBase

from compose_v4.benchmark.oracles.forest import FrozenForest
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.parent_edit_cycles import pmo_oracle
from compose_v4.experiments.pmo_target_program_wave import (
    build_task_programs,
    locked_queries,
    score_values,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

KIND = "pmo_property_program_wave"
CONTRACT = f"configs/{KIND}.json"
OUTPUT = f"diagnostics/{KIND}"
CURRICULUM_NAME = "curriculum_v2.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _output(root: Path, output: Path) -> Path:
    return output if output.is_absolute() else root / output


def implementation_sha256(root: Path) -> dict[str, str]:
    paths = (
        "src/compose_v4/experiments/pmo_property_program_wave.py",
        "src/compose_v4/experiments/pmo_target_program_wave.py",
        "src/compose_v4/experiments/pmo_winner_program_curriculum.py",
        "src/compose_v4/experiments/parent_edit_cycles.py",
        "src/compose_v4/experiments/winner_paths.py",
        "src/compose_v4/control/edit_program.py",
        "src/compose_v4/control/program_task.py",
        "src/compose_v4/benchmark/oracles/forest.py",
        "src/compose_v4/rewrite/kernel.py",
        "src/compose_v4/rewrite/action_codec_v4.py",
        "src/compose_v4/rewrite/trace_shard.py",
        "src/compose_v4/chem/molecular_graph.py",
    )
    return {path: sha256_file(root / path) for path in paths}


def load_contract(root: Path) -> dict:
    contract = _read(root / CONTRACT)
    if contract.get("schema_version") != "pmo_property_program_wave_contract_v1":
        raise ValueError("unexpected property-program wave contract schema")
    if tuple(sorted(contract.get("tasks", {}))) != ("gsk3b", "jnk3", "qed"):
        raise ValueError("property-program task set changed")
    if contract.get("support") != {
        "active_atoms_max": 40,
        "persistent_slots": 48,
        "formal_charge_changes": False,
        "stereochemistry": "out_of_scope",
        "max_primitives_per_complete_program": 96,
        "max_blocks_per_complete_program": 2,
    }:
        raise ValueError("property-program support changed")
    oracle = contract.get("oracle", {})
    expected = (16, 15, 48, 10000, 100, True, "0.3.6", "2024.03.5")
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
        raise ValueError("property-program oracle or metric contract changed")
    verify_file(
        root / contract["source_root"]["path"], contract["source_root"]["sha256"]
    )
    verify_file(root / oracle["forest_manifest_path"], oracle["forest_manifest_sha256"])
    manifest = _read(root / oracle["forest_manifest_path"])
    for name, row in sorted(contract["tasks"].items()):
        molecule = Chem.MolFromSmiles(row["anchor_smiles"])
        if molecule is None or molecule.GetNumHeavyAtoms() > 40:
            raise ValueError(f"invalid or unsupported property anchor for {name}")
        if any(atom.GetFormalCharge() for atom in molecule.GetAtoms()):
            raise ValueError(f"charged property anchor is outside this wave: {name}")
        if row["oracle_kind"] == "frozen_tdc_forest":
            verify_file(root / row["parameters_path"], row["parameters_sha256"])
            if manifest[name]["parameters_npz_sha256"] != row["parameters_sha256"]:
                raise ValueError(f"forest parameter manifest mismatch for {name}")
            if (
                manifest[name]["provenance"]["pickle_sha256"]
                != row["official_pickle_sha256"]
            ):
                raise ValueError(f"official pickle identity mismatch for {name}")
        elif row["oracle_kind"] != "pytdc_qed":
            raise ValueError(f"unknown oracle kind for {name}")
        for key in ("no_prescreen_auc_top10", "prescreen_auc_top10"):
            if not 0 <= row[key] <= 1:
                raise ValueError(f"invalid {key} comparator for {name}")
    return contract


def build_curriculum(root: Path, contract: dict) -> dict:
    source_row = _read(root / contract["source_root"]["path"])["result"]
    source_payload = source_row["source_state"]
    source_smiles = canonical_state_key(decode_state(source_payload))
    tasks = {
        name: build_task_programs(
            source_payload, source_smiles, name, row["anchor_smiles"], contract
        )
        for name, row in sorted(contract["tasks"].items())
    }
    gate = {
        "tasks_expected": 3,
        "tasks_complete": len(tasks),
        "programs_expected": 45,
        "programs_replayed": sum(len(row["programs"]) for row in tasks.values()),
        "unique_within_each_task": all(
            len({p["receipt"]["endpoint"] for p in row["programs"]}) == 15
            for row in tasks.values()
        ),
    }
    gate["passed"] = gate == {
        **gate,
        "tasks_complete": 3,
        "programs_replayed": 45,
        "unique_within_each_task": True,
    }
    if not gate["passed"]:
        raise ValueError(f"property-program structural gate failed: {gate}")
    return {
        "schema_version": "pmo_property_program_wave_curriculum_v1",
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


def _adapters(root: Path, contract: dict):
    adapters = {}
    for name, row in sorted(contract["tasks"].items()):
        if row["oracle_kind"] == "pytdc_qed":
            adapters[name] = pmo_oracle(name, {name: "pinned_property_program_wave_v1"})
        else:
            forest = FrozenForest(root / row["parameters_path"])
            adapters[name] = forest.score
    return adapters


def verified_adapters(root: Path, contract: dict):
    """Install PyTDC compatibility before importing its pinned oracle source."""

    adapters = _adapters(root, contract)
    import tdc.chem_utils.oracle.oracle as oracle_module

    verify_file(
        Path(oracle_module.__file__), contract["oracle"]["pytdc_oracle_source_sha256"]
    )
    return adapters


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
        "schema_version": "pmo_property_program_wave_result_v1",
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
