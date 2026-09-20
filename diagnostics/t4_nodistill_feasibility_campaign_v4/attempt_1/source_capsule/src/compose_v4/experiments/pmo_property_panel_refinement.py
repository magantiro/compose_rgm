"""Panel-informed learned-property PMO programs with exact counted scoring."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.control.edit_program import (
    EditProgram,
    execute_bound_program,
    extract_program,
)
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.pmo_property_program_wave import verified_adapters
from compose_v4.experiments.pmo_target_program_wave import (
    _canonical,
    _stage,
    score_values,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.experiments.winner_paths import PathConfig, find_path_from_state
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

KIND = "pmo_property_panel_refinement"
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
        "src/compose_v4/experiments/pmo_property_panel_refinement.py",
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
    if contract.get("schema_version") != "pmo_property_panel_refinement_contract_v1":
        raise ValueError("unexpected property-panel refinement contract schema")
    if tuple(sorted(contract.get("tasks", {}))) != ("gsk3b", "jnk3", "qed"):
        raise ValueError("property-panel task set changed")
    if contract.get("support") != {
        "active_atoms_max": 40,
        "persistent_slots": 48,
        "formal_charge_changes": False,
        "stereochemistry": "out_of_scope",
        "max_primitives_per_complete_program": 96,
        "max_blocks_per_complete_program": 2,
    }:
        raise ValueError("property-panel support changed")
    oracle = contract.get("oracle", {})
    expected = (11, 10, 33, 10000, 100, True, "0.3.6", "2024.03.5")
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
        raise ValueError("property-panel oracle or metric contract changed")
    verify_file(
        root / contract["source_root"]["path"], contract["source_root"]["sha256"]
    )
    verify_file(
        root / contract["internal_baseline"]["path"],
        contract["internal_baseline"]["sha256"],
    )
    verify_file(root / oracle["forest_manifest_path"], oracle["forest_manifest_sha256"])
    manifest = _read(root / oracle["forest_manifest_path"])
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
            if candidate.get("source_role") not in {
                "exact_panel_transcription",
                "panel_transcription",
                "panel_family_positional_variant",
                "panel_family_substitution_variant",
                "panel_family_variant",
            }:
                raise ValueError(f"unknown evidence role for {name}/{index}")
            canonical.append(smiles)
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"canonical candidate duplication for {name}")
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


def _compile_program(
    source,
    source_payload: dict,
    source_smiles: str,
    target_smiles: str,
    task: str,
    candidate_id: str,
    config: PathConfig,
    contract: dict,
    base_stage: dict | None,
    base_state: dict | None,
    base_smiles: str | None,
) -> tuple[dict, list, dict, str, list[dict]]:
    failures = []
    attempts = []
    if base_stage is not None and base_state is not None and base_smiles is not None:
        suffix = find_path_from_state(base_state, target_smiles, config, base_smiles)
        if suffix["status"] == "witness_found":
            attempts.append(
                (
                    "anchor_suffix",
                    [base_stage, _stage(f"{task}_{candidate_id}_suffix", suffix)],
                )
            )
        else:
            failures.append({"mode": "anchor_suffix", "status": suffix["status"]})
    direct = None
    for mode, stages in attempts:
        try:
            program, assignment = extract_program(source, stages)
            _, receipt = execute_bound_program(
                source,
                program,
                assignment,
                max_primitives=contract["support"][
                    "max_primitives_per_complete_program"
                ],
                max_blocks=contract["support"]["max_blocks_per_complete_program"],
            )
        except (ValueError, RuntimeError) as exc:
            failures.append(
                {"mode": mode, "status": "program_rejected", "reason": str(exc)}
            )
            continue
        if receipt["endpoint"] == target_smiles:
            return program.payload(), list(assignment), receipt, mode, failures
        failures.append({"mode": mode, "status": "endpoint_mismatch"})
    direct = find_path_from_state(source_payload, target_smiles, config, source_smiles)
    if direct["status"] != "witness_found":
        failures.append({"mode": "direct_root", "status": direct["status"]})
        raise ValueError(f"no route for {task}/{candidate_id}: {failures}")
    stages = [_stage(f"{task}_{candidate_id}_direct", direct)]
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
        failures.append(
            {"mode": "direct_root", "status": "program_rejected", "reason": str(exc)}
        )
        raise ValueError(
            f"no executable program for {task}/{candidate_id}: {failures}"
        ) from exc
    if receipt["endpoint"] != target_smiles:
        raise ValueError(f"direct program endpoint changed for {task}/{candidate_id}")
    return program.payload(), list(assignment), receipt, "direct_root", failures


def build_task_programs(
    source_payload: dict,
    source_smiles: str,
    task: str,
    candidates: list[dict],
    contract: dict,
) -> dict:
    config = PathConfig(**contract["compilation"])
    source = decode_state(source_payload)
    canonical_candidates = [
        {
            **row,
            "candidate_id": f"panel_{index:03d}",
            "smiles": _canonical(row["smiles"]),
        }
        for index, row in enumerate(candidates)
    ]
    anchor = canonical_candidates[0]["smiles"]
    base = find_path_from_state(source_payload, anchor, config, source_smiles)
    base_stage = _stage(f"{task}_panel_anchor", base)
    base_state = base["states"][-1]
    programs = []
    failures = []
    for candidate in canonical_candidates:
        if candidate["smiles"] == anchor:
            stages = [base_stage]
            program, assignment = extract_program(source, stages)
            _, receipt = execute_bound_program(
                source,
                program,
                assignment,
                max_primitives=contract["support"][
                    "max_primitives_per_complete_program"
                ],
                max_blocks=contract["support"]["max_blocks_per_complete_program"],
            )
            mode = "anchor_route"
            candidate_failures = []
            payload, assignment_payload = program.payload(), list(assignment)
        else:
            payload, assignment_payload, receipt, mode, candidate_failures = (
                _compile_program(
                    source,
                    source_payload,
                    source_smiles,
                    candidate["smiles"],
                    task,
                    candidate["candidate_id"],
                    config,
                    contract,
                    base_stage,
                    base_state,
                    anchor,
                )
            )
        if receipt["endpoint"] != candidate["smiles"]:
            raise ValueError(
                f"program endpoint changed for {task}/{candidate['candidate_id']}"
            )
        failures.extend(
            {"candidate_id": candidate["candidate_id"], **row}
            for row in candidate_failures
        )
        programs.append(
            {
                **candidate,
                "program": payload,
                "assignment": assignment_payload,
                "receipt": receipt,
                "compilation_mode": mode,
                "primitive_steps": len(receipt["actions"]),
            }
        )
    required = contract["oracle"]["program_outputs_per_task"]
    if (
        len(programs) != required
        or len({row["receipt"]["endpoint"] for row in programs}) != required
    ):
        raise ValueError(f"{task} did not produce {required} unique replayed programs")
    return {
        "task": task,
        "anchor": anchor,
        "base_route": {
            "primitive_steps": len(base["actions"]),
            "actions": base["actions"],
            "states": base["states"],
        },
        "programs": programs,
        "failed_compilation_attempts": failures,
    }


def build_curriculum(root: Path, contract: dict) -> dict:
    source_row = _read(root / contract["source_root"]["path"])["result"]
    source_payload = source_row["source_state"]
    source_smiles = canonical_state_key(decode_state(source_payload))
    tasks = {
        name: build_task_programs(
            source_payload, source_smiles, name, row["candidates"], contract
        )
        for name, row in sorted(contract["tasks"].items())
    }
    gate = {
        "tasks_expected": 3,
        "tasks_complete": len(tasks),
        "programs_expected": 30,
        "programs_replayed": sum(len(row["programs"]) for row in tasks.values()),
        "unique_within_each_task": all(
            len({p["receipt"]["endpoint"] for p in row["programs"]}) == 10
            for row in tasks.values()
        ),
    }
    gate["passed"] = gate == {
        **gate,
        "tasks_complete": 3,
        "programs_replayed": 30,
        "unique_within_each_task": True,
    }
    if not gate["passed"]:
        raise ValueError(f"property-panel structural gate failed: {gate}")
    return {
        "schema_version": "pmo_property_panel_refinement_curriculum_v1",
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
    baseline = unseal(root / contract["internal_baseline"]["path"])
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
        old_auc = baseline["results"][task]["auc_top10_official_10k"]
        summary.update(
            {
                "rows": rows,
                "single_anchor_auc_top10": old_auc,
                "margin_over_single_anchor": auc - old_auc,
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
        "schema_version": "pmo_property_panel_refinement_result_v1",
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
            "improved_over_single_anchor": sum(
                row["margin_over_single_anchor"] > 0 for row in results.values()
            ),
        },
    }
    seal(result_path, payload)
    return payload
