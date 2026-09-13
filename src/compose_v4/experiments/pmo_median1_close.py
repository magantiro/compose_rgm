"""One-query prospective closeout of the Median1 PMO development gap."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.edit_program import EditProgram, execute_bound_program
from compose_v4.experiments.continuation_profile import (
    canonical_bytes,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.pmo_ivg_oracle_parity import (
    load_contract as load_parity_contract,
)
from compose_v4.experiments.pmo_ivg_oracle_parity import (
    score_values,
    verify_environment,
    verify_pytdc_sources,
)
from compose_v4.experiments.pmo_property_panel_refinement import build_task_programs
from compose_v4.experiments.pmo_target_program_wave import _canonical
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import encode_state

KIND = "pmo_median1_close"
CONTRACT = f"configs/{KIND}.json"
OUTPUT = f"diagnostics/{KIND}"
LOCK_NAME = "query_lock.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _output(root: Path, output: Path) -> Path:
    return output if output.is_absolute() else root / output


def _identity(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def implementation_sha256(root: Path) -> dict[str, str]:
    paths = (
        "src/compose_v4/experiments/pmo_median1_close.py",
        "src/compose_v4/experiments/pmo_ivg_oracle_parity.py",
        "src/compose_v4/experiments/pmo_property_panel_refinement.py",
        "src/compose_v4/experiments/pmo_target_program_wave.py",
        "src/compose_v4/control/edit_program.py",
        "src/compose_v4/control/program_task.py",
        "src/compose_v4/rewrite/kernel.py",
        "src/compose_v4/rewrite/trace_shard.py",
        "src/compose_v4/chem/molecular_graph.py",
        "src/compose_v4/chem/state.py",
    )
    return {path: sha256_file(root / path) for path in paths}


def load_contract(root: Path) -> dict:
    contract = _read(root / CONTRACT)
    if contract.get("schema_version") != "pmo_median1_close_contract_v1":
        raise ValueError("unexpected Median1 closeout contract schema")
    if contract.get("support") != {
        "active_atoms_max": 40,
        "persistent_slots": 48,
        "formal_charge_changes": False,
        "stereochemistry": "out_of_scope",
        "max_primitives_per_complete_program": 96,
        "max_blocks_per_complete_program": 2,
    }:
        raise ValueError("Median1 closeout support changed")
    oracle = contract.get("oracle", {})
    if oracle != {
        "task": "median1",
        "new_query_ceiling": 1,
        "cumulative_query_count": 12,
        "automatic_retries": 0,
        "metric_budget": 10000,
        "metric_frequency": 100,
        "finish": True,
    }:
        raise ValueError("Median1 closeout oracle contract changed")
    regime = contract.get("information_regime", {})
    if regime.get("candidate_score_observed_before_lock") is not False:
        raise ValueError("Median1 candidate must remain unscored at lock time")
    if any(
        regime.get(key) is not False
        for key in (
            "held_out_claim",
            "autonomous_search_claim",
            "general_pmo_claim",
            "t4_affected",
        )
    ):
        raise ValueError("Median1 closeout claim boundary broadened")
    source = contract["source_root"]
    candidate = contract["candidate"]
    source_smiles = _canonical(source["smiles"])
    candidate_smiles = _canonical(candidate["smiles"])
    if source.get("source_role") != "predeclared_task_specific_generic_root":
        raise ValueError("Median1 source role changed")
    if candidate.get("source_role") != "pre_score_panel_series_extrapolation":
        raise ValueError("Median1 candidate role changed")
    if source_smiles == candidate_smiles:
        raise ValueError("Median1 source and candidate duplicate")
    for label, smiles in (("source", source_smiles), ("candidate", candidate_smiles)):
        graph = smiles_to_molecular_graph(smiles)
        if graph.n_real_atoms > 40 or any(graph.formal_charges):
            raise ValueError(f"unsupported Median1 {label}")
    verify_file(
        root / contract["prior_result"]["path"],
        contract["prior_result"]["sha256"],
    )
    verify_file(
        root / contract["parity_contract"]["path"],
        contract["parity_contract"]["sha256"],
    )
    prior = unseal(root / contract["prior_result"]["path"])
    task = prior["results"]["median1"]
    if len(task["rows"]) != contract["prior_result"]["expected_queries"]:
        raise ValueError("Median1 prior chronology changed")
    if (
        task["auc_top10_official_10k"]
        != contract["baseline_and_ablation"]["prior_auc_top10"]
    ):
        raise ValueError("Median1 prior AUC changed")
    return contract


def build_query_lock(root: Path, contract: dict) -> dict:
    source_smiles = _canonical(contract["source_root"]["smiles"])
    source = pad_molecular_graph(smiles_to_molecular_graph(source_smiles), 48)
    source_payload = encode_state(source)
    source_endpoint = canonical_state_key(source)
    compile_contract = {
        **contract,
        "oracle": {"program_outputs_per_task": 1},
    }
    task = build_task_programs(
        source_payload,
        source_endpoint,
        "median1",
        [contract["candidate"]],
        compile_contract,
    )
    row = task["programs"][0]
    program = EditProgram.from_payload(row["program"])
    _, receipt = execute_bound_program(
        source,
        program,
        tuple(row["assignment"]),
        max_primitives=contract["support"]["max_primitives_per_complete_program"],
        max_blocks=contract["support"]["max_blocks_per_complete_program"],
    )
    if receipt != row["receipt"]:
        raise ValueError("Median1 closeout replay changed")
    expected = _canonical(contract["candidate"]["smiles"])
    if receipt["endpoint"] != expected:
        raise ValueError("Median1 closeout endpoint changed")
    body = {
        "task": "median1",
        "query_index": 0,
        "smiles": expected,
        "source_role": contract["candidate"]["source_role"],
        "selection_rule": contract["candidate"]["selection_rule"],
        "program_id": program.program_id,
    }
    return {
        "schema_version": "pmo_median1_close_query_lock_v1",
        "contract_path": CONTRACT,
        "contract_sha256": sha256_file(root / CONTRACT),
        "implementation_sha256": implementation_sha256(root),
        "code_revision": _git(root, "rev-parse", "HEAD"),
        "code_worktree_status": _git(root, "status", "--porcelain"),
        "information_regime": contract["information_regime"],
        "source": {
            "smiles": source_endpoint,
            "state": source_payload,
            "source_role": contract["source_root"]["source_role"],
        },
        "program": row["program"],
        "assignment": row["assignment"],
        "receipt": receipt,
        "compilation_mode": row["compilation_mode"],
        "query": {**body, "query_id": _identity(body)},
        "new_oracle_calls": 0,
    }


def prepare(root: Path, output: Path) -> dict:
    output = _output(root, output)
    destination = output / LOCK_NAME
    if destination.exists():
        raise ValueError(f"Median1 closeout lock already exists: {destination}")
    if _git(root, "status", "--porcelain"):
        raise ValueError("Median1 closeout preparation requires clean committed source")
    payload = build_query_lock(root, load_contract(root))
    seal(destination, payload)
    return payload


def _prior_values(root: Path, contract: dict) -> list[float]:
    prior = unseal(root / contract["prior_result"]["path"])
    rows = prior["results"]["median1"]["rows"]
    values = [float(row["parity_score"]) for row in rows]
    if len(values) != 11:
        raise ValueError("Median1 prior score count changed")
    return values


def run(root: Path, output: Path) -> dict:
    output = _output(root, output)
    contract = load_contract(root)
    lock_path = output / LOCK_NAME
    lock = unseal(lock_path)
    if lock["contract_sha256"] != sha256_file(root / CONTRACT):
        raise ValueError("Median1 closeout lock contract changed")
    if lock["implementation_sha256"] != implementation_sha256(root):
        raise ValueError("Median1 closeout lock implementation changed")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Median1 scoring requires unchanged tracked source")

    parity_contract = load_parity_contract(root)
    environment = verify_environment(parity_contract)
    source_files = verify_pytdc_sources(parity_contract)
    result_path = output / "result.json"
    if result_path.exists():
        return unseal(result_path)

    from tdc import Oracle

    query = lock["query"]
    folder = output / "oracle" / "query_000000"
    started = folder / "started.json"
    completed = folder / "result.json"
    failed = folder / "failure.json"
    if started.exists() or completed.exists() or failed.exists():
        raise RuntimeError("Median1 closeout query has prior state and is not retried")
    seal(
        started,
        {
            "task": "median1",
            "query_index": 0,
            "query_id": query["query_id"],
            "query": query,
            "automatic_retry": False,
        },
    )
    began = perf_counter()
    try:
        value = float(Oracle(name="median1")(query["smiles"]))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Median1 score outside [0,1]: {value}")
    except Exception as error:
        seal(
            failed,
            {
                "task": "median1",
                "query_index": 0,
                "query_id": query["query_id"],
                "query": query,
                "seconds": perf_counter() - began,
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_retry": False,
            },
        )
        raise
    new_row = {
        "task": "median1",
        "query_index": 0,
        "query_id": query["query_id"],
        "query": query,
        "score": value,
        "seconds": perf_counter() - began,
        "status": "complete",
    }
    seal(completed, new_row)

    prior_values = _prior_values(root, contract)
    cumulative = score_values(prior_values + [value], parity_contract)
    comparator = contract["baseline_and_ablation"]
    auc = cumulative["auc_top10_official_10k"]
    payload = {
        "schema_version": "pmo_median1_close_result_v1",
        "contract": contract,
        "contract_sha256": sha256_file(root / CONTRACT),
        "query_lock_path": str(lock_path.relative_to(root)),
        "query_lock_sha256": sha256_file(lock_path),
        "implementation_sha256": implementation_sha256(root),
        "code_revision": _git(root, "rev-parse", "HEAD"),
        "software": {
            **environment,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "rdkit_runtime": rdBase.rdkitVersion,
            "pytdc_runtime": importlib.metadata.version("PyTDC"),
        },
        "verified_pytdc_source_sha256": source_files,
        "new_oracle_calls": 1,
        "automatic_retries": 0,
        "new_observation": new_row,
        "prior_query_count": len(prior_values),
        "cumulative": cumulative,
        "prior_auc_top10": comparator["prior_auc_top10"],
        "auc_improvement": auc - comparator["prior_auc_top10"],
        "ivg_no_prescreen_auc_top10": comparator["ivg_no_prescreen_auc_top10"],
        "ivg_prescreen_auc_top10": comparator["ivg_prescreen_auc_top10"],
        "margin_no_prescreen": auc - comparator["ivg_no_prescreen_auc_top10"],
        "margin_prescreen": auc - comparator["ivg_prescreen_auc_top10"],
        "beats_ivg_no_prescreen": auc > comparator["ivg_no_prescreen_auc_top10"],
        "beats_ivg_prescreen": auc > comparator["ivg_prescreen_auc_top10"],
        "information_regime": contract["information_regime"],
    }
    seal(result_path, payload)
    return payload
