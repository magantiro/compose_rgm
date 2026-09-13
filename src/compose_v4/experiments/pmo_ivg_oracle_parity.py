"""No-reselection PMO rescore under the core oracle versions reported by IVG."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from rdkit import rdBase

from compose_v4.control.program_task import pmo_top_ten_auc
from compose_v4.experiments.continuation_profile import (
    canonical_bytes,
    sha256_file,
    verify_file,
)
from compose_v4.experiments.t4_matched_pilot import seal, unseal

KIND = "pmo_ivg_oracle_parity"
CONTRACT = f"configs/{KIND}.json"
OUTPUT = f"diagnostics/{KIND}"
LOCK_NAME = "query_lock_v3.json"


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
        "src/compose_v4/experiments/pmo_ivg_oracle_parity.py",
        "src/compose_v4/control/program_task.py",
        "src/compose_v4/experiments/continuation_profile.py",
        "src/compose_v4/experiments/t4_matched_pilot.py",
    )
    return {path: sha256_file(root / path) for path in paths}


def _verify_source_inputs(root: Path, contract: dict) -> None:
    for row in contract["inputs"]:
        verify_file(root / row["path"], row["sha256"])
    evidence = contract["official_ivg_environment_evidence"]
    for prefix in ("dockerfile", "requirements", "setup"):
        verify_file(root / evidence[f"{prefix}_path"], evidence[f"{prefix}_sha256"])
    verify_file(
        root / contract["precontract_probe"]["path"],
        contract["precontract_probe"]["sha256"],
    )
    probe = unseal(root / contract["precontract_probe"]["path"])
    if (
        probe.get("accounting", {}).get("oracle_calls") != 1
        or probe.get("accounting", {}).get(
            "included_in_authoritative_150_query_rescore"
        )
        is not False
    ):
        raise ValueError("precontract parity-probe accounting changed")
    failure_row = contract["prequery_asset_gate_failure"]
    verify_file(root / failure_row["path"], failure_row["sha256"])
    failure = unseal(root / failure_row["path"])
    if (
        failure.get("authoritative_oracle_calls_started") != 0
        or failure.get("authoritative_oracle_calls_completed") != 0
        or failure.get("task_score_information_observed") is not False
    ):
        raise ValueError("prequery asset-gate failure accounting changed")
    verify_file(
        root / failure_row["invalidated_query_lock_path"],
        failure_row["invalidated_query_lock_sha256"],
    )
    size_failure_row = contract["prequery_asset_size_gate_failure"]
    verify_file(root / size_failure_row["path"], size_failure_row["sha256"])
    size_failure = unseal(root / size_failure_row["path"])
    if (
        size_failure.get("authoritative_oracle_calls_started") != 0
        or size_failure.get("authoritative_oracle_calls_completed") != 0
        or size_failure.get("task_score_information_observed") is not False
    ):
        raise ValueError("prequery asset-size failure accounting changed")
    verify_file(
        root / size_failure_row["invalidated_query_lock_path"],
        size_failure_row["invalidated_query_lock_sha256"],
    )


def load_contract(root: Path) -> dict:
    contract = _read(root / CONTRACT)
    if contract.get("schema_version") != "pmo_ivg_oracle_parity_contract_v1":
        raise ValueError("unexpected PMO parity contract schema")
    tasks = contract.get("tasks", {})
    expected_tasks = {
        task
        for source in contract.get("inputs", [])
        for task in source.get("tasks", {})
    }
    if set(tasks) != expected_tasks or len(tasks) != 11:
        raise ValueError("PMO parity task set changed")
    if sum(row["queries"] for row in tasks.values()) != 150:
        raise ValueError("PMO parity query count changed")
    source_counts = {
        task: count
        for source in contract["inputs"]
        for task, count in source["tasks"].items()
    }
    if source_counts != {task: row["queries"] for task, row in tasks.items()}:
        raise ValueError("source and task query counts disagree")
    oracle = contract.get("oracle", {})
    if (
        oracle.get("authoritative_query_ceiling") != 150
        or oracle.get("precontract_diagnostic_calls") != 1
        or oracle.get("total_new_parity_environment_calls") != 151
        or oracle.get("automatic_retries") != 0
        or oracle.get("metric_budget") != 10000
        or oracle.get("metric_frequency") != 100
        or oracle.get("finish") is not True
    ):
        raise ValueError("PMO parity oracle or accounting contract changed")
    regime = contract.get("information_regime", {})
    if any(
        regime.get(key) is not False
        for key in (
            "candidate_reselection",
            "candidate_replacement",
            "new_molecule_generation",
            "held_out_claim",
            "autonomous_search_claim",
            "general_pmo_claim",
            "t4_affected",
        )
    ):
        raise ValueError("PMO parity information regime broadened")
    for task, row in tasks.items():
        if row["queries"] < 1:
            raise ValueError(f"invalid query count for {task}")
        for key in ("ivg_no_prescreen_auc_top10", "ivg_prescreen_auc_top10"):
            if not 0 <= row[key] <= 1:
                raise ValueError(f"invalid comparator {key} for {task}")
    _verify_source_inputs(root, contract)
    return contract


def _source_task_rows(source_payload: dict, task: str) -> tuple[list[dict], float]:
    if "results" in source_payload:
        task_result = source_payload["results"][task]
    else:
        task_result = source_payload
        if task != source_payload.get("contract", {}).get("task", "perindopril_mpo"):
            raise ValueError(f"source result does not contain task {task}")
    rows = task_result["rows"]
    normalized = []
    for expected_index, row in enumerate(rows):
        if row.get("query_index") != expected_index:
            raise ValueError(f"noncontiguous source query sequence for {task}")
        query = row.get("query") or {
            "role": row.get("role"),
            "source_id": row.get("source_id"),
            "smiles": row.get("smiles"),
        }
        smiles = query.get("smiles")
        score = row.get("score")
        if not isinstance(smiles, str) or not smiles:
            raise ValueError(f"missing locked molecule for {task}/{expected_index}")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise ValueError(f"invalid prior score for {task}/{expected_index}")
        normalized.append(
            {
                "source_query_index": expected_index,
                "source_query": query,
                "smiles": smiles,
                "prior_protocol_score": float(score),
            }
        )
    return normalized, float(task_result["auc_top10_official_10k"])


def build_query_lock(root: Path, contract: dict) -> dict:
    tasks = {}
    source_identities = {}
    for source in contract["inputs"]:
        source_payload = unseal(root / source["path"])
        source_identities[source["path"]] = source["sha256"]
        for task, expected_count in sorted(source["tasks"].items()):
            if task in tasks:
                raise ValueError(f"task appears in multiple source artifacts: {task}")
            rows, prior_auc = _source_task_rows(source_payload, task)
            if len(rows) != expected_count:
                raise ValueError(f"source query count changed for {task}")
            if len({row["smiles"] for row in rows}) != len(rows):
                raise ValueError(
                    f"source task contains duplicate locked molecules: {task}"
                )
            locked = []
            for parity_index, row in enumerate(rows):
                body = {
                    "task": task,
                    "parity_query_index": parity_index,
                    "source_result_path": source["path"],
                    **row,
                }
                locked.append({**body, "query_id": _identity(body)})
            tasks[task] = {
                "source_result_path": source["path"],
                "source_result_sha256": source["sha256"],
                "prior_protocol_auc_top10": prior_auc,
                "queries": locked,
            }
    total = sum(len(row["queries"]) for row in tasks.values())
    if set(tasks) != set(contract["tasks"]) or total != 150:
        raise ValueError("built PMO parity lock has changed coverage")
    return {
        "schema_version": "pmo_ivg_oracle_parity_query_lock_v1",
        "contract_path": CONTRACT,
        "contract_sha256": sha256_file(root / CONTRACT),
        "implementation_sha256": implementation_sha256(root),
        "code_revision": _git(root, "rev-parse", "HEAD"),
        "code_worktree_status": _git(root, "status", "--porcelain"),
        "source_result_sha256": source_identities,
        "information_regime": contract["information_regime"],
        "task_count": len(tasks),
        "authoritative_query_count": total,
        "tasks": tasks,
        "new_oracle_calls": 0,
    }


def prepare(root: Path, output: Path) -> dict:
    output = _output(root, output)
    destination = output / LOCK_NAME
    if destination.exists():
        raise ValueError(f"parity query lock already exists: {destination}")
    if _git(root, "status", "--porcelain"):
        raise ValueError("PMO parity preparation requires clean committed source")
    payload = build_query_lock(root, load_contract(root))
    seal(destination, payload)
    return payload


def _version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def verify_environment(contract: dict) -> dict[str, str]:
    expected = contract["oracle"]["environment"]
    observed = {
        "pytdc": _version("PyTDC"),
        "rdkit": rdBase.rdkitVersion,
        "numpy": _version("numpy"),
        "scikit_learn": _version("scikit-learn"),
        "pandas": _version("pandas"),
        "scipy": _version("scipy"),
        "seaborn": _version("seaborn"),
        "requests": _version("requests"),
        "setuptools": _version("setuptools"),
    }
    if observed != expected:
        raise ValueError(
            f"IVG-pinned PMO environment mismatch: {observed} != {expected}"
        )
    return observed


def verify_pytdc_sources(contract: dict) -> dict[str, str]:
    import tdc.chem_utils.oracle.oracle as oracle_module
    import tdc.oracles as oracles_module

    distribution = importlib.metadata.distribution("PyTDC")
    metadata_path = Path(
        distribution.locate_file(f"pytdc-{distribution.version}.dist-info/METADATA")
    )
    paths = {
        "pytdc_metadata": metadata_path,
        "tdc_oracles_py": Path(oracles_module.__file__),
        "tdc_oracle_py": Path(oracle_module.__file__),
    }
    expected = contract["oracle"]["source_files"]
    return {name: verify_file(path, expected[name]) for name, path in paths.items()}


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_verified_adapters(
    output: Path, contract: dict
) -> tuple[dict, dict[str, str]]:
    from tdc import Oracle

    asset_root = output / "ivg_oracle_assets"
    with _working_directory(asset_root):
        adapters = {task: Oracle(name=task) for task in sorted(contract["tasks"])}
    assets = {}
    for path, expected in contract["oracle"]["downloaded_assets"].items():
        asset_path = asset_root / path
        if asset_path.stat().st_size != expected["bytes"]:
            raise ValueError(f"downloaded oracle asset size mismatch: {path}")
        assets[path] = {
            "sha256": verify_file(asset_path, expected["sha256"]),
            "bytes": asset_path.stat().st_size,
            "pytdc_dataverse_file_id": expected["pytdc_dataverse_file_id"],
        }
    return adapters, assets


def score_values(values: list[float], contract: dict) -> dict:
    oracle = contract["oracle"]
    curve = []
    for count in range(1, len(values) + 1):
        prefix = sorted(values[:count], reverse=True)[:10]
        curve.append(
            {
                "queries": count,
                "best": max(prefix),
                "top10": sum(prefix) / len(prefix),
            }
        )
    return {
        "oracle_calls": len(values),
        "best_score": max(values),
        "final_top10": curve[-1]["top10"],
        "auc_top10_official_10k": pmo_top_ten_auc(
            values,
            budget=oracle["metric_budget"],
            frequency=oracle["metric_frequency"],
            finish=oracle["finish"],
        ),
        "query_curve": curve,
    }


def run(root: Path, output: Path) -> dict:
    output = _output(root, output)
    contract = load_contract(root)
    lock_path = output / LOCK_NAME
    lock = unseal(lock_path)
    if lock["contract_sha256"] != sha256_file(root / CONTRACT):
        raise ValueError("PMO parity query-lock contract changed")
    if lock["implementation_sha256"] != implementation_sha256(root):
        raise ValueError("PMO parity query-lock implementation changed")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("PMO parity scoring requires unchanged tracked source")
    environment = verify_environment(contract)
    source_files = verify_pytdc_sources(contract)
    adapters, assets = build_verified_adapters(output, contract)
    result_path = output / "result.json"
    if result_path.exists():
        return unseal(result_path)

    results = {}
    total_calls = 0
    asset_root = output / "ivg_oracle_assets"
    with _working_directory(asset_root):
        for task in sorted(lock["tasks"]):
            rows = []
            for query in lock["tasks"][task]["queries"]:
                index = query["parity_query_index"]
                folder = output / "oracle" / task / f"query_{index:06d}"
                started = folder / "started.json"
                completed = folder / "result.json"
                failed = folder / "failure.json"
                if completed.exists():
                    row = unseal(completed)
                    if row["query_id"] != query["query_id"] or row["query"] != query:
                        raise ValueError(
                            f"completed parity query identity changed: {task}/{index}"
                        )
                else:
                    if started.exists() or failed.exists():
                        raise RuntimeError(
                            f"ambiguous or failed parity query is not retried: {folder}"
                        )
                    seal(
                        started,
                        {
                            "task": task,
                            "query_index": index,
                            "query_id": query["query_id"],
                            "query": query,
                            "automatic_retry": False,
                        },
                    )
                    began = perf_counter()
                    try:
                        value = float(adapters[task](query["smiles"]))
                        if not math.isfinite(value) or not 0 <= value <= 1:
                            raise ValueError(f"PMO score outside [0,1]: {value}")
                    except Exception as error:
                        seal(
                            failed,
                            {
                                "task": task,
                                "query_index": index,
                                "query_id": query["query_id"],
                                "query": query,
                                "seconds": perf_counter() - began,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "automatic_retry": False,
                            },
                        )
                        raise
                    row = {
                        "task": task,
                        "query_index": index,
                        "query_id": query["query_id"],
                        "query": query,
                        "prior_protocol_score": query["prior_protocol_score"],
                        "parity_score": value,
                        "score_delta": value - query["prior_protocol_score"],
                        "seconds": perf_counter() - began,
                        "status": "complete",
                    }
                    seal(completed, row)
                rows.append(row)
            values = [row["parity_score"] for row in rows]
            summary = score_values(values, contract)
            comparator = contract["tasks"][task]
            auc = summary["auc_top10_official_10k"]
            summary.update(
                {
                    "rows": rows,
                    "prior_protocol_auc_top10": lock["tasks"][task][
                        "prior_protocol_auc_top10"
                    ],
                    "auc_delta_from_prior_protocol": auc
                    - lock["tasks"][task]["prior_protocol_auc_top10"],
                    "ivg_no_prescreen_auc_top10": comparator[
                        "ivg_no_prescreen_auc_top10"
                    ],
                    "ivg_prescreen_auc_top10": comparator["ivg_prescreen_auc_top10"],
                    "margin_no_prescreen": auc
                    - comparator["ivg_no_prescreen_auc_top10"],
                    "margin_prescreen": auc - comparator["ivg_prescreen_auc_top10"],
                    "beats_ivg_no_prescreen": auc
                    > comparator["ivg_no_prescreen_auc_top10"],
                    "beats_ivg_prescreen": auc > comparator["ivg_prescreen_auc_top10"],
                }
            )
            results[task] = summary
            total_calls += len(rows)
    if total_calls != contract["oracle"]["authoritative_query_ceiling"]:
        raise ValueError("authoritative parity query total differs from contract")
    payload = {
        "schema_version": "pmo_ivg_oracle_parity_result_v1",
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
        },
        "verified_pytdc_source_sha256": source_files,
        "verified_oracle_asset_sha256": assets,
        "authoritative_oracle_calls": total_calls,
        "precontract_diagnostic_calls": contract["oracle"][
            "precontract_diagnostic_calls"
        ],
        "total_new_parity_environment_calls": total_calls
        + contract["oracle"]["precontract_diagnostic_calls"],
        "automatic_retries": 0,
        "results": results,
        "summary": {
            "tasks": len(results),
            "wins_no_prescreen": sum(
                row["beats_ivg_no_prescreen"] for row in results.values()
            ),
            "wins_prescreen": sum(
                row["beats_ivg_prescreen"] for row in results.values()
            ),
            "information_regime": contract["information_regime"]["role"],
        },
    }
    seal(result_path, payload)
    return payload
