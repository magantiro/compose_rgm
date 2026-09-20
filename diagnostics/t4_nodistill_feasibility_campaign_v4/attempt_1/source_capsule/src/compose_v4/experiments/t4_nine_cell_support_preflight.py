"""Zero-oracle production proposal preflight for the nine missing T4 cells."""

from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from rdkit import rdBase

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.route_distilled_goal_expert import (
    RouteDistilledGoalExpert,
    propose_route_expert_candidates,
)
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.t4_proposal_funnel_audit import (
    assess_endpoint,
    audit_expansion_lane,
    audit_route_candidates,
)

SCHEMA_VERSION = "t4_nine_cell_support_preflight_v1"
EXPERTS = ("shallow", "anchored_replacement", "route_complete_region")
EXPECTED_CHECKPOINT_SHA256 = (
    "cb0d0bd0130b31f956c320ccf8171ce897f797506c8cf1a524a7f697c749865e"
)
EXPECTED_CHECKPOINT_PAYLOAD_SHA256 = (
    "5476be572dd40ee3f068cc8f1df238eec54e23a82cb84f803905ac891701e07d"
)


def identity(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_envelope(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        raise ValueError(f"refusing to overwrite preflight artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(path)
    return envelope["payload_sha256"]


def _write_jsonl_gzip(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ValueError(f"refusing to overwrite preflight ledger: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
    ):
        for row in sorted(rows, key=lambda value: str(value.get("smiles", ""))):
            compressed.write(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
    temporary.replace(path)
    return sha256_file(path)


def _revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _distribution(values: list[int | str]) -> dict[str, int]:
    return dict(sorted(Counter(map(str, values)).items()))


def _margin_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float] | None]:
    result: dict[str, dict[str, float] | None] = {}
    for key in ("similarity_margin", "qed_margin", "sa_margin"):
        values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
        result[key] = (
            {
                "minimum": values[0],
                "median": float(median(values)),
                "maximum": values[-1],
            }
            if values
            else None
        )
    heavy = sorted(
        40 - int(row["heavy"]) for row in rows if row.get("heavy") is not None
    )
    result["heavy_atom_margin"] = (
        {
            "minimum": float(heavy[0]),
            "median": float(median(heavy)),
            "maximum": float(heavy[-1]),
        }
        if heavy
        else None
    )
    return result


def validate_contract(root: Path, contract_path: Path) -> tuple[dict[str, Any], dict]:
    contract = unseal(contract_path)
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected nine-cell preflight schema")
    if contract.get("status") != "AUTHORIZED_ZERO_ORACLE_PRODUCTION_PREFLIGHT":
        raise ValueError("nine-cell preflight is not zero-oracle authorized")
    cells = contract.get("cells")
    expected = {
        *(f"parp1_{seed}_d0p4" for seed in range(3)),
        *(f"braf_{seed}_d0p4" for seed in range(3)),
        *(f"braf_{seed}_d0p6" for seed in range(3)),
    }
    if not isinstance(cells, list) or {row.get("cell_id") for row in cells} != expected:
        raise ValueError("nine-cell preflight cell census drift")
    if len(cells) != 9 or len({row["cell_id"] for row in cells}) != 9:
        raise ValueError("nine-cell preflight cells must be unique")
    if contract.get("experts") != list(EXPERTS):
        raise ValueError("production expert census drift")
    proposal = contract.get("proposal")
    if proposal != {
        "shallow": {"draws": 480, "horizon": 3},
        "anchored_replacement": {"draws": 512, "horizon": 3},
        "route_complete_region": {
            "pool_size": 192,
            "realization_limit": 96,
            "beam_width": 48,
            "expansion_width": 48,
            "max_bindings_per_template": 4,
            "maximum_expansions": 4000,
            "scale_balanced": True,
        },
    }:
        raise ValueError("production proposal settings drift")
    if contract.get("deferred_joint_planning") != {
        "enabled": False,
        "reason": "not_a_production_expert_and_cells_are_not_the_motivating_5ht1b_cell",
    }:
        raise ValueError("deferred-joint exclusion drift")
    if int(contract.get("workers", 0)) < 1 or int(contract["workers"]) > 3:
        raise ValueError("preflight worker bound drift")

    for relative, expected_sha in sorted(contract.get("inputs_sha256", {}).items()):
        actual = sha256_file(root / relative)
        if actual != expected_sha:
            raise ValueError(f"material input mismatch for {relative}: {actual}")
    checkpoint_binding = contract.get("shared_route_checkpoint", {})
    if checkpoint_binding != {
        "path": "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json",
        "sha256": EXPECTED_CHECKPOINT_SHA256,
        "payload_sha256": EXPECTED_CHECKPOINT_PAYLOAD_SHA256,
    }:
        raise ValueError("all-77 shared checkpoint binding drift")
    checkpoint_path = root / checkpoint_binding["path"]
    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("all-77 shared checkpoint physical hash mismatch")
    checkpoint_envelope = json.loads(checkpoint_path.read_text())
    if (
        identity(checkpoint_envelope.get("payload"))
        != EXPECTED_CHECKPOINT_PAYLOAD_SHA256
    ):
        raise ValueError("all-77 shared checkpoint payload hash mismatch")
    checkpoint = checkpoint_envelope["payload"]
    if (
        checkpoint.get("training_routes") != 77
        or checkpoint.get("training_regions") != 147
        or checkpoint.get("runtime_target_conditioning") is not False
        or len(checkpoint.get("expert", {}).get("templates", ())) != 137
    ):
        raise ValueError("all-77 shared checkpoint census drift")

    registry = json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())
    counters: defaultdict[str, int] = defaultdict(int)
    registry_by_cell = {}
    for row in registry:
        target = str(row["target"])
        cell = f"{target}_{counters[target]}"
        counters[target] += 1
        registry_by_cell[cell] = row
    for row in cells:
        source = registry_by_cell.get(row["cell"])
        if source is None or source["idx"] != row["source_global_index"]:
            raise ValueError(f"source registry mismatch for {row['cell_id']}")
        expected_controller_seed = 2026091900 + int(row["source_global_index"])
        if row["controller_seed"] != expected_controller_seed:
            raise ValueError(f"controller seed drift for {row['cell_id']}")
        expected_seeds = {
            expert: expected_controller_seed + 1_000_003 + 101 * index
            for index, expert in enumerate(EXPERTS)
        }
        if row["proposal_seeds"] != expected_seeds:
            raise ValueError(f"first-round proposal seed drift for {row['cell_id']}")
    return contract, checkpoint


def _lane_result(
    summary: dict[str, Any],
    assessments: list[dict[str, Any]],
    *,
    ledger: str,
    ledger_sha256: str,
) -> dict[str, Any]:
    cumulative = summary["cumulative_funnel"]
    eligible = [row for row in assessments if row["compose_valid_pass"]]
    executed = [row for row in assessments if row["parseable"] and row["connected"]]
    lane = summary["lane"]
    size: dict[str, Any]
    if lane == "route_complete_region":
        size = {
            "realized_primitive_counts": _distribution(
                [row["realized_primitives"] for row in assessments]
            ),
            "realized_primitive_bands": _distribution(
                [row.get("realized_primitive_band") for row in assessments]
            ),
            "rewrite_scales": _distribution(
                [row.get("rewrite_scale") for row in assessments]
            ),
            "region_counts": _distribution([row["regions"] for row in assessments]),
        }
        statuses = dict(summary["realization_status_counts"])
        abstentions = {
            "realization_status_counts": statuses,
            "noncommitted_realizations": sum(
                value for key, value in statuses.items() if key != "committed"
            ),
            "proposal_pool_shortfall": int(summary.get("proposal_pool_shortfall", 0)),
        }
    else:
        admitted = [row for row in assessments if row.get("production_candidate")]
        size = {
            "configured_module_horizon": int(summary["horizon"]),
            "admitted_module_counts": _distribution(
                [row["program_module_count"] for row in admitted]
            ),
            "admitted_region_counts": _distribution(
                [row["regions"] for row in admitted]
            ),
            "admitted_created_plus_deleted": _distribution(
                [row["created"] + row["deleted"] for row in admitted]
            ),
        }
        abstentions = {
            "program_synthesis_failures": int(summary["program_synthesis_failures"]),
            "goal_extraction_failures": int(summary["goal_extraction_failures"]),
            "instantiation_failures": int(summary["instantiation_failures"]),
            "distinct_exact_but_ineligible": int(
                cumulative["distinct_executed_endpoints"]
            )
            - int(cumulative["all_compose_valid"]),
        }
    return {
        "expert": lane,
        "proposal_seed": summary.get("proposal_seed"),
        "attempted": int(summary["programs_attempted"]),
        "generated": int(summary["programs_synthesized"]),
        "exact_unique": int(cumulative["distinct_executed_endpoints"]),
        "valid_unique": int(summary["independent_gate_counts"]["structurally_valid"]),
        "eligible_unique": int(cumulative["all_compose_valid"]),
        "exact_execution_precision": (
            float(summary["exact_realization_precision"])
            if "exact_realization_precision" in summary
            else 1.0 if executed else None
        ),
        "gate_funnel": cumulative,
        "failure_combinations": summary["failure_combinations"],
        "margins_all_exact": _margin_summary(executed),
        "margins_eligible": _margin_summary(eligible),
        "scale_and_program_length": size,
        "abstentions": abstentions,
        "wall_seconds": float(summary["elapsed_seconds"]),
        "endpoint_ledger": ledger,
        "endpoint_ledger_sha256": ledger_sha256,
    }


def _cell_worker(
    root_text: str,
    contract_path_text: str,
    spec: dict[str, Any],
    staging_text: str,
) -> dict[str, Any]:
    root = Path(root_text)
    contract_path = Path(contract_path_text)
    staging = Path(staging_text)
    contract, checkpoint = validate_contract(root, contract_path)
    registry = json.loads((root / "docs/GENMOL_T4_SEEDS.json").read_text())
    source_row = registry[int(spec["source_global_index"])]
    smiles = str(source_row["smiles"])
    delta = float(spec["delta"])
    support = str(contract["support"])
    started = perf_counter()
    expert_results = []
    eligible_sets = []
    for expert_name in EXPERTS:
        lane_started = perf_counter()
        if expert_name == "route_complete_region":
            settings = contract["proposal"][expert_name]
            model = RouteDistilledGoalExpert.from_checkpoint(checkpoint["expert"])
            source = pad_molecular_graph(smiles_to_molecular_graph(smiles), 48)
            candidates, telemetry = propose_route_expert_candidates(
                source,
                model,
                pool_size=settings["pool_size"],
                realization_limit=settings["realization_limit"],
                beam_width=settings["beam_width"],
                expansion_width=settings["expansion_width"],
                max_bindings_per_template=settings["max_bindings_per_template"],
                maximum_expansions=settings["maximum_expansions"],
                scale_balanced=settings["scale_balanced"],
            )
            summary, assessments = audit_route_candidates(
                cell=spec["cell"],
                seed_smiles=smiles,
                delta=delta,
                support=support,
                proposed_pool=settings["pool_size"],
                realization_limit=settings["realization_limit"],
                telemetry=telemetry,
                candidates=candidates,
            )
            summary["proposal_seed"] = spec["proposal_seeds"][expert_name]
            summary["elapsed_seconds"] = perf_counter() - lane_started
            summary["proposal_pool_shortfall"] = int(telemetry["candidate_shortfall"])
            denominator = int(telemetry["exact_realization_precision_denominator"])
            summary["exact_realization_precision"] = (
                int(telemetry["exact_realization_precision_numerator"]) / denominator
                if denominator
                else None
            )
        else:
            settings = contract["proposal"][expert_name]
            summary, assessments = audit_expansion_lane(
                cell=spec["cell"],
                seed_smiles=smiles,
                delta=delta,
                support=support,
                lane=expert_name,
                proposal_seed=spec["proposal_seeds"][expert_name],
                draws=settings["draws"],
                horizon=settings["horizon"],
            )
        relative = f"cells/{spec['cell_id']}/{expert_name}.jsonl.gz"
        ledger_sha = _write_jsonl_gzip(staging / relative, assessments)
        result = _lane_result(
            summary,
            assessments,
            ledger=relative,
            ledger_sha256=ledger_sha,
        )
        expert_results.append(result)
        eligible_sets.append(
            {row["smiles"] for row in assessments if row["compose_valid_pass"]}
        )
        print(
            f"[{spec['cell_id']}:{expert_name}] attempted={result['attempted']} "
            f"generated={result['generated']} exact={result['exact_unique']} "
            f"valid={result['valid_unique']} eligible={result['eligible_unique']} "
            f"wall={result['wall_seconds']:.3f}s",
            flush=True,
        )
    source_assessment = assess_endpoint(smiles, smiles, delta=delta, support=support)
    pooled = set().union(*eligible_sets)
    payload = {
        "schema_version": "t4_nine_cell_support_preflight_cell_v1",
        "contract_payload_sha256": identity(contract),
        "cell_id": spec["cell_id"],
        "cell": spec["cell"],
        "source_global_index": spec["source_global_index"],
        "delta": delta,
        "controller_seed": spec["controller_seed"],
        "source_properties": {
            key: source_assessment[key]
            for key in (
                "smiles",
                "similarity",
                "qed",
                "sa",
                "heavy",
                "similarity_margin",
                "qed_margin",
                "sa_margin",
                "compose_valid_pass",
            )
        },
        "registry_source_properties": {
            "seed_qed": source_row["seed_qed"],
            "seed_sa": source_row["seed_sa"],
            "heavy": source_row["heavy"],
        },
        "experts": expert_results,
        "pooled_eligible_unique": len(pooled),
        "pooled_eligible_sha256": identity(sorted(pooled)),
        "gate": {"nonzero_eligible_support": bool(pooled)},
        "wall_seconds": perf_counter() - started,
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
    }
    relative_result = staging / "cells" / spec["cell_id"] / "summary.json"
    _write_envelope(relative_result, payload)
    print(
        f"[{spec['cell_id']}] COMPLETE pooled_eligible={len(pooled)} "
        f"gate={'PASS' if pooled else 'FAIL'} wall={payload['wall_seconds']:.3f}s",
        flush=True,
    )
    return payload


def scientific_projection(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove operational timing only; the remaining payload is byte-stable."""

    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in sorted(value.items())
                if key not in {"wall_seconds", "elapsed_seconds"}
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    projected = clean(payload)
    projected["schema_version"] = "t4_nine_cell_support_preflight_scientific_v1"
    return projected


def aggregate_cells(
    *,
    root: Path,
    contract: dict[str, Any],
    cells: list[dict[str, Any]],
    wall_seconds: float,
) -> dict[str, Any]:
    cells = sorted(cells, key=lambda row: row["cell_id"])
    failed = [
        row["cell_id"] for row in cells if not row["gate"]["nonzero_eligible_support"]
    ]
    expert_totals = {}
    for expert in EXPERTS:
        rows = [
            next(item for item in cell["experts"] if item["expert"] == expert)
            for cell in cells
        ]
        expert_totals[expert] = {
            key: sum(int(row[key]) for row in rows)
            for key in (
                "attempted",
                "generated",
                "exact_unique",
                "valid_unique",
                "eligible_unique",
            )
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_status": "computed_zero_oracle_current_production_support_preflight",
        "scientific_problem": contract["scientific_problem"],
        "primary_model_output": contract["primary_model_output"],
        "central_claim_under_test": contract["central_claim_under_test"],
        "experimental_setting": contract["experimental_setting"],
        "claim_boundary": contract["claim_boundary"],
        "code_revision": _revision(root),
        "contract_payload_sha256": identity(contract),
        "contract_file_sha256": sha256_file(
            root / "configs/t4_nine_cell_support_preflight_v1.json"
        ),
        "inputs_sha256": dict(sorted(contract["inputs_sha256"].items())),
        "shared_route_checkpoint": contract["shared_route_checkpoint"],
        "proposal": contract["proposal"],
        "deferred_joint_planning": False,
        "cells": cells,
        "expert_totals": expert_totals,
        "gate": {
            "complete_nine_cell_census": len(cells) == 9,
            "zero_oracle": True,
            "every_cell_has_nonzero_eligible_support": not failed,
            "zero_eligible_cells": failed,
            "passed": len(cells) == 9 and not failed,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "workers": contract["workers"],
            "precision": "native CPU graph execution",
        },
        "wall_seconds": wall_seconds,
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "gpu_seconds": 0,
        },
    }


def _report(payload: dict[str, Any]) -> str:
    lines = [
        "# Nine-cell T4 production support preflight",
        "",
        f"**Decision: {'PASS' if payload['gate']['passed'] else 'FAIL'}**",
        "",
        (
            "This is a zero-oracle proposal and exact-execution preflight. It used the "
            "all-77 shared checkpoint and the three frozen production experts. It did "
            "not use teacher endpoints, docking, Modal, live runs, or the deferred-joint lane."
        ),
        "",
        "| Cell | Expert | Attempted | Generated | Exact unique | Valid unique | Eligible unique | Wall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in payload["cells"]:
        for expert in cell["experts"]:
            lines.append(
                f"| {cell['cell_id']} | {expert['expert']} | {expert['attempted']} | "
                f"{expert['generated']} | {expert['exact_unique']} | {expert['valid_unique']} | "
                f"{expert['eligible_unique']} | {expert['wall_seconds']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Cell gate",
            "",
            "| Cell | Source QED | Source SA | Source heavy | Pooled eligible unique | Gate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in payload["cells"]:
        source = cell["source_properties"]
        lines.append(
            f"| {cell['cell_id']} | {source['qed']:.6f} | {source['sa']:.6f} | "
            f"{source['heavy']} | {cell['pooled_eligible_unique']} | "
            f"{'PASS' if cell['gate']['nonzero_eligible_support'] else 'FAIL'} |"
        )
    if payload["gate"]["zero_eligible_cells"]:
        lines.extend(
            [
                "",
                "## Negative finding",
                "",
                "The following cells have zero eligible autonomous proposal support: "
                + ", ".join(payload["gate"]["zero_eligible_cells"])
                + ". The preflight fails closed. No scored launch is supported.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "A passing cell establishes only free proposal support inside the official "
                "endpoint fiber. It does not establish docking utility. Full endpoint ledgers "
                "contain the similarity, QED, SA, heavy-atom margins, scale, length and abstention evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_preflight(
    root: Path,
    contract_path: Path,
    output: Path,
    *,
    compare_scientific: Path | None = None,
) -> dict[str, Any]:
    contract, _ = validate_contract(root, contract_path)
    if output.exists():
        raise ValueError(f"refusing to overwrite preflight output: {output}")
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if staging.exists():
        raise ValueError(f"preflight staging path already exists: {staging}")
    staging.mkdir(parents=True)
    started = perf_counter()
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(contract["workers"]), mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    _cell_worker,
                    str(root),
                    str(contract_path),
                    cell,
                    str(staging),
                )
                for cell in contract["cells"]
            ]
            cells = [future.result() for future in futures]
        payload = aggregate_cells(
            root=root,
            contract=contract,
            cells=cells,
            wall_seconds=perf_counter() - started,
        )
        _write_envelope(staging / "result.json", payload)
        scientific = scientific_projection(payload)
        scientific_sha = _write_envelope(staging / "scientific_result.json", scientific)
        if compare_scientific is not None:
            previous = json.loads(compare_scientific.read_text())
            if (
                previous.get("payload_sha256") != scientific_sha
                or previous.get("payload") != scientific
            ):
                raise RuntimeError("scientific rerun is not byte-identical")
        (staging / "INTERPRETATION.md").write_text(_report(payload))
        staging.replace(output)
        return payload
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "EXPERTS",
    "SCHEMA_VERSION",
    "aggregate_cells",
    "run_preflight",
    "scientific_projection",
    "validate_contract",
]
