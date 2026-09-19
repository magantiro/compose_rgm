"""Fit and audit the shared scale-balanced retained-subgraph proposal expert."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.docking_value import identity
from compose_v4.control.graph_geometry import topology
from compose_v4.control.route_distilled_goal_expert import (
    RouteDistilledGoalExpert,
    make_route_expert,
    propose_route_expert_candidates,
)
from compose_v4.control.structural_subgoal_policy import fit_marginal_subgoal_policy
from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_fiber_campaign import COMPOSE_VALID, Fiber
from compose_v4.rewrite.kernel import canonical_state_key
from tools.t4_structural_subgoal_audit import LIBRARY
from tools.t4_structural_subgoal_policy import _rows, _template_vocabulary

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "t4_shared_retained_rewrite_gate_v1"
CHECKPOINT_SCHEMA = "t4_shared_retained_rewrite_checkpoint_v1"
DEFAULT_OUTPUT = ROOT / "diagnostics/t4_shared_retained_rewrite_v1/gate.json"
DEFAULT_CHECKPOINT = ROOT / "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"
CONTRACT = ROOT / "configs/t4_shared_retained_rewrite_v1.json"


def _publish(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite shared rewrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def _revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _contract() -> dict:
    envelope = json.loads(CONTRACT.read_text())
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(payload):
        raise ValueError("shared retained-rewrite contract is not self-hashed")
    if payload["oracle"]["calls_authorized"] != 0:
        raise ValueError("shared retained-rewrite gate must remain zero-oracle")
    for row in payload["inputs"].values():
        path = ROOT / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"shared retained-rewrite input changed: {path}")
    return payload


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _payload_keys(value) -> set[str]:
    if isinstance(value, dict):
        keys = set(map(str, value))
        for row in value.values():
            keys.update(_payload_keys(row))
        return keys
    if isinstance(value, list):
        keys = set()
        for row in value:
            keys.update(_payload_keys(row))
        return keys
    return set()


def _build_expert() -> tuple[RouteDistilledGoalExpert, list[dict], dict[str, dict]]:
    routes, metadata = _rows()
    if len(routes) != 77 or len({row["source_group"] for row in routes}) != 15:
        raise RuntimeError("locked T4 route census changed")
    marginal = fit_marginal_subgoal_policy(routes, exploration_floor=0.10)
    templates = _template_vocabulary(routes)
    training_evidence_identity = identity(
        {
            "schema_version": "shared_retained_rewrite_training_identity_v1",
            "route_corpus_sha256": sha256_file(LIBRARY),
            "source_groups": sorted(row["source_group"] for row in routes),
            "template_ids": sorted(
                template.template_id for row in routes for template in row["templates"]
            ),
        }
    )
    return (
        make_route_expert(
            templates,
            marginal,
            training_evidence_identity=training_evidence_identity,
        ),
        routes,
        metadata,
    )


def _seeds() -> list[dict]:
    raw = json.loads((ROOT / "docs/GENMOL_T4_SEEDS.json").read_text())
    counters: defaultdict[str, int] = defaultdict(int)
    rows = []
    for seed in raw:
        target = str(seed["target"])
        source_index = counters[target]
        counters[target] += 1
        rows.append({**seed, "cell": f"{target}_{source_index}"})
    if len(rows) != 15 or any(count != 3 for count in counters.values()):
        raise RuntimeError("T4 seed census changed")
    return rows


def _teacher_endpoints_by_source(routes: list[dict]) -> dict[str, set[str]]:
    endpoints: defaultdict[str, set[str]] = defaultdict(set)
    for row in routes:
        endpoints[canonical_state_key(row["source"])].add(canonical_state_key(row["endpoint"]))
    return dict(endpoints)


def _candidate_geometry(source, candidate: dict) -> dict:
    endpoint = pad_molecular_graph(smiles_to_molecular_graph(candidate["smiles"]), source.n_atoms)
    before = topology(source)
    after = topology(endpoint)
    return {
        "heavy_atoms": after["n_heavy"],
        "delta_heavy_atoms": after["n_heavy"] - before["n_heavy"],
        "cycle_rank": after["cycle_rank"],
        "delta_cycle_rank": after["cycle_rank"] - before["cycle_rank"],
        "endpoint_key": canonical_state_key(endpoint),
    }


def run(
    output: Path,
    checkpoint: Path,
    *,
    selected_cells: set[str] | None = None,
    deltas: tuple[float, ...] = (0.4, 0.6),
    pool_size: int = 192,
    realization_limit: int = 96,
    beam_width: int = 48,
    expansion_width: int = 48,
    max_bindings_per_template: int = 4,
    maximum_expansions: int = 4_000,
    workers: int = 1,
) -> dict:
    started = perf_counter()
    contract = _contract()
    expected_proposal = contract["proposal"]
    observed_proposal = {
        "pool_size": pool_size,
        "realization_limit": realization_limit,
        "beam_width": beam_width,
        "expansion_width": expansion_width,
        "max_bindings_per_template": max_bindings_per_template,
        "maximum_expansions": maximum_expansions,
    }
    if any(expected_proposal[key] != value for key, value in observed_proposal.items()):
        raise ValueError("runtime proposal settings disagree with the frozen contract")
    if tuple(map(float, contract["deltas"])) != deltas:
        raise ValueError("runtime deltas disagree with the frozen contract")
    if selected_cells is None and workers != int(contract["workers"]):
        raise ValueError("authoritative worker count disagrees with the frozen contract")
    expert, routes, _ = _build_expert()
    checkpoint_payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "expert": expert.checkpoint(),
        "training_scope": "all_locked_t4_routes_shared_task_independent",
        "training_routes": len(routes),
        "training_regions": sum(len(row["templates"]) for row in routes),
        "runtime_target_conditioning": False,
        "new_oracle_calls": 0,
    }
    forbidden_keys = {
        "route_id",
        "source_group",
        "endpoint",
        "smiles",
        "target_name",
        "cell",
        "target",
        "source_graph",
    }
    leaked_keys = forbidden_keys & _payload_keys(checkpoint_payload)
    if leaked_keys:
        raise RuntimeError(f"runtime checkpoint leaked forbidden fields {sorted(leaked_keys)}")
    _publish(checkpoint, checkpoint_payload)

    teacher_endpoints = _teacher_endpoints_by_source(routes)
    seeds = [seed for seed in _seeds() if selected_cells is None or seed["cell"] in selected_cells]
    if not seeds:
        raise ValueError("selected T4 cell set is empty")

    def evaluate(seed: dict) -> dict:
        cell_started = perf_counter()
        source = pad_molecular_graph(smiles_to_molecular_graph(seed["smiles"]), 48)
        candidates, telemetry = propose_route_expert_candidates(
            source,
            expert,
            pool_size=pool_size,
            realization_limit=realization_limit,
            beam_width=beam_width,
            expansion_width=expansion_width,
            max_bindings_per_template=max_bindings_per_template,
            maximum_expansions=maximum_expansions,
            scale_balanced=True,
        )
        source_key = canonical_state_key(source)
        expected_endpoints = teacher_endpoints.get(source_key, set())
        rows = []
        exact_teacher_endpoint_recovery = 0
        primitive_bands = Counter()
        rewrite_bands = Counter()
        for candidate in candidates:
            geometry = _candidate_geometry(source, candidate)
            exact_teacher = geometry["endpoint_key"] in expected_endpoints
            exact_teacher_endpoint_recovery += int(exact_teacher)
            primitive_bands[candidate["realized_primitive_band"]] += 1
            rewrite_bands[candidate["rewrite_scale"]] += 1
            eligibility = {}
            for delta in deltas:
                properties = Fiber(seed["smiles"], delta, support=COMPOSE_VALID).check(
                    candidate["smiles"]
                )
                eligibility[str(delta)] = {
                    "eligible": properties is not None,
                    "properties": properties,
                }
            rows.append(
                {
                    **candidate,
                    **geometry,
                    "exact_teacher_endpoint": exact_teacher,
                    "eligibility": eligibility,
                }
            )
        return {
            "cell": seed["cell"],
            "source_global_index": seed["idx"],
            "source_heavy_atoms": source.n_real_atoms,
            "known_teacher_endpoints": len(expected_endpoints),
            "exact_teacher_endpoint_recovery": exact_teacher_endpoint_recovery,
            "complete_programs_committed": len(rows),
            "primitive_band_counts": dict(sorted(primitive_bands.items())),
            "rewrite_band_counts": dict(sorted(rewrite_bands.items())),
            "eligible_by_delta": {
                str(delta): sum(row["eligibility"][str(delta)]["eligible"] for row in rows)
                for delta in deltas
            },
            "maximum_heavy_atoms": max((row["heavy_atoms"] for row in rows), default=None),
            "maximum_cycle_rank_increase": max(
                (row["delta_cycle_rank"] for row in rows), default=None
            ),
            "telemetry": telemetry,
            "wall_seconds": perf_counter() - cell_started,
            "candidates": rows,
        }

    if workers < 1:
        raise ValueError("worker count must be positive")
    if workers == 1:
        cells = [evaluate(seed) for seed in seeds]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(seeds))) as executor:
            cells = list(executor.map(evaluate, seeds))

    parp1_teacher_scale = any(
        row["cell"] == "parp1_1"
        and any(
            candidate["realized_primitive_band"] == "large"
            and candidate["heavy_atoms"] >= 27
            and candidate["delta_cycle_rank"] >= 2
            for candidate in row["candidates"]
        )
        for row in cells
    )
    global_primitive_bands = Counter(
        candidate["realized_primitive_band"] for row in cells for candidate in row["candidates"]
    )
    gate = {
        "same_task_independent_checkpoint_every_cell": True,
        "complete_program_committed_every_selected_cell": all(
            row["complete_programs_committed"] > 0 for row in cells
        ),
        "exact_realization_precision_one": all(
            row["telemetry"]["exact_realization_precision_numerator"]
            == row["telemetry"]["exact_realization_precision_denominator"]
            for row in cells
        ),
        "small_medium_large_realized": all(
            global_primitive_bands[band] > 0 for band in ("small", "medium", "large")
        ),
        "parp1_teacher_scale_geometry_realized": parp1_teacher_scale,
    }
    gate["passed"] = bool(cells) and all(gate.values())
    code_inputs = (
        "src/compose_v4/control/route_distilled_goal_expert.py",
        "src/compose_v4/control/structural_subgoal_policy.py",
        "src/compose_v4/control/complete_region_program.py",
        "src/compose_v4/control/structural_subgoal_realizer.py",
        "tools/t4_shared_retained_rewrite_gate.py",
    )
    payload = {
        "schema_version": SCHEMA,
        "evidence": "zero-oracle actual-production shared retained-subgraph proposal gate",
        "code_revision": _revision(),
        "working_tree_dirty": True,
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_payload_sha256": identity(contract),
        "inputs_sha256": {
            "route_corpus": sha256_file(LIBRARY),
            "source_registry": sha256_file(ROOT / "docs/GENMOL_T4_SEEDS.json"),
            **{path: sha256_file(ROOT / path) for path in code_inputs},
        },
        "checkpoint": _display_path(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_payload_sha256": identity(checkpoint_payload),
        "configuration": {
            "selected_cells": None if selected_cells is None else sorted(selected_cells),
            "deltas": list(deltas),
            "pool_size": pool_size,
            "realization_limit": realization_limit,
            "beam_width": beam_width,
            "expansion_width": expansion_width,
            "max_bindings_per_template": max_bindings_per_template,
            "maximum_expansions": maximum_expansions,
            "workers": workers,
            "scale_balanced": True,
            "realized_primitive_bands": {
                "small": [1, 3],
                "medium": [4, 11],
                "large": [12, 32],
            },
        },
        "cells": cells,
        "global_primitive_band_counts": dict(sorted(global_primitive_bands.items())),
        "gate": gate,
        "wall_seconds": perf_counter() - started,
        "costs": {
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
            "gpu_seconds": 0,
        },
    }
    _publish(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--cells",
        help="optional comma-separated T4 cells; default is all fifteen",
    )
    parser.add_argument("--pool-size", type=int, default=192)
    parser.add_argument("--realization-limit", type=int, default=96)
    parser.add_argument("--beam-width", type=int, default=48)
    parser.add_argument("--expansion-width", type=int, default=48)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    selected = None if not args.cells else set(args.cells.split(","))
    result = run(
        args.output.resolve(),
        args.checkpoint.resolve(),
        selected_cells=selected,
        pool_size=args.pool_size,
        realization_limit=args.realization_limit,
        beam_width=args.beam_width,
        expansion_width=args.expansion_width,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "gate": result["gate"],
                "global_primitive_band_counts": result["global_primitive_band_counts"],
                "cells": [
                    {
                        key: row[key]
                        for key in (
                            "cell",
                            "complete_programs_committed",
                            "primitive_band_counts",
                            "eligible_by_delta",
                            "exact_teacher_endpoint_recovery",
                            "maximum_heavy_atoms",
                            "maximum_cycle_rank_increase",
                            "wall_seconds",
                        )
                    }
                    for row in result["cells"]
                ],
                "wall_seconds": result["wall_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
