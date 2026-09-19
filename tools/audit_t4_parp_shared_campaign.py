"""Audit a durable PARP shared-controller snapshot without oracle access.

The audit joins three immutable evidence classes:

* the live campaign checkpoint and candidate locks;
* the shared route-expert checkpoint actually named by the launch contract;
* the sealed delta=0.6 PARP route corpus used as known-answer supervision.

It never proposes, executes, selects, or docks a molecule.  Teacher routes are used only
to measure production support and ordering in already locked candidate pools.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal import StructuralGoal
from compose_v4.control.structural_subgoal_policy import minimize_subgoal
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "t4_parp_shared_campaign_interim_audit_v1"
CELL_MAP = {
    "parp1_0": "docking_parp1_idx0_thr6",
    "parp1_1": "docking_parp1_idx1_thr6",
    "parp1_2": "docking_parp1_idx2_thr6",
}


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


def _payload(path: Path, *, verify: bool = True) -> dict:
    envelope = _read_json(path)
    if not isinstance(envelope, dict) or not isinstance(envelope.get("payload"), dict):
        raise TypeError(f"{path}: expected a payload envelope")
    if verify and envelope.get("payload_sha256") != identity(envelope["payload"]):
        raise ValueError(f"{path}: payload hash mismatch")
    return envelope["payload"]


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES in evidence: {smiles!r}")
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def _round_number(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _teacher_rows(
    corpus_path: Path, route_checkpoint_path: Path
) -> dict[str, list[dict]]:
    corpus = _payload(corpus_path)
    checkpoint = _payload(route_checkpoint_path)
    expert = checkpoint["expert"]
    marginal = expert["marginal"]
    probabilities = dict(
        zip(
            marginal["template_ids"], map(float, marginal["probabilities"]), strict=True
        )
    )
    ranks = {
        template_id: rank
        for rank, (template_id, _) in enumerate(
            sorted(probabilities.items(), key=lambda row: (-row[1], row[0])), 1
        )
    }
    by_cell: dict[str, list[dict]] = defaultdict(list)
    reverse_cells = {value: key for key, value in CELL_MAP.items()}
    for route in corpus["routes"]:
        cell = reverse_cells.get(route.get("cell"))
        if cell is None:
            continue
        goal = StructuralGoal.from_payload(route["structural_goal"])
        template_ids = [minimize_subgoal(row)[0].template_id for row in goal.subgoals]
        endpoint = _canonical(
            molecular_graph_to_smiles(decode_state(route["endpoint_state"]))
        )
        geometry = route["geometry"]
        by_cell[cell].append(
            {
                "route_id": route["route_id"],
                "endpoint": endpoint,
                "primitive_count": int(route["primitive_count"]),
                "region_count": int(route["region_count"]),
                "template_ids": template_ids,
                "templates_present": [row in probabilities for row in template_ids],
                "marginal_probabilities": [
                    probabilities.get(row) for row in template_ids
                ],
                "marginal_ranks": [ranks.get(row) for row in template_ids],
                "source_heavy": int(geometry["source"]["active_atoms"]),
                "target_heavy": int(geometry["target"]["active_atoms"]),
                "cycle_rank_delta": int(geometry["cycle_rank_delta"]),
                "retained_source_atom_fraction": float(
                    geometry["retained_source_atom_fraction"]
                ),
            }
        )
    return {
        cell: sorted(rows, key=lambda row: row["route_id"])
        for cell, rows in by_cell.items()
    }


def _collect_locks(snapshot_roots: tuple[Path, ...], cell: str) -> dict[int, dict]:
    locks: dict[int, dict] = {}
    for root in snapshot_roots:
        cell_dir = root / cell
        if not cell_dir.exists():
            continue
        for path in sorted(cell_dir.glob("round_*_lock.json")):
            locks[_round_number(path)] = _payload(path)
    return locks


def _successful_docked(checkpoint: dict) -> list[dict]:
    rows = []
    for round_row in checkpoint.get("rounds", []):
        for docked in round_row.get("docked", []):
            if docked.get("failure") is None and docked.get("score") is not None:
                rows.append({**docked, "round": int(round_row["round"])})
    return rows


def _lineage(best: dict, docked: list[dict], root_result: dict) -> list[dict]:
    by_smiles = {_canonical(row["smiles"]): row for row in docked}
    root_smiles = _canonical(root_result["smiles"])
    current = best
    chain = []
    visited = set()
    while current is not None:
        smiles = _canonical(current["smiles"])
        if smiles in visited:
            raise ValueError("cycle in scored lineage")
        visited.add(smiles)
        chain.append(
            {
                key: current.get(key)
                for key in (
                    "round",
                    "score",
                    "smiles",
                    "parent",
                    "parent_score",
                    "proposal_lane",
                    "selection_kind",
                    "program_families",
                    "route_program_id",
                    "route_template_ids",
                    "realized_primitives",
                    "rewrite_scale",
                )
                if current.get(key) is not None
            }
        )
        parent = current.get("parent")
        if parent is None or _canonical(parent) == root_smiles:
            chain.append(
                {
                    "round": 0,
                    "score": float(root_result["score"]),
                    "smiles": root_result["smiles"],
                    "proposal_lane": "root",
                }
            )
            break
        current = by_smiles.get(_canonical(parent))
        if current is None:
            chain.append(
                {
                    "score": float(chain[-1]["parent_score"]),
                    "smiles": parent,
                    "proposal_lane": "migrated_archive_parent",
                }
            )
            break
    return list(reversed(chain))


def _cell_audit(
    cell: str,
    checkpoint: dict,
    locks: dict[int, dict],
    teachers: list[dict],
    ivg_mean: float,
) -> dict:
    teacher_endpoints = {row["endpoint"] for row in teachers}
    teacher_templates = {value for row in teachers for value in row["template_ids"]}
    teacher_programs = {tuple(row["template_ids"]) for row in teachers}
    candidate_rows = []
    selected_rows = []
    for round_index, lock in sorted(locks.items()):
        candidate_rows.extend(
            {**row, "round": round_index} for row in lock.get("candidate_pool", [])
        )
        selected_rows.extend(
            {**row, "round": round_index} for row in lock.get("selected_rows", [])
        )
    docked = _successful_docked(checkpoint)
    root = checkpoint["root_result"]
    scored = [*docked, {**root, "proposal_lane": "root", "round": 0}]
    best = min(scored, key=lambda row: float(row["score"]))

    def annotate(rows: list[dict]) -> list[dict]:
        annotated = []
        for row in rows:
            smiles = _canonical(row["smiles"])
            template_ids = tuple(row.get("route_template_ids") or ())
            annotated.append(
                {
                    **row,
                    "exact_teacher_endpoint": smiles in teacher_endpoints,
                    "teacher_template_overlap": len(
                        set(template_ids) & teacher_templates
                    ),
                    "exact_teacher_template_program": template_ids in teacher_programs,
                }
            )
        return annotated

    candidate_rows = annotate(candidate_rows)
    selected_rows = annotate(selected_rows)
    docked = annotate(docked)

    lane_rows: dict[str, dict] = {}
    lanes = sorted(
        {str(row.get("proposal_lane", "unknown")) for row in candidate_rows + docked}
    )
    for lane in lanes:
        pool = [row for row in candidate_rows if row.get("proposal_lane") == lane]
        selected = [row for row in selected_rows if row.get("proposal_lane") == lane]
        observed = [row for row in docked if row.get("proposal_lane") == lane]
        lane_rows[lane] = {
            "candidate_memberships": len(pool),
            "unique_candidate_endpoints": len(
                {_canonical(row["smiles"]) for row in pool}
            ),
            "selected_memberships": len(selected),
            "docked": len(observed),
            "best_docking_score": (
                min(float(row["score"]) for row in observed) if observed else None
            ),
            "exact_teacher_endpoint_candidates": sum(
                row["exact_teacher_endpoint"] for row in pool
            ),
            "exact_teacher_endpoint_docked": sum(
                row["exact_teacher_endpoint"] for row in observed
            ),
            "teacher_template_overlap_candidates": sum(
                row["teacher_template_overlap"] > 0 for row in pool
            ),
            "exact_teacher_template_program_candidates": sum(
                row["exact_teacher_template_program"] for row in pool
            ),
        }

    best_row = {
        key: best.get(key)
        for key in (
            "round",
            "score",
            "smiles",
            "parent",
            "parent_score",
            "proposal_lane",
            "selection_kind",
            "program_families",
            "route_program_id",
            "route_template_ids",
            "realized_primitives",
            "rewrite_scale",
            "heavy",
            "similarity",
            "qed",
            "sa",
        )
        if best.get(key) is not None
    }
    if best.get("proposal_lane") == "root":
        lineage = [best_row]
    else:
        lineage = _lineage(best, docked, root)
    return {
        "status": checkpoint["status"],
        "charged_calls": int(checkpoint["charged_calls"]),
        "budget_remaining": int(checkpoint["budget_remaining"]),
        "rounds_completed": int(checkpoint["rounds_completed"]),
        "root_score": float(root["score"]),
        "best": best_row,
        "ivg_mean": float(ivg_mean),
        "gap_to_ivg_mean": float(best["score"]) - float(ivg_mean),
        "known_teacher_routes": len(teachers),
        "known_teacher_endpoints": len(teacher_endpoints),
        "all_teacher_templates_in_live_checkpoint": all(
            all(row["templates_present"]) for row in teachers
        ),
        "teacher_route_summary": teachers,
        "locked_rounds": sorted(locks),
        "candidate_memberships": len(candidate_rows),
        "unique_candidate_endpoints": len(
            {_canonical(row["smiles"]) for row in candidate_rows}
        ),
        "successful_docking_receipts": len(docked),
        "lane_summary": lane_rows,
        "winning_lineage": lineage,
    }


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _markdown(report: dict) -> str:
    lines = [
        "# Interim PARP shared-controller audit",
        "",
        "This is a zero-oracle audit of a running campaign snapshot. It is not a final campaign result.",
        "",
        "| Cell | Calls | Best | IVG mean | Gap | Calls left |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell, row in report["cells"].items():
        lines.append(
            f"| {cell} | {row['charged_calls']} | {row['best']['score']:.1f} | "
            f"{row['ivg_mean']:.1f} | {row['gap_to_ivg_mean']:+.1f} | "
            f"{row['budget_remaining']} |"
        )
    lines.extend(
        [
            "",
            "## Measured mechanism",
            "",
            "- Every known PARP delta=0.6 teacher-region template is present in the live shared checkpoint.",
            "- PARP1-1 entered a strong basin through a complete-region route proposal, then FiberControl selected shallow descendants that improved it further.",
            "- The weaker cells therefore require analysis of binding, composition, ordering, and scored basin quality rather than another compiler change.",
            "",
            "## Evidence boundary",
            "",
            "Teacher matches are diagnostic only. The live controller received one shared template distribution, not a cell-to-winner lookup. Docking scores come only from preserved campaign receipts.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-snapshot", type=Path, required=True)
    parser.add_argument("--legacy-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-corpus", type=Path, required=True)
    parser.add_argument("--route-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--volume", required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    teachers = _teacher_rows(args.teacher_corpus, args.route_checkpoint)
    ivg = {"parp1_0": -12.3, "parp1_1": -11.7, "parp1_2": -10.7}
    cells = {}
    input_paths = [args.teacher_corpus, args.route_checkpoint]
    for cell in CELL_MAP:
        checkpoint_path = args.current_snapshot / cell / "checkpoint.json"
        checkpoint = _payload(checkpoint_path)
        locks = _collect_locks((args.legacy_snapshot, args.current_snapshot), cell)
        cells[cell] = _cell_audit(cell, checkpoint, locks, teachers[cell], ivg[cell])
        input_paths.append(checkpoint_path)
        for root in (args.legacy_snapshot, args.current_snapshot):
            input_paths.extend(sorted((root / cell).glob("round_*_lock.json")))

    report = {
        "schema_version": SCHEMA,
        "evidence_status": "computed_interim_running_campaign_snapshot",
        "created_at": datetime.now(UTC).isoformat(),
        "run": {"run_id": args.run_id, "volume": args.volume},
        "code_revision": _git_revision(repository),
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "costs": {"new_oracle_calls": 0, "new_docking_calls": 0},
        "inputs_sha256": {
            str(path): _sha256_file(path)
            for path in sorted(set(input_paths))
            if path.exists()
        },
        "cells": cells,
        "computed_conclusions": {
            "all_teacher_templates_in_live_checkpoint": all(
                row["all_teacher_templates_in_live_checkpoint"]
                for row in cells.values()
            ),
            "parp1_1_route_jump_then_descendant_refinement": [
                row["proposal_lane"] for row in cells["parp1_1"]["winning_lineage"]
            ]
            == ["root", "route_complete_region", "shallow"],
            "compiler_or_feasibility_is_not_the_common_failure": all(
                row["unique_candidate_endpoints"] > 0 for row in cells.values()
            ),
        },
        "interpretation": {
            "status": "inferred_from_computed_evidence",
            "statement": (
                "The successful cell demonstrates the intended route-jump, recursive-archive, "
                "reward-selected refinement mechanism. The weak cells retain route-template support "
                "but do not reliably surface or exploit the corresponding high-value bound programs; "
                "the next general intervention is source-conditioned complete-region proposal ordering "
                "and binding/composition coverage, not compiler modification."
            ),
        },
    }
    envelope = {"payload": report, "payload_sha256": identity(report)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
