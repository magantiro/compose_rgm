"""Derive a template-balanced allocation from a sealed joint bound census."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from compose_v4.control.docking_value import identity

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "t4_virtual_joint_5ht1b0_template_balanced_allocation_v1"
INPUT_SCHEMA = "t4_virtual_joint_5ht1b0_support_gate_v1"
CHECKPOINT = ROOT / "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _envelope(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text())
    payload = envelope.get("payload")
    payload_sha256 = envelope.get("payload_sha256")
    if not isinstance(payload, dict) or payload_sha256 != identity(payload):
        raise ValueError(f"artifact is not self-hashed: {path}")
    return payload, str(payload_sha256)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _allocate(
    rows: list[dict[str, Any]],
    probabilities: dict[str, float],
    width: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[str]]:
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        template_id = str(row["template_id"])
        if template_id not in probabilities:
            raise ValueError(f"bound template is absent from checkpoint: {template_id}")
        by_template[template_id].append(row)
    template_order = sorted(
        by_template,
        key=lambda template_id: (-probabilities[template_id], template_id),
    )
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < width:
        added = False
        for template_id in template_order:
            template_rows = by_template[template_id]
            if depth >= len(template_rows):
                continue
            selected.append(template_rows[depth])
            added = True
            if len(selected) == width:
                break
        if not added:
            break
        depth += 1
    return selected, dict(by_template), template_order


def run(result_path: Path) -> dict[str, Any]:
    result, result_payload_sha256 = _envelope(result_path)
    if result.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("input is not the sealed joint support gate")
    checkpoint, checkpoint_payload_sha256 = _envelope(CHECKPOINT)
    marginal = checkpoint["expert"]["marginal"]
    probabilities = dict(
        zip(
            map(str, marginal["template_ids"]),
            map(float, marginal["probabilities"]),
            strict=True,
        )
    )
    telemetry = result["autonomous_joint"]["telemetry"]
    rows = telemetry.get("bound_constituent_ranked_census")
    if not isinstance(rows, list) or len(rows) != int(
        telemetry["bound_constituent_rank_rows"]
    ):
        raise ValueError("joint result omitted its complete bound constituent census")
    if [int(row["rank"]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("bound constituent ranks are not contiguous")
    width = int(result["configuration"]["joint_budgets"]["expansion_width"])
    if width != 48:
        raise ValueError("template-balanced diagnostic requires frozen width 48")

    selected, by_template, template_order = _allocate(rows, probabilities, width)
    selected_ranks = {
        row["constituent_key"]: rank for rank, row in enumerate(selected, 1)
    }
    within_template = {
        row["constituent_key"]: rank
        for template_rows in by_template.values()
        for rank, row in enumerate(template_rows, 1)
    }
    teachers = result["teacher_constituent_support"]["constituents"]
    teacher_rows = []
    for teacher in teachers:
        template_id = str(teacher["template_id"])
        key = str(teacher["constituent_key"])
        teacher_rows.append(
            {
                "route_id": teacher["route_id"],
                "region_index": int(teacher["region_index"]),
                "template_id": template_id,
                "constituent_key": key,
                "marginal_template_probability": probabilities[template_id],
                "original_bound_rank": int(teacher["bound_rank"]),
                "rank_within_template": within_template[key],
                "bound_constituents_for_template": len(by_template[template_id]),
                "rewrite_event_count": int(
                    teacher["size_signature"]["rewrite_event_count"]
                ),
                "rewrite_scale": teacher["size_signature"]["rewrite_scale"],
                "default_selected_for_expansion": bool(
                    teacher["selected_for_expansion"]
                ),
                "template_balanced_selected": key in selected_ranks,
                "template_balanced_selection_rank": selected_ranks.get(key),
            }
        )

    all_scale = Counter(str(row["rewrite_scale"]) for row in rows)
    selected_scale = Counter(str(row["rewrite_scale"]) for row in selected)
    selected_teacher_count = sum(
        row["template_balanced_selected"] for row in teacher_rows
    )
    return {
        "schema_version": SCHEMA,
        "evidence": (
            "derived zero-oracle fair-allocation diagnostic over the sealed fixed-run "
            "bound constituent census"
        ),
        "claim_boundary": (
            "this reorders already bound constituents only; it does not rebind, plan, "
            "realize, learn a rule or modify the joint proposer"
        ),
        "code_revision": _revision(),
        "configuration": {
            "width": width,
            "template_order": (
                "descending frozen marginal template probability, then template ID"
            ),
            "within_template_order": "sealed original bound constituent rank",
            "allocation": (
                "one constituent per bound template in template order, then the second "
                "per template, continuing round-robin to width"
            ),
            "new_learned_rule": False,
            "budget_widening": False,
        },
        "source_result": {
            "path": _relative(result_path),
            "sha256": _sha256(result_path),
            "payload_sha256": result_payload_sha256,
            "bound_constituent_census_sha256": telemetry[
                "bound_constituent_rank_census_sha256"
            ],
        },
        "checkpoint": {
            "path": _relative(CHECKPOINT),
            "sha256": _sha256(CHECKPOINT),
            "payload_sha256": checkpoint_payload_sha256,
        },
        "costs": {
            "binding_calls": 0,
            "planning_calls": 0,
            "realization_calls": 0,
            "oracle_calls": 0,
            "docking_calls": 0,
            "modal_launches": 0,
        },
        "bound_constituents": len(rows),
        "bound_templates": len(by_template),
        "selected_constituents": len(selected),
        "template_order_sha256": identity(template_order),
        "selection_sha256": identity(
            [row["constituent_key"] for row in selected]
        ),
        "scale_census": {
            scale: {
                "all_bound": all_scale[scale],
                "template_balanced_selected_at_width_48": selected_scale[scale],
            }
            for scale in ("local", "medium", "large")
        },
        "teacher_constituents": teacher_rows,
        "teacher_selection": {
            "default_selected_numerator": sum(
                row["default_selected_for_expansion"] for row in teacher_rows
            ),
            "template_balanced_selected_numerator": selected_teacher_count,
            "denominator": len(teacher_rows),
            "any_selected": selected_teacher_count > 0,
            "all_selected": selected_teacher_count == len(teacher_rows),
        },
        "implementation_inputs_sha256": {
            "tools/t4_virtual_joint_template_allocation_diagnostic.py": _sha256(
                Path(__file__)
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite allocation diagnostic: {output}")
    payload = run(source)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    )
    temporary.replace(output)
    print(json.dumps(envelope, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
