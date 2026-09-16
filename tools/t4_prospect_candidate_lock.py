"""Assess prospected pools for free, then freeze a score-blind candidate lock.

Runs the two stages that must happen between generating a pool and charging a call:
match the pool against the recovered historical observations so rediscovered endpoints
hand over their measured scores at no cost, then select novel members under a rule
fixed before the pool existed and seal the result.

Zero oracle calls. Publishing a lock does not launch anything.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
from pathlib import Path

from rdkit import rdBase

from compose_v4.experiments.continuation_profile import sha256_file
from compose_v4.experiments.t4_matched_pilot import _stamp, seal, unseal
from compose_v4.experiments.t4_prospect_assessment import (
    free_labels,
    shape_comparison,
)
from compose_v4.experiments.t4_prospect_lock import build_lock, select, verify_lock

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/t4_objective_reset_runtime_v1.json"
CORPUS = ROOT / "diagnostics/t4_proposal_prior/dataset_v1/records.jsonl.gz"


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _report(payload: dict) -> str:
    lines = [
        "# Prospected candidate lock",
        "",
        (
            f"Lock `{payload['lock']['lock_id'][:16]}`, "
            f"{payload['lock']['charged_calls']} charged calls against a limit of "
            f"{payload['lock']['new_oracle_call_limit']}."
        ),
        "Zero oracle calls were made to produce this lock.",
        "",
        "## Pools, and what the corpus already knows about them",
        "",
        "| cell | attempts | eligible pool | rediscovered | best rediscovered | novel | charged |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell, row in sorted(payload["cells"].items()):
        labels, chosen = row["free_labels"], row["selection"]
        best = labels["best_rediscovered_score"]
        lines.append(
            f"| {cell} | {row['attempted']} | {labels['pool_size']} "
            f"| {labels['rediscovered']} | {'none' if best is None else f'{best:.1f}'} "
            f"| {chosen['novel']} | {len(chosen['take'])} |"
        )
    lines += [
        "",
        "A rediscovered endpoint carries the score of a molecule already charged",
        "historically. It is evidence about this pool and is not a new result, and it says",
        "nothing about the members that were never charged. Rediscovered endpoints are",
        "excluded from the charged set so the budget goes to novel molecules.",
        "",
        "## Shape against the cell's best historical constructions",
        "",
        "| cell | pool primitives | pool changed | pool created | best-30 primitives | changed | created |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell, row in sorted(payload["cells"].items()):
        shape = row["shape"]
        if not shape.get("comparable"):
            lines.append(f"| {cell} | — | — | — | — | — | — |")
            continue
        p, h = shape["pool"], shape["historical_best"]
        lines.append(
            f"| {cell} | {p['median_primitives']:.0f} | {p['median_changed_originals']:.0f} "
            f"| {p['median_created']:.0f} | {h['median_primitives']:.0f} "
            f"| {h['median_changed_originals']:.0f} | {h['median_created']:.0f} |"
        )
    lines += [
        "",
        "Shape only. Occupying the same size regime as high-scoring constructions is not",
        "evidence of scoring well: structural proximity to a good molecule was measured at",
        "Spearman 0.12 on PARP1 and is not a usable surrogate for docking utility.",
        "",
        "## Limits",
        "",
        "- These endpoints are eligible and have never been docked. The lock is a",
        "  prospective commitment, not a result.",
        "- Selection used no docking score of any kind; ranking is by program size.",
        "- A scored outcome from this lock is bounded development evidence on answer-known",
        "  cells, not a held-out benchmark result and not an IVG comparison.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", action="append", required=True, metavar="CELL=PATH")
    parser.add_argument("--take", type=int, default=20, help="charged calls per cell")
    parser.add_argument("--per-signature", type=int, default=3)
    parser.add_argument("--call-limit", type=int, required=True)
    parser.add_argument("--authorization-ceiling", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with gzip.open(CORPUS, "rt") as handle:
        records = [json.loads(line) for line in handle]
    units = {u["cell"]: u for u in unseal(REGISTRY)["units"]}
    registry = unseal(REGISTRY)

    cells, selections, protocols, inputs = (
        {},
        {},
        {},
        {str(REGISTRY.relative_to(ROOT)): sha256_file(REGISTRY)},
    )
    for spec in args.pool:
        if "=" not in spec:
            parser.error(f"--pool expects CELL=PATH, got {spec!r}")
        cell, raw = spec.split("=", 1)
        path = Path(raw)
        if cell not in units:
            parser.error(f"unknown cell {cell}")
        if not path.exists():
            parser.error(f"pool artifact is missing: {path}")
        pool_payload = unseal(path)
        inputs[str(path)] = sha256_file(path)
        charged = {record["endpoint_sha256"] for record in records if record["cell"] == cell}
        labels = free_labels(pool_payload["pool"], records, cell=cell)
        chosen = select(
            pool_payload["pool"], charged, take=args.take, per_signature=args.per_signature
        )
        cells[cell] = {
            "attempted": pool_payload["attempted"],
            "eligible_per_attempt": pool_payload["eligible_per_attempt"],
            "free_labels": labels,
            "shape": shape_comparison(pool_payload["pool"], records, cell=cell),
            "selection": chosen,
        }
        selections[cell] = chosen
        protocols[cell] = {
            "delta": registry["delta"],
            "oracle_protocol": units[cell]["oracle_protocol"],
            "docking_seed": units[cell]["docking_seed"],
            "box": registry["docking_boxes"][units[cell]["target"]],
        }

    lock = build_lock(
        selections,
        call_limit=args.call_limit,
        authorization_ceiling=args.authorization_ceiling,
        docking_seed=1701,
        required_rdkit="2024.03.5",
        oracle_protocols=protocols,
        input_sha256=inputs,
    )
    verify_lock(lock)
    payload = {
        "schema_version": "t4_prospect_candidate_lock_publication_v1",
        "lock": lock,
        "cells": cells,
        "new_oracle_calls": 0,
        "code_revision": _revision(),
        "runtime": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "completed_at_utc": _stamp(),
    }
    seal(args.output, payload)
    (args.output.parent / "REPORT.md").write_text(_report(payload))
    print(
        json.dumps(
            {
                "lock_id": lock["lock_id"],
                "charged_calls": lock["charged_calls"],
                "per_cell": {
                    cell: {
                        "pool": row["free_labels"]["pool_size"],
                        "rediscovered": row["free_labels"]["rediscovered"],
                        "best_rediscovered": row["free_labels"]["best_rediscovered_score"],
                        "charged": len(row["selection"]["take"]),
                        "shortfall": row["selection"]["shortfall"],
                    }
                    for cell, row in sorted(cells.items())
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
