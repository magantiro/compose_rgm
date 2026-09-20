"""Build the delta=0.4 arm of a frozen 250-call held-target T4 contract.

The delta=0.4 arm is derived from the protein's own frozen delta=0.6 250-call
contract by changing exactly one executable field -- ``delta`` -- so that a paired
per-cell comparison between the two arms differs in the similarity corridor alone.
Every other executable field (cells, route/shallow/anchored proposal widths, parents,
exploration, expert-floor schedule, batch, value penalty, docking box, docking seed,
evaluator hashes, call budget) is inherited byte-for-byte from the delta=0.6 arm.

Prose that names the corridor is rewritten so it agrees with the executable field, and
the reported InVirtuoGen comparison is RESOLVED IN CODE from the authoritative published
registry by exact ``(target, seed_index, delta)`` identity.  Nothing is transcribed.

``resume_predecessor`` is dropped: the delta=0.4 arm launches onto a fresh volume with no
checkpoints, so it has no ancestor contract whose state could legitimately be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.t4_matched_pilot import unseal

ROOT = Path(__file__).resolve().parents[1]
APP = "modal_apps/t4_integrated_route_fiber_parp1_app.py"
BASELINE = "configs/t4_published_invirtuogen_baseline.json"
SEEDS = "docs/GENMOL_T4_SEEDS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_published(target: str, cells: list[dict]) -> dict[str, float]:
    """Join the published delta=0.4 table by exact (target, seed_index) identity."""
    registry = json.loads((ROOT / BASELINE).read_text())
    table = registry["delta_0_4"][target]
    resolved = {}
    for index, cell in enumerate(cells):
        if cell["cell"] != f"{target}_{index}":
            raise ValueError(f"cell {cell['cell']!r} is not at seed index {index}")
        resolved[cell["cell"]] = table[index]
    if len(resolved) != len(table):
        raise ValueError(f"{target}: {len(resolved)} cells against {len(table)} published rows")
    return resolved


def check_seed_binding(target: str, cells: list[dict]) -> None:
    """Every cell's SMILES must be the published seed row it claims."""
    seeds = {row["idx"]: row for row in json.loads((ROOT / SEEDS).read_text())}
    for cell in cells:
        row = seeds[cell["source_global_index"]]
        if row["target"] != target or row["smiles"] != cell["smiles"]:
            raise ValueError(f"{cell['cell']} does not bind published seed {row['idx']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--base", required=True, help="the frozen delta=0.6 250-call contract")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target = args.target
    base_payload = unseal(ROOT / args.base)
    if base_payload["delta"] != 0.6:
        raise ValueError(f"base {args.base} is at delta {base_payload['delta']}, expected 0.6")
    if base_payload["charged_calls_per_cell"] != 250:
        raise ValueError("base is not a 250-call contract")

    payload = dict(base_payload)
    cells = payload["cells"]
    check_seed_binding(target, cells)
    published = resolve_published(target, cells)

    payload.pop("resume_predecessor", None)
    payload.pop("reported_ivg_delta_0_6", None)
    payload.pop("promotion_criteria", None)

    payload.update(
        {
            "delta": 0.4,
            "schema_version": f"t4_held_target_distilled_{target}_d04_250_contract_v1",
            "status": "AUTHORIZED_BY_OWNER_FOR_250_CALL_DELTA_0_4_PANEL",
            "charged_calls_per_cell": 250,
            "total_charged_call_ceiling": 250 * len(cells),
            "claim_boundary": (
                f"held-target {target} panel at delta 0.4 with 250 charged calls per cell; "
                f"no {target} route enters the prior"
            ),
            "central_claim_under_test": (
                "the same frozen held-target controller can use the broader delta=0.4 corridor "
                f"to match or exceed reported InVirtuoGen {target} utility"
            ),
            "experimental_setting": (
                f"prospective three-cell {target} delta=0.4 held-target panel; fixed 250-call "
                "ceiling per cell; candidate exhaustion and abstention are valid outcomes; this "
                f"is the paired delta=0.4 arm of the frozen {target} delta=0.6 250-call contract "
                "and differs from it in the similarity corridor alone"
            ),
            "primary_baselines": (
                "read-only measured roots and the authoritative published InVirtuoGen delta=0.4 "
                "table resolved by exact (target, seed_index, delta) identity; no comparator "
                "value enters runtime"
            ),
            "reported_ivg_delta_0_4": published,
            "reported_ivg_source": {
                "artifact": BASELINE,
                "artifact_sha256": sha256(ROOT / BASELINE),
                "join": "exact (target, seed_index, delta=0.4) identity, resolved in code",
                "sign_convention": "docking score, more negative is better; gap = COMPOSE - IVG, negative means COMPOSE wins",
                "aggregation": "published InVirtuoGen per-cell mean, used only after the run for read-only comparison",
                "supersedes": "CSV-derived values in configs/t4_frozen_program_benchmark.json must not be substituted",
            },
            "paired_delta_arm": {
                "delta_0_6_contract": args.base,
                "delta_0_6_contract_payload_sha256": identity(base_payload),
                "only_executable_change": "delta 0.6 -> 0.4",
                "inherited_unchanged": [
                    "cells",
                    "proposal",
                    "parents",
                    "parent_explore",
                    "exploration",
                    "expert_floor_rounds",
                    "batch",
                    "value_penalty",
                    "support",
                    "docking_box",
                    "docking_seed",
                    "evaluator_sha256",
                    "charged_calls_per_cell",
                    "total_charged_call_ceiling",
                ],
            },
            "runtime_inputs_sha256": dict(payload["runtime_inputs_sha256"]),
        }
    )
    payload["frozen_from"] = {
        **payload.get("frozen_from", {}),
        "allowed_adapter_changes": [
            f"leave-{target}-out route checkpoint",
            f"{target} cells and docking evaluator",
            "held-target delta=0.4 output namespace",
            "delta=0.4 payload-bound contract identity",
        ],
    }

    runtime = payload["runtime_inputs_sha256"]
    runtime[args.checkpoint] = sha256(ROOT / args.checkpoint)
    runtime[APP] = sha256(ROOT / APP)
    for relative in tuple(runtime):
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        runtime[relative] = sha256(path)

    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    output = (ROOT / args.output).resolve()
    if output.exists() and not args.force:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "path": str(output.relative_to(ROOT)),
                "target": target,
                "delta": payload["delta"],
                "payload_sha256": envelope["payload_sha256"],
                "reported_ivg_delta_0_4": published,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
