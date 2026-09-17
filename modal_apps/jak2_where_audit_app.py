"""Parallel WHERE-support audit across containers. Zero oracle calls.

REGION DEFINITION, and the correction that forced it. A first version of this audit took
the teacher region to be every slot appearing in the region's primitive payloads. That set
includes CREATED slot indices -- atoms the transformation inserts, numbered at or above
the source's real-atom count -- and 80.4% of teacher regions carry at least one. The
runtime region is `assignment` plus `changed_original_slots`, which by construction only
ever names SOURCE slots. So four fifths of the comparison was uncoverable no matter what
the sampler proposed, and it reported 10.6% coverage with 0 of 42 single-region routes
covered. That number was an artifact of comparing two different objects.

What a WHERE proposal can be judged on is the ORIGINAL slots a transformation must
occupy. Created atoms are a consequence of the program, not a precondition of the region,
so they are excluded here.

The question is support coverage, not teacher imitation: for every strong-route structural
region, does the production sampler ever propose a region that CONTAINS it? If it does
not, reward was never given the chance to select that region, and no acquisition rule
could have rescued it.

Fanned out because the measurement is embarrassingly parallel and slow serially -- 77
routes at a few hundred proposals each is roughly half an hour on one core and about a
minute across containers. Each container takes a stride of routes and returns rows; no
container sees a docking score or a teacher endpoint.
"""

from __future__ import annotations

import json

import modal

from modal_apps.run_process_v2_p50_app import REMOTE_ROOT, ROOT
from modal_apps.run_process_v2_p50_app import image as _base_image

image = (
    _base_image
    .add_local_dir(ROOT / "diagnostics/ivg_winner_paths/pairs",
                   str(REMOTE_ROOT / "diagnostics/ivg_winner_paths/pairs"), copy=True)
    .add_local_dir(ROOT / "diagnostics/t4_route_decision_audit",
                   str(REMOTE_ROOT / "diagnostics/t4_route_decision_audit"), copy=True)
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src:{REMOTE_ROOT}", "OMP_NUM_THREADS": "1"})
)

app = modal.App("jak2-where-audit")


@app.function(image=image, cpu=2, timeout=60 * 40)
def audit_shard(payload: str) -> str:
    """Audit one stride of the strong routes."""
    import glob
    import gzip
    from pathlib import Path

    import numpy as np

    from compose_v4.chem.molecular_graph import is_element
    from compose_v4.control.dependency_region_program import dependency_region_program
    from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
    from compose_v4.rewrite.trace_shard import decode_state

    request = json.loads(payload)
    shard, shards, draws = request["shard"], request["shards"], request["draws"]
    root = Path(str(REMOTE_ROOT))

    targets = {}
    for path in glob.glob(str(root / "diagnostics/t4_route_decision_audit/shard_*.json")):
        with open(path) as handle:
            declared = json.load(handle)["payload"]["routes"]
        for row in declared:
            # receipts were recorded with absolute local paths; re-root them here
            name = Path(row["receipt"]).name
            targets[root / "diagnostics/ivg_winner_paths/pairs" / name] = row["target"]

    ordered = sorted(targets)[shard::shards]
    rows, refused = [], 0
    for receipt in ordered:
        try:
            payload_path = json.loads(gzip.decompress(Path(receipt).read_bytes()))["payload"]["path"]
            states, actions = tuple(payload_path["states"]), tuple(payload_path["actions"])
            source = decode_state(states[0])
            program = dependency_region_program(states, actions)
        except (ValueError, KeyError, IndexError, TypeError, OSError):
            refused += 1
            continue

        real_atoms = int(np.count_nonzero(is_element(source.atom_types)))
        wanted = []
        for component in program["components"]:
            touched = set()
            for index in component["primitive_indices"]:
                inner = actions[index].get("payload") or {}
                for key in ("v", "a", "b", "u", "slot", "target"):
                    value = inner.get(key)
                    if isinstance(value, int):
                        touched.add(value)
                for neighbour in (inner.get("neighbors") or inner.get("neighbours") or []):
                    slot = neighbour[0] if isinstance(neighbour, (list, tuple)) else neighbour
                    if isinstance(slot, int):
                        touched.add(slot)
            # keep only ORIGINAL slots: created indices cannot appear in a runtime region
            original = {slot for slot in touched if slot < real_atoms}
            if original:
                wanted.append(frozenset(original))
        if not wanted:
            continue

        rng = np.random.default_rng(abs(hash(str(receipt))) % (2**31))
        proposed = []
        for _ in range(draws):
            try:
                _, _, assignment, trace, _ = synthesize_dynamic_program(
                    source, rng, max_modules=3
                )
            except (ValueError, KeyError, IndexError, TypeError) as error:
                refused += 1
                del error
                continue
            slots = {int(s) for s in (assignment or ())}
            changed = (trace.get("actual_changes") or {}).get("changed_original_slots") or []
            slots |= {int(s) for s in changed}
            if slots:
                proposed.append(frozenset(slots))
        if not proposed:
            continue

        for region in wanted:
            overlap = [len(region & p) / len(region) for p in proposed]
            rows.append({
                "target": targets[receipt],
                "region_size": len(region),
                "proposals": len(proposed),
                "fully_covered": sum(1 for p in proposed if region <= p),
                "best_overlap": max(overlap),
                "mean_overlap": float(np.mean(overlap)),
                "multi_region": len(wanted) > 1,
            })
    return json.dumps({"shard": shard, "routes": len(ordered), "refused": refused, "rows": rows})
