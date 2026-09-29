"""Arm C pre-launch development panel: IS THE INTERVENTION INFORMATIVE AND EXECUTABLE?

Runs BEFORE any scored campaign and reads no PMO score.  Panel programs are taken from
a completed arm-A snapshot in INSERTION ORDER; the `score` and `static_score` fields are
never consulted, and nothing here charges an oracle call.

Reports, per the pre-registered questions:
  * how often an alternative created-atom binding EXISTS at all
  * how often the intervention changes the executed MOLECULE
  * completion rate, and whether a failure is ORIGINAL-CONSTRUCTION or REBINDING
  * throughput, as rebinding wall time against baseline execution wall time
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path.home() / "compose_pmo_chain" / "src"))

from compose_v4.rewrite.trace_shard import decode_state  # noqa: E402
from compose_v4.control.edit_program import EditProgram, execute_bound_program  # noqa: E402
from compose_v4.control.edit_program_graph import (  # noqa: E402
    compile_program_graph,
    scheduled_program,
)
from compose_v4.control.pmo_binding_intervention import (  # noqa: E402
    binding_rng,
    execute_rebound_program,
)
from compose_v4.rewrite.kernel import canonical_state_key  # noqa: E402

RUN_SEED = 20260925
#: Production PMO budgets (`ProgramSearchConfig` defaults), not the library defaults.
MAX_PRIMITIVES = 32
MAX_BLOCKS = 8


def panel(snapshot_path: Path, limit: int):
    entries = json.load(snapshot_path.open())["snapshot"]["entries"]
    # Insertion order of the snapshot mapping; no score field is read anywhere.
    return [(key, entries[key]) for key in list(entries)[:limit]]


def main(snapshot: str, limit: int = 60):
    rows, failures = [], Counter()
    for entry_id, entry in panel(Path(snapshot), limit):
        source = decode_state(entry["source_state"])
        original = EditProgram.from_payload(entry["program"])
        assignment = tuple(entry["assignment"])
        try:
            graph = compile_program_graph(original)
            program, _order, _timeline = scheduled_program(graph, source.n_real_atoms)
        except Exception as error:  # noqa: BLE001
            failures["original_construction"] += 1
            rows.append({"entry": entry_id, "stage": "construction", "error": str(error)[:120]})
            continue

        began = perf_counter()
        try:
            _, base = execute_bound_program(
                source, program, assignment,
                max_primitives=MAX_PRIMITIVES, max_blocks=MAX_BLOCKS,
            )
        except Exception as error:  # noqa: BLE001
            failures["original_execution"] += 1
            rows.append({"entry": entry_id, "stage": "baseline", "error": str(error)[:120]})
            continue
        baseline_seconds = perf_counter() - began

        proposal_id = f"{program.program_id}:{canonical_state_key(source)}:{assignment}"

        # Development check: an IDENTITY intervention must reproduce the original
        # program exactly and must not advance the generator it was handed.
        idle = binding_rng(RUN_SEED, proposal_id)
        before = idle.bit_generator.state["state"]["state"]
        _, same = execute_rebound_program(
            source, program, assignment, rng=idle, identity=True,
            max_primitives=MAX_PRIMITIVES, max_blocks=MAX_BLOCKS,
        )
        identity_ok = (
            same["endpoint"] == base["endpoint"]
            and tuple(same["marks"]) == tuple(program.marks)
            and idle.bit_generator.state["state"]["state"] == before
        )

        rng = binding_rng(RUN_SEED, proposal_id)
        began = perf_counter()
        try:
            _, bound = execute_rebound_program(
                source, program, assignment, rng=rng,
                max_primitives=MAX_PRIMITIVES, max_blocks=MAX_BLOCKS,
            )
        except Exception as error:  # noqa: BLE001
            stage = getattr(error, "receipt", {}).get("failure_stage", "unknown")
            failures[f"rebinding:{stage}"] += 1
            rows.append(
                {
                    "entry": entry_id,
                    "stage": "rebinding",
                    "failure_stage": stage,
                    "identity_ok": identity_ok,
                    "error": str(error)[:120],
                }
            )
            continue
        rebind_seconds = perf_counter() - began
        audit = bound["rebinding"]
        rows.append(
            {
                "entry": entry_id,
                "stage": "complete",
                "identity_ok": identity_ok,
                "primitives": audit["steps"],
                "steps_with_created_operands": audit["steps_with_created_operands"],
                "steps_with_alternatives": audit["steps_with_alternatives"],
                "steps_changed": audit["steps_changed"],
                "steps_cap_bound": audit["steps_cap_bound"],
                "endpoint_changed": bound["endpoint"] != base["endpoint"],
                "baseline_seconds": round(baseline_seconds, 4),
                "rebind_seconds": round(rebind_seconds, 4),
            }
        )

    done = [r for r in rows if r["stage"] == "complete"]
    with_ops = [r for r in done if r["steps_with_created_operands"]]
    with_alt = [r for r in done if r["steps_with_alternatives"]]
    report = {
        "panel_size": len(rows),
        "panel_selection": "snapshot insertion order; no score field read",
        "run_seed": RUN_SEED,
        "completed": len(done),
        "completion_rate": round(len(done) / len(rows), 4) if rows else None,
        "failures": dict(failures),
        "identity_check_passed": sum(1 for r in rows if r.get("identity_ok")),
        "identity_check_failed": sum(
            1 for r in rows if r.get("identity_ok") is False
        ),
        "programs_with_created_operands": len(with_ops),
        "programs_with_an_alternative_binding": len(with_alt),
        "programs_whose_endpoint_changed": sum(1 for r in done if r["endpoint_changed"]),
        "step_totals": {
            "primitives": sum(r["primitives"] for r in done),
            "with_created_operands": sum(r["steps_with_created_operands"] for r in done),
            "with_alternatives": sum(r["steps_with_alternatives"] for r in done),
            "changed": sum(r["steps_changed"] for r in done),
            "cap_bound": sum(r["steps_cap_bound"] for r in done),
        },
        "throughput": {
            "baseline_seconds_total": round(sum(r["baseline_seconds"] for r in done), 3),
            "rebind_seconds_total": round(sum(r["rebind_seconds"] for r in done), 3),
        },
        "rows": rows,
    }
    print(json.dumps(report, indent=2)[:4000])
    out = Path.home() / "compose_pmo_ablation" / "armC" / "dev_panel_v1.json"
    out.write_text(json.dumps(report, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60)
