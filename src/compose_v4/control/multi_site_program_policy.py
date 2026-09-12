"""Three proposal ablations with one executor and no intermediate score pruning.

Joint mode retrieves complete demonstrated programs, including their dependent
attachments. Independent mode draws site programs against the same parent;
serial mode conditions subsequent draws on the already executed exact prefix.
No mode reads a target molecule or claims to implement the frozen reference law.
The caller retains the existing single-region and generic proposal channels.
"""

from __future__ import annotations

from time import perf_counter

from compose_v4.control.edit_program import ProgramExecutionError, extract_program
from compose_v4.control.edit_program_graph import (
    combine_bound_programs,
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.edit_program_policy import ProgramEntry, propose_programs
from compose_v4.rewrite.trace_shard import encode_state

MODES = ("uninterrupted_serial", "independent_multi_site", "joint_multi_site")


def generate_program_candidates(
    source,
    entries: tuple[ProgramEntry, ...],
    *,
    mode: str,
    seed: int,
    attempts: int = 16,
    sites: int = 2,
    max_primitives: int = 24,
    max_blocks: int = 5,
    wall_seconds: float = 120,
    joint_entries: tuple[ProgramEntry, ...] | None = None,
) -> dict:
    """Bounded proposal pool with all attempts, failures, choices and exact traces.

    Joint entries must be prepared from verified coordinated examples, not supplied
    as endpoint molecules. This is non-neural program retrieval, not a qualified
    task-adapted joint decoder. Unique endpoints are locked only by the caller.
    """
    if mode not in MODES or type(attempts) is not int or attempts < 1:
        raise ValueError("invalid multi-site proposal mode/attempt count")
    if type(sites) is not int or not 2 <= sites <= 4:
        raise ValueError("multi-site proposals require two to four planned components")
    if any(type(v) is not int or v < 1 for v in (max_primitives, max_blocks)):
        raise ValueError("program work caps must be positive integers")
    if not 0 < wall_seconds < float("inf"):
        raise ValueError("proposal wall limit must be positive and finite")
    if mode == "joint_multi_site" and not joint_entries:
        raise ValueError("joint proposal requires a separately prepared coordinated-program bank")
    original = encode_state(source)
    started = perf_counter()
    rows, products = [], {}
    selected_entries = joint_entries if mode == "joint_multi_site" else entries
    # Static-parent modes share one binding census across all draws. This is the
    # main throughput saving; no per-attempt reconstruction of the same frontier.
    static = (
        None
        if mode == "uninterrupted_serial"
        else propose_programs(
            source,
            selected_entries,
            seed=seed,
            count=attempts * (sites if mode == "independent_multi_site" else 1),
        )
    )
    for attempt in range(attempts):
        if perf_counter() - started >= wall_seconds:
            break
        choices, stages, consumed, blocks = [], [], 0, 0
        current = source
        failure = None
        pieces = []
        # Site count is NOT the serial decision horizon. A serial control must
        # have access to the same five-stage route as a five-block joint program.
        width = (
            1
            if mode == "joint_multi_site"
            else max_blocks
            if mode == "uninterrupted_serial"
            else sites
        )
        for site in range(width):
            if mode == "uninterrupted_serial" and blocks == max_blocks:
                break
            selection = (
                static
                if static is not None
                else propose_programs(
                    current,
                    entries,
                    seed=seed + attempt * max_blocks + site,
                    count=1,
                )
            )
            if not selection["draws"]:
                failure = {"reason_code": "no_bindings", "selection": selection}
                break
            draw = (
                selection["draws"][attempt * width + site]
                if static is not None
                else selection["draws"][0]
            )
            program = selected_entries[draw["program_index"]].program
            assignment = tuple(draw["assignment"])
            choices.append({**draw, "program_id": program.program_id})
            consumed += len(program.marks)
            blocks += len(program.blocks)
            if consumed > max_primitives or blocks > max_blocks:
                failure = {"reason_code": "complete_program_budget"}
                break
            pieces.append((program, assignment))
            if mode == "uninterrupted_serial":
                try:
                    current, receipt = execute_program_graph(
                        current,
                        compile_program_graph(program),
                        assignment,
                        max_primitives=max_primitives,
                        max_blocks=max_blocks,
                    )
                except (ValueError, ProgramExecutionError) as error:
                    failure = {
                        "reason_code": "serial_execution_rejected",
                        "message": str(error),
                        "prefix": getattr(error, "receipt", None),
                    }
                    break
                start = 0
                for block in program.blocks:
                    stages.append(
                        {
                            "name": block.label,
                            "actions": receipt["actions"][start : block.stop],
                            "states": receipt["states"][start : block.stop + 1],
                            # Extraction validates each exact state; boundary identity
                            # comes from that state, not the eventual task endpoint.
                            "endpoint": _endpoint(receipt["states"][block.stop]),
                        }
                    )
                    start = block.stop
        if failure is None:
            try:
                if mode == "uninterrupted_serial":
                    combined, binding = extract_program(source, stages)
                elif mode == "independent_multi_site":
                    combined, binding = combine_bound_programs(
                        source, tuple(pieces), max_primitives=max_primitives, max_blocks=max_blocks
                    )
                else:
                    combined, binding = pieces[0]
                dag = compile_program_graph(combined)
                _, receipt = execute_program_graph(
                    source, dag, binding, max_primitives=max_primitives, max_blocks=max_blocks
                )
            except ValueError as error:
                failure = {
                    "reason_code": "joint_execution_rejected",
                    "message": str(error),
                    "prefix": getattr(error, "receipt", None),
                }
        if failure is not None:
            rows.append({"attempt": attempt, "choices": choices, "status": "rejected", **failure})
        else:
            rows.append(
                {
                    "attempt": attempt,
                    "choices": choices,
                    "status": "complete",
                    "receipt": receipt,
                    "program": combined.payload(),
                    "graph": dag.payload(),
                }
            )
            products.setdefault(receipt["endpoint"], attempt)
    if encode_state(source) != original:
        raise RuntimeError("multi-site proposal mutated its parent")
    return {
        "schema_version": "multi_site_proposal_pool_v1",
        "mode": mode,
        "configuration": {
            "seed": seed,
            "attempts": attempts,
            "sites": sites,
            "max_primitives": max_primitives,
            "max_blocks": max_blocks,
            "wall_seconds": wall_seconds,
        },
        "rows": rows,
        "unique_endpoint_first_attempt": products,
        "static_binding_census": None if static is None else static["censuses"],
        "attempts_completed": len(rows),
        "attempts_unstarted": attempts - len(rows),
        "proposal_seconds": perf_counter() - started,
        "inference_target_supplied": False,
        "intermediate_task_evaluations": 0,
        "limitation": "program-retrieval proposal; not a learned task-value or reference-sampling guarantee",
    }


def _endpoint(state):
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.trace_shard import decode_state

    return canonical_state_key(decode_state(state))
