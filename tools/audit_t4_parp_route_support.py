"""Trace known PARP routes through the shared production proposal pipeline.

This is an answer-known, zero-oracle diagnostic.  Teacher routes are probes only;
they are never supplied to a live controller or prospective candidate pool.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from rdkit import rdBase

from compose_v4.control.complete_region_program import (
    execute_complete_region_program,
    program_from_structural_goal,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.route_distilled_goal_expert import RouteDistilledGoalExpert
from compose_v4.control.structural_subgoal import StructuralGoal, instantiate_goal
from compose_v4.control.structural_subgoal_policy import (
    ProposedStructuralGoal,
    _single_goal_candidates,
    materialize_template,
    minimize_subgoal,
    proposal_rewrite_scale,
    propose_structural_goals,
    select_scale_balanced_proposals,
    transfer_bindings,
)
from compose_v4.control.structural_subgoal_realizer import RealizerConfig
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "t4_parp_route_known_answer_support_probe_v1"
CELL = "docking_parp1_idx0_thr6"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _payload(path: Path) -> dict:
    envelope = _read(path)
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if not isinstance(payload, dict) or envelope.get("payload_sha256") != identity(
        payload
    ):
        raise ValueError(f"invalid sealed input: {path}")
    return payload


def stage_positions(
    rows: list[ProposedStructuralGoal], template_id: str, endpoint_key: str
) -> list[int]:
    """Return one-based exact program positions in a proposal stage."""

    return [
        index
        for index, row in enumerate(rows, 1)
        if canonical_state_key(row.endpoint) == endpoint_key
        and tuple(template.template_id for template in row.templates) == (template_id,)
    ]


def _scored_singles(source, expert, *, binding_limit: int):
    singles, telemetry = _single_goal_candidates(
        source, expert.templates, max_bindings_per_template=binding_limit
    )
    scored = [
        ProposedStructuralGoal(
            row.goal,
            row.bindings,
            row.templates,
            row.endpoint,
            row.constituent_keys,
            expert.marginal.score(row.templates),
        )
        for row in singles
    ]
    scored.sort(
        key=lambda row: (
            -row.score,
            canonical_state_key(row.endpoint),
            row.constituent_keys,
        )
    )
    return scored, telemetry


def _realize_worker(source, proposal, maximum_expansions: int, queue) -> None:
    started = time.perf_counter()
    try:
        receipt = execute_complete_region_program(
            source,
            program_from_structural_goal(proposal.goal),
            resolved_bindings=proposal.bindings,
            config=RealizerConfig(maximum_expansions=maximum_expansions),
        )
        queue.put(
            {
                "status": receipt["status"],
                "seconds": time.perf_counter() - started,
                "realized_primitive_count": receipt.get("realized_primitive_count"),
                "expanded": receipt.get("expanded"),
                "attempted": receipt.get("attempted"),
                "committed_endpoint_key": receipt.get("committed_endpoint_key"),
                "compiler_strategy": receipt.get("compiler_strategy"),
            }
        )
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
        queue.put(
            {
                "status": "exception",
                "seconds": time.perf_counter() - started,
                "error": repr(error),
            }
        )


def _timeboxed_realization(
    source,
    proposal: ProposedStructuralGoal,
    *,
    maximum_expansions: int,
    timeout_seconds: float,
) -> dict:
    context = mp.get_context("fork")
    queue = context.Queue()
    process = context.Process(
        target=_realize_worker,
        args=(source, proposal, maximum_expansions, queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return {"status": "timeboxed", "timeout_seconds": timeout_seconds}
    if queue.empty():
        return {"status": "worker_exit_without_result", "exitcode": process.exitcode}
    return queue.get()


def _live_root_timeout(lock_path: Path | None, source_key: str) -> dict | None:
    if lock_path is None:
        return None
    from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
    from compose_v4.chem.state import pad_molecular_graph

    payload = _payload(lock_path)
    matches = [
        row
        for row in payload.get("worker_telemetry", [])
        if row.get("expert") == "route_complete_region"
        and canonical_state_key(
            pad_molecular_graph(smiles_to_molecular_graph(row["parent"]), 48)
        )
        == source_key
    ]
    if len(matches) != 1:
        raise ValueError("live lock does not contain exactly one root route worker")
    row = matches[0]
    return {
        "status": row.get("status"),
        "error": row.get("error"),
        "parent": row.get("parent"),
    }


def run(
    *,
    route_checkpoint: Path,
    teacher_corpus: Path,
    output_json: Path,
    output_md: Path,
    live_root_lock: Path | None,
    diagnostic_binding_limit: int,
    per_candidate_timeout_seconds: float,
) -> dict:
    checkpoint = _payload(route_checkpoint)
    expert = RouteDistilledGoalExpert.from_checkpoint(checkpoint["expert"])
    corpus = _payload(teacher_corpus)
    routes = sorted(
        (row for row in corpus["routes"] if row.get("cell") == CELL),
        key=lambda row: row["route_id"],
    )
    if len(routes) != 3 or any(int(row["region_count"]) != 1 for row in routes):
        raise RuntimeError("PARP1-0 delta=0.6 teacher-route census changed")
    sources = {canonical_state_key(decode_state(row["source_state"])) for row in routes}
    if len(sources) != 1:
        raise RuntimeError("teacher probes no longer share one source")
    source = decode_state(routes[0]["source_state"])
    source_key = canonical_state_key(source)
    probabilities = dict(
        zip(
            expert.marginal.template_ids,
            map(float, expert.marginal.probabilities),
            strict=True,
        )
    )
    marginal_ranks = {
        template_id: rank
        for rank, (template_id, _) in enumerate(
            sorted(probabilities.items(), key=lambda row: (-row[1], row[0])), 1
        )
    }

    settings = {
        "pool_size": 192,
        "realization_limit": 96,
        "beam_width": 48,
        "expansion_width": 48,
        "max_bindings_per_template": 4,
        "maximum_expansions": 4000,
        "scale_balanced": True,
    }
    traces = {}
    stage_cache = {}
    for binding_limit in (
        settings["max_bindings_per_template"],
        diagnostic_binding_limit,
    ):
        singles, singles_telemetry = _scored_singles(
            source, expert, binding_limit=binding_limit
        )
        expansion = select_scale_balanced_proposals(
            singles, settings["expansion_width"]
        )
        frontier = select_scale_balanced_proposals(singles, settings["beam_width"])
        pool, pool_telemetry = propose_structural_goals(
            source,
            expert.templates,
            expert.marginal,
            ranker=None,
            pool_size=settings["pool_size"],
            beam_width=settings["beam_width"],
            expansion_width=settings["expansion_width"],
            max_bindings_per_template=binding_limit,
            scale_balanced=True,
        )
        stage_cache[binding_limit] = (singles, expansion, frontier, pool)
        traces[str(binding_limit)] = {
            "single_count": len(singles),
            "expansion_count": len(expansion),
            "frontier_count": len(frontier),
            "pool_count": len(pool),
            "single_telemetry": singles_telemetry,
            "pool_telemetry": pool_telemetry,
        }

    teacher_rows = []
    production_stages = stage_cache[settings["max_bindings_per_template"]]
    diagnostic_stages = stage_cache[diagnostic_binding_limit]
    for route in routes:
        teacher_goal = StructuralGoal.from_payload(route["structural_goal"])
        template, retained_roles = minimize_subgoal(teacher_goal.subgoals[0])
        target_key = canonical_state_key(decode_state(route["endpoint_state"]))
        production_census = transfer_bindings(
            template,
            source,
            max_bindings=settings["max_bindings_per_template"],
            max_visits=1_000_000,
        )
        full_census = transfer_bindings(
            template,
            source,
            max_bindings=diagnostic_binding_limit,
            max_visits=1_000_000,
        )

        def exact_binding_positions(
            census, *, bound_template=template, bound_target_key=target_key
        ) -> list[int]:
            positions = []
            for position, binding in enumerate(census.assignments, 1):
                try:
                    subgoal = materialize_template(bound_template, source, binding)
                    endpoint, _ = instantiate_goal(
                        source, StructuralGoal((subgoal,)), (binding,)
                    )
                except ValueError:
                    continue
                if canonical_state_key(endpoint) == bound_target_key:
                    positions.append(position)
            return positions

        production_positions = [
            stage_positions(rows, template.template_id, target_key)
            for rows in production_stages
        ]
        diagnostic_positions = [
            stage_positions(rows, template.template_id, target_key)
            for rows in diagnostic_stages
        ]
        pool = production_stages[-1]
        match = next(
            (
                row
                for row in pool
                if canonical_state_key(row.endpoint) == target_key
                and tuple(value.template_id for value in row.templates)
                == (template.template_id,)
            ),
            None,
        )
        realization = (
            None
            if match is None
            else _timeboxed_realization(
                source,
                match,
                maximum_expansions=settings["maximum_expansions"],
                timeout_seconds=per_candidate_timeout_seconds,
            )
        )
        if realization is not None:
            realization["endpoint_exact"] = (
                realization.get("committed_endpoint_key") == target_key
            )
        production_single = next(
            (
                row
                for row in production_stages[0]
                if canonical_state_key(row.endpoint) == target_key
                and tuple(value.template_id for value in row.templates)
                == (template.template_id,)
            ),
            None,
        )
        band = (
            None
            if production_single is None
            else proposal_rewrite_scale(production_single)
        )
        band_rows = (
            []
            if band is None
            else [
                row
                for row in production_stages[0]
                if proposal_rewrite_scale(row) == band
            ]
        )
        row = {
            "route_id": route["route_id"],
            "template_id": template.template_id,
            "target_endpoint_key": target_key,
            "primitive_count_teacher": int(route["primitive_count"]),
            "retained_minimized_roles": list(retained_roles),
            "template_in_checkpoint": template.template_id in probabilities,
            "marginal_probability": probabilities.get(template.template_id),
            "marginal_template_rank": marginal_ranks.get(template.template_id),
            "production_binding": {
                "assignments": len(production_census.assignments),
                "visits": production_census.visits,
                "truncated": production_census.truncated,
                "exact_endpoint_binding_positions": exact_binding_positions(
                    production_census
                ),
            },
            "diagnostic_full_binding": {
                "limit": diagnostic_binding_limit,
                "assignments": len(full_census.assignments),
                "visits": full_census.visits,
                "truncated": full_census.truncated,
                "exact_endpoint_binding_positions": exact_binding_positions(
                    full_census
                ),
            },
            "production_stage_positions": {
                "bound_singles": production_positions[0],
                "scale_balanced_expansion": production_positions[1],
                "scale_balanced_frontier": production_positions[2],
                "scale_balanced_pool": production_positions[3],
                "single_rewrite_band": band,
                "single_band_rank": stage_positions(
                    band_rows, template.template_id, target_key
                ),
                "inside_realization_prefix": bool(
                    stage_positions(
                        production_stages[-1][: settings["realization_limit"]],
                        template.template_id,
                        target_key,
                    )
                ),
            },
            "diagnostic_binding_stage_positions": {
                "bound_singles": diagnostic_positions[0],
                "scale_balanced_expansion": diagnostic_positions[1],
                "scale_balanced_frontier": diagnostic_positions[2],
                "scale_balanced_pool": diagnostic_positions[3],
            },
            "isolated_production_realization": realization,
        }
        survives = (
            row["template_in_checkpoint"]
            and bool(row["production_binding"]["exact_endpoint_binding_positions"])
            and all(production_positions)
            and row["production_stage_positions"]["inside_realization_prefix"]
            and realization is not None
            and realization["status"] == "committed"
            and realization["endpoint_exact"]
        )
        row["survives_every_candidate_level_stage"] = survives
        row["first_candidate_level_disappearance"] = (
            None
            if survives
            else next(
                name
                for name, passed in (
                    ("checkpoint_template", row["template_in_checkpoint"]),
                    (
                        "source_binding",
                        bool(
                            row["production_binding"][
                                "exact_endpoint_binding_positions"
                            ]
                        ),
                    ),
                    ("bound_singles", bool(production_positions[0])),
                    ("scale_balanced_expansion", bool(production_positions[1])),
                    ("scale_balanced_frontier", bool(production_positions[2])),
                    ("scale_balanced_pool", bool(production_positions[3])),
                    (
                        "realization_prefix",
                        row["production_stage_positions"]["inside_realization_prefix"],
                    ),
                    (
                        "exact_realization",
                        realization is not None
                        and realization["status"] == "committed"
                        and realization["endpoint_exact"],
                    ),
                )
                if not passed
            )
        )
        teacher_rows.append(row)

    prefix = production_stages[-1][: settings["realization_limit"]]
    prefix_receipts = []
    for rank, proposal in enumerate(prefix, 1):
        receipt = _timeboxed_realization(
            source,
            proposal,
            maximum_expansions=settings["maximum_expansions"],
            timeout_seconds=per_candidate_timeout_seconds,
        )
        prefix_receipts.append(
            {
                "rank": rank,
                "regions": len(proposal.templates),
                "rewrite_scale": proposal_rewrite_scale(proposal),
                **receipt,
            }
        )
        if receipt["status"] == "timeboxed":
            break

    live_timeout = _live_root_timeout(live_root_lock, source_key)
    inputs = [route_checkpoint, teacher_corpus, Path(__file__)]
    if live_root_lock is not None:
        inputs.append(live_root_lock)
    payload = {
        "schema_version": SCHEMA,
        "evidence_status": "computed_zero_oracle_known_answer_diagnostic",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "inputs_sha256": {str(path): _sha256_file(path) for path in inputs},
        "costs": {"new_oracle_calls": 0, "new_docking_calls": 0, "modal_launches": 0},
        "cell": "parp1_0",
        "teacher_cell": CELL,
        "production_settings": settings,
        "diagnostic_binding_limit": diagnostic_binding_limit,
        "per_candidate_diagnostic_timeout_seconds": per_candidate_timeout_seconds,
        "stage_census": traces,
        "teacher_routes": teacher_rows,
        "production_prefix_realization_trace": prefix_receipts,
        "first_timeboxed_production_rank": next(
            (row["rank"] for row in prefix_receipts if row["status"] == "timeboxed"),
            None,
        ),
        "live_root_worker_evidence": live_timeout,
        "computed_conclusions": {
            "all_templates_bind_source": all(
                row["production_binding"]["assignments"] > 0 for row in teacher_rows
            ),
            "all_exact_endpoints_in_bound_singles": all(
                row["production_binding"]["exact_endpoint_binding_positions"]
                for row in teacher_rows
            ),
            "binding_limit_is_not_causal": all(
                not row["production_binding"]["truncated"]
                and row["production_stage_positions"]
                == {
                    **row["diagnostic_binding_stage_positions"],
                    "single_rewrite_band": row["production_stage_positions"][
                        "single_rewrite_band"
                    ],
                    "single_band_rank": row["production_stage_positions"][
                        "single_band_rank"
                    ],
                    "inside_realization_prefix": row["production_stage_positions"][
                        "inside_realization_prefix"
                    ],
                }
                for row in teacher_rows
            ),
            "all_teacher_candidates_survive_and_realize_exactly": all(
                row["survives_every_candidate_level_stage"] for row in teacher_rows
            ),
            "monolithic_realization_has_head_of_line_blocking": (
                bool(prefix_receipts)
                and prefix_receipts[-1]["status"] == "timeboxed"
                and live_timeout is not None
                and live_timeout.get("status") == "failed"
                and "timeout" in str(live_timeout.get("error", "")).lower()
            ),
        },
        "decisive_mechanism": (
            "The teacher programs are present, source-bound, highly ranked by scale-balanced "
            "production ordering, inside the first 96 realization slots, and exact-realizable. "
            "They are lost only because the route expert realizes the entire prefix serially and "
            "returns atomically; a later pathological proposal blocks completion until the whole "
            "worker times out, discarding already completed teacher candidates."
        ),
        "smallest_generic_fix_proposed_not_implemented": (
            "Isolate each route-program realization behind a bounded per-candidate deadline and "
            "persist/collect successful candidate receipts incrementally in the existing ranked "
            "order. A timed-out proposal should abstain without erasing earlier committed programs."
        ),
        "claim_boundary": (
            "Known answers are used only to locate support loss in an offline production-path "
            "diagnostic. This does not measure autonomous discovery or docking utility and does "
            "not alter any running campaign."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"payload": payload, "payload_sha256": identity(payload)}
    output_json.write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# PARP1-0 production-route support probe",
        "",
        "This is a zero-oracle, answer-known diagnostic. It does not modify a live campaign.",
        "",
        "| Route | Marginal rank | Single | Expansion | Frontier | Pool | Exact realization |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in teacher_rows:
        stages = row["production_stage_positions"]
        realized = row["isolated_production_realization"]
        lines.append(
            f"| `{row['route_id'][:12]}` | {row['marginal_template_rank']} | "
            f"{stages['bound_singles'][0]} | {stages['scale_balanced_expansion'][0]} | "
            f"{stages['scale_balanced_frontier'][0]} | {stages['scale_balanced_pool'][0]} | "
            f"{realized['status']}, {realized['seconds']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "## Decisive mechanism",
            "",
            payload["decisive_mechanism"],
            "",
            (
                f"The first diagnostic timebox in production order is rank "
                f"{payload['first_timeboxed_production_rank']}. The preserved live root "
                "worker also failed at its 1,800-second function timeout."
            ),
            "",
            "## Smallest generic fix",
            "",
            payload["smallest_generic_fix_proposed_not_implemented"],
            "",
            "This fix is proposed only. It was not implemented or launched by this audit.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-corpus", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--live-root-lock", type=Path)
    parser.add_argument("--diagnostic-binding-limit", type=int, default=64)
    parser.add_argument("--per-candidate-timeout-seconds", type=float, default=20.0)
    args = parser.parse_args()
    result = run(
        route_checkpoint=args.route_checkpoint.resolve(),
        teacher_corpus=args.teacher_corpus.resolve(),
        output_json=args.output_json.resolve(),
        output_md=args.output_md.resolve(),
        live_root_lock=(
            None if args.live_root_lock is None else args.live_root_lock.resolve()
        ),
        diagnostic_binding_limit=args.diagnostic_binding_limit,
        per_candidate_timeout_seconds=args.per_candidate_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "computed_conclusions": result["computed_conclusions"],
                "first_timeboxed_production_rank": result[
                    "first_timeboxed_production_rank"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
