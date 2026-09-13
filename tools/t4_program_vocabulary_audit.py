"""Audit specificity and cross-cell applicability of the frozen T4 program bank.

This is a zero-oracle, read-only audit. It does not alter the frozen bank or the
running T4 benchmark and does not treat public-winner reconstruction as discovery.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from rdkit import rdBase

from compose_v4.control.docking_value import identity
from compose_v4.control.edit_program import (
    EditProgram,
    ProgramExecutionError,
    attachment_bindings,
    execute_bound_program,
)
from compose_v4.control.edit_program_policy import ProgramEntry
from compose_v4.experiments.continuation_profile import sha256_file, verify_file
from compose_v4.experiments.t4_matched_pilot import seal, unseal
from compose_v4.rewrite.trace_shard import decode_state

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/t4_frozen_program_benchmark_v2.json"
SEEDS = ROOT / "docs/GENMOL_T4_SEEDS.json"
CENSUS = ROOT / "diagnostics/ivg_t4_census/census.json"
RETRIEVAL = ROOT / "diagnostics/t4_program_retrieval/attempt_2/result.json"
ARCHIVES = {
    cell: ROOT / "diagnostics/t4_program_curriculum/attempt_1" / f"{cell}_archive.json"
    for cell in ("5ht1b_0", "braf_1", "fa7_0", "jak2_1")
}
IMPLEMENTATION = (
    "src/compose_v4/control/edit_program.py",
    "src/compose_v4/control/edit_program_policy.py",
    "src/compose_v4/control/program_transfer.py",
    "tools/prepare_t4_program_transfer.py",
    "tools/t4_program_retrieval_probe.py",
    "tools/t4_program_vocabulary_audit.py",
)
MAX_BINDINGS = 64
MAX_VISITS = 4096
MAX_PRIMITIVES = 32
MAX_BLOCKS = 8


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_tree_status() -> list[str]:
    return subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).splitlines()


def _winner_index(census: dict) -> dict[str, dict[str, float]]:
    winners: dict[str, dict[str, float]] = {}
    for row in census["cells"]:
        if row["delta"] != 0.4:
            continue
        cell = f"{row['target']}_{row['source_idx']}"
        values: dict[str, float] = {}
        for run in row["runs"]:
            for winner in run["winners"]:
                endpoint = winner["canonical_smiles"]
                values[endpoint] = min(
                    values.get(endpoint, float("inf")),
                    run["reported_docking_score"],
                )
        winners[cell] = values
    if len(winners) != 15:
        raise ValueError("public census lacks one or more delta-0.4 T4 cells")
    return winners


def source_group_map(contract: dict, seed_registry: list[dict]) -> dict[str, dict]:
    """Invert the declared source identity over the finite public 15-cell registry."""
    by_index = {row["idx"]: row for row in seed_registry}
    if len(by_index) != 15:
        raise ValueError("T4 seed registry must contain 15 unique global indices")
    result = {}
    for cell, saved in sorted(contract["cells"].items()):
        row = by_index.get(saved["global_index"])
        if row is None or (row["target"], row["smiles"]) != (
            saved["target"],
            saved["original_seed"],
        ):
            raise ValueError(f"seed registry and frozen source disagree: {cell}")
        group = identity(
            {
                "original_benchmark_seed": saved["original_seed"],
                "target": saved["target"],
            }
        )
        if group in result:
            raise ValueError(f"source-group collision in public registry: {group}")
        result[group] = {
            "cell": cell,
            "target": saved["target"],
            "source_idx": saved["source_idx"],
            "global_index": saved["global_index"],
        }
    return result


def origin_relation(recipient_cell: str, origins: list[dict]) -> str:
    recipient_target = recipient_cell.rsplit("_", 1)[0]
    origin_cells = {row["cell"] for row in origins}
    origin_targets = {row["target"] for row in origins}
    if recipient_cell in origin_cells:
        return "includes_same_cell"
    if recipient_target in origin_targets:
        return "includes_same_target_other_seed"
    return "cross_target_only"


def _walk(value):
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)


def _slot_operands(record: dict) -> list[object]:
    rule, payload = record["executor_rule"], record["payload"]
    if rule == "atom_insert":
        return [payload["slot"], *(row[0] for row in payload["neighbors"])]
    if rule in ("atom_delete", "atom_restate_semantic"):
        return [payload["v"]]
    if rule in ("cycle_close", "cycle_open", "bond_reorder"):
        return [payload["a"], payload["b"]]
    if rule == "bond_reroute":
        return [payload[field] for field in ("a", "b", "u", "v")]
    if rule == "ring_system_restate":
        return [change[field] for change in payload["changes"] for field in ("a", "b")]
    raise ValueError(f"unsupported frozen program executor rule: {rule!r}")


def inspect_program_payload(
    payload: dict,
    *,
    targets: set[str],
    seeds: set[str],
    winner_endpoints: set[str],
) -> dict:
    """Inspect literal fields and address/reference form without interpreting scores."""
    program = EditProgram.from_payload(payload)
    expanded = {**payload, "marks": [json.loads(text) for text in payload["marks"]]}
    scalars = list(_walk(expanded))
    keys = [
        str(key).lower()
        for node in _walk(expanded)
        if isinstance(node, dict)
        for key in node
    ]
    literal_targets = sorted(
        targets & {value for value in scalars if isinstance(value, str)}
    )
    literal_seeds = sorted(
        seeds & {value for value in scalars if isinstance(value, str)}
    )
    literal_winners = sorted(
        winner_endpoints & {value for value in scalars if isinstance(value, str)}
    )
    records = [json.loads(text) for text in program.marks]
    operands = [operand for record in records for operand in _slot_operands(record)]
    typed = [
        operand
        for operand in operands
        if isinstance(operand, dict)
        and len(operand) == 1
        and next(iter(operand)) in ("input", "created")
        and type(next(iter(operand.values()))) is int
    ]
    raw_integer_operands = [operand for operand in operands if type(operand) is int]
    malformed_operands = [
        operand
        for operand in operands
        if operand not in typed and type(operand) is not int
    ]
    operators = Counter(record["executor_rule"] for record in records)
    labels = [block.label for block in program.blocks]
    return {
        "program_id": program.program_id,
        "marks": len(program.marks),
        "blocks": len(program.blocks),
        "block_labels": labels,
        "input_atoms": len(program.input_atoms),
        "operator_counts": dict(sorted(operators.items())),
        "typed_atom_operands": len(typed),
        "raw_integer_atom_operands": len(raw_integer_operands),
        "malformed_atom_operands": len(malformed_operands),
        "literal_target_values": literal_targets,
        "literal_seed_smiles": literal_seeds,
        "literal_winner_endpoint_smiles": literal_winners,
        "target_named_fields": sorted({key for key in keys if "target" in key}),
        "seed_named_fields": sorted({key for key in keys if "seed" in key}),
        "score_named_fields": sorted(
            {key for key in keys if "score" in key or "docking" in key}
        ),
        "endpoint_named_fields": sorted(
            {key for key in keys if "endpoint" in key or key == "smiles"}
        ),
        "compiled_complete_winner_route": labels
        == ["compiled_complete_transformation"],
        "static_parameter_form": (
            "fully_specified_primitive_payloads_with_typed_atom_handles; "
            "no open parameter variables"
        ),
    }


def _library_entries(library_rows: list[dict]) -> tuple[ProgramEntry, ...]:
    entries = tuple(
        ProgramEntry(
            EditProgram.from_payload(row["program"]), tuple(row["source_groups"])
        )
        for row in library_rows
    )
    ids = [entry.program.program_id for entry in entries]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("frozen library must be program-unique and ID-sorted")
    return entries


def _applicability(
    entries: tuple[ProgramEntry, ...], contract: dict, group_map: dict[str, dict]
) -> tuple[list[dict], dict]:
    source_states = {
        cell: decode_state(saved["source_state"])
        for cell, saved in sorted(contract["cells"].items())
    }
    program_rows = []
    pair_counts = Counter()
    target_counts = defaultdict(Counter)
    total_binding_visits = total_execution_attempts = 0
    for program_index, entry in enumerate(entries):
        origins = [group_map[group] for group in entry.source_groups]
        cells = []
        for cell, source in source_states.items():
            census = attachment_bindings(
                entry.program,
                source,
                max_bindings=MAX_BINDINGS,
                max_visits=MAX_VISITS,
                contextual=True,
            )
            total_binding_visits += census.visits
            relation = origin_relation(cell, origins)
            failures = Counter()
            attempts = 0
            success = False
            for assignment in census.assignments:
                attempts += 1
                total_execution_attempts += 1
                try:
                    execute_bound_program(
                        source,
                        entry.program,
                        assignment,
                        max_primitives=MAX_PRIMITIVES,
                        max_blocks=MAX_BLOCKS,
                    )
                except ProgramExecutionError as error:
                    failures[f"ProgramExecutionError:{error.step}"] += 1
                except ValueError as error:
                    failures[type(error).__name__] += 1
                else:
                    success = True
                    break
            pair_counts["pairs"] += 1
            pair_counts["binding_available"] += bool(census.assignments)
            pair_counts["exact_executable"] += success
            pair_counts[f"relation:{relation}:pairs"] += 1
            pair_counts[f"relation:{relation}:binding"] += bool(census.assignments)
            pair_counts[f"relation:{relation}:executable"] += success
            target = contract["cells"][cell]["target"]
            target_counts[target]["pairs"] += 1
            target_counts[target]["binding_available"] += bool(census.assignments)
            target_counts[target]["exact_executable"] += success
            cells.append(
                {
                    "cell": cell,
                    "relation_to_origin": relation,
                    "binding_count": len(census.assignments),
                    "binding_visits": census.visits,
                    "binding_truncated": census.truncated,
                    "execution_attempts_until_first_success": attempts,
                    "exact_executable": success,
                    "observed_failed_bindings_before_success": sum(failures.values()),
                    "failure_counts": dict(sorted(failures.items())),
                }
            )
        binding_cells = [row["cell"] for row in cells if row["binding_count"]]
        executable_cells = [row["cell"] for row in cells if row["exact_executable"]]
        binding_targets = sorted({cell.rsplit("_", 1)[0] for cell in binding_cells})
        executable_targets = sorted(
            {cell.rsplit("_", 1)[0] for cell in executable_cells}
        )
        origin_targets = sorted({row["target"] for row in origins})
        program_rows.append(
            {
                "program_index": program_index,
                "program_id": entry.program.program_id,
                "origin_source_groups": list(entry.source_groups),
                "origins": origins,
                "origin_targets": origin_targets,
                "binding_cells": binding_cells,
                "binding_targets": binding_targets,
                "exact_executable_cells": executable_cells,
                "exact_executable_targets": executable_targets,
                "cross_target_binding": bool(
                    set(binding_targets) - set(origin_targets)
                ),
                "cross_target_exact_execution": bool(
                    set(executable_targets) - set(origin_targets)
                ),
                "cells": cells,
            }
        )
    by_target = {
        target: dict(sorted(counts.items())) for target, counts in target_counts.items()
    }
    summary = {
        **dict(sorted(pair_counts.items())),
        "programs": len(entries),
        "programs_with_any_binding": sum(
            bool(row["binding_cells"]) for row in program_rows
        ),
        "programs_with_any_exact_execution": sum(
            bool(row["exact_executable_cells"]) for row in program_rows
        ),
        "programs_with_cross_target_binding": sum(
            row["cross_target_binding"] for row in program_rows
        ),
        "programs_with_cross_target_exact_execution": sum(
            row["cross_target_exact_execution"] for row in program_rows
        ),
        "total_binding_visits": total_binding_visits,
        "total_execution_attempts_until_first_success": total_execution_attempts,
        "observed_first_success_fraction": (
            pair_counts["exact_executable"] / total_execution_attempts
            if total_execution_attempts
            else None
        ),
        "stopping_policy": "try context-ranked bindings until first exact success per program-cell",
        "by_recipient_target": by_target,
    }
    return program_rows, summary


def _draw_origins(
    cell: str,
    draws: list[dict],
    entries: tuple[ProgramEntry, ...],
    group_map: dict[str, dict],
) -> list[dict]:
    result = []
    for draw in draws:
        index = draw["program_index"]
        if type(index) is not int or not 0 <= index < len(entries):
            raise ValueError(f"invalid archived program index for {cell}: {index!r}")
        entry = entries[index]
        origins = [group_map[group] for group in entry.source_groups]
        result.append(
            {
                "program_index": index,
                "program_id": entry.program.program_id,
                "origins": origins,
                "relation_to_recipient": origin_relation(cell, origins),
            }
        )
    return result


def _audit_retrieval(
    winners: dict[str, dict[str, float]],
    entries: tuple[ProgramEntry, ...],
    group_map: dict[str, dict],
) -> dict:
    payload = unseal(RETRIEVAL)
    base = RETRIEVAL.parent
    recovered = []
    for row in payload["rows"]:
        cell = row["cell"]
        artifact = base / row["retrieval"]["artifact"]
        verify_file(artifact, row["retrieval"]["artifact_sha256"])
        batch = unseal(artifact)
        candidates = {
            candidate["endpoint"]: candidate for candidate in batch["candidates"]
        }
        for endpoint in row["new_winner_recovery"]:
            if endpoint not in candidates or endpoint not in winners[cell]:
                raise ValueError(
                    f"retrieval winner reconciliation failed: {cell} {endpoint}"
                )
            draws = candidates[endpoint]["provenance"]["metadata"]["draws"]
            recovered.append(
                {
                    "cell": cell,
                    "endpoint": endpoint,
                    "external_reported_score": winners[cell][endpoint],
                    "program_origins": _draw_origins(cell, draws, entries, group_map),
                }
            )
    summary = payload["summary"]
    if len(recovered) != summary["new_winners_recovered"]:
        raise ValueError("sealed retrieval winner total does not reconcile")
    return {
        "artifact_schema_version": payload["schema_version"],
        "sealed_summary": summary,
        "recovered": recovered,
        "origin_relation_counts": dict(
            sorted(
                Counter(
                    origin["relation_to_recipient"]
                    for row in recovered
                    for origin in row["program_origins"]
                ).items()
            )
        ),
        "evidence_class": (
            "answer-known direct reconstruction against public IVG winner endpoints; "
            "not autonomous discovery"
        ),
    }


def _audit_strong_archives(
    entries: tuple[ProgramEntry, ...], group_map: dict[str, dict]
) -> list[dict]:
    rows = []
    library_ids = {entry.program.program_id for entry in entries}
    for cell, path in sorted(ARCHIVES.items()):
        artifact = json.loads(path.read_text())
        optimizer = artifact["optimizer"]
        observations = list(optimizer["observations"].values())
        if not observations:
            raise ValueError(f"empty measured development archive: {path}")
        best = min(observations, key=lambda row: (row["score"], row["endpoint"]))
        matches = [
            row
            for row in optimizer["entries"].values()
            if row["endpoint"] == best["endpoint"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"best archive endpoint has {len(matches)} entries: {path}"
            )
        entry = matches[0]
        draws = entry["provenance"]["metadata"]["draws"]
        archived_program = EditProgram.from_payload(entry["program"])
        rows.append(
            {
                "cell": cell,
                "endpoint": best["endpoint"],
                "measured_docking_score": best["score"],
                "candidate_id": entry["candidate_id"],
                "proposal_channel": entry["provenance"]["channel"],
                "mutations": entry["provenance"]["metadata"].get("mutations", []),
                "archived_program_id": archived_program.program_id,
                "archived_program_is_unmodified_library_entry": archived_program.program_id
                in library_ids,
                "base_program_origins": _draw_origins(cell, draws, entries, group_map),
            }
        )
    return rows


def _write_report(output: Path, body: dict) -> None:
    summary = body["summary"]
    literal = body["literal_and_representation_audit"]["summary"]
    applicability = body["applicability"]["summary"]
    retrieval = body["direct_retrieval_audit"]["sealed_summary"]
    report = f"""# Frozen T4 Program-Vocabulary Audit

Generated from commit `{body['code_revision']}` with zero oracle calls.

## Outcome

The frozen bank is address-free and context-rebound, but it is not only an
abstract grammar. It is a shared, winner-informed executable program bank that
contains {summary['compiled_complete_winner_routes']} compiled complete
winner-route programs and {summary['other_programs']} other programs. Static
program payloads are fully specified primitive sequences; mutation and
recombination provide runtime variation.

- Literal target values: {literal['programs_with_literal_target_values']}
- Literal seed SMILES: {literal['programs_with_literal_seed_smiles']}
- Literal winner endpoint SMILES: {literal['programs_with_literal_winner_endpoints']}
- Programs with raw integer atom operands: {literal['programs_with_raw_integer_atom_operands']}
- Opaque source groups mapped to public T4 cells: {summary['mapped_source_groups']}/15
- Programs with exact execution on at least one cell: {applicability['programs_with_any_exact_execution']}/{summary['programs']}
- Programs with exact execution on another target: {applicability['programs_with_cross_target_exact_execution']}/{summary['programs']}
- Existing direct retrieval recovered {retrieval['new_winners_recovered']} new known
  public winners across {retrieval['cells_with_new_winner_recovery']}/15 cells.

## Interpretation

The literal payload audit supports the implementation claim that physical source
slot addresses, explicit endpoint molecules, numeric scores and target-routing
fields are absent from program payloads. The source-group hashes are nevertheless
uniquely invertible by enumerating the public 15-cell registry, and complete
winner-derived primitive routes remain in the bank. Direct known-winner recovery
shows that some entries preserve enough route specificity to reconstruct public
answers. Cross-cell binding and exact execution measure transfer applicability,
not held-out task value.

The frozen official T4 controller uses one shared bank and disables direct
retrieval (`cold_start_retrieval_candidates=0`), while mutation and recombination
remain active. These facts reduce literal replay in that run but do not erase the
winner-informed prior. The scientifically accurate description is therefore
"shared, address-free, context-bound, winner-informed executable program bank,"
not a target-specific lookup table and not a purified task-independent grammar.
"""
    temporary = output / "report.md.tmp"
    temporary.write_text(report)
    temporary.replace(output / "report.md")


def run(output: Path) -> dict:
    if output.exists():
        raise ValueError(
            f"output already exists; use a new immutable attempt: {output}"
        )
    dirty = _source_tree_status()
    if dirty:
        raise ValueError(f"audit requires a clean committed source tree: {dirty}")
    began = perf_counter()
    code_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    contract = unseal(CONTRACT)
    if contract["schema_version"] != "t4_frozen_program_benchmark_v2":
        raise ValueError("unexpected frozen T4 contract schema")
    library_path = ROOT / contract["library_path"]
    verify_file(SEEDS, contract["inputs"][str(SEEDS.relative_to(ROOT))])
    verify_file(CENSUS, contract["inputs"][str(CENSUS.relative_to(ROOT))])
    verify_file(library_path, contract["inputs"][contract["library_path"]])
    library_rows = json.loads(library_path.read_text())
    if len(library_rows) != contract["library_programs"] != 146:
        raise ValueError("frozen T4 library count is not exactly 146")
    entries = _library_entries(library_rows)
    registry = json.loads(SEEDS.read_text())
    groups = source_group_map(contract, registry)
    library_groups = {group for entry in entries for group in entry.source_groups}
    if library_groups != set(groups):
        raise ValueError(
            f"library source groups do not equal the 15-cell registry: "
            f"missing={set(groups) - library_groups}, unknown={library_groups - set(groups)}"
        )
    census = json.loads(CENSUS.read_text())
    winners = _winner_index(census)
    all_winner_endpoints = {
        endpoint for values in winners.values() for endpoint in values
    }
    target_names = {saved["target"] for saved in contract["cells"].values()}
    seed_smiles = {saved["original_seed"] for saved in contract["cells"].values()}
    inspected = [
        inspect_program_payload(
            row["program"],
            targets=target_names,
            seeds=seed_smiles,
            winner_endpoints=all_winner_endpoints,
        )
        for row in library_rows
    ]
    literal_summary = {
        "programs_with_literal_target_values": sum(
            bool(row["literal_target_values"]) for row in inspected
        ),
        "programs_with_literal_seed_smiles": sum(
            bool(row["literal_seed_smiles"]) for row in inspected
        ),
        "programs_with_literal_winner_endpoints": sum(
            bool(row["literal_winner_endpoint_smiles"]) for row in inspected
        ),
        "programs_with_target_named_fields": sum(
            bool(row["target_named_fields"]) for row in inspected
        ),
        "programs_with_seed_named_fields": sum(
            bool(row["seed_named_fields"]) for row in inspected
        ),
        "programs_with_score_named_fields": sum(
            bool(row["score_named_fields"]) for row in inspected
        ),
        "programs_with_endpoint_named_fields": sum(
            bool(row["endpoint_named_fields"]) for row in inspected
        ),
        "programs_with_raw_integer_atom_operands": sum(
            bool(row["raw_integer_atom_operands"]) for row in inspected
        ),
        "programs_with_malformed_atom_operands": sum(
            bool(row["malformed_atom_operands"]) for row in inspected
        ),
        "typed_atom_operands": sum(row["typed_atom_operands"] for row in inspected),
    }
    applicability_rows, applicability_summary = _applicability(
        entries, contract, groups
    )
    retrieval = _audit_retrieval(winners, entries, groups)
    strong = _audit_strong_archives(entries, groups)
    route_count = sum(row["compiled_complete_winner_route"] for row in inspected)
    controller = contract["controller"]
    body = {
        "schema_version": "t4_program_vocabulary_audit_v1",
        "generated_at": _stamp(),
        "code_revision": code_revision,
        "inputs_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                CONTRACT,
                SEEDS,
                CENSUS,
                RETRIEVAL,
                library_path,
                *ARCHIVES.values(),
            )
        },
        "implementation_sha256": {
            path: sha256_file(ROOT / path) for path in IMPLEMENTATION
        },
        "configuration": {
            "cells": 15,
            "programs": 146,
            "max_bindings": MAX_BINDINGS,
            "max_binding_visits": MAX_VISITS,
            "max_primitives": MAX_PRIMITIVES,
            "max_blocks": MAX_BLOCKS,
            "binding_order": "production contextual ranking",
            "execution_stopping": "first exact success per program-cell",
            "new_oracle_calls": 0,
        },
        "frozen_controller_facts": {
            "proposal_mode": controller["proposal_mode"],
            "channel_probabilities": controller["channel_probabilities"],
            "cold_start_retrieval_candidates": controller.get(
                "cold_start_retrieval_candidates", 0
            ),
            "direct_retrieval_enabled": bool(
                controller.get("cold_start_retrieval_candidates", 0)
            ),
            "require_broad_runtime": controller["require_broad_runtime"],
            "broad_channel_probability": controller["channel_probabilities"][2],
            "mutation_sampling": controller["mutation_sampling"],
            "decompose_programs": controller["decompose_programs"],
            "interpretation": (
                "one shared bank; direct retrieval and broad runtime are disabled in "
                "this frozen configuration; mutation and recombination remain"
            ),
        },
        "source_group_audit": {
            "encoding": (
                "SHA-256 identity of canonical JSON containing original_benchmark_seed "
                "and target"
            ),
            "opaque_in_library": True,
            "uniquely_mapped_by_public_15_cell_enumeration": True,
            "groups": [{"source_group": key, **groups[key]} for key in sorted(groups)],
            "use_in_production_policy": "source-balanced prior only; no recipient-target lookup",
        },
        "literal_and_representation_audit": {
            "summary": literal_summary,
            "programs": inspected,
            "address_finding": (
                "all authoritative atom operands use typed input/created handles; "
                "physical source slots are absent"
            ),
            "parameter_finding": (
                "bank entries are fixed primitive programs, not open templates; separate "
                "runtime mutation and recombination operators create variants"
            ),
        },
        "applicability": {
            "summary": applicability_summary,
            "programs": applicability_rows,
            "claim_boundary": (
                "binding and exact execution are structural applicability, not task-score "
                "utility or held-out generalization"
            ),
        },
        "direct_retrieval_audit": retrieval,
        "strong_development_candidate_provenance": strong,
        "summary": {
            "programs": len(entries),
            "compiled_complete_winner_routes": route_count,
            "other_programs": len(entries) - route_count,
            "mapped_source_groups": len(groups),
            "origin_source_group_multiplicity": dict(
                sorted(Counter(len(entry.source_groups) for entry in entries).items())
            ),
            "scientific_description": (
                "shared, address-free, context-bound, winner-informed executable "
                "program bank containing transferable modules and complete route programs"
            ),
            "not_supported": [
                "purified task-independent grammar",
                "target-specific lookup table",
                "held-out generalization",
                "autonomous winner discovery from this audit",
            ],
        },
        "evidence_ledger": {
            "computed": [
                "literal/schema/address audit",
                "source-group inversion over the declared 15-cell registry",
                "bounded cross-cell binding and exact-execution applicability",
                "archive provenance reconciliation",
            ],
            "reported": ["public IVG winner endpoint identities and docking scores"],
            "measured_prior": ["four COMPOSE development archive docking scores"],
            "inferred": ["scientific description and claim boundary"],
        },
        "limitations": [
            "All T4 sources and public winners were available during development.",
            "Binding search is capped at 64 assignments and 4,096 visits per pair.",
            "Execution stops after the first success, so observed success fraction is not full-binding precision.",
            "Structural applicability does not establish docking usefulness.",
            "The four strong-candidate provenance records are selected development examples, not independent validation.",
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {"device": "cpu", "workers": 1, "machine": platform.machine()},
        "costs": {
            "wall_seconds": perf_counter() - began,
            "binding_visits": applicability_summary["total_binding_visits"],
            "execution_attempts": applicability_summary[
                "total_execution_attempts_until_first_success"
            ],
            "new_oracle_calls": 0,
            "modal_calls": 0,
        },
    }
    output.mkdir(parents=True)
    seal(output / "result.json", body)
    _write_report(output, body)
    print(json.dumps({**body["summary"], **applicability_summary}, indent=2))
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
