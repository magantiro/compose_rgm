"""Reproduce the bounded, zero-oracle JAK2-0 delta=0.6 forensic audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import rdkit

from compose_v4.gates.med_chem_gate import validity_reasons

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).with_name("result.json")
IVG = -9.7

INPUTS = {
    "current_matrix": (
        "diagnostics/t4_shared_controller_matrix_20260919/result.json",
        "1b0c8ec8e202e19b758a653a0a23f07c1f539410c49b098c5489181b6dac3c8c",
    ),
    "current_final": (
        "diagnostics/t4_integrated_route_fiber_v1/final_result.json",
        "fa1fbae94f7d8fc9f08cfa383b8d435be132a237a321e41792478b15600aed01",
    ),
    "current_contract": (
        "configs/t4_integrated_route_fiber_v1_1.json",
        "74fd3a1c0fce95ee0489107cbe951a23825af44bef2feb1a0c8dfd8850f5d96e",
    ),
    "current_route_checkpoint": (
        "diagnostics/t4_integrated_route_fiber_v1/route_expert_checkpoint.json",
        "d9771225c7a9dc855330318ea4475246b7328107c9ad95da11c976098ca3fb25",
    ),
    "historical_scored_rows": (
        "diagnostics/t4_strategy_reset/20260916/scored_rows.jsonl.gz",
        "0e78c3b19839c8535bf342d4120ec72f1086d11b9260b46e0850878621153bcf",
    ),
    "historical_audit": (
        "diagnostics/t4_strategy_reset/20260916/audit.json",
        "c78d2a9af4961a98208289d1b362b71ca383b3e0a24877c575e944692c3fcdc3",
    ),
    "old_route_forensics": (
        "diagnostics/t4_known_good_transformation_forensics/attempt_1/result.json",
        "b332e768d2ffd4b2b298d565aea4270fcef890c0e6c6e4b01add10ea8cc33724",
    ),
    "shared_route_checkpoint": (
        "diagnostics/t4_shared_retained_rewrite_v1/checkpoint.json",
        "cb0d0bd0130b31f956c320ccf8171ce897f797506c8cf1a524a7f697c749865e",
    ),
    "shared_jak2_0_support": (
        "diagnostics/t4_shared_retained_rewrite_v1/shards/jak2_0.json",
        "0e0c9b70eb3ac590e09803545e6fdc7dfca9ef3f39ab821fad09e9bcc9cc4755",
    ),
    "shared_jak2_contract": (
        "configs/t4_shared_retained_fiber_jak2_v1.json",
        "6884b616e1b896001a7e0e21d45049da8ae7a29b6acfd1c1b77e5f3cc7705388",
    ),
    "compose_endpoint_gate": (
        "src/compose_v4/gates/med_chem_gate.py",
        "4503e1e72d7c8022b3b5cf24f36bfd2e63dea5a4eb261cb018ce06d508928946",
    ),
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def payload_sha(envelope: dict[str, Any]) -> str | None:
    if "payload" not in envelope or "payload_sha256" not in envelope:
        return None
    actual = hashlib.sha256(canonical_bytes(envelope["payload"])).hexdigest()
    if actual != envelope["payload_sha256"]:
        raise ValueError(f"payload hash mismatch: {actual} != {envelope['payload_sha256']}")
    return actual


def checked_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    loaded: dict[str, Any] = {}
    ledger: dict[str, Any] = {}
    for name, (relative, expected) in INPUTS.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required local input absent: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"physical hash mismatch for {relative}: {actual} != {expected}")
        data = load_json(path) if path.suffix == ".json" else None
        loaded[name] = data
        ledger[name] = {
            "path": relative,
            "physical_sha256": actual,
            "payload_sha256": payload_sha(data) if isinstance(data, dict) else None,
        }
    return loaded, ledger


def historical_rows() -> list[dict[str, Any]]:
    path = ROOT / INPUTS["historical_scored_rows"][0]
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("cell") == "jak2_0":
                rows.append(row)
    return rows


def arm_summary(
    rows: list[dict[str, Any]], arm: str, shared_route_endpoints: set[str]
) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    best = min(selected, key=lambda row: row["score"])
    meeting = [row for row in selected if row["score"] <= IVG]
    overlap = [row for row in selected if row["endpoint"] in shared_route_endpoints]
    by_replicate = []
    for replicate in sorted({int(row["replicate"]) for row in selected}):
        group = [row for row in selected if int(row["replicate"]) == replicate]
        winner = min(group, key=lambda row: row["score"])
        by_replicate.append(
            {
                "replicate": replicate,
                "scored_rows": len(group),
                "best_reported_historical_score": winner["score"],
                "best_endpoint": winner["endpoint"],
                "query": winner["query"],
                "query_verified": winner["query_verified"],
            }
        )
    return {
        "arm": arm,
        "evidence_class": "reported_historical_labels_not_current_oracle_evidence",
        "scored_rows": len(selected),
        "unique_endpoints": len({row["endpoint"] for row in selected}),
        "best_reported_historical_score": best["score"],
        "best_endpoint": best["endpoint"],
        "reported_historical_rows_at_or_better_than_ivg": len(meeting),
        "unique_reported_historical_endpoints_at_or_better_than_ivg": len(
            {row["endpoint"] for row in meeting}
        ),
        "query_verified_census": dict(
            sorted(Counter(str(row["query_verified"]).lower() for row in selected).items())
        ),
        "shared_route_pool_overlap": {
            "scored_rows": len(overlap),
            "unique_endpoints": len({row["endpoint"] for row in overlap}),
            "best_reported_historical_score": min((row["score"] for row in overlap), default=None),
            "any_reported_historical_score_at_or_better_than_ivg": any(
                row["score"] <= IVG for row in overlap
            ),
            "best_arm_endpoint_is_in_shared_route_pool": (
                best["endpoint"] in shared_route_endpoints
            ),
        },
        "by_replicate": by_replicate,
    }


def git_commit_exists(revision: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data, input_ledger = checked_inputs()
    final = data["current_final"]["payload"]
    contract = data["current_contract"]["payload"]
    current_checkpoint = data["current_route_checkpoint"]["payload"]
    shared_checkpoint = data["shared_route_checkpoint"]["payload"]
    support = data["shared_jak2_0_support"]["payload"]
    old_forensics = data["old_route_forensics"]["payload"]
    shared_contract = data["shared_jak2_contract"]["payload"]

    current_cell = next(row for row in final["cells"] if row["cell"] == "jak2_0")
    matrix_cell = next(
        row
        for row in data["current_matrix"]["records"]
        if row["cell"] == "jak2_0" and row["delta"] == 0.6
    )
    if current_cell["final_best"] != matrix_cell["compose"]:
        raise ValueError("current matrix and authoritative run result disagree")

    old_ids = set(current_checkpoint["expert"]["marginal"]["template_ids"])
    shared_ids = set(shared_checkpoint["expert"]["marginal"]["template_ids"])
    support_cell = next(row for row in support["cells"] if row["cell"] == "jak2_0")
    candidates = support_cell["candidates"]
    candidate_by_endpoint = {row["smiles"]: row for row in candidates}

    rows = historical_rows()
    rows_by_endpoint: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_endpoint.setdefault(row["endpoint"], []).append(row)

    receipt_ledger = []
    known_routes = []
    for route in old_forensics["routes"]:
        if route["cell"] != "jak2_0":
            continue
        receipt_path = ROOT / route["teacher_receipt"]
        receipt_hash = sha256(receipt_path)
        if receipt_hash != route["teacher_receipt_sha256"]:
            raise ValueError(f"teacher receipt hash mismatch: {route['teacher_receipt']}")
        receipt = load_json(receipt_path)
        payload_sha(receipt)
        target = receipt["pair"]["target"]
        label = route["known_labels"][0]
        candidate = candidate_by_endpoint.get(target)
        historical = [
            {
                "arm": row["arm"],
                "replicate": row["replicate"],
                "reported_historical_score": row["score"],
                "evidence_class": "reported_historical_label_not_current_oracle_evidence",
                "query": row["query"],
                "query_verified": row["query_verified"],
                "protocol": row["protocol"],
                "entry_id": row["entry_id"],
            }
            for row in sorted(
                rows_by_endpoint.get(target, []),
                key=lambda value: (value["arm"], value["replicate"], value["score"]),
            )
        ]
        if candidate is None:
            runtime = {"autonomously_realized_in_shared_support_shard": False}
        else:
            template_ids = candidate["route_template_ids"]
            runtime = {
                "autonomously_realized_in_shared_support_shard": True,
                "route_proposal_rank": candidate["route_proposal_rank"],
                "regions": candidate["regions"],
                "realized_primitives": candidate["realized_primitives"],
                "realized_primitive_band": candidate["realized_primitive_band"],
                "rewrite_scale": candidate["rewrite_scale"],
                "exact_teacher_endpoint": candidate["exact_teacher_endpoint"],
                "eligible_delta_0_6": candidate["eligibility"]["0.6"]["eligible"],
                "compose_validity_reasons": validity_reasons(target),
                "template_ids": template_ids,
                "exact_template_present_in_scored_run_checkpoint": all(
                    value in old_ids for value in template_ids
                ),
                "exact_template_present_in_shared_checkpoint": all(
                    value in shared_ids for value in template_ids
                ),
            }
        known_routes.append(
            {
                "probe_id": route["probe_id"],
                "endpoint": target,
                "reported_external_label": label["reported_external_docking_score"],
                "reported_label_delta": label["delta"],
                "reported_label_seed": label["run_seed"],
                "reported_label_meets_or_beats_ivg": (
                    label["delta"] == 0.6 and label["reported_external_docking_score"] <= IVG
                ),
                "dependency_regions": route["dependency_regions"],
                "primitive_count": route["primitive_count"],
                "old_standalone_representation": route["representation"],
                "old_standalone_generator_support": {
                    "whole_macro_supported": route["generator"]["whole_macro_supported"],
                    "whole_macro_support_absence": route["generator"][
                        "whole_macro_support_absence"
                    ],
                    "single_invocation_runtime": route["generator"]["runtime_support"][
                        "actual_current_hybrid_runtime"
                    ],
                },
                "old_standalone_compiler": route["compiler"],
                "shared_route_runtime": runtime,
                "recovered_historical_archive_scores": historical,
            }
        )
        receipt_ledger.append(
            {
                "path": route["teacher_receipt"],
                "physical_sha256": receipt_hash,
                "payload_sha256": receipt["payload_sha256"],
            }
        )

    strong_delta_06 = [
        row
        for row in known_routes
        if row["reported_label_delta"] == 0.6 and row["reported_label_meets_or_beats_ivg"]
    ]
    eligible_exact_strong = [
        row
        for row in strong_delta_06
        if row["shared_route_runtime"].get("exact_teacher_endpoint")
        and row["shared_route_runtime"].get("eligible_delta_0_6")
    ]

    support_scales = {}
    for field, output_name in (
        ("realized_primitive_band", "primitive_bands"),
        ("rewrite_scale", "rewrite_scales"),
    ):
        support_scales[output_name] = {}
        for name in sorted({row[field] for row in candidates}):
            selected = [row for row in candidates if row[field] == name]
            support_scales[output_name][name] = {
                "realized": len(selected),
                "eligible_delta_0_6": sum(
                    row["eligibility"]["0.6"]["eligible"] for row in selected
                ),
            }

    deferred_commits = {
        "defer_joint_region_validity_until_stop": ("b5dcafa4d887d971a383aabab3fe1316ff9ada37"),
        "preserve_depth_coverage_in_joint_planning": ("412c9f03d1f8b3c252b2155c64ae790c203b3a71"),
    }
    if not all(git_commit_exists(value) for value in deferred_commits.values()):
        raise ValueError("required local joint-planning commits are absent")

    missing_prefix = f"{final['remote_path']}/jak2_0"
    payload = {
        "schema_version": "t4_jak2_0_d06_forensic_audit_v1",
        "status": "PARTIAL_ABSTENTION_MISSING_CURRENT_PROPOSAL_LOCKS",
        "audit_date": "2026-09-19",
        "scientific_problem": (
            "Localize the current shared-controller JAK2 seed-0 delta=0.6 "
            "0.1 docking-score miss without new scored evidence."
        ),
        "primary_model_output": (
            "A state-dependent distribution over complete executable molecular edit programs."
        ),
        "central_claim_under_test": (
            "Whether known strong JAK2 transformations were available to, proposed by, "
            "and retained by the scored shared controller."
        ),
        "experimental_setting": (
            "Read-only reconciliation of one 49-call JAK2 delta=0.6 campaign, frozen "
            "reported historical labels and routes, and later zero-oracle shared-route support."
        ),
        "declared_support": {
            "source": "jak2_0 / global source index 12",
            "delta": 0.6,
            "runtime_support": contract["support"],
            "representation_limit_heavy_atoms": 40,
            "route_primitive_limit": 32,
            "route_region_limit": 4,
            "limitations": [
                "Historical Full146 score labels are not current-run docking observations.",
                "The exact current candidate pools and selected-record ancestry are unavailable locally.",
                "The later all-route support shard postdates the scored delta=0.6 run.",
            ],
        },
        "costs": {
            "new_oracle_calls": 0,
            "new_docking_calls": 0,
            "modal_launches": 0,
            "live_run_accesses": 0,
            "network_calls": 0,
        },
        "inputs": input_ledger,
        "teacher_receipts": sorted(receipt_ledger, key=lambda row: row["path"]),
        "implementation": {
            "repository_revision_at_audit": ("412c9f03d1f8b3c252b2155c64ae790c203b3a71"),
            "audit_script_sha256": sha256(Path(__file__)),
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "historical_scored_rows": len(rows),
        },
        "current_scored_run": {
            "run_id": final["run_id"],
            "code_revision": final["code_revision"],
            "remote_volume": final["remote_volume"],
            "remote_path": final["remote_path"],
            "contract_payload_sha256": final["contract_payload_sha256"],
            "cell_result_physical_sha256": final["authoritative_artifact_sha256"][
                "jak2_0/result.json"
            ],
            "source_global_index": contract["cells"][0]["source_global_index"],
            "source_smiles": contract["cells"][0]["smiles"],
            "controller_seed": contract["cells"][0]["controller_seed"],
            "docking_seed": contract["docking_seed"],
            "charged_calls": current_cell["charged_calls"],
            "best_by_call": current_cell["best_by_call"],
            "final_best": current_cell["final_best"],
            "ivg_reported": current_cell["ivg_reported_delta_0_6"],
            "gap": current_cell["compose_minus_ivg"],
            "any_current_scored_candidate_meets_or_beats_ivg": (current_cell["final_best"] <= IVG),
            "best_smiles": current_cell["best_smiles"],
            "winner": {
                "proposal_experts": current_cell["winner_proposal_experts"],
                "selection_kind": current_cell["winner_selection_kind"],
                "parent_score": current_cell["winner_parent_score"],
                "round": current_cell["winner_round"],
            },
            "proposal_policy": {
                "families_scheduled_for_every_parent": [
                    "shallow",
                    "anchored_replacement",
                    "route_complete_region",
                ],
                "parents_per_round": contract["parents"],
                "batch": contract["batch"],
                "exploration_per_round": contract["exploration"],
                "expert_floor_rounds": contract["expert_floor_rounds"],
                "parent_explore": contract["parent_explore"],
                "proposal_parameters": contract["proposal"],
                "route_scales": (
                    "route records carried local/medium/large rewrite_scale, but this "
                    "revision had no scale floor; exact offered/selected scale census "
                    "requires the missing round locks"
                ),
                "aggregate_observation": final["mechanism_findings"],
            },
        },
        "missing_current_run_assets": {
            "effect": (
                "Exact candidate pools, per-round eligible/selected family and scale "
                "censuses, selected-record parent/program ancestry, and whether a "
                "historically known endpoint appeared before selection cannot be reconstructed."
            ),
            "required": [
                {
                    "identity": (f"modal://{final['remote_volume']}{missing_prefix}/result.json"),
                    "physical_sha256": final["authoritative_artifact_sha256"]["jak2_0/result.json"],
                },
                *[
                    {
                        "identity": (
                            f"modal://{final['remote_volume']}{missing_prefix}/"
                            f"round_{round_index:03d}_lock.json"
                        ),
                        "physical_sha256": (
                            "not recorded in the local aggregate; the payload hash is "
                            "referenced by the missing cell result"
                        ),
                    }
                    for round_index in range(1, 7)
                ],
            ],
            "not_accessed_because": "bounded audit forbids live-run or Modal access",
        },
        "historical_scored_evidence": {
            "ivg_threshold": IVG,
            "dynamic_v0_delta_0_6": arm_summary(rows, "v0d06", set(candidate_by_endpoint)),
            "full146": arm_summary(rows, "full146", set(candidate_by_endpoint)),
            "claim_boundary": (
                "Dynamic-v0 delta=0.6 is protocol-aligned by delta and reached only -9.5. "
                "Full146 contains many labels below -9.7, but those rows use its historical "
                "protocol, lack recovered exact call indices, and are not current-run oracle "
                "observations. "
                "Endpoint overlap with the later route shard does not establish support in "
                "the scored run or in Dynamic's stochastic proposal law."
            ),
        },
        "known_route_evidence": {
            "routes": sorted(known_routes, key=lambda row: row["probe_id"]),
            "delta_0_6_routes_meeting_ivg": len(strong_delta_06),
            "delta_0_6_routes_exactly_realized_and_eligible_in_shared_support": len(
                eligible_exact_strong
            ),
            "shared_checkpoint": {
                "training_scope": shared_checkpoint["training_scope"],
                "training_routes": shared_checkpoint["training_routes"],
                "training_regions": shared_checkpoint["training_regions"],
                "runtime_target_conditioning": shared_checkpoint["runtime_target_conditioning"],
                "template_count": len(shared_ids),
                "bound_by_later_shared_jak2_contract": (
                    shared_contract["runtime_inputs_sha256"][INPUTS["shared_route_checkpoint"][0]]
                    == INPUTS["shared_route_checkpoint"][1]
                ),
            },
            "scored_run_checkpoint": {
                "split_audit": current_checkpoint["split_audit"],
                "template_count": len(old_ids),
                "training_split": contract["proposal"]["route_complete_region"]["training_split"],
            },
            "shared_jak2_0_support": {
                "code_revision": support["code_revision"],
                "working_tree_dirty": support["working_tree_dirty"],
                "complete_programs": len(candidates),
                "eligible_delta_0_6": sum(
                    row["eligibility"]["0.6"]["eligible"] for row in candidates
                ),
                "exact_teacher_endpoints": sum(row["exact_teacher_endpoint"] for row in candidates),
                "exact_teacher_endpoints_eligible_delta_0_6": sum(
                    row["exact_teacher_endpoint"] and row["eligibility"]["0.6"]["eligible"]
                    for row in candidates
                ),
                "scale_census": support_scales,
                "exact_realization_precision": {
                    "numerator": support_cell["telemetry"]["exact_realization_precision_numerator"],
                    "denominator": support_cell["telemetry"][
                        "exact_realization_precision_denominator"
                    ],
                },
            },
        },
        "localization": {
            "observed": [
                (
                    "The current 49-call run scored no candidate at or below -9.7; its "
                    "best was -9.6 from shallow exploration."
                ),
                (
                    "The scored run used a leave-JAK2-out 111-template checkpoint. The "
                    "exact templates for all autonomously recovered JAK2 teacher endpoints "
                    "in the later support shard are absent from that checkpoint."
                ),
                (
                    "The later all-route shared checkpoint autonomously realized two "
                    "one-region, delta=0.6-eligible reported JAK2 routes at ranks 3 and 6; "
                    "their reported external labels are -10.6 and -10.8."
                ),
                (
                    "A third reported delta=0.6 route at -10.5 is exactly realized at "
                    "rank 12 but rejected by the current compose_valid endpoint support; "
                    "the route receipt records benchmark feasibility at delta=0.6."
                ),
                (
                    "The old standalone forensic represented all six JAK2 routes exactly "
                    "but its earlier one-to-two-event generator did not support the complete "
                    "control sequence and its compiler result was an explicit missing-payload abstention."
                ),
            ],
            "inferred": [
                (
                    "For the two eligible strong routes, the latest evidence rules out "
                    "binding, exact compilation, primitive budget, and joint-region "
                    "composition as the present support loss: each is one region and was "
                    "exactly emitted within 12 primitives."
                ),
                (
                    "Their absence from the scored delta=0.6 result is first explained by "
                    "checkpoint lineage: the scored route prior deliberately excluded JAK2. "
                    "The missing locks prevent a stronger claim that no alternative template "
                    "produced the same endpoints or that selection, rather than proposal, lost them."
                ),
                (
                    "Archive continuation is not required to establish a historically strong "
                    "direct route: the external endpoint labels already beat -9.7. It may still "
                    "matter for stronger Full146 descendants, but current-run ancestry is missing."
                ),
            ],
        },
        "deferred_joint_planning": {
            "commits": deferred_commits,
            "relevance": "not causal for the identified delta=0.6 strong-route miss",
            "basis": (
                "All three reported delta=0.6 JAK2 routes have one dependency region, and "
                "the two current-support-eligible routes were already exactly emitted as "
                "one-region candidates before the deferred multi-region planning commits. "
                "Those commits can affect other multi-region proposals only."
            ),
            "evidence_class": "inferred from observed region counts, artifact chronology, and commit scope",
        },
        "decision": {
            "exact_current_pool_audit": "ABSTAIN_MISSING_LOCKS",
            "current_scored_candidate_at_or_below_ivg": False,
            "historical_dynamic_v0_delta_0_6_at_or_below_ivg": False,
            "known_reported_delta_0_6_routes_at_or_below_ivg": len(strong_delta_06),
            "known_reported_delta_0_6_routes_in_current_shared_support": len(eligible_exact_strong),
            "primary_localized_loss": (
                "scored-run route-checkpoint lineage excluded the known JAK2 exact templates"
            ),
            "residual_unknown": (
                "selection and archive-continuation behavior in the exact scored pool"
            ),
        },
    }

    envelope = {"payload": payload}
    envelope["payload_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
