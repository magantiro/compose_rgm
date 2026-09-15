"""Prospective zero-oracle request lock for selector-ranked T4 macros."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from compose_v4.control.docking_value import identity

CONTRACT_SCHEMA = "t4_selector_utility_prelock_contract_v1"
SELECTOR_SCHEMA = "t4_complete_macro_selector_candidate_lock_v1"
ELIGIBILITY_SCHEMA = "t4_compositional_generator_utility_eligibility_ledger_v1"
PRIOR_REQUEST_SCHEMA = "t4_compositional_generator_utility_request_lock_v1"
CANDIDATE_SCHEMA = "t4_selector_utility_candidate_lock_v1"
REQUEST_SCHEMA = "t4_selector_utility_request_lock_v1"
ABSTENTION_SCHEMA = "t4_selector_utility_abstention_ledger_v1"
RESULT_SCHEMA = "t4_selector_utility_prelock_result_v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def _git_bytes(root: Path, revision: str, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path}"], cwd=root, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip()
        raise ValueError(f"bound Git input is unavailable: {revision}:{path}: {detail}") from exc


def _decode_json(data: bytes, path: str) -> dict:
    if path.endswith(".gz"):
        import gzip

        data = gzip.decompress(data)
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sealed_payload(data: bytes, path: str, expected: str) -> dict:
    envelope = _decode_json(data, path)
    payload = envelope.get("payload")
    claimed = envelope.get("payload_sha256", envelope.get("contract_sha256"))
    if not isinstance(payload, dict) or claimed != identity(payload):
        raise ValueError(f"input is not self-hashed: {path}")
    if claimed != expected:
        raise ValueError(f"payload identity changed: {path}")
    return payload


def load_contract(path: Path) -> dict:
    envelope = _decode_json(path.read_bytes(), str(path))
    payload = envelope.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CONTRACT_SCHEMA
        or envelope.get("contract_sha256") != identity(payload)
    ):
        raise ValueError(f"invalid selector utility prelock contract: {path}")
    authority = payload.get("authority", {})
    if (
        authority.get("oracle_calls_authorized") != 0
        or authority.get("docking_calls_authorized") != 0
        or authority.get("modal_launches_authorized") != 0
        or authority.get("live_run_access_authorized") is not False
        or authority.get("scored_launch_authorized") is not False
    ):
        raise ValueError("selector utility prelock must remain zero-oracle")
    if len(payload.get("cells", ())) != 5:
        raise ValueError("selector utility prelock requires exactly five cells")
    return payload


def load_bound_inputs(root: Path, contract: dict) -> tuple[dict, dict, dict, dict]:
    """Validate physical Git objects before returning the three used payloads."""

    loaded: dict[str, dict] = {}
    for name, entry in sorted(contract["immutable_inputs"].items()):
        data = _git_bytes(root, entry["source_revision"], entry["path"])
        if sha256_bytes(data) != entry["sha256"]:
            raise ValueError(f"bound physical input changed: {name}")
        local_path = root / entry["path"]
        if local_path.exists() and sha256_file(local_path) != entry["sha256"]:
            raise ValueError(f"local input differs from its bound Git object: {name}")
        if "payload_sha256" in entry:
            loaded[name] = _sealed_payload(data, entry["path"], entry["payload_sha256"])

    selector = loaded["selector_candidate_lock"]
    eligibility = loaded["prior_delta04_eligibility_ledger"]
    prior_request = loaded["prior_delta04_request_lock"]
    prior_result = loaded["prior_delta04_result"]
    if selector.get("schema_version") != SELECTOR_SCHEMA:
        raise ValueError("unexpected selector candidate-lock schema")
    if eligibility.get("schema_version") != ELIGIBILITY_SCHEMA:
        raise ValueError("unexpected prior eligibility-ledger schema")
    if prior_request.get("schema_version") != PRIOR_REQUEST_SCHEMA:
        raise ValueError("unexpected prior request-lock schema")
    if (
        selector.get("teacher_fields_present") is not False
        or selector.get("task_identity_present") is not False
        or selector.get("new_oracle_calls") != 0
        or prior_request.get("scoring_status") != "UNAUTHORIZED_NOT_LAUNCHED"
        or prior_request.get("new_oracle_calls") != 0
        or prior_request.get("new_docking_calls") != 0
        or prior_request.get("modal_launches") != 0
        or prior_request.get("live_run_accesses") != 0
    ):
        raise ValueError("bound inputs violate the zero-score boundary")
    summary = prior_result.get("summary", {})
    if (
        summary.get("unique_future_requests") != 4
        or summary.get("selected_memberships") != 8
        or summary.get("new_docking_calls") != 0
    ):
        raise ValueError("prior delta-0.4 result census changed")
    return selector, eligibility, prior_request, prior_result


def _request_identity(row: dict, counterpart: dict) -> dict:
    return {
        "schema_version": "t4_docking_request_identity_v1",
        "target": row["target"],
        "canonical_smiles": row["canonical_smiles"],
        "docking_seed": counterpart["docking_seed"],
        "evaluator": counterpart["evaluator"],
    }


def _index_selector(selector: dict) -> dict[str, list[dict]]:
    cases: dict[str, list[dict]] = {}
    for fold in selector.get("folds", ()):
        if fold.get("status") != "fit":
            raise ValueError(f"selector fold is not fit: {fold.get('fold')}")
        for case in fold.get("cases", ()):
            source_case_id = case.get("source_case_id")
            rows = case.get("ranked_candidates")
            if source_case_id in cases or not isinstance(rows, list):
                raise ValueError("selector lock contains a duplicate or malformed case")
            if case.get("candidate_count") != 128 or len(rows) != 128:
                raise ValueError(f"selector case is not a complete 128-pool: {source_case_id}")
            if sorted(row.get("selector_rank") for row in rows) != list(range(1, 129)):
                raise ValueError(f"selector ranks are not complete: {source_case_id}")
            cases[source_case_id] = sorted(
                rows,
                key=lambda row: (
                    row["selector_rank"],
                    row["original_rank"],
                    row["candidate_identity"],
                ),
            )
    return cases


def _index_eligibility(eligibility: dict, declared_cells: list[str]) -> dict[str, list[dict]]:
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for row in eligibility.get("candidates", ()):
        if (
            row.get("generator_revision") == "baseline"
            and row.get("arm_id") == "baseline_learned"
            and row.get("policy_id") == "balanced_joint_autoregressive"
        ):
            by_cell[row["cell"]].append(row)
    if set(by_cell) != set(declared_cells):
        raise ValueError("baseline learned eligibility cells differ from the contract")
    for cell, rows in sorted(by_cell.items()):
        if len(rows) != 128 or sorted(row["rank"] for row in rows) != list(range(1, 129)):
            raise ValueError(f"eligibility ledger is not a complete 128-pool: {cell}")
        if len({row["candidate_id"] for row in rows}) != 128:
            raise ValueError(f"eligibility ledger contains duplicate candidates: {cell}")
        for row in rows:
            if any(key in row for key in ("docking_score", "task_score")):
                raise ValueError("eligibility row contains a prohibited score")
            if row.get("eligible") != (not row.get("eligibility_exclusion_reasons")):
                raise ValueError(f"eligibility Boolean is inconsistent: {cell}/{row['rank']}")
    return by_cell


def _raw_counterparts(prior_request: dict) -> tuple[dict[str, dict], set[str], set[str]]:
    counterparts: dict[str, dict] = {}
    request_ids: set[str] = set()
    candidate_ids: set[str] = set()
    for request in prior_request.get("requests", ()):
        request_ids.add(request["request_id"])
        expected = {
            "schema_version": request["schema_version"],
            "target": request["target"],
            "canonical_smiles": request["canonical_smiles"],
            "docking_seed": request["docking_seed"],
            "evaluator": request["evaluator"],
        }
        if identity(expected) != request["request_id"]:
            raise ValueError(f"prior request identity is invalid: {request['request_id']}")
        for membership in request.get("memberships", ()):
            candidate_ids.add(membership["candidate_id"])
            if membership.get("arm_id") == "baseline_learned":
                cell = membership["cell"]
                if cell in counterparts:
                    raise ValueError(f"duplicate raw learned counterpart: {cell}")
                counterparts[cell] = {"request": request, "membership": membership}
    return counterparts, request_ids, candidate_ids


def build_payloads(
    *,
    contract: dict,
    selector: dict,
    eligibility: dict,
    prior_request: dict,
    provenance: dict,
) -> tuple[dict, dict, dict]:
    """Select an incremental panel without reading any prospective score."""

    cells = list(contract["cells"])
    selector_cases = _index_selector(selector)
    eligible_cases = _index_eligibility(eligibility, cells)
    counterparts, prior_request_ids, prior_candidate_ids = _raw_counterparts(prior_request)
    baseline_payload = selector.get("baseline_candidate_lock_payload_sha256")
    if baseline_payload != eligibility.get("candidate_locks", {}).get("baseline", {}).get(
        "payload_sha256"
    ):
        raise ValueError("selector and eligibility ledger bind different baseline pools")

    selections = []
    outcomes = []
    for cell in cells:
        pool = eligible_cases[cell]
        source_ids = {row["source_case_id"] for row in pool}
        if len(source_ids) != 1:
            raise ValueError(f"eligibility cell has multiple sources: {cell}")
        source_case_id = next(iter(source_ids))
        ranked = selector_cases.get(source_case_id)
        if ranked is None:
            raise ValueError(f"selector case is absent: {cell}")
        rows_by_identity = {row["candidate_id"]: row for row in pool}
        if {row["candidate_identity"] for row in ranked} != set(rows_by_identity):
            raise ValueError(f"selector/eligibility candidate join is not total: {cell}")
        for selector_row in ranked:
            ledger_row = rows_by_identity[selector_row["candidate_identity"]]
            if ledger_row["rank"] != selector_row["original_rank"]:
                raise ValueError(f"original rank changed across locks: {cell}")

        eligible_count = sum(bool(row["eligible"]) for row in pool)
        base_outcome = {
            "cell": cell,
            "fold": pool[0]["fold"],
            "target": pool[0]["target"],
            "source_case_id": source_case_id,
            "pool_candidates": 128,
            "eligible_candidates": eligible_count,
            "ineligible_candidates": 128 - eligible_count,
        }
        if eligible_count == 0:
            outcomes.append(
                {
                    **base_outcome,
                    "status": "abstain_no_eligible_baseline_learned_candidate",
                    "selected_requests": 0,
                }
            )
            continue
        counterpart = counterparts.get(cell)
        if counterpart is None:
            outcomes.append(
                {
                    **base_outcome,
                    "status": "abstain_missing_raw_learned_counterpart",
                    "selected_requests": 0,
                }
            )
            continue

        skipped_ineligible = skipped_candidate = skipped_request = 0
        selected = None
        for selector_row in ranked:
            row = rows_by_identity[selector_row["candidate_identity"]]
            if not row["eligible"]:
                skipped_ineligible += 1
                continue
            if row["candidate_id"] in prior_candidate_ids:
                skipped_candidate += 1
                continue
            request_body = _request_identity(row, counterpart["request"])
            request_id = identity(request_body)
            if request_id in prior_request_ids:
                skipped_request += 1
                continue
            selected = (row, selector_row, request_body, request_id, counterpart)
            break
        if selected is None:
            outcomes.append(
                {
                    **base_outcome,
                    "status": "abstain_no_new_request_after_prior_deduplication",
                    "selected_requests": 0,
                    "skipped_ineligible": skipped_ineligible,
                    "skipped_prior_candidate_identity": skipped_candidate,
                    "skipped_prior_request_identity": skipped_request,
                }
            )
            continue

        row, selector_row, request_body, request_id, counterpart = selected
        fields = (
            "candidate_uid",
            "candidate_id",
            "fold",
            "cell",
            "target",
            "source_case_id",
            "rank",
            "source_state",
            "source_state_sha256",
            "source_canonical_smiles",
            "endpoint_state",
            "endpoint_state_sha256",
            "actions",
            "goal",
            "bindings",
            "construction_dependencies",
            "patch_ids",
            "realization",
            "proposal_score",
            "proposal_score_role",
            "exact_action_replay",
            "canonical_smiles",
            "active_atoms",
            "heavy_atoms",
            "non_self",
            "morgan_radius2_2048_similarity_to_source",
            "qed",
            "sa",
        )
        membership = {field: row[field] for field in fields}
        membership.update(
            {
                "generator_revision": "baseline",
                "arm_id": contract["arm"]["arm_id"],
                "source_policy_id": contract["arm"]["source_policy_id"],
                "ordering": contract["arm"]["ordering"],
                "selector_rank": selector_row["selector_rank"],
                "selector_score": selector_row["selector_score"],
                "selector_score_role": "offline complete-macro ordering provenance; not a task or docking score",
                "complete_macro_signature": selector_row["complete_macro_signature"],
                "eligibility_source": "immutable_prior_delta04_eligibility_ledger",
                "selection_rule": "first_eligible_selector_rank_after_prior_candidate_and_request_deduplication",
                "request_id": request_id,
                "docking_seed": request_body["docking_seed"],
                "evaluator": request_body["evaluator"],
                "evaluator_payload": counterpart["request"]["evaluator_payload"],
                "raw_counterpart": {
                    "request_id": counterpart["request"]["request_id"],
                    "membership_id": counterpart["membership"]["membership_id"],
                    "candidate_id": counterpart["membership"]["candidate_id"],
                    "rank": counterpart["membership"]["rank"],
                    "role": "immutable prior baseline_learned raw-order counterpart",
                },
            }
        )
        membership_id = identity(membership)
        selections.append({"membership_id": membership_id, **membership})
        outcomes.append(
            {
                **base_outcome,
                "status": "selector_candidate_locked",
                "selected_requests": 1,
                "selector_rank": selector_row["selector_rank"],
                "original_rank": selector_row["original_rank"],
                "skipped_ineligible": skipped_ineligible,
                "skipped_prior_candidate_identity": skipped_candidate,
                "skipped_prior_request_identity": skipped_request,
            }
        )

    expected_supported = set(contract["selection_semantics"]["predeclared_maximum_supported_cells"])
    observed_supported = {row["cell"] for row in selections}
    if not observed_supported <= expected_supported:
        raise RuntimeError("selection escaped the predeclared supported cells")
    if len(selections) > contract["selection_semantics"]["maximum_new_requests"]:
        raise RuntimeError("selector utility prelock exceeded its request ceiling")

    common = {
        "prelock": provenance["prelock"],
        "immutable_inputs": provenance["immutable_inputs"],
        "implementation": provenance["implementation"],
        "selection_semantics": contract["selection_semantics"],
        "selection_inputs": {
            "selector_rank_accessed": True,
            "selector_score_role": "ordering provenance only",
            "docking_score_accessed": False,
            "task_score_accessed": False,
            "teacher_endpoint_accessed": False,
            "teacher_transformation_accessed": False,
            "teacher_metric_accessed": False,
            "evaluation_manifest_accessed": False,
            "candidate_regeneration": False,
            "candidate_reranking": False,
            "eligibility_recomputed": False,
        },
        "new_oracle_calls": 0,
        "new_docking_calls": 0,
        "modal_launches": 0,
        "live_run_accesses": 0,
    }
    selections.sort(key=lambda row: (row["cell"], row["selector_rank"]))
    outcomes.sort(key=lambda row: row["cell"])
    candidate_payload = {
        "schema_version": CANDIDATE_SCHEMA,
        **common,
        "memberships": selections,
    }
    candidate_payload_sha256 = identity(candidate_payload)

    requests = []
    for membership in selections:
        request_body = {
            "schema_version": "t4_docking_request_identity_v1",
            "target": membership["target"],
            "canonical_smiles": membership["canonical_smiles"],
            "docking_seed": membership["docking_seed"],
            "evaluator": membership["evaluator"],
        }
        if identity(request_body) != membership["request_id"]:
            raise RuntimeError("selected request identity drifted")
        requests.append(
            {
                "request_id": membership["request_id"],
                **request_body,
                "evaluator_payload": membership["evaluator_payload"],
                "membership_ids": [membership["membership_id"]],
                "memberships": [
                    {
                        key: membership[key]
                        for key in (
                            "membership_id",
                            "arm_id",
                            "fold",
                            "cell",
                            "source_case_id",
                            "rank",
                            "selector_rank",
                            "candidate_id",
                            "candidate_uid",
                        )
                    }
                ],
                "raw_counterpart": membership["raw_counterpart"],
            }
        )
    requests.sort(key=lambda row: (row["target"], row["canonical_smiles"]))
    if len({row["request_id"] for row in requests}) != len(requests):
        raise RuntimeError("incremental selector requests are not unique")
    request_payload = {
        "schema_version": REQUEST_SCHEMA,
        **common,
        "candidate_lock_payload_sha256": candidate_payload_sha256,
        "scoring_status": "UNAUTHORIZED_NOT_LAUNCHED",
        "call_ceiling_if_later_authorized": len(requests),
        "maximum_permitted_call_ceiling": 2,
        "automatic_retries": 0,
        "replacement_or_backfill": False,
        "requests": requests,
        "interpretation_limit": contract["interpretation_limit"],
    }
    abstention_payload = {
        "schema_version": ABSTENTION_SCHEMA,
        **common,
        "candidate_lock_payload_sha256": candidate_payload_sha256,
        "cell_outcomes": outcomes,
    }
    return candidate_payload, request_payload, abstention_payload


def _envelope(payload: dict) -> dict:
    return {"payload": payload, "payload_sha256": identity(payload)}


def _readme(result: dict) -> str:
    summary = result["summary"]
    artifacts = result["output_artifacts"]
    statuses = "\n".join(f"- `{row['cell']}`: `{row['status']}`" for row in result["cell_outcomes"])
    return f"""# T4 selector utility prelock

This prospective development lock adds {summary["new_requests"]} new score-blind
selector-ranked requests to the immutable delta-0.4 raw panel. The selector was
chosen using offline teacher/component diagnostics, so any later scored outcome
is development evidence rather than an independent final benchmark.

The existing raw request lock is unchanged. Its baseline-learned memberships
are the frozen raw counterparts. Candidate and physical request identities from
that lock were excluded before selecting the first eligible selector-ranked
candidate in each supported cell.

## Cell outcomes

{statuses}

## Immutable request lock

- new call ceiling if separately authorized: {summary["new_requests"]}
- request-lock physical SHA-256: `{artifacts["request_lock.json"]["sha256"]}`
- request-lock payload SHA-256: `{artifacts["request_lock.json"]["payload_sha256"]}`
- candidate-lock physical SHA-256: `{artifacts["candidate_lock.json"]["sha256"]}`
- candidate-lock payload SHA-256: `{artifacts["candidate_lock.json"]["payload_sha256"]}`

No docking or task score was inspected. No oracle or docking function was
called, Modal was not launched, and no live run was accessed. This request lock
is not launch authority. A later launch requires explicit authorization of the
exact request-lock hashes and exact call ceiling, with no retry, replacement or
backfill.
"""


def run(root: Path, contract_path: Path, output: Path) -> dict[str, str]:
    contract = load_contract(contract_path)
    selector, eligibility, prior_request, _ = load_bound_inputs(root, contract)
    code_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    implementation_path = Path("src/compose_v4/experiments/t4_selector_utility_prelock.py")
    provenance = {
        "prelock": {
            "path": str(contract_path.relative_to(root)),
            "sha256": sha256_file(contract_path),
            "contract_sha256": identity(contract),
        },
        "immutable_inputs": contract["immutable_inputs"],
        "implementation": {
            "path": str(implementation_path),
            "sha256": sha256_file(root / implementation_path),
            "code_revision": code_revision,
        },
    }
    candidate, request, abstention = build_payloads(
        contract=contract,
        selector=selector,
        eligibility=eligibility,
        prior_request=prior_request,
        provenance=provenance,
    )
    payloads = {
        "candidate_lock.json": candidate,
        "request_lock.json": request,
        "abstention_ledger.json": abstention,
    }
    encoded = {name: _json_bytes(_envelope(payload)) for name, payload in payloads.items()}
    artifact_refs = {
        name: {"sha256": sha256_bytes(data), "payload_sha256": identity(payloads[name])}
        for name, data in sorted(encoded.items())
    }
    outcomes = abstention["cell_outcomes"]
    result = {
        "schema_version": RESULT_SCHEMA,
        "prelock": provenance["prelock"],
        "immutable_inputs": provenance["immutable_inputs"],
        "implementation": provenance["implementation"],
        "summary": {
            "declared_cells": 5,
            "supported_cells": sum(
                row["status"] == "selector_candidate_locked" for row in outcomes
            ),
            "eligibility_abstentions": sum(
                row["status"] == "abstain_no_eligible_baseline_learned_candidate"
                for row in outcomes
            ),
            "new_requests": len(request["requests"]),
            "raw_counterparts_added": 0,
            "prior_requests_preserved": len(prior_request["requests"]),
            "new_oracle_calls": 0,
            "new_docking_calls": 0,
            "modal_launches": 0,
            "live_run_accesses": 0,
        },
        "cell_outcomes": outcomes,
        "output_artifacts": artifact_refs,
        "evidence_status": contract["evidence_status"],
        "interpretation_limit": contract["interpretation_limit"],
    }
    result_bytes = _json_bytes(_envelope(result))
    encoded["result.json"] = result_bytes
    artifact_refs["result.json"] = {
        "sha256": sha256_bytes(result_bytes),
        "payload_sha256": identity(result),
    }
    encoded["README.md"] = _readme(result).encode()

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selector utility lock: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, data in sorted(encoded.items()):
            (temporary / name).write_bytes(data)
        for name, reference in artifact_refs.items():
            if sha256_file(temporary / name) != reference["sha256"]:
                raise RuntimeError(f"published artifact failed physical validation: {name}")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {name: sha256_file(output / name) for name in sorted(encoded)}


__all__ = [
    "ABSTENTION_SCHEMA",
    "CANDIDATE_SCHEMA",
    "CONTRACT_SCHEMA",
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "build_payloads",
    "load_bound_inputs",
    "load_contract",
    "run",
]
