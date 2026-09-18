"""Zero-oracle recognition of known complete programs by FiberControl.

The probe program is inserted into a same-source menu only for diagnosis.  Its score is
never used before its rank is computed.  Protected primitive intermediates are not
pretended to be FiberControl decisions; every record here is one complete program with one
queryable endpoint.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles
from compose_v4.control.docking_value import identity
from compose_v4.control.fiber_control import (
    ProgramValue,
    SearchState,
    endpoint_utility,
    program_features,
)
from compose_v4.experiments.t4_fiber_campaign import Fiber, expand
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA_VERSION = "t4_fiber_route_recognition_result_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Rank:
    optimistic: int
    pessimistic: int
    denominator: int
    tied: int
    predicted_endpoint: float


def _canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid archived SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule)


def _archive_rows(archive: dict, *, delta: float) -> tuple[str, float, list[dict]]:
    receipts = sorted(archive["initialization_receipts"], key=lambda row: row["index"])
    roots = [row for row in receipts if row["role"] == "seed_control"]
    if len(roots) != 1 or roots[0]["failure"] is not None:
        raise ValueError("archive must contain one successful seed control")
    root_smiles = _canonical_smiles(roots[0]["smiles"])
    root_score = float(roots[0]["ds"])
    receipt_by_index = {int(row["index"]): row for row in receipts}

    rows = []
    source_keys = set()
    for entry_id, entry in archive["optimizer"]["entries"].items():
        trace = entry["trace"]
        if not trace.get("complete"):
            raise ValueError(f"incomplete route entry {entry_id}")
        source = decode_state(entry["source_state"])
        source_keys.add(canonical_state_key(source))
        if _canonical_smiles(molecular_graph_to_smiles(source)) != root_smiles:
            raise ValueError(f"route {entry_id} does not start at the seed control")
        endpoint = _canonical_smiles(entry["endpoint"])
        traced_endpoint = _canonical_smiles(
            molecular_graph_to_smiles(decode_state(trace["states"][-1]))
        )
        if endpoint != traced_endpoint:
            raise ValueError(f"route {entry_id} endpoint differs from exact trace")
        query_index = int(entry["scored_receipt"]["index"])
        receipt = receipt_by_index.get(query_index)
        if (
            receipt is None
            or receipt["role"] != "candidate"
            or receipt["failure"] is not None
            or _canonical_smiles(receipt["smiles"]) != endpoint
            or float(receipt["ds"]) != float(entry["score"])
        ):
            raise ValueError(f"route {entry_id} lacks a matching scored receipt")
        properties = entry["provenance"]["properties"]
        changes = entry["provenance"]["actual_changes"]
        if not properties.get("oracle_eligible"):
            raise ValueError(f"route {entry_id} endpoint was not queryable")
        rows.append(
            {
                "entry_id": entry_id,
                "candidate_id": entry["candidate_id"],
                "endpoint": endpoint,
                "score": float(entry["score"]),
                "query_index": query_index,
                "parent": root_smiles,
                "parent_score": root_score,
                "similarity": float(properties["sim"]),
                "qed": float(properties["qed"]),
                "sa": float(properties["sa"]),
                "delta": float(delta),
                "regions": int(changes["changed_site_count"]),
                "created": int(changes["surviving_new_atoms"]),
                "deleted": int(changes["deleted_original_atoms"]),
                # The historical complete-route archive predates FiberControl's
                # intervention-family field.  Do not infer a label from the answer.
                "families": [],
                "family_features": "unavailable_zeroed_not_inferred",
                "program_id": trace["program_id"],
                "primitive_edits": int(trace["primitive_edits"]),
                "trace_complete": True,
                "source_state_identity": canonical_state_key(source),
            }
        )
    if len(source_keys) != 1:
        raise ValueError("curriculum archive contains multiple source states")
    rows.sort(key=lambda row: (row["query_index"], row["entry_id"]))
    return root_smiles, root_score, rows


def choose_probe(rows: list[dict], *, minimum_prior_outcomes: int) -> dict:
    eligible = [row for position, row in enumerate(rows) if position >= minimum_prior_outcomes]
    if not eligible:
        raise ValueError("no route probe has enough chronological prior outcomes")
    return min(eligible, key=lambda row: (row["score"], row["query_index"], row["entry_id"]))


def _with_fingerprint(record: dict, fiber: Fiber, state: SearchState) -> dict:
    molecule = Chem.MolFromSmiles(record["endpoint"])
    if molecule is None:
        raise ValueError(f"invalid candidate endpoint: {record['endpoint']!r}")
    return {
        **record,
        "smiles": record["endpoint"],
        "features": program_features(record, state),
        "fingerprint": set(fiber.generator.GetFingerprint(molecule).GetOnBits()),
    }


def rank_probe(
    menu: list[dict],
    *,
    probe_endpoint: str,
    value: ProgramValue,
    state: SearchState,
    fiber: Fiber,
    tolerance: float,
) -> Rank:
    prepared = [_with_fingerprint(row, fiber, state) for row in menu]
    matches = [i for i, row in enumerate(prepared) if row["endpoint"] == probe_endpoint]
    if len(matches) != 1:
        raise ValueError("candidate menu must contain the probe exactly once")
    utilities = endpoint_utility(prepared, value)
    probe_utility = float(utilities[matches[0]])
    better = int(np.sum(utilities > probe_utility + tolerance))
    no_worse = int(np.sum(utilities >= probe_utility - tolerance))
    tied = int(np.sum(np.abs(utilities - probe_utility) <= tolerance))
    return Rank(
        optimistic=better + 1,
        pessimistic=no_worse,
        denominator=len(prepared),
        tied=tied,
        predicted_endpoint=-probe_utility,
    )


def _ordinary_candidates(
    root_smiles: str,
    root_score: float,
    *,
    delta: float,
    draws: int,
    horizon: int,
    support: str,
    seed: int,
) -> list[dict]:
    fiber = Fiber(root_smiles, delta, support=support)
    generated = expand(
        root_smiles,
        root_score,
        fiber,
        np.random.default_rng(seed),
        draws=draws,
        horizon=horizon,
        proposal_lane="shallow",
    )
    return [
        {
            **row,
            "endpoint": row["smiles"],
            "candidate_source": "ordinary_shallow_compose",
        }
        for row in generated
    ]


def analyze_case(
    archive: dict,
    *,
    cell: str,
    delta: float,
    minimum_prior_outcomes: int,
    ordinary_records: list[dict],
    penalty: float,
    tolerance: float,
) -> dict:
    root_smiles, root_score, rows = _archive_rows(archive, delta=delta)
    probe = choose_probe(rows, minimum_prior_outcomes=minimum_prior_outcomes)
    prefix = [row for row in rows if row["query_index"] < probe["query_index"]]
    fiber = Fiber(root_smiles, delta, support="compose_valid")
    state = SearchState(archive={root_smiles: root_score})
    value = ProgramValue(penalty=penalty)
    features: list[np.ndarray] = []
    improvements: list[float] = []

    route_endpoints = {row["endpoint"] for row in rows}
    ordinary = []
    seen = set(route_endpoints)
    for row in ordinary_records:
        endpoint = _canonical_smiles(row.get("endpoint", row["smiles"]))
        if endpoint in seen or endpoint == root_smiles:
            continue
        seen.add(endpoint)
        ordinary.append({**row, "endpoint": endpoint})

    chronology = []
    observed_ids = set()
    for observed in prefix:
        training_record = _with_fingerprint(observed, fiber, state)
        features.append(training_record["features"])
        improvements.append(root_score - observed["score"])
        observed_ids.add(observed["entry_id"])
        state.archive[observed["endpoint"]] = observed["score"]
        value.fit(features, improvements)
        remaining_routes = [row for row in rows if row["entry_id"] not in observed_ids]
        rank = rank_probe(
            [*remaining_routes, *ordinary],
            probe_endpoint=probe["endpoint"],
            value=value,
            state=state,
            fiber=fiber,
            tolerance=tolerance,
        )
        chronology.append(
            {
                "outcomes_available": len(improvements),
                "last_observed_query_index": observed["query_index"],
                "last_observed_score": observed["score"],
                "model_fitted": value.weights is not None,
                "probe_rank": rank.__dict__,
            }
        )

    if value.weights is None:
        raise RuntimeError("probe selection failed to provide four prior outcomes")
    final_rank = Rank(**chronology[-1]["probe_rank"])
    return {
        "cell": cell,
        "root": {"smiles": root_smiles, "score": root_score},
        "archive_candidates": len(rows),
        "ordinary_shallow_candidates": len(ordinary),
        "probe": {
            "entry_id": probe["entry_id"],
            "candidate_id": probe["candidate_id"],
            "program_id": probe["program_id"],
            "endpoint": probe["endpoint"],
            "historical_score_report_only": probe["score"],
            "query_index": probe["query_index"],
            "primitive_edits": probe["primitive_edits"],
            "regions": probe["regions"],
            "created": probe["created"],
            "deleted": probe["deleted"],
            "family_features": probe["family_features"],
            "exact_complete_trace": probe["trace_complete"],
        },
        "frozen_recognition_before_probe_outcome": {
            "outcomes_available": len(prefix),
            "probe_score_used_in_fit": False,
            "later_scores_used_in_fit": False,
            "rank": final_rank.__dict__,
            "top1_optimistic": final_rank.optimistic <= 1,
            "top5_pessimistic": final_rank.pessimistic <= 5,
            "top8_pessimistic": final_rank.pessimistic <= 8,
        },
        "chronological_online_update": chronology,
        "limitations": [
            "The complete program is injected for retrospective recognition; this is not autonomous proposal support.",
            "The archive contains one queryable endpoint per protected route, so no primitive intermediate is treated as a FiberControl decision.",
            "Historical complete-route records lack FiberControl intervention-family labels; those indicators are zeroed rather than inferred from the answer.",
            "Ordinary shallow distractors are zero-oracle current-code proposals and have no historical scores.",
        ],
    }


def run(root: Path, contract_path: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract.get("schema_version") != "t4_fiber_route_recognition_contract_v1":
        raise ValueError("unexpected route-recognition contract schema")
    delta = float(contract["runtime"]["delta"])
    ordinary_spec = contract["candidate_menu"]["ordinary_shallow"]
    cases = []
    for offset, cell in enumerate(contract["selection_rule"]["cells"]):
        spec = contract["inputs"][cell]
        path = root / spec["path"]
        actual_hash = sha256_file(path)
        if actual_hash != spec["sha256"]:
            raise ValueError(f"archive hash changed for {cell}: {actual_hash}")
        archive = json.loads(path.read_text())
        root_smiles, root_score, _ = _archive_rows(archive, delta=delta)
        ordinary = _ordinary_candidates(
            root_smiles,
            root_score,
            delta=delta,
            draws=int(ordinary_spec["draws"]),
            horizon=int(ordinary_spec["horizon"]),
            support=str(ordinary_spec["support"]),
            seed=int(ordinary_spec["seed"]) + offset,
        )
        cases.append(
            analyze_case(
                archive,
                cell=cell,
                delta=delta,
                minimum_prior_outcomes=int(contract["selection_rule"]["minimum_prior_outcomes"]),
                ordinary_records=ordinary,
                penalty=float(contract["runtime"]["program_value_penalty"]),
                tolerance=float(contract["runtime"]["rank_tolerance"]),
            )
        )
    top5 = sum(
        case["frozen_recognition_before_probe_outcome"]["top5_pessimistic"] for case in cases
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence": "retrospective zero-oracle known-answer recognition diagnostic",
        "contract": {
            "path": str(contract_path.relative_to(root)),
            "sha256": sha256_file(contract_path),
            "payload_sha256": identity(contract),
        },
        "implementation": {
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "analyzer_sha256": sha256_file(Path(__file__)),
            "fiber_control_sha256": sha256_file(root / "src/compose_v4/control/fiber_control.py"),
            "rdkit_version": rdBase.rdkitVersion,
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0, "modal_launches": 0},
        "cases": cases,
        "summary": {
            "probes": len(cases),
            "top5_pessimistic": top5,
            "promotion_rule": "rank <= 5 on at least two of three probes",
            "promotion_pass": top5 >= 2,
        },
        "global_limitations": [
            "Probe selection is answer-known and retrospective by design.",
            "No future probe or later-route score enters a fit, but the exact probe program is injected into the menu.",
            "This diagnostic evaluates recognition only; it does not evaluate proposal probability, autonomous route generation, or prospective optimization.",
        ],
    }
