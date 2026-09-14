#!/usr/bin/env python3
"""Collect immutable policy folds and seal the route-distilled launch decision."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.experiments.continuation_profile import publish_json, sha256_file
from tools.t4_route_policy_comparison import _combined_metrics

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/t4_route_policy_comparison_parallel/attempt_1"
LOCAL = ROOT / "diagnostics/t4_route_policy_parallel/attempt_1"
OUTPUT = ROOT / "diagnostics/t4_route_policy_selection/attempt_1/result.json"
ACTOR = ROOT / "diagnostics/t4_route_distillation/attempt_3/actor.json.gz"
EXPECTED_REVISION = "173fb52e161dade6004c2f68be74c70459a10a85"


def _load_envelope(path: Path, *, compressed=False):
    raw = gzip.decompress(path.read_bytes()) if compressed else path.read_bytes()
    envelope = json.loads(raw)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise ValueError(f"invalid policy artifact envelope: {path}")
    if identity(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError(f"policy artifact payload changed: {path}")
    return envelope["payload"]


def collect() -> None:
    import modal

    volume = modal.Volume.from_name("compose-v4-artifacts")
    for fold in range(3):
        folder = LOCAL / f"fold_{fold}"
        for name in ("result.json", "models.json.gz"):
            destination = folder / name
            if destination.exists():
                raise ValueError(
                    f"preserve already collected fold artifact: {destination}"
                )
            raw = b"".join(volume.read_file(f"{REMOTE}/fold_{fold}/{name}"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(raw)
            temporary.replace(destination)
    print(json.dumps({"status": "collected", "folds": 3, "oracle_calls": 0}))


def actor_selection_gate(autonomous: dict) -> dict[str, bool]:
    """Apply the frozen held-source actor-versus-marginal selection rule."""

    generic = autonomous["generic_marginal"]
    actor = autonomous["context_module_prototype"]
    return {
        "transformation_recall_at_128_not_lower": (
            actor["cutoffs"]["128"]["transformation_recall"]
            >= generic["cutoffs"]["128"]["transformation_recall"]
        ),
        "transformation_mrr_strictly_higher": (
            actor["source_balanced_transformation_mrr"]
            > generic["source_balanced_transformation_mrr"]
        ),
        "execution_precision_retains_half": (
            actor["execution_precision"] >= 0.5 * generic["execution_precision"]
        ),
        "unique_endpoint_yield_retains_half": (
            actor["unique_endpoint_yield"] >= 0.5 * generic["unique_endpoint_yield"]
        ),
    }


def merge() -> dict:
    if OUTPUT.exists():
        raise ValueError("route-policy selection exists; do not reselect")
    reports, model_rows = [], []
    for fold in range(3):
        folder = LOCAL / f"fold_{fold}"
        result = _load_envelope(folder / "result.json")
        models = _load_envelope(folder / "models.json.gz", compressed=True)
        if (
            result.get("costs") != {"oracle_calls": 0, "docking_calls": 0}
            or result.get("implementation", {}).get("revision") != EXPECTED_REVISION
            or result.get("implementation", {}).get("working_tree_dirty") is not False
            or result.get("predeclared_split", {}).get("evaluated_fold_ids") != [fold]
            or len(result.get("fold_reports", ())) != 1
            or len(models.get("fold_models", ())) != 1
            or models["fold_models"][0].get("fold") != fold
        ):
            raise ValueError(f"fold {fold} is incomplete or provenance-incompatible")
        reports.append(result["fold_reports"][0])
        model_rows.extend(models["fold_models"])
    test_sources = [
        source for row in reports for source in row["split"]["test_sources"]
    ]
    if len(test_sources) != 15 or len(set(test_sources)) != 15:
        raise ValueError("policy folds do not cover each T4 source exactly once")
    autonomous = _combined_metrics(reports, "autonomous_generation")
    shared = _combined_metrics(reports, "shared_panel_reranking")
    actor_gate = actor_selection_gate(autonomous)
    actor_payload = _load_envelope(ACTOR, compressed=True)
    selected = all(actor_gate.values())
    body = {
        "schema_version": "t4_route_policy_selection_v1",
        "decision": (
            "route_distilled_actor_selected_for_scored_qualification"
            if selected
            else "route_distilled_actor_failed_zero_oracle_selection_gate"
        ),
        "selected_policy": "context_module_prototype" if selected else None,
        "selection_gate": actor_gate,
        "selection_rule": {
            "primary": (
                "held-source autonomous transformation recall@128 must not fall "
                "and transformation MRR must strictly improve over generic marginal"
            ),
            "safeguards": (
                "execution precision and unique endpoint yield must each retain at "
                "least half of generic marginal"
            ),
            "shared_panel_role": "reported reranking evidence only",
        },
        "aggregate_results": {
            "autonomous_generation": autonomous,
            "shared_panel_reranking": shared,
        },
        "folds": reports,
        "model_census": [
            {
                "fold": row["fold"],
                "marginal_training_identity": row["marginal"]["training_identity"],
                "actor_training_identity": row["actor"]["training_identity"],
                "hybrid_training_identity": row["hybrid"]["training_identity"],
            }
            for row in model_rows
        ],
        "selected_all_route_checkpoint": (
            {
                "path": str(ACTOR.relative_to(ROOT)),
                "sha256": sha256_file(ACTOR),
                "payload_sha256": identity(actor_payload),
                "training_identity": actor_payload["training_identity"],
            }
            if selected
            else None
        ),
        "inputs": {
            str((LOCAL / f"fold_{fold}" / name).relative_to(ROOT)): sha256_file(
                LOCAL / f"fold_{fold}" / name
            )
            for fold in range(3)
            for name in ("result.json", "models.json.gz")
        },
        "costs": {"oracle_calls": 0, "docking_calls": 0},
        "evidence": "answer-known T4 route-policy development with held-source folds",
        "limitations": [
            "The selected actor is subsequently fit on all 77 public T4 teacher routes.",
            "Transformation equivalence is a bounded structural diagnostic, not task utility.",
            "The graph contrastive model is reported as a candidate ranker, not a decoder.",
        ],
    }
    body["input_set_sha256"] = hashlib.sha256(
        json.dumps(body["inputs"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    publish_json(OUTPUT, body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "merge"))
    args = parser.parse_args()
    if args.action == "collect":
        collect()
    else:
        result = merge()
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "selected_policy": result["selected_policy"],
                    "selection_gate": result["selection_gate"],
                    "costs": result["costs"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
