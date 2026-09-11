"""Audit the two local PMO interventions without additional oracle evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.molecular_search_codec import decode_search_state
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.experiments.pmo_archive_pilot import Store
from compose_v4.experiments.pmo_chronological import RECIPE, state_features
from compose_v4.experiments.t4_matched_pilot import unseal
from compose_v4.experiments.winner_paths import replay
from compose_v4.rewrite import action_codec, action_codec_v4
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from tools.pmo_cross_parent_selection import decision, predict_frozen, transfer_arms
from tools.pmo_local_query_selection import metrics, partition, query_arms
from tools.pmo_persistent_lookahead import best_first, continuation_tasks

ROOT = Path(__file__).resolve().parents[1]


def provenance(directory):
    store = Store(directory, lambda: None)
    config, result = (store.read(k) for k in ("configuration", "result"))
    verify_file(directory / "configuration.json", result["configuration_sha256"])
    for path, h in config.get("input_sha256", config.get("inputs", {})).items():
        verify_file(Path(path), h)
    with tarfile.open(directory / "source_snapshot.tar.gz") as archive:
        for p, h in config["implementation"].items():
            if hashlib.sha256(archive.extractfile(p).read()).hexdigest() != h:
                raise ValueError(f"executed source snapshot does not match: {p}")
            if p.startswith("src/"):
                verify_file(ROOT / p, h)
    return store, config, result


def receipts(store, result, known):
    for i, q in enumerate(result["oracle_rows"]):
        if q["status"] != "complete" or q["index"] != i or q["smiles"] in known:
            raise ValueError("incomplete or repeated physical query")
        if store.read(f"oracle/{i:04}/result") != q:
            raise ValueError("physical oracle receipt differs from result ledger")
        verify_file(Path(q["lock_path"]), q["lock_sha256"])
        known[q["smiles"]] = q["score"]
    if len(result["oracle_rows"]) != result["new_calls"]:
        raise ValueError("physical query accounting mismatch")
    return known


def selection_report(directory):
    store, config, result = provenance(directory)
    for key in ("query_lock", "calibration"):
        verify_file(directory / f"{key}.json", result[f"{key}_sha256"])
    lock, calibration = (store.read(k) for k in ("query_lock", "calibration"))
    old_path = next(Path(p) for p in config["input_sha256"] if p.endswith("/result.json"))
    old = unseal(old_path)
    generation = unseal(old_path.with_name("generation_lock.json"))
    known = {r["smiles"]: r["score"] for r in old["rows"]}
    if config["split"] != partition(list(known)) or lock["training_smiles"] != sorted(known):
        raise ValueError("training or calibration assignment drift")
    molecules = sorted(p["smiles"] for p in generation["products"])
    if lock["all_smiles"] != molecules:
        raise ValueError("query allocation silently changed molecular support")
    kernel, _ = state_features(molecules)
    lookup = {s: i for i, s in enumerate(molecules)}

    def verify_model(model, prediction):
        indices = model["train_indices"]
        means = np.array([known[molecules[i]] for i in indices])
        coeff = np.array(model["coefficients"])
        if model["mean"] != float(means.mean()):
            raise ValueError("endpoint model centering uses nontraining labels")
        residual = (
            kernel[np.ix_(indices, indices)] + RECIPE["ridge"] * np.eye(len(indices))
        ) @ coeff - (means - model["mean"])
        if np.max(np.abs(residual)) > 1e-10:
            raise ValueError("stored model fails its declared training normal equations")
        if (
            identity({k: v for k, v in model.items() if k != "snapshot_sha256"})
            != model["snapshot_sha256"]
        ):
            raise ValueError("model coefficient identity mismatch")
        keys = list(prediction)
        expected = model["mean"] + kernel[np.ix_([lookup[s] for s in keys], indices)] @ coeff
        np.testing.assert_allclose(expected, [prediction[s] for s in keys], rtol=0, atol=1e-12)
        return float(np.max(np.abs(expected - [prediction[s] for s in keys])))

    error = max(
        verify_model(lock["model"], lock["predictions"]),
        verify_model(
            calibration["model"], {r["smiles"]: r["predicted"] for r in calibration["predictions"]}
        ),
    )
    if lock["arms"] != query_arms(list(lock["predictions"]), list(lock["predictions"].values())):
        raise ValueError("prospective candidate allocation does not reproduce")
    if lock["new"] != sorted(set().union(*map(set, lock["arms"].values()))) or set(
        lock["new"]
    ) & set(known):
        raise ValueError("queried union is repeated or differs from locked arms")
    if [q["smiles"] for q in result["oracle_rows"]] != lock["new"]:
        raise ValueError("query order differs from lock")
    receipts(store, result, known)
    expected = {
        arm: metrics(rows, known, old["summary"]["initial_score"])
        for arm, rows in lock["arms"].items()
    }
    if result["arms"] != expected:
        raise ValueError("reported arm outcomes differ from physical labels")
    graph, system = decode_state(generation["source"]), editing_v2_semantic_rewrite_system()
    for row in result["rows"]:
        a = row["witnesses"][0]
        codec = action_codec_v4 if a["schema_version"] == 4 else action_codec
        actual = system.apply(graph, *codec.decode_action(a))
        if (
            row["state"] != encode_state(actual)
            or row["smiles"] != canonical_state_key(actual)
            or row["score"] != known[row["smiles"]]
        ):
            raise ValueError("prospective product lacks exact executor/score replay")
    return {
        **{
            k: result[k]
            for k in (
                "arms",
                "new_calls",
                "best",
                "prior_champion",
                "decision",
                "prepare_seconds",
                "scoring_and_replay_seconds",
                "oracle_seconds",
                "peak_memory_bytes",
            )
        },
        "calibration": {
            k: calibration[k]
            for k in (
                "selected",
                "uniform_expected",
                "positive_recall",
                "mae",
                "kernel_seconds",
                "decision",
            )
        },
        "numerical_prediction_replay_error": error,
        "checks": "input/source snapshots; fixed split and training equations; query allocation/union/order; physical receipts; every scored endpoint replay",
    }, config


def lookahead_report(directory):
    store, config, result = provenance(directory)
    verify_file(directory / "prepared.json", result["prepared_sha256"])
    for name, h in result["phase_lock_sha256"].items():
        verify_file(directory / f"{name}.json", h)
    data = store.read("prepared")
    known = data["known"].copy()
    receipts(store, result, known)
    plans = store.read("phase2_parents")
    selected = [
        best_first(root, rows) for root, rows in zip(data["roots"], result["first"], strict=True)
    ]
    if identity(selected) != identity(plans["selected"]) or identity(plans["plans"]) != identity(
        [continuation_tasks(i, rows, selected[i]) for i, rows in enumerate(result["first"])]
    ):
        raise ValueError("immediate/lookahead parent allocation drift")
    replayed, statuses, depths, releases = 0, {}, [], []
    for key, summary in result["attempts"].items():
        saved = store.read(f"draws/{key}")
        program = saved["program"]
        statuses[program["status"]] = statuses.get(program["status"], 0) + 1
        if program["status"] != summary["status"]:
            raise ValueError("attempt status differs from durable program")
        if program["status"] == "compiled":
            source = saved["task"]["parent"]["node"]["graph"]
            states = replay(source, program["actions"], program["smiles"])
            if states != program["states"] or states[-1] != saved["candidate"]["node"]["graph"]:
                raise ValueError("compiled program lacks exact persistent-state replay")
            replayed += len(program["actions"])
            depths.append(len(program["actions"]))
            releases.append(program["released_fraction"])
    for arm, rows in result["second"].items():
        archive = {
            p["smiles"]: p
            for p in data["roots"] + [p for group in result["first"] for p in group] + rows
            if p is not None
        }
        for p in archive.values():
            if (
                known[p["smiles"]] != p["score"]
                or canonical_state_key(decode_search_state(p["node"]).graph) != p["smiles"]
            ):
                raise ValueError("lookahead archive differs from exact graphs and observed scores")
        if result["arms"][arm]["best"] != max(p["score"] for p in archive.values()):
            raise ValueError("lookahead best not reproduced")
        winner = result["arms"][arm]["winner"]
        if winner != archive[winner["smiles"]]:
            raise ValueError("selected winning path was not preserved")
    return {
        **{
            k: result[k]
            for k in ("new_calls", "seconds", "proposal_seconds", "oracle_seconds", "decision")
        },
        "arms": {
            a: {k: v for k, v in r.items() if k != "winner"} for a, r in result["arms"].items()
        },
        "attempts": len(result["attempts"]),
        "statuses": statuses,
        "primitive_transitions_replayed": replayed,
        "primitive_depth_min_max": [min(depths), max(depths)] if depths else None,
        "intended_release_min_max": [min(releases), max(releases)] if releases else None,
        "checks": "source/input hashes; locks and paid ledger; parent allocation; every compiled primitive path; retained winners",
        "historical_prescreen_calls": data["historical_prescreen_calls"],
        "historical_development_physical_calls": data["historical_development_physical_calls"],
    }, config


def cross_parent_report(directory):
    store, config, result = provenance(directory)
    verify_file(directory / "query_lock.json", result["query_lock_sha256"])
    lock = store.read("query_lock")
    model_path = next(Path(p) for p in config["inputs"] if p.endswith("/query_lock.json"))
    history_path = next(Path(p) for p in config["inputs"] if p.endswith("/prepared.json"))
    model, history = unseal(model_path), unseal(history_path)
    prior = unseal(history_path.with_name("result.json"))
    known = history["known"].copy()
    for q in prior["oracle_rows"]:
        if q["smiles"] in known or q["status"] != "complete":
            raise ValueError("cross-parent history contains unresolved or repeated queries")
        known[q["smiles"]] = q["score"]
    query_union = sorted(
        {s for batch in lock["batches"] for rows in batch["arms"].values() for s in rows}
    )
    if query_union != lock["new"] or set(query_union) & set(known) or len(query_union) > 64:
        raise ValueError(
            "cross-parent query union differs from fixed allocations or repeats history"
        )
    if [q["smiles"] for q in result["oracle_rows"]] != query_union:
        raise ValueError("cross-parent physical query order differs from lock")
    prior_best = max(known.values())
    receipts(store, {**result, "new_calls": result["summary"]["new_calls"]}, known)
    measured, details, max_error = [], [], 0.0
    system = editing_v2_semantic_rewrite_system()
    for i, batch in enumerate(lock["batches"]):
        path = directory / f"generation/{i}.json"
        verify_file(path, lock["generation_sha256"][i])
        generation = store.read(f"generation/{i}")
        products = {p["smiles"]: p for p in generation["products"]}
        historical = set(known) - set(query_union)
        if batch["pool"] != sorted(set(products) - historical):
            raise ValueError("cross-parent allocation omitted unqueried supported products")
        predicted = predict_frozen(model, batch["pool"])
        recorded = [batch["predictions"][s] for s in batch["pool"]]
        np.testing.assert_allclose(predicted, recorded, rtol=0, atol=1e-12)
        max_error = max(max_error, float(np.max(np.abs(predicted - recorded))))
        if batch["arms"] != transfer_arms(batch["pool"], predicted, i):
            raise ValueError("cross-parent candidate selection does not reproduce")
        measured.append(
            {a: metrics(rows, known, batch["parent"]["score"]) for a, rows in batch["arms"].items()}
        )
        source = decode_state(generation["source"])
        if generation["source"] != batch["parent"]["node"]["graph"]:
            raise ValueError("cross-parent source lost exact slot identity")
        from compose_v4.control.graph_geometry import topology

        before = topology(source)
        arm_details = {}
        for arm, rows in batch["arms"].items():
            for s in rows:
                row, a = products[s], products[s]["witnesses"][0]
                codec = action_codec_v4 if a["schema_version"] == 4 else action_codec
                actual = system.apply(source, *codec.decode_action(a))
                if encode_state(actual) != row["state"] or canonical_state_key(actual) != s:
                    raise ValueError("cross-parent scored product lacks primitive replay")
            arm_details[arm] = {
                "selected": [
                    {
                        "smiles": s,
                        "score": known[s],
                        "prediction": batch["predictions"][s],
                        "topology": products[s]["topology"],
                        "primitive": products[s]["witnesses"][0],
                    }
                    for s in rows
                ],
                "mae": float(np.mean([abs(known[s] - batch["predictions"][s]) for s in rows])),
                "pool_coverage": len(rows) / len(products),
                "improvement_recall": "unknown for unqueried candidates",
            }
        details.append(
            {
                "parent_smiles": batch["parent"]["smiles"],
                "parent_score": batch["parent"]["score"],
                "parent_topology": before,
                "available_products": len(products),
                "new_pool": len(batch["pool"]),
                "generation_seconds": generation["generation_seconds"],
                "prediction_seconds": batch["prediction_seconds"],
                "arms": arm_details,
            }
        )
    expected_best = max(prior_best, max(known[s] for s in query_union))
    summary = result["summary"]
    if (
        summary["per_parent"] != measured
        or summary["decision"] != decision(measured)
        or summary["best"] != expected_best
        or summary["prior_champion"] != prior_best
    ):
        raise ValueError("cross-parent outcomes/decision do not match paid receipts")
    return {
        **summary,
        "details": details,
        "numerical_prediction_replay_error": max_error,
        "parent_in_model_training": config["parent_in_model_training"],
        "historical_prescreen_calls": config["historical_prescreen_calls"],
        "historical_development_physical_calls": config["historical_development_physical_calls"],
        "oracle_seconds": sum(q["oracle_seconds"] for q in result["oracle_rows"]),
        "checks": "input/source hashes; unchanged fitted coefficients and predictions; full pool allocation; query locks/union/physical receipts; exact primitive replay; outcomes and decision",
    }, config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("selection", "lookahead", "cross_parent"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report, config = {
        "selection": selection_report,
        "lookahead": lookahead_report,
        "cross_parent": cross_parent_report,
    }[a.mode](a.input)
    report.update(
        schema_version=f"pmo_{a.mode}_local_audit_v1",
        input_sha256={
            str(a.input / name): sha256_file(a.input / name)
            for name in ("configuration.json", "result.json", "source_snapshot.tar.gz")
        },
        analyzer_sha256=sha256_file(Path(__file__)),
        executed_code_revision=config["code_revision"],
        software=config["software"],
        hardware=config["hardware"],
        new_audit_oracle_calls=0,
    )
    publish_json(a.output, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k in ("decision", "arms", "new_calls", "checks")}
        )
    )


if __name__ == "__main__":
    main()
