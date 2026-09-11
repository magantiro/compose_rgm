"""Prepare paid complete-edit receipts, then evaluate one fixed CPU policy.

No oracle dependency or remote launch surface. Later runs are an exposed
chronological development audit, never a sealed benchmark or off-policy estimate.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from compose_v4.control.docking_value import graph_kernel, identity, molecular_features
from compose_v4.control.donor_memory import build_memory
from compose_v4.control.donor_program import transplant_plan
from compose_v4.control.edit_chooser import (
    RECIPE,
    EditChooser,
    EditFeatures,
    fit_chooser,
    fit_regression,
)
from compose_v4.control.edit_replay import count_tanimoto, cut_from_payload, fit_replay
from compose_v4.experiments.continuation_profile import publish_json, sha256_file, verify_file
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

SEED = 20261010
RAW = Path("/private/tmp/compose-pmo-edit-replay-20260911a")
PROBE = Path("/private/tmp/compose-pmo-donor-probe-20260911a")
RUNS = (
    Path("/private/tmp/compose-pmo-local-guidance-runs")
    / "2f58bc8ace9d1591520d0c2a670a0b04cb1bd1cad4879f1ca4420361c8570e72/result.json",
    Path("/private/tmp/compose-pmo-local-guidance-replication-runs")
    / "83d84e96ce3d961d09816b3a7829103a1b608380147416c6b0996e9643806b77/result.json",
)


def read_input(path, inputs):
    path = Path(path)
    inputs[str(path)] = sha256_file(path)
    value = json.loads(path.read_text())
    if set(value) == {"payload", "payload_sha256"}:
        if identity(value["payload"]) != value["payload_sha256"]:
            raise ValueError(f"sealed payload mismatch: {path}")
        return value["payload"]
    return value


def _row(parent, donor, left, right, *, status, smiles, score, origin):
    return {
        "parent_smiles": parent["smiles"],
        "parent_score": parent["score"],
        "source": parent.get("state", parent.get("node", {}).get("graph")),
        "donor": donor,
        "source_cut": left,
        "donor_cut": right,
        "status": status,
        "smiles": smiles,
        "score": score,
        "origin": origin,
    }


def collect(root):
    inputs = {}
    prior = read_input(RAW / "training.json", inputs)
    for path, expected in prior["inputs_sha256"].items():
        verify_file(Path(path), expected)
        inputs[path] = expected
    rows = [{**r, "status": "compiled", "wave": "earlier"} for r in prior["rows"]]
    batch, probe = (
        read_input(PROBE / "batch.json", inputs),
        read_input(PROBE / "result.json", inputs),
    )
    labels = {r["parent"]: r["score"] for r in probe["parity"]}
    for attempt in probe["attempts"]:
        if attempt["status"] == "compiled":
            continue
        parent = {**batch["parents"][attempt["parent"]], "score": labels[attempt["parent"]]}
        rows.append(
            _row(
                parent,
                batch["parents"][attempt["donor"]]["state"],
                attempt["parent_cut"],
                attempt["donor_cut"],
                status=attempt["status"],
                smiles=None,
                score=None,
                origin=[str(PROBE / "result.json"), attempt["index"]],
            )
            | {"wave": "earlier"}
        )
    prepared, result = (
        read_input(RAW / "prepared.json", inputs),
        read_input(RAW / "report.json", inputs),
    )
    verify_file(RAW / "candidate_lock.json", result["candidate_lock_sha256"])
    locked = read_input(RAW / "candidate_lock.json", inputs)
    scored = {r["id"]: r for arm in result["arms"].values() for r in arm["candidates"]}
    missing = []
    for attempt in locked["attempts"]:
        proposal = attempt["proposal"]
        origin = [
            str(RAW / "candidate_lock.json"),
            attempt["arm"],
            attempt["slot"],
            attempt["draw"],
        ]
        if proposal is None:
            missing.append(
                {"origin": origin, "reason": "no_attempt_operands", "status": attempt["status"]}
            )
            continue
        name = f"draws/{attempt['arm']}_{attempt['slot']:02}_{attempt['draw']:02}"
        child = scored.get(name)
        if (child is not None) != (attempt["status"] == "compiled"):
            raise ValueError(f"completion/score receipt mismatch: {name}")
        rows.append(
            _row(
                prepared["parents"][attempt["slot"]],
                prepared["donors"][proposal["donor_index"]],
                proposal["source_cut"],
                proposal["donor_cut"],
                status=attempt["status"],
                smiles=child["smiles"] if child else None,
                score=child["score"] if child else None,
                origin=origin,
            )
            | {"wave": "earlier"}
        )
    for path in RUNS:
        run = read_input(path, inputs)
        if run["status"] != "complete_development":
            raise ValueError(f"incomplete chronological run: {path}")
        spec = run["configuration"]["prepared"]
        verify_file(root / spec["path"], spec["sha256"])
        data = read_input(root / spec["path"], inputs)
        memory = build_memory(data["initial_donor_memory"], {}, mode="fixed")
        parents = {(r["smiles"], tuple(r["chain"])): r for r in run["initial_parents"]}
        for rd in run["rounds"]:
            for arm, outcome in rd["arms"].items():
                for child in outcome["proposals"]:
                    if child is None:
                        continue
                    parent = parents[(child["parent_smiles"], tuple(child["chain"][:-1]))]
                    if parent["score"] != child["parent_score"]:
                        raise ValueError("chronological source score changed")
                    b = child["bundle"]
                    if b["option"] == "donor_transplant":
                        donor = memory["rows"][b["donor_index"]]
                        if (
                            b["donor_memory_id"] != memory["memory_id"]
                            or b["donor_smiles"] != donor["smiles"]
                        ):
                            raise ValueError("chronological donor identity mismatch")
                        rows.append(
                            _row(
                                parent,
                                donor["state"],
                                b["source_cut"],
                                b["donor_cut"],
                                status="compiled",
                                smiles=child["smiles"],
                                score=child["score"],
                                origin=[str(path), arm, rd["boundary"], child["id"]],
                            )
                            | {"wave": "chronological"}
                        )
                    parents[(child["smiles"], tuple(child["chain"]))] = child
    return rows, inputs, missing


def reconcile(rows):
    """One outcome edge per source/product, distinct failure plans, full origins."""
    unique, labels = {}, {}
    for row in rows:
        for smi, score in (
            (row["parent_smiles"], row["parent_score"]),
            (row["smiles"], row["score"]),
        ):
            if score is None:
                continue
            if not isinstance(smi, str) or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("malformed paid label")
            if smi in labels and abs(labels[smi] - score) > 1e-12:
                raise ValueError(f"conflicting paid label: {smi}")
            labels[smi] = score
        if row["status"] != "compiled" and row["score"] is not None:
            raise ValueError("noncompiled attempt carries an oracle target")
        key = (
            identity([row["parent_smiles"], row["smiles"]])
            if row["score"] is not None
            else identity(
                [row["source"], row["donor"], row["source_cut"], row["donor_cut"], row["status"]]
            )
        )
        if key not in unique:
            unique[key] = {**row, "row_id": key, "origins": [row["origin"]]}
        else:
            unique[key]["origins"].append(row["origin"])
    return [unique[k] for k in sorted(unique)]


def assign_splits(rows):
    """Assign groups and remove cross-role scored-molecule leakage before features."""
    calibration = {
        r["parent_smiles"]
        for r in rows
        if r["wave"] == "earlier" and int(identity([SEED, r["parent_smiles"]])[:16], 16) % 5 == 0
    }
    protected = calibration | {
        r["smiles"] for r in rows if r["parent_smiles"] in calibration and r["score"] is not None
    }
    for row in rows:
        row["split"] = (
            "chronological"
            if row["wave"] == "chronological"
            else ("calibration" if row["parent_smiles"] in calibration else "train")
        )
        if row["split"] == "train" and (
            row["parent_smiles"] in protected or row["smiles"] in protected
        ):
            row.update(split="excluded", exclusion="training_molecule_overlap_with_calibration")
    fitted = {
        s
        for r in rows
        if r["split"] == "train"
        for s in (r["parent_smiles"], r["smiles"])
        if s is not None
    }
    for row in rows:
        if row["split"] == "chronological" and ({row["parent_smiles"], row["smiles"]} & fitted):
            row.update(split="excluded", exclusion="chronological_molecule_seen_during_fit")
    return rows


def validate_edge(row):
    source, donor = decode_state(row["source"]), decode_state(row["donor"])
    if canonical_state_key(source) != row["parent_smiles"]:
        raise ValueError(f"exact parent identity mismatch: {row['row_id']}")
    plan = transplant_plan(
        source, donor, cut_from_payload(row["source_cut"]), cut_from_payload(row["donor_cut"])
    )
    if row["score"] is not None and (
        row["status"] != "compiled"
        or plan["status"] != "planned"
        or canonical_state_key(plan["target"]) != row["smiles"]
    ):
        raise ValueError(f"paid product differs from recorded edit plan: {row['row_id']}")


def _metadata(root, inputs):
    paths = [
        Path(__file__),
        root / "src/compose_v4/control/edit_chooser.py",
        root / "src/compose_v4/control/edit_replay.py",
        root / "src/compose_v4/control/donor_program.py",
        root / "src/compose_v4/control/docking_value.py",
        root / "src/compose_v4/chem/molecular_graph.py",
        root / "src/compose_v4/rewrite/trace_shard.py",
        root / "src/compose_v4/rewrite/kernel.py",
        root / "docs/PMO_EDIT_CHOOSER.md",
    ]
    return {
        "schema_version": "paid_edit_chooser_preparation_v1",
        "seed": SEED,
        "seed_derivation": "sha256(canonical_json([seed,parent_smiles])) first 64 bits mod 5",
        "inputs_sha256": inputs,
        "implementation_sha256": {str(p.relative_to(root)): sha256_file(p) for p in paths},
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=root)
        ).hexdigest(),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "hardware": {
            "processor": platform.machine(),
            "device": "cpu",
            "precision": "float64",
            "blas_threads": 1,
        },
        "new_oracle_calls": 0,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "historical_prescreen_calls": 249455,
        "historical_development_physical_calls": 2217,
        "limitations": "exposed related-scaffold warm development; legacy bank origin chain incomplete; failed latest donor draws not in extracted result records",
    }


def prepare(root, output):
    started = perf_counter()
    rows, inputs, missing = collect(root)
    raw_count = len(rows)
    rows = assign_splits(reconcile(rows))
    split = [
        {k: r[k] for k in ("row_id", "parent_smiles", "split")} | {"reason": r.get("exclusion")}
        for r in rows
    ]
    # Publish and bind assignments before any reusable learned representation.
    split_hash = publish_json(output / "split.json", split)
    features = EditFeatures()
    for row in rows:
        if perf_counter() - started > 300:
            raise TimeoutError("offline edit preparation exceeded 300 seconds")
        validate_edge(row)
        row["features"] = features(row)
    value = {
        **_metadata(root, inputs),
        "recipe": RECIPE,
        "raw_rows": raw_count,
        "rows": rows,
        "split_sha256": split_hash,
        "missing_operands": missing,
        "counts": dict(Counter(r["split"] for r in rows)),
        "seconds": perf_counter() - started,
    }
    publish_json(output / "prepared.json", value)
    return {k: value[k] for k in ("raw_rows", "counts", "seconds", "new_oracle_calls")}


def ranking_metrics(rows, scores):
    groups = defaultdict(list)
    for row, value in zip(rows, scores, strict=True):
        groups[row["parent_smiles"]].append((row, float(value)))
    pools = []
    for parent, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda t: (-t[1], t[0]["smiles"]))
        gains = np.array([r["score"] - r["parent_score"] for r, _ in group])
        winner = ordered[0][0]
        mixture = EditChooser.distribution([v for _, v in group], np.ones(len(group)))
        pools.append(
            {
                "parent": parent,
                "choices": len(group),
                "available_improvers": int((gains > 1e-12).sum()),
                "uniform_gain": float(gains.mean()),
                "best_available_gain": float(gains.max()),
                "uniform_improvement_probability": float(np.mean(gains > 1e-12)),
                "selected": winner["smiles"],
                "selected_gain": winner["score"] - winner["parent_score"],
                "mixture_gain": float(mixture @ gains),
                "mixture_improvement_probability": float(mixture @ (gains > 1e-12)),
            }
        )
    positives = sum(p["available_improvers"] > 0 for p in pools)
    hits = sum(p["selected_gain"] > 1e-12 for p in pools)
    return {
        "rows": len(rows),
        "parents": len(groups),
        "eligible_pools": len(pools),
        "singleton_pools": sum(len(g) == 1 for g in groups.values()),
        "pools_with_improvers": positives,
        "selected_improvers": hits,
        "improvement_precision": hits / len(pools) if pools else None,
        "improver_parent_coverage": hits / positives if positives else None,
        "distinct_selected_molecules": len({p["selected"] for p in pools}),
        **{
            k: float(np.mean([p[k] for p in pools])) if pools else None
            for k in (
                "selected_gain",
                "uniform_gain",
                "best_available_gain",
                "mixture_gain",
                "uniform_improvement_probability",
                "mixture_improvement_probability",
            )
        },
        "pools": pools,
    }


def evaluate(root, output):
    started = perf_counter()
    path = output / "prepared.json"
    data = json.loads(path.read_text())
    verify_file(output / "split.json", data["split_sha256"])
    for relative, expected in data["implementation_sha256"].items():
        verify_file(root / relative, expected)
    train = [r for r in data["rows"] if r["split"] == "train"]
    fit_started = perf_counter()
    payload = fit_chooser(train, [r["features"] for r in train], input_identity=sha256_file(path))
    publish_json(output / "model.json", payload)
    chooser = EditChooser(payload)
    scored_train = [r for r in train if r["score"] is not None]
    endpoint_features = [molecular_features(r["smiles"])[1] for r in scored_train]
    endpoint_fit = fit_regression(
        graph_kernel(endpoint_features, endpoint_features),
        [r["score"] for r in scored_train],
        [r["parent_smiles"] for r in scored_train],
    )
    replay = fit_replay(scored_train)
    fit_seconds = perf_counter() - fit_started
    score_started = perf_counter()
    results, predictions = {}, []
    for role in ("calibration", "chronological"):
        subset = [r for r in data["rows"] if r["split"] == role]
        pred, completion = chooser.predict([r["features"] for r in subset])
        scored = [r for r in subset if r["score"] is not None]
        gains = [v for r, v in zip(subset, pred, strict=True) if r["score"] is not None]
        endpoint = (
            endpoint_fit["mean"]
            + graph_kernel([molecular_features(r["smiles"])[1] for r in scored], endpoint_features)
            @ endpoint_fit["alpha"]
            - np.array([r["parent_score"] for r in scored])
        )
        replay_values = []
        for row in scored:
            score = sum(
                w
                * np.prod(
                    [
                        0.05
                        + 0.95 * count_tanimoto(row["features"]["bags"][i], entry["features"][k])
                        for i, k in enumerate(("context", "removed"))
                    ]
                )
                ** 2
                for entry, w in zip(replay["entries"], replay["weights"], strict=True)
            )
            replay_values.append(score)
        results[role] = {
            "edit_chooser": ranking_metrics(scored, gains),
            "endpoint_only": ranking_metrics(scored, endpoint),
            "positive_context_only": ranking_metrics(scored, replay_values),
            "gain_mae": float(
                np.mean(np.abs(np.array(gains) - [r["score"] - r["parent_score"] for r in scored]))
            )
            if scored
            else None,
            "completion_brier": float(
                np.mean((completion - [r["status"] == "compiled" for r in subset]) ** 2)
            )
            if subset
            else None,
            "attempt_statuses": dict(Counter(r["status"] for r in subset)),
        }
        predictions.extend(
            {
                "row_id": r["row_id"],
                "split": role,
                "predicted_gain": float(p),
                "completion_estimate_diagnostic_only": float(c),
            }
            for r, p, c in zip(subset, pred, completion, strict=True)
        )
    observed = [results[k]["edit_chooser"] for k in results]
    sufficient = all(m["eligible_pools"] >= 5 and m["pools_with_improvers"] > 0 for m in observed)
    better = all(
        results[k]["edit_chooser"]["selected_gain"] is not None
        and results[k]["edit_chooser"]["selected_gain"] > results[k]["edit_chooser"]["uniform_gain"]
        and results[k]["edit_chooser"]["selected_improvers"]
        >= results[k]["endpoint_only"]["selected_improvers"]
        for k in results
    )
    decision = (
        "inconclusive_for_improvement_discovery"
        if not sufficient
        else ("supports_bounded_fresh_proposal_comparison" if better else "stop_fixed_recipe")
    )
    scaffold = lambda smi: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi))
    train_scaffolds = {scaffold(r["parent_smiles"]) for r in train}
    report = {
        **{k: v for k, v in data.items() if k not in ("rows", "seconds")},
        "schema_version": "paid_edit_chooser_report_v1",
        "decision": decision,
        "prepared_sha256": sha256_file(path),
        "model_sha256": payload["model_sha256"],
        "training_counts": payload["training_counts"],
        "results": results,
        "related_scaffold_audit": {
            role: sum(
                scaffold(s) in train_scaffolds
                for s in {r["parent_smiles"] for r in data["rows"] if r["split"] == role}
            )
            for role in results
        },
        "preparation_seconds": data["seconds"],
        "fit_seconds": fit_seconds,
        "evaluation_seconds": perf_counter() - started,
        "scoring_and_reporting_seconds": perf_counter() - score_started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "predictions": predictions,
    }
    if report["evaluation_seconds"] > 300:
        raise TimeoutError("offline fit/evaluation exceeded 300 seconds")
    publish_json(output / "report.json", report)
    return {
        "decision": decision,
        "training_counts": payload["training_counts"],
        "seconds": report["evaluation_seconds"],
        "results": {
            k: {
                a: {n: v for n, v in m.items() if n != "pools"} if isinstance(m, dict) else m
                for a, m in value.items()
            }
            for k, value in results.items()
        },
    }
