"""Separate complete-plan pool recall from delayed-credit evidence without new labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

SCHEMA = "pmo_plan_failure_audit_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_map(rows, *, label: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in rows:
        smiles, score = row
        if not isinstance(smiles, str) or not smiles or not isinstance(score, (int, float)):
            raise ValueError(f"{label} contains an invalid score row")
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{label} contains an out-of-range score for {smiles}")
        if smiles in scores and scores[smiles] != score:
            raise ValueError(f"{label} contains conflicting scores for {smiles}")
        scores[smiles] = score
    return scores


def _decision_rows(result, pre_scores, all_scores):
    membership: dict[str, set[tuple[str, int]]] = {}
    for round_ in result["rounds"]:
        boundary = int(round_["boundary"])
        for arm, arm_result in round_["arms"].items():
            for proposal in arm_result["proposals"]:
                if proposal is None:
                    continue
                decision_id = proposal.get("bundle", {}).get("plan_decision_id")
                if decision_id is not None:
                    membership.setdefault(decision_id, set()).add((arm, boundary))

    rows = []
    seen = set()
    for worker in result["workers"]:
        decision = worker.get("plan_decision")
        if decision is None:
            continue
        decision_id = decision["decision_id"]
        if decision_id in seen:
            raise ValueError(f"duplicate plan decision {decision_id}")
        seen.add(decision_id)
        products = decision["products"]
        probabilities = [float(value) for value in decision["probabilities"]]
        selected = int(decision["selected"])
        if (
            not products
            or len(products) != len(set(products))
            or len(products) != len(probabilities)
            or not 0 <= selected < len(products)
            or any(not math.isfinite(value) or value <= 0 for value in probabilities)
            or not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-10)
        ):
            raise ValueError(f"invalid recorded behavior law for {decision_id}")
        source = decision["source"]
        if source not in all_scores:
            raise ValueError(f"decision source lacks a paid score: {source}")
        parent_score = all_scores[source]
        preknown = [index for index, product in enumerate(products) if product in pre_scores]
        preknown_improvers = [
            index for index in preknown if pre_scores[products[index]] > parent_score
        ]
        best_preknown = (
            max(preknown, key=lambda index: (pre_scores[products[index]], -index))
            if preknown
            else None
        )
        selected_product = products[selected]
        selected_score = all_scores.get(selected_product)
        rows.append(
            {
                "decision_id": decision_id,
                "worker_id": worker["worker_id"],
                "boundary": int(worker["phase"]),
                "slot": int(worker["slot"]),
                "memberships": [
                    {"arm": arm, "boundary": boundary}
                    for arm, boundary in sorted(membership.get(decision_id, set()))
                ],
                "source": source,
                "parent_score": parent_score,
                "pool_size": len(products),
                "preknown_count": len(preknown),
                "preknown_fraction": len(preknown) / len(products),
                "preknown_improver_count": len(preknown_improvers),
                "preknown_improver_probability_mass": sum(
                    probabilities[index] for index in preknown_improvers
                ),
                "best_preknown_score": (
                    None if best_preknown is None else pre_scores[products[best_preknown]]
                ),
                "best_preknown_gain": (
                    None
                    if best_preknown is None
                    else pre_scores[products[best_preknown]] - parent_score
                ),
                "best_preknown_probability": (
                    None if best_preknown is None else probabilities[best_preknown]
                ),
                "best_preknown_probability_rank": (
                    None
                    if best_preknown is None
                    else 1
                    + sum(
                        probability > probabilities[best_preknown] + 1e-15
                        for probability in probabilities
                    )
                ),
                "selected_product": selected_product,
                "selected_probability": probabilities[selected],
                "selected_preknown": selected_product in pre_scores,
                "selected_score_after_run": selected_score,
                "selected_gain_after_run": (
                    None if selected_score is None else selected_score - parent_score
                ),
                "compiled_into_arm_archive": bool(membership.get(decision_id)),
            }
        )
    return rows


def _lineage_audit(result):
    arms = {}
    for arm in sorted(result["arms"]):
        proposals = [
            proposal
            for round_ in result["rounds"]
            for proposal in round_["arms"][arm]["proposals"]
            if proposal is not None and proposal.get("score") is not None
        ]
        by_id = {proposal["id"]: proposal for proposal in proposals}
        if len(by_id) != len({proposal["id"] for proposal in proposals}):
            # Re-emitting an existing state is legitimate; use its first identical receipt.
            for proposal in proposals:
                previous = by_id[proposal["id"]]
                if previous["smiles"] != proposal["smiles"] or previous["score"] != proposal["score"]:
                    raise ValueError("one proposal id identifies inconsistent scored states")
        continued, temporary_loss_parents, recoveries = 0, 0, 0
        details = []
        for child in proposals:
            ancestors = [
                by_id[node_id] for node_id in child.get("chain", [])[:-1] if node_id in by_id
            ]
            if not ancestors:
                continue
            parent = ancestors[-1]
            if not math.isclose(
                float(child["parent_score"]), float(parent["score"]), rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("continued lineage parent score does not match its receipt")
            continued += 1
            if float(parent["score"]) < float(parent["parent_score"]):
                temporary_loss_parents += 1
                recovered = float(child["score"]) > float(parent["parent_score"])
                recoveries += int(recovered)
                details.append(
                    {
                        "parent_id": parent["id"],
                        "child_id": child["id"],
                        "root_score": parent["parent_score"],
                        "intermediate_score": parent["score"],
                        "child_score": child["score"],
                        "recovered_above_root": recovered,
                    }
                )
        arms[arm] = {
            "scored_proposals": len(proposals),
            "continued_scored_edges": continued,
            "continued_from_temporary_loss": temporary_loss_parents,
            "recovered_above_pre_loss_score": recoveries,
            "temporary_loss_continuations": details,
        }
    return arms


def analyze(prepared, result):
    if result.get("status") != "complete_development" or result.get("schema_version") != "option_population_result_v2":
        raise ValueError("failure audit requires one complete plan-policy development result")
    if result["configuration"].get("artifact_kind") != "pmo_plan_policy":
        raise ValueError("input is not a plan-policy result")
    pre_scores = _score_map(prepared["observed"].items(), label="prepared paid labels")
    new_rows = [
        (row["smiles"], row["score"])
        for row in result["oracle_rows"]
        if row.get("status") == "complete"
    ]
    run_scores = _score_map(new_rows, label="run oracle rows")
    conflicts = [
        smiles for smiles in set(pre_scores) & set(run_scores) if pre_scores[smiles] != run_scores[smiles]
    ]
    if conflicts:
        raise ValueError(f"run conflicts with pre-run score for {conflicts[0]}")
    all_scores = {**pre_scores, **run_scores}
    decisions = _decision_rows(result, pre_scores, all_scores)
    if not decisions:
        raise ValueError("result contains no complete-plan decisions")
    product_slots = sum(row["pool_size"] for row in decisions)
    known_slots = sum(row["preknown_count"] for row in decisions)
    products = {
        product for worker in result["workers"] if worker.get("plan_decision") for product in worker["plan_decision"]["products"]
    }
    known_products = products & set(pre_scores)
    compiled = [row for row in decisions if row["compiled_into_arm_archive"]]
    selected_scored = [row for row in decisions if row["selected_score_after_run"] is not None]
    summary = {
        "decision_pools": len(decisions),
        "compiled_selected_decisions": len(compiled),
        "selected_decisions_with_score_after_run": len(selected_scored),
        "mean_pool_size": mean(row["pool_size"] for row in decisions),
        "median_pool_size": median(row["pool_size"] for row in decisions),
        "product_slots": product_slots,
        "pre_run_labeled_product_slots": known_slots,
        "pre_run_labeled_slot_fraction": known_slots / product_slots,
        "unique_pool_products": len(products),
        "unique_pre_run_labeled_pool_products": len(known_products),
        "unique_pre_run_labeled_product_fraction": len(known_products) / len(products),
        "pools_with_any_pre_run_label": sum(row["preknown_count"] > 0 for row in decisions),
        "pools_with_pre_run_known_improver": sum(
            row["preknown_improver_count"] > 0 for row in decisions
        ),
        "mean_behavior_mass_on_pre_run_known_improvers": mean(
            row["preknown_improver_probability_mass"] for row in decisions
        ),
        "selected_parent_improvements_after_run": sum(
            row["selected_gain_after_run"] is not None and row["selected_gain_after_run"] > 0
            for row in decisions
        ),
        "selected_parent_losses_after_run": sum(
            row["selected_gain_after_run"] is not None and row["selected_gain_after_run"] < 0
            for row in decisions
        ),
        "selected_ties_after_run": sum(
            row["selected_gain_after_run"] is not None and row["selected_gain_after_run"] == 0
            for row in decisions
        ),
        "compile_failures_after_selection": len(decisions) - len(compiled),
    }
    coverage = summary["pre_run_labeled_slot_fraction"]
    if coverage < 0.2:
        verdict = "label_coverage_too_low_to_separate_pool_recall_from_value"
        next_action = (
            "Lock a small set of complete pools and score enough randomly selected pool endpoints "
            "to estimate improving-endpoint prevalence before implementing a new ranker."
        )
    elif summary["pools_with_pre_run_known_improver"] == 0:
        verdict = "proposal_recall_failure_on_labeled_support"
        next_action = "Change multi-step proposal construction before adding future-value ranking."
    else:
        verdict = "ranking_headroom_exists_on_labeled_support"
        next_action = (
            "Compare endpoint ranking with rollout-derived continuation value on these fixed pools."
        )
    return {
        "schema_version": SCHEMA,
        "evidence_class": "retrospective_exposed_development",
        "new_oracle_calls": 0,
        "summary": summary,
        "lineages": _lineage_audit(result),
        "verdict": verdict,
        "next_action": next_action,
        "limitations": [
            "Only products labeled before this run support counterfactual pool evaluation.",
            "Selected products scored during the run do not label unselected alternatives.",
            "This audit measures sampled complete donor-plan pools, not all COMPOSE successors.",
            "No future-value or Doob controller is evaluated here.",
        ],
        "decisions": decisions,
    }


def publish(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run(prepared_path: Path, result_path: Path, output_path: Path, root: Path):
    prepared = json.loads(prepared_path.read_text())
    result = json.loads(result_path.read_text())
    report = analyze(prepared, result)
    report.update(
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "prepared": {"path": str(prepared_path), "sha256": sha256_file(prepared_path)},
            "result": {
                "path": str(result_path),
                "sha256": sha256_file(result_path),
                "durable_prefix": (
                    f"compose-v4-artifacts/pmo_plan_policy/{result['run_id']}"
                ),
            },
        },
        run_id=result["run_id"],
        source_commit=result["image_revision"]["commit"],
        analysis_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        implementation_sha256=sha256_file(Path(__file__)),
        software={"python": platform.python_version()},
    )
    publish(output_path, report)
    return report
