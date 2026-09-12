from compose_v4.experiments.pmo_plan_failure_audit import analyze


def _proposal(identifier, parent_id, parent_score, score, decision_id):
    return {
        "id": identifier,
        "smiles": f"mol-{identifier}",
        "parent_score": parent_score,
        "score": score,
        "chain": ([] if parent_id is None else [parent_id]) + [identifier],
        "bundle": {"plan_decision_id": decision_id},
    }


def test_audit_separates_labeled_pool_recall_and_delayed_recovery():
    prepared = {"observed": {"source": 0.5, "known-good": 0.7, "known-bad": 0.2}}
    first = _proposal("one", None, 0.5, 0.4, "decision-1")
    second = _proposal("two", "one", 0.4, 0.6, "decision-2")
    result = {
        "status": "complete_development",
        "schema_version": "option_population_result_v2",
        "configuration": {"artifact_kind": "pmo_plan_policy"},
        "arms": {"frozen": {}, "learning": {}},
        "oracle_rows": [
            {"status": "complete", "smiles": "picked-1", "score": 0.4},
            {"status": "complete", "smiles": "picked-2", "score": 0.6},
        ],
        "workers": [
            {
                "worker_id": "worker-1",
                "phase": 1,
                "slot": 0,
                "plan_decision": {
                    "decision_id": "decision-1",
                    "source": "source",
                    "products": ["known-good", "known-bad", "picked-1"],
                    "probabilities": [0.1, 0.2, 0.7],
                    "selected": 2,
                },
            },
            {
                "worker_id": "worker-2",
                "phase": 2,
                "slot": 0,
                "plan_decision": {
                    "decision_id": "decision-2",
                    "source": "picked-1",
                    "products": ["picked-2"],
                    "probabilities": [1.0],
                    "selected": 0,
                },
            },
        ],
        "rounds": [
            {
                "boundary": 1,
                "arms": {
                    "frozen": {"proposals": [first]},
                    "learning": {"proposals": [first]},
                },
            },
            {
                "boundary": 2,
                "arms": {
                    "frozen": {"proposals": [second]},
                    "learning": {"proposals": [second]},
                },
            },
        ],
    }
    report = analyze(prepared, result)
    assert report["summary"]["pools_with_pre_run_known_improver"] == 1
    assert report["summary"]["mean_behavior_mass_on_pre_run_known_improvers"] == 0.05
    assert report["lineages"]["learning"]["continued_from_temporary_loss"] == 1
    assert report["lineages"]["learning"]["recovered_above_pre_loss_score"] == 1
    assert report["verdict"] == "ranking_headroom_exists_on_labeled_support"


def test_low_counterfactual_label_coverage_is_inconclusive():
    prepared = {"observed": {"source": 0.5}}
    result = {
        "status": "complete_development",
        "schema_version": "option_population_result_v2",
        "configuration": {"artifact_kind": "pmo_plan_policy"},
        "arms": {"frozen": {}},
        "oracle_rows": [{"status": "complete", "smiles": "picked", "score": 0.6}],
        "workers": [
            {
                "worker_id": "worker",
                "phase": 1,
                "slot": 0,
                "plan_decision": {
                    "decision_id": "decision",
                    "source": "source",
                    "products": ["unknown-a", "unknown-b", "picked"],
                    "probabilities": [0.2, 0.3, 0.5],
                    "selected": 2,
                },
            }
        ],
        "rounds": [
            {
                "boundary": 1,
                "arms": {
                    "frozen": {"proposals": [_proposal("picked", None, 0.5, 0.6, "decision")]}
                },
            }
        ],
    }
    report = analyze(prepared, result)
    assert report["summary"]["pre_run_labeled_slot_fraction"] == 0
    assert report["verdict"] == "label_coverage_too_low_to_separate_pool_recall_from_value"
