from compose_v4.experiments.t4_fiber_campaign import Fiber
from compose_v4.experiments.t4_proposal_funnel_audit import (
    assess_endpoint,
    audit_route_candidates,
)

BRAF_2 = "FC(F)(F)c4cc(NC(=O)Nc3ccc(Oc2ccnc(C(=O)Nc1cccnc1)c2)cc3)ccc4Cl"
ELIGIBLE_SHRINK = "CC(=O)NC(=O)Nc1ccc(Oc2ccnc(C(=O)Nc3cccnc3)c2)cc1"


def test_endpoint_assessment_matches_production_fiber():
    observed = assess_endpoint(BRAF_2, ELIGIBLE_SHRINK, delta=0.6, support="compose_valid")
    production = Fiber(BRAF_2, 0.6, support="compose_valid").check(ELIGIBLE_SHRINK)

    assert production is not None
    assert observed["compose_valid_pass"]
    assert observed["similarity_margin"] >= 0.0
    assert observed["qed_margin"] >= 0.0
    assert observed["sa_margin"] >= 0.0
    assert observed["heavy_delta"] == -8


def test_route_audit_separates_realization_from_endpoint_eligibility():
    candidate = {
        "smiles": ELIGIBLE_SHRINK,
        "created": 0,
        "deleted": 8,
        "regions": 1,
        "route_proposal_rank": 2,
        "route_template_ids": ["template"],
        "realized_primitives": 8,
    }
    summary, assessments = audit_route_candidates(
        cell="braf_2",
        seed_smiles=BRAF_2,
        delta=0.6,
        support="compose_valid",
        proposed_pool=64,
        realization_limit=32,
        telemetry={
            "complete_programs_committed": 1,
            "realization_status_counts": {
                "committed": 1,
                "realizer_frontier_exhausted_abstention": 31,
            },
        },
        candidates=[candidate],
    )

    assert summary["programs_attempted"] == 64
    assert summary["programs_synthesized"] == 1
    assert summary["cumulative_funnel"]["all_compose_valid"] == 1
    assert summary["cumulative_funnel"]["admitted_to_candidate_pool"] == 1
    assert assessments[0]["route_template_ids"] == ["template"]
