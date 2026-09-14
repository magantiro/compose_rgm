import numpy as np

from compose_v4.control.dynamic_program_synthesis_v1 import GENERIC_MODULES
from compose_v4.experiments.t4_route_candidate_utility import _marginal, _ranker
from compose_v4.experiments.t4_route_policy_comparison import (
    ContrastiveRanker,
    MarginalPolicy,
)


def test_policy_checkpoints_round_trip_for_candidate_locking():
    family_probability = 1.0 / len(GENERIC_MODULES)
    marginal = MarginalPolicy(
        tuple(family_probability for _ in GENERIC_MODULES),
        (0.2, 0.3, 0.5),
        0.1,
        "training-marginal",
    )
    restored_marginal = _marginal(marginal.checkpoint())
    assert restored_marginal == marginal

    ranker = ContrastiveRanker(
        (1.0, 2.0),
        (2.0, 4.0),
        (0.5, -0.25),
        0.1,
        "training-ranker",
    )
    restored_ranker = _ranker(ranker.checkpoint())
    assert restored_ranker == ranker
    assert restored_ranker.score(np.asarray((3.0, 6.0))) == 0.25
