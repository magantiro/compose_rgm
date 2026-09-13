import numpy as np
import pytest

from compose_v4.control.demonstration_prior import DemonstrationPrior
from compose_v4.control.option_selector import balanced_option_prior
from compose_v4.control.ring_program import RingSpec


def test_prior_keeps_unseen_channels_and_actual_applicability():
    option = RingSpec("pendant", 6, (6, 0, 0), "aromatic").option
    prior = DemonstrationPrior.fit(
        [{"source_id": "a", "role": "train", "option": option}], training_identity="fixture"
    )
    names = ("generic", "rebuild", option)
    reference = balanced_option_prior(names)
    q = prior.distribution(names, reference)
    assert q.proposal[-1] > reference[-1]
    assert np.all(np.asarray(q.proposal) >= 0.1 * reference - 1e-12)
    assert q.kl <= 1.0
    # If the demonstrated action is inapplicable, preserve the full supplied law.
    absent = prior.distribution(names[:2], [0.6, 0.4])
    assert absent.proposal == pytest.approx([0.6, 0.4])
    with pytest.raises(ValueError, match="training-role"):
        DemonstrationPrior.fit(
            [{"source_id": "a", "role": "heldout_source_diagnostic", "option": option}],
            training_identity="fixture",
        )
