"""Falsifier for the section-9.3 ring-hazard reformulation.

The unconditional small-ring/bridged excess (Lineage B "miss 3") is the
support-collapse renormalization defect: when a state stops supporting the
common six-membered ring groups, the *hierarchical* factorization keeps the ring
family's probability fixed (it is driven by ``family_head`` alone) and the
within-ring template softmax renormalizes the residual rare small-ring templates
to probability one -- so rare rings are over-fired precisely where nothing else
is legal.

The audit's fix ("masked, unnormalized topology-group intensities so loss of
common support lowers ring hazard rather than transferring it to rare small
rings") is realized by the ``superposed`` rate factorization combined with the
``topology_cycle_hierarchical`` ring-template factorization -- NOT by a new
``ring_family_mass_mode`` (that add-to-the-family-logit hack cancels against the
within-ring normalizer in hierarchical mode, which is why it was rejected).

This test exercises the *production* rate functions
``_masked_family_logits`` and ``_hierarchical_ring_template_logits`` on a
controlled two-group support setup and asserts:

* superposed: removing the common six-ring group's support LOWERS the ring
  family's mass fraction (hazard flows to non-ring families);
* hierarchical: removing that same support leaves the ring family's mass
  fraction essentially unchanged -- the defect -- while the within-ring
  distribution collapses onto the rare small ring.

No model weights, no GPU: it isolates the mechanism algebraically so the fix is
proven before any warm-start training is spent.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from compose_v4.model.factorized_tracelet_rate_model import (
    _hierarchical_ring_template_logits,
    _masked_family_logits,
)

# Two topology groups: group 0 = common six-membered rings (templates 0, 1),
# group 1 = a rare three-membered ring (template 2).  A high/low corpus log
# prior stands in for the offset/base measure; the learned residual is zero here
# so the test measures the structural mechanism, not a fitted head.
_MEMBERS: tuple[tuple[int, ...], ...] = ((0, 1), (2,))
_GROUP_LOG_PRIOR = torch.tensor([[math.log(0.9), math.log(0.1)]])
_TEMPLATE_LOGITS = torch.zeros((1, 3))

# Support at two states: one that supports the common six-ring group and the
# rare ring (COMMON), one that only supports the rare three-ring (RARE_ONLY),
# e.g. a congested junction where only a strained closure is legal.
_SUPPORT_COMMON = torch.tensor([[True, True, True]])
_SUPPORT_RARE_ONLY = torch.tensor([[False, False, True]])

# One competing non-ring family with a fixed within-family partition, so the
# family softmax has something to shift mass toward when ring support shrinks.
_RAW_FAMILY_LOGITS = torch.tensor([[0.0, 0.0]])  # [ring, other]
_OTHER_ACTION_LOG_Z = torch.tensor([[0.5]])


def _ring_mass_fraction(support: torch.Tensor, factorization: str) -> float:
    """Return the ring family's share of the total hazard for a support mask."""
    ring_template_logits = _hierarchical_ring_template_logits(
        _TEMPLATE_LOGITS,
        support,
        _GROUP_LOG_PRIOR,
        _MEMBERS,
    )
    # The ring family's within-family log partition is the log-sum-exp over its
    # (already -inf-masked) template logits -- the exact ``action_log_z`` the
    # production family composition consumes.
    ring_action_log_z = torch.logsumexp(ring_template_logits, dim=1, keepdim=True)
    action_log_z = torch.cat([ring_action_log_z, _OTHER_ACTION_LOG_Z], dim=1)
    enabled = torch.tensor([[True, True]])
    family_scores = _masked_family_logits(
        _RAW_FAMILY_LOGITS,
        action_log_z,
        enabled,
        rate_factorization=factorization,
    )
    family_fraction = F.softmax(family_scores, dim=1)
    return float(family_fraction[0, 0])


def test_superposed_lowers_ring_mass_when_common_support_is_lost() -> None:
    common = _ring_mass_fraction(_SUPPORT_COMMON, "superposed")
    rare_only = _ring_mass_fraction(_SUPPORT_RARE_ONLY, "superposed")
    # Losing the common six-ring group must reduce total ring hazard, not merely
    # reshuffle it inside the ring family.
    assert rare_only < common - 0.05, (rare_only, common)


def test_hierarchical_keeps_ring_mass_fixed_the_defect() -> None:
    common = _ring_mass_fraction(_SUPPORT_COMMON, "hierarchical")
    rare_only = _ring_mass_fraction(_SUPPORT_RARE_ONLY, "hierarchical")
    # The defect: in hierarchical mode the ring family's mass is driven by the
    # family head alone and does not fall when only rare small rings remain
    # legal -- so the within-ring softmax over-fires the rare ring.
    assert abs(rare_only - common) < 1e-6, (rare_only, common)


def test_within_ring_collapses_onto_rare_ring_under_support_loss() -> None:
    # Independent of factorization, when only the rare three-ring is legal the
    # within-ring distribution puts all of its (conditional) mass there; the fix
    # works by shrinking the ring family's *total* mass, not this conditional.
    logits = _hierarchical_ring_template_logits(
        _TEMPLATE_LOGITS,
        _SUPPORT_RARE_ONLY,
        _GROUP_LOG_PRIOR,
        _MEMBERS,
    )
    within_ring = F.softmax(logits, dim=1)
    assert float(within_ring[0, 2]) > 0.999
    assert float(within_ring[0, 0]) < 1e-6
    assert float(within_ring[0, 1]) < 1e-6
