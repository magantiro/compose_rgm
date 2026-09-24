"""The production PMO proposal channel names, in one place.

`pmo_population_controller` owns these, but importing it from `pmo_contextual_macro` would
close an import cycle (the controller imports the value model). The names are duplicated
here and a test asserts the two agree, so a rename cannot silently split them.
"""

from __future__ import annotations

import os

SHALLOW_CHANNEL = "shallow_program_channel"
STRUCTURED_CHANNEL = "structured_program_channel"
JUMP_CHANNEL = "joint_dependency_region_jump"
# T4 samples a third generic lane that PMO never imported.  Measured on the 15-cell
# held-target audit, round one: it supplied 921 of ~2,383 eligible endpoints (39%) and
# 16 of 74 selections, at a higher gate-call yield than `shallow` (1.6% vs 1.2%).  That
# is a T4 -> PMO regression, not a shared sampling weakness, so the lane is restored
# from the existing T4 implementation rather than reimplemented.
ANCHORED_CHANNEL = "anchored_replacement_channel"

#: Transplant was reachable only as half of the `recombination` proposal KIND, which is
#: itself one kind among many: MEASURED, that left it at 5.8% of attempts even at
#: `transplant_share=0.5`, and 51.3% of those were duplicates -- an effective novel
#: footprint near 2.8%. A mechanism that small cannot move AUC, so the matched A/B was
#: underpowered by construction rather than informative about the chemistry. As a PEER
#: LANE it gets the same generic scheduling and an explicit, visible candidate budget.
#:
#: OPT-IN, and APPENDED LAST on purpose: absent, the tuple is byte-identical to every run
#: so far, and when present the existing channels keep their indices, so the two arms
#: differ by the added lane rather than by a re-indexed RNG.
TRANSPLANT_CHANNEL = "transplant_program_channel"


def transplant_lane_enabled() -> bool:
    """Whether the dedicated transplant lane is active for this process."""
    return bool(os.environ.get("PMO_TRANSPLANT_LANE"))


CHANNELS: tuple[str, ...] = (
    SHALLOW_CHANNEL,
    STRUCTURED_CHANNEL,
    ANCHORED_CHANNEL,
    JUMP_CHANNEL,
) + ((TRANSPLANT_CHANNEL,) if transplant_lane_enabled() else ())
