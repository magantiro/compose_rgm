"""Arm names and stage partition. DEPENDENCY-FREE ON PURPOSE.

`modal run` imports the app module in the LAUNCHER's interpreter, which is not
the image and does not have RDKit. Anything the app touches at module scope
must therefore be importable without the chemistry stack. The arm names are
just strings, so they live here and are imported by both
`pathwise_arms.py` (which needs RDKit) and `modal_apps/pathwise_constraints_app.py`
(which must not, at module scope).

Do not add imports to this file.
"""

from __future__ import annotations

#: Every arm without lookahead rollouts. Stage A resolves the vacuity gate,
#: the support-removal rate, feasibility, and the budget-matched price of the
#: mask -- everything except the planning subclaim.
STAGE_A = (
    "unconstrained_greedy",
    "endpoint_only",
    "pathwise_greedy",
    "pathwise_stochastic",
    "mask_only_sampling",
)

#: The two remaining-budget lookahead arms. Worth paying for only once stage A
#: has passed its gate.
STAGE_B = ("unconstrained_verified", "pathwise_verified")

ALL_ARMS = STAGE_A + STAGE_B
