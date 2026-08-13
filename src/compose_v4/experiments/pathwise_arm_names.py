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

# ---------------------------------------------------------------------------
# Stage B (corridor) -- a 2x2 of {where enforced} x {how navigated}
# ---------------------------------------------------------------------------
#: The four PRIMARY causal arms. Exactly four; no fifth causal arm is added,
#: because each extra arm is another comparison a reader must be stopped from
#: making.
STAGE_B_CORRIDOR_ARMS = (
    "endpoint_greedy",      # enforce at t=H only, navigate myopically
    "pathwise_greedy",      # enforce at every t, navigate myopically
    "endpoint_verified",    # enforce at t=H only, navigate with lookahead
    "pathwise_verified",    # enforce at every t, navigate with lookahead
)

#: DESCRIPTIVE ONLY. Never enters a causal contrast, and it is the arm whose
#: visited states define the support-retention census, so the SUPPORT_TIGHT
#: classification cannot depend on any constrained arm's outcome.
STAGE_B_DESCRIPTIVE_ARM = "unconstrained_potency"

STAGE_B_ALL = STAGE_B_CORRIDOR_ARMS + (STAGE_B_DESCRIPTIVE_ARM,)

#: (mask, endpoint_only, controller) for each stage-B arm.
STAGE_B_SPEC: dict[str, tuple[bool, bool, str]] = {
    "endpoint_greedy": (True, True, "greedy"),
    "pathwise_greedy": (True, False, "greedy"),
    "endpoint_verified": (True, True, "verified"),
    "pathwise_verified": (True, False, "verified"),
    "unconstrained_potency": (False, False, "greedy"),
}
