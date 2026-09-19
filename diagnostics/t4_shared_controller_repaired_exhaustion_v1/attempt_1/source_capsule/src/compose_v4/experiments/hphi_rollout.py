"""Region-`h_φ` rollout corpus: label semantics and termination convention.

Preregistered in `docs/HPHI_V1_CORPUS_PREREGISTRATION.md`. Pure and importable
so the three execution invariants are testable without Modal or `R_θ`.

INVARIANT 1 — THE ROLLOUT LAW IS THE CONTROLLER'S BASE CHAIN
------------------------------------------------------------
`H = 6` means six **committed productive edits** under the frozen canonical
`R_θ` law. Not six raw mark draws, not six attempts. A mark whose canonical
image equals the current state is VIRTUAL: it is redrawn, and it does not
consume a step. Direct-mark sampling is the intended implementation, and it is
checked against the canonical-successor law by a parity fixture before launch.

INVARIANT 2 — TERMINAL REACHABILITY, NOT "EVER HIT THE GOAL"
-------------------------------------------------------------
For `x_0 -> ... -> x_6`, the example at prefix `x_t` carries `b = 6 - t` and
its label for region `z` is

    g_z(x_6)              the TERMINAL state

    NOT  1[ exists k >= t : x_k in B_z ]

This is load-bearing. A trajectory that crosses QED 0.90 at step 4 and falls
back below it by step 6 is a NEGATIVE terminal-reachability label. Labelling it
positive would train a hitting-probability controller rather than the
finite-horizon Doob value we claim.

Similarity is ALWAYS measured against the immutable original `x_src`, never
against the previous state or the prefix.

INVARIANT 3 — EARLY TERMINATION IS EXPLICIT
--------------------------------------------
A rollout that cannot make six committed productive edits does NOT have its last
molecule silently promoted to an `H=6` endpoint. It terminates into the
CEMETERY convention below, and that convention is identical in training and at
inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

#: The preregistered 5x4 goal grid. Frozen before any census was read.
QED_THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.95)
SIMILARITY_FLOORS = (0.30, 0.40, 0.50, 0.60)
HORIZON = 6  # DEFAULT ONLY. The frozen corpus is H24 -- pass horizon= explicitly.

#: The benchmark region is ONE MEMBER of the grid, not the target it was
#: designed around.
BENCHMARK_REGION = (0.90, 0.40)


def registered_regions() -> tuple[tuple[float, float], ...]:
    return tuple((q, s) for q in QED_THRESHOLDS for s in SIMILARITY_FLOORS)


@dataclass(frozen=True)
class Termination:
    """How a rollout ended. Identical semantics in training and inference."""

    #: "complete" -- six committed productive edits were made.
    #: "cemetery" -- the chain could not continue (empty fiber / all virtual).
    kind: str
    #: Committed productive edits actually made, 0..HORIZON.
    edits: int
    reason: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.kind == "complete"


def terminal_label(
    terminal_qed: float,
    terminal_similarity: float,
    region: tuple[float, float],
    termination: Termination,
) -> int:
    """`g_z(x_6)` — INVARIANT 2. Terminal membership only.

    A cemetery termination is a NEGATIVE label for every region. It did not
    reach the goal in the allotted budget, and pretending its short trajectory
    ended at `H=6` would silently redefine the horizon.
    """
    if not termination.is_complete:
        return 0
    q_min, s_min = region
    return int(terminal_qed >= q_min and terminal_similarity >= s_min)


def first_hit_step(
    qed_of: Sequence[float],
    similarity_to_source: Sequence[float],
    region: tuple[float, float],
) -> int | None:
    """The stopping time `tau_z = min{t : g_z(x_t) = 1}`, or None.

    This is a POLICY, not retrospective selection: at `x_t` the controller sees
    that the predeclared region is satisfied and executes STOP. It never
    consults a future state to decide whether an earlier one was better.
    """
    q_min, s_min = region
    for t, (q, s) in enumerate(zip(qed_of, similarity_to_source)):
        if q >= q_min and s >= s_min:
            return t
    return None


def prefix_examples(
    trajectory: Sequence[str],
    qed_of: Sequence[float],
    similarity_to_source: Sequence[float],
    termination: Termination,
    horizon: int = HORIZON,
) -> list[dict[str, object]]:
    """One training example per prefix, labelled against every registered region.

    `trajectory[0]` is the source. `b = horizon - t` is the remaining budget at
    prefix `t`. Every example takes the TERMINAL label, not its own membership.

    `horizon` MUST be passed for any corpus that is not H6. The frozen corpus is
    H24, and the module default of 6 would raise on every trajectory past step 6.
    """
    n = len(trajectory)
    if n != len(qed_of) or n != len(similarity_to_source):
        raise ValueError("trajectory and property sequences must align")
    if n == 0:
        return []
    term_q, term_s = qed_of[-1], similarity_to_source[-1]

    out: list[dict[str, object]] = []
    for t, state in enumerate(trajectory):
        b = horizon - t
        if b < 0:
            raise ValueError(f"prefix {t} exceeds horizon {horizon}")
        out.append({
            "state": state,
            "budget_remaining": b,
            "state_qed": float(qed_of[t]),
            "state_similarity_to_source": float(similarity_to_source[t]),
            # BOTH label definitions are stored. Terminal is the fixed-horizon
            # Doob object; hit is first-passage for the anytime controller.
            "labels": {
                f"{q:.2f}_{s:.2f}": terminal_label(term_q, term_s, (q, s), termination)
                for q, s in registered_regions()
            },
            "labels_hit": {
                f"{q:.2f}_{s:.2f}": ever_hit_label(
                    qed_of, similarity_to_source, (q, s), start=t)
                for q, s in registered_regions()
            },
            "termination": termination.kind,
        })
    return out


def ever_hit_label(
    qed_of: Sequence[float],
    similarity_to_source: Sequence[float],
    region: tuple[float, float],
    start: int = 0,
) -> int:
    """`1[exists k >= start : x_k in B_z]` — FIRST-PASSAGE reachability.

    PROMOTED TO FIRST-CLASS. An earlier version of this module called this "the
    wrong label." That was too narrow: it is the wrong label for the FIXED-
    HORIZON Doob object, and the RIGHT one for the anytime controller.

        h_terminal_b(x) = P( X_b in B_z )                fixed horizon
        h_hit_b(x)      = P( exists t <= b : X_t in B_z )  first passage

    with the clean recursion `h_hit_0 = g_z` and, for `b > 0`,
    `h_hit_b(x) = 1` if `x in B_z`, else `sum_y R(y|x) h_hit_{b-1}(y)`.

    Equivalently: the same terminal-Doob framework on an ABSORBED chain — once
    a state enters `B_z`, stop there, and "hit within b" becomes "the terminal
    state lies in `B_z`."

    Both labels are stored. The same corpus trains either value definition.
    """
    q_min, s_min = region
    return int(any(
        qed_of[k] >= q_min and similarity_to_source[k] >= s_min
        for k in range(start, len(qed_of))
    ))
