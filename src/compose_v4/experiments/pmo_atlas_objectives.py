"""Counted diagnostic objectives for the answer-known PMO atlas.

WHY THIS EXISTS
---------------
Two rules govern every number the atlas programme produces.

1. **A cheap objective is still an oracle.**  Every newly computed objective
   value is a diagnostic evaluation and is counted.  :class:`DiagnosticOracle`
   keeps a ledger; a caller cannot obtain a value without incrementing it.
2. **Construction success is not scoring success.**  An oracle that builds is
   not an oracle that scores: PyTDC's ``Oracle.__call__`` wraps its evaluator in
   a bare ``except`` and returns ``default_property`` (0.0) on any failure, so a
   broken asset produces a plausible all-zero ledger rather than an error.
   :func:`assert_reference_panel` therefore CALLS each oracle against pinned
   molecules with expected values before the first diagnostic call, including a
   nontrivial intermediate value so that a constant-output oracle fails.

EVALUATOR BOUNDARY
------------------
The recorded curriculum scores were produced under PyTDC 0.3.6 / RDKit 2024.03.5,
and ``gsk3b``/``jnk3`` there used a LOCAL FROZEN FOREST (``oracle_kind ==
"frozen_tdc_forest"``), not a PyTDC ``Oracle`` object.  The production PMO
container pins PyTDC 1.1.15 / RDKit 2023.9.6 with an ``rdkit.six`` shim.  These
are different evaluator builds.  Values computed here are NOT comparable to the
recorded ones and no recorded value is ever carried across.

INVARIANTS MAINTAINED AND TESTED
--------------------------------
* the ledger counts every distinct molecule evaluated, and a repeat is served
  from cache and counted once (matching ``ProgramQueryLedger`` semantics);
* :func:`assert_reference_panel` raises unless every pinned expectation is met
  within tolerance, and the panel contains at least one value strictly between
  0 and 1 so a constant oracle cannot pass.
"""

from __future__ import annotations

import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: The eleven tasks the atlas covers, mapped to their PyTDC oracle names.
ATLAS_TASK_ORACLES: dict[str, str] = {
    "albuterol_similarity": "albuterol_similarity",
    "celecoxib_rediscovery": "celecoxib_rediscovery",
    "gsk3b": "gsk3b",
    "isomers_c7h8n2o2": "isomers_c7h8n2o2",
    "jnk3": "jnk3",
    "median1": "median1",
    "mestranol_similarity": "mestranol_similarity",
    "perindopril_mpo": "perindopril_mpo",
    "qed": "qed",
    "thiothixene_rediscovery": "thiothixene_rediscovery",
    "troglitazone_rediscovery": "troglitazone_rediscovery",
}

#: Tasks whose PyTDC evaluator loads a pickled model from a relative path.  For
#: these the recorded curricula used a different evaluator object entirely.
ASSET_BACKED_TASKS: frozenset[str] = frozenset({"gsk3b", "jnk3"})


class ReferencePanelError(RuntimeError):
    """Raised when an oracle fails its pinned positive control."""


def install_rdkit_six_shim() -> None:
    """Reproduce the production container's ``rdkit.six`` shim.

    PyTDC 1.1.15 imports ``rdkit.six``, which modern RDKit no longer ships.  The
    production PMO image installs exactly this shim; without it every oracle
    import fails with a message blaming the RDKit installation.
    """

    import rdkit

    if "rdkit.six" in sys.modules:
        return
    shim = types.ModuleType("rdkit.six")
    shim.iteritems = lambda value, **kwargs: iter(value.items())
    shim.itervalues = lambda value, **kwargs: iter(value.values())
    shim.iterkeys = lambda value, **kwargs: iter(value.keys())
    shim.string_types = (str,)
    sys.modules["rdkit.six"] = shim
    rdkit.six = shim


@dataclass
class DiagnosticOracle:
    """A counted, cached scalar objective.

    ``calls`` is the number of DISTINCT molecules evaluated, which is the figure
    a diagnostic budget is denominated in.  ``lookups`` counts every request,
    so the cache hit rate stays visible rather than hiding repeated work.
    """

    task: str
    evaluate: Callable[[str], float]
    calls: int = 0
    lookups: int = 0
    cache: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    history: list[tuple[str, float]] = field(default_factory=list)

    def __call__(self, smiles: str) -> float:
        self.lookups += 1
        if smiles in self.cache:
            return self.cache[smiles]
        begin = time.time()
        value = float(self.evaluate(smiles))
        self.seconds += time.time() - begin
        self.cache[smiles] = value
        self.calls += 1
        self.history.append((smiles, value))
        return value

    def ledger(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "distinct_molecules_evaluated": self.calls,
            "lookups": self.lookups,
            "cache_hits": self.lookups - self.calls,
            "seconds": round(self.seconds, 3),
        }


def build_oracle(task: str) -> DiagnosticOracle:
    """Construct the PyTDC oracle for one atlas task."""

    if task not in ATLAS_TASK_ORACLES:
        raise KeyError(f"{task!r} is not one of the eleven atlas tasks")
    install_rdkit_six_shim()
    from tdc import Oracle

    oracle = Oracle(name=ATLAS_TASK_ORACLES[task])
    return DiagnosticOracle(task=task, evaluate=lambda smiles: oracle(smiles))


@dataclass(frozen=True)
class ReferenceExpectation:
    smiles: str
    value: float
    label: str


def assert_reference_panel(
    oracle: DiagnosticOracle,
    panel: tuple[ReferenceExpectation, ...],
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Call the oracle against pinned molecules before any diagnostic call.

    The panel must contain a value strictly inside ``(0, 1)``; an oracle that
    returns a constant, or one whose asset failed to load and is silently
    returning 0.0, cannot satisfy that.
    """

    if not panel:
        raise ReferencePanelError(f"{oracle.task}: an empty reference panel proves nothing")
    if not any(0.0 < item.value < 1.0 for item in panel):
        raise ReferencePanelError(
            f"{oracle.task}: the panel has no nontrivial intermediate value, so a "
            "constant-output oracle would pass it"
        )
    observed: list[dict[str, Any]] = []
    worst = 0.0
    for item in panel:
        value = float(oracle.evaluate(item.smiles))
        delta = abs(value - item.value)
        worst = max(worst, delta)
        observed.append(
            {"label": item.label, "expected": item.value, "observed": value, "delta": delta}
        )
        if delta > tolerance:
            raise ReferencePanelError(
                f"{oracle.task}: reference {item.label!r} expected {item.value!r}, "
                f"observed {value!r} (delta {delta})"
            )
    return {
        "task": oracle.task,
        "checks": len(panel),
        "max_abs_delta": worst,
        "tolerance": tolerance,
        "observations": observed,
        "charged_to_diagnostic_budget": False,
        "note": (
            "Reference-panel calls bypass the counted ledger deliberately: they are "
            "an instrument check, and they are reported separately from the "
            "diagnostic budget."
        ),
    }
