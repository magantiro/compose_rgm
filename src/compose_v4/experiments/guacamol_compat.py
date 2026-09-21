"""Compatibility shims that let the OFFICIAL ``guacamol`` package import under a
modern scientific-Python stack.

Why this module exists
----------------------
The benchmark numbers COMPOSE reports must come from the official
``guacamol`` implementation -- a benchmark scored by a local reimplementation is
not a comparison.  ``guacamol==0.5.5`` predates two upstream removals and pulls
in one heavyweight optional dependency, so a bare ``import guacamol`` fails on
this stack.  Rather than edit the installed package (which would make the
scoring path unauditable), every fix here is applied to ``sys.modules`` *before*
``guacamol`` is imported, leaving the official bytes untouched.

Two shims, both narrow:

``scipy.histogram``
    Removed from SciPy in 1.3.  It was only ever a re-export of
    :func:`numpy.histogram`, so rebinding the same callable is semantically
    exact rather than an approximation.

``fcd``
    ``guacamol.frechet_benchmark`` does ``import fcd`` at module scope, and
    ``guacamol.benchmark_suites`` imports that module transitively, so the goal
    directed suite cannot be constructed without it.  ``fcd`` is only *used*
    inside :class:`~guacamol.frechet_benchmark.FrechetBenchmark`, which appears
    exclusively in the DISTRIBUTION-LEARNING suite.  We therefore install a
    guard module that satisfies the import and raises loudly on any attribute
    access, so the stub can never silently contribute to a reported number: a
    goal-directed run that touches it fails instead of scoring.

Invariant maintained (and asserted in tests): after
:func:`install_guacamol_compat`, constructing and running the goal-directed
benchmark suite never reads an attribute of the ``fcd`` guard.
"""

from __future__ import annotations

import sys
import types
from typing import Any

FCD_MODULE_NAME = "fcd"


class FcdGuardModule(types.ModuleType):
    """Stands in for the optional ``fcd`` package and refuses to be used.

    Importing succeeds; touching anything raises.  That asymmetry is the whole
    point -- it unblocks the goal-directed suite without creating a silent path
    by which a stubbed Frechet computation could reach a reported score.
    """

    #: Set to the attribute name if the guard is ever touched, for diagnostics.
    touched_attribute: str | None = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        FcdGuardModule.touched_attribute = name
        raise RuntimeError(
            "The 'fcd' guard stub was used (attribute "
            f"{name!r}). 'fcd' backs only the distribution-learning Frechet "
            "benchmark; a goal-directed run must never reach it. Install the "
            "real 'fcd' package if you intend to run distribution learning."
        )


def install_guacamol_compat(*, install_fcd_guard: bool = True) -> dict[str, bool]:
    """Apply the shims. Idempotent; safe to call more than once.

    Args:
        install_fcd_guard: install the raising ``fcd`` stub when the real
            package is absent. Pass ``False`` if the genuine ``fcd`` is
            installed and distribution-learning benchmarks are wanted.

    Returns:
        Which shims this call actually installed, for the run manifest.
    """
    applied = {"scipy_histogram": False, "fcd_guard": False}

    import numpy
    import scipy

    if not hasattr(scipy, "histogram"):
        # Exactly the object SciPy used to re-export, not a reimplementation.
        scipy.histogram = numpy.histogram
        applied["scipy_histogram"] = True

    if install_fcd_guard and FCD_MODULE_NAME not in sys.modules:
        try:
            import fcd  # noqa: F401
        except ImportError:
            sys.modules[FCD_MODULE_NAME] = FcdGuardModule(FCD_MODULE_NAME)
            applied["fcd_guard"] = True

    return applied


def fcd_guard_was_touched() -> bool:
    """True if the ``fcd`` guard was ever read -- must stay False for a run."""
    return FcdGuardModule.touched_attribute is not None
