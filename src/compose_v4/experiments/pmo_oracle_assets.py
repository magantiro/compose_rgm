"""Call-time asset resolution and positive controls for PMO oracles.

PyTDC loads some estimator pickles from paths relative to the working directory.
Lazy loads happen on the first score request, not when an oracle is constructed.
``AssetPinnedOracle`` pins the asset directory for every call and primes the lazy
load there. A positive control compares graded scores with the versioned
28-molecule fixture in ``experiments/pmo/assets/oracle_reference_panel.json``.
Construction alone or a nonzero score is not an adequate asset check.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# ---- asset layout ----------------------------------------------------------

# PyTDC's own relative layout: the evaluator opens "oracle/<name>.pkl" against cwd.
ORACLE_ASSET_SUBDIR = "oracle"

# Tasks whose score comes from a pickled estimator opened by RELATIVE path.  Measured
# across all 23 PMO tasks under PyTDC 1.1.15. The other 20 open no model file.
# The measured classification is in experiments/pmo/assets/oracle_asset_audit.json.
ASSET_BACKED_PMO_TASKS = frozenset({"drd2", "gsk3b", "jnk3"})


class OraclePositiveControlFailure(RuntimeError):
    """An asset-backed oracle did not reproduce its pinned reference scores."""


@contextmanager
def pinned_working_directory(directory: Path):
    """Run the block with `directory` as the working directory, then restore it.

    Restoration is unconditional so a raising call cannot leave the process pointing
    somewhere else; that is the whole reason the production defect was invisible.
    """
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield directory
    finally:
        os.chdir(previous)


@dataclass(frozen=True)
class OracleReference:
    """One pinned molecule with the raw score the frozen panel records for it.

    `tolerance` is an absolute tolerance on the RAW oracle output -- no direction
    transform, no normalisation.  `role` is the panel group plus what this row is
    here to catch.
    """

    smiles: str
    expected: float
    tolerance: float
    role: str


def _references(
    rows: Iterable[tuple[str, float, str]], tolerance: float
) -> tuple[OracleReference, ...]:
    return tuple(OracleReference(s, e, tolerance, r) for s, e, r in rows)


# Random-forest oracles return a mean over 100 pure-leaf trees, so the raw value is
# k/100 and reproduces bit-for-bit under the pinned versions; the frozen panel's own
# parity check measured max_abs_delta 0.0 for gsk3b/jnk3 and 4.6e-14 for drd2.  A
# 1e-6 absolute tolerance is therefore far looser than the observed drift while still
# refusing anything that has actually changed model, fingerprint or asset.
_FOREST_TOLERANCE = 1e-6
_SVM_TOLERANCE = 1e-6


# The GSK3B panel includes high, intermediate and inactive molecules. This
# distinguishes a correct oracle from a constant or thresholded response.
_GSK3B = _references(
    (
        ("C1(C2=CN(CCCO)C=3C2=CC=CN3)=C(C(=O)NC1=O)C4=NC=CC=C4", 1.0, "gsk3b_active/high"),
        (
            "C12=C(C=3C=C(C(F)(F)F)C=CC3N1)CC(=O)NC=4C2=CC(/C=C/C(=O)OC)=CC4",
            1.0,
            "gsk3b_active/high",
        ),
        ("ClC1=NC(=NC(C=2C3=C(NC2)N=CC=C3)=C1)NC4C(N)CCCC4", 1.0, "gsk3b_active/high"),
        ("BrC1=C2N(N=C1)C(NCC3=CC=CN=C3)=CC(=N2)C=4C=CC=CC4", 0.99, "gsk3b_active/high"),
        ("C=1(C(=C(C=C(C1Br)Br)/C=C/C(=O)O)Br)Br", 0.58, "gsk3b_active/intermediate"),
        ("O1[C@H](CCCC(=O)CCCC=CC=2C(C1=O)=C(O)C=C(O)C2)C", 0.58, "gsk3b_active/intermediate"),
        ("S(=O)(=O)(NC=1C=C2N=C(C(=NC2=CC1)C)C)C3=CC=C(N)C=C3", 0.51, "gsk3b_active/intermediate"),
        (
            "NS(=O)(=O)C1=CC=C(N/N=C/2\\C(=O)NC3=CC=C4N=CC=CC4=C23)C=C1",
            0.49,
            "gsk3b_active/intermediate",
        ),
        ("O=C([O-])C1(Cc2ccccc2Cl)CCN(c2ncnc3[nH]ccc23)CC1", 0.66, "zinc/graded"),
        ("CC(C)NC(=O)C(=O)NCc1nc(-c2cccc(F)c2)c(-c2ccccn2)[nH]1", 0.31, "zinc/graded"),
        ("COc1ccc(-c2n[nH]cc2C[NH+]2CCC[C@@H](N3CCNC3=O)C2)cc1", 0.23, "zinc/graded"),
        ("C#CCN(Cc1ccccc1Cl)C1CCCC1", 0.0, "zinc/inactive"),
        ("CC(=O)CCC(C(C)=O)C(C)=O", 0.0, "zinc/inactive"),
    ),
    _FOREST_TOLERANCE,
)

# jnk3: asset-backed and relative-path, but EAGER -- `class jnk3.__init__` calls
# `load_pickled_model("oracle/jnk3_current.pkl")`, so the asset is read when the
# evaluator is constructed and cached on the instance.  Measured consequence: jnk3
# raises FileNotFoundError at construction instead of returning a swallowed 0.0, and
# a correctly constructed jnk3 is NOT cwd-dependent at call time.  It is audited and
# controlled here anyway, because the relative path still has to resolve once.
_JNK3 = _references(
    (
        (
            "C1(=C(C=2C=CN(N2)C)C=C(N=C1)NC=3C=CC(N4N=C(N=C4)N5CCOCC5)=CC3)F",
            1.0,
            "jnk3_active/high",
        ),
        ("C1(C(F)(F)F)=C(NC=2C=CC=CC2C(=O)N)C=C(OC3=CC=CC(=C3)OC)N=C1", 1.0, "jnk3_active/high"),
        ("C1=C(Cl)C(=NC(N[C@@H]2CCCN(C2)C(=O)NCC)=N1)C=3C4=C(C=CC=C4)NC3", 1.0, "jnk3_active/high"),
        ("C=1C(C(=O)NCC)=CC=2N(C(=C(C2C1)N=O)O)CC3=C4C(=CC(=C3)F)COCO4", 0.99, "jnk3_active/high"),
        (
            "C=1C(=CC(=NC1)NC2=C(C=CC=C2)OC)C3=CC(=NC=C3)NC4=CC=CC=C4OC",
            0.59,
            "jnk3_active/intermediate",
        ),
        ("C=1C=C(S(/C=C/C#N)(=O)=O)C=CC1C", 0.59, "jnk3_active/intermediate"),
        ("O=[N+]([O-])c1ccc(/C=N/Nc2ccc(C(F)(F)F)cc2[N+](=O)[O-])o1", 0.14, "zinc/graded"),
        ("CC(C)[C@@H](O)c1cccc([N+](=O)[O-])c1", 0.11, "zinc/graded"),
        ("C#CCN(Cc1ccccc1Cl)C1CCCC1", 0.0, "zinc/inactive"),
        (
            "C=CCN1C(=O)C(C#N)=C(C)/C(=C\\c2cn(-c3ccccc3)nc2-c2ccc(OCC)cc2)C1=O",
            0.0,
            "zinc/inactive",
        ),
    ),
    _FOREST_TOLERANCE,
)

# drd2: an RBF-SVM rather than a forest, so the reference values are continuous and a
# wrong fingerprint shows up as a small offset rather than a zero.  Measured to carry
# the SAME lazy relative-path defect as gsk3b: from an unpinned working directory the
# first reference scores 0.0, and 0.7537053303355454 with the directory pinned.
_DRD2 = _references(
    (
        ("C=1C(=CC=C(C1)OCCN2CCCC2)NC3=NC=C(S3)C4=CC=CC=C4", 0.7537053303355454, "drd2/high"),
        (
            "C=1C=CC(N2CCN(CC2)CCCNC(=O)C=3ON=C(N3)C=4C=NC=CC4)=C(C1)F",
            0.5475828952606038,
            "drd2/high",
        ),
        ("CCc1noc(CC)c1CC(=O)N1c2ccccc2C[C@H]1C", 0.18060387163257433, "drd2/intermediate"),
        ("CC(C)Cc1ccc2c(c1)[C@@H](Cl)CC2", 0.16003438414126, "drd2/intermediate"),
        ("Cc1cc(OCC(=O)Nc2ccc([N-]S(C)(=O)=O)cc2)no1", 1.8113384239191436e-05, "drd2/inactive"),
        ("C=1(C=CSC1C(OCC)=O)NC(CC2=CC=CC=C2)=O", 3.0298196625955345e-05, "drd2/inactive"),
    ),
    _SVM_TOLERANCE,
)

POSITIVE_CONTROLS: dict[str, tuple[OracleReference, ...]] = {
    "gsk3b": _GSK3B,
    "jnk3": _JNK3,
    "drd2": _DRD2,
}

# The pinned PyTDC environment reproduced all 29 task-specific reference values.
# The complete call-time audit is summarized in the versioned asset audit.
POSITIVE_CONTROL_STATUS: dict[str, str] = {
    "gsk3b": "MEASURED",
    "jnk3": "MEASURED",
    "drd2": "MEASURED",
}

# How each asset-backed task resolves its asset, MEASURED in the pinned image.  The
# distinction matters: a LAZY load fails silently into `default_property`, an EAGER
# one raises at construction.  Both need the asset directory to resolve; only the
# lazy form can produce a complete, plausible, entirely uninformative ledger.
ASSET_RESOLUTION: dict[str, str] = {
    "gsk3b": "LAZY_RELATIVE_AT_CALL",
    "drd2": "LAZY_RELATIVE_AT_CALL",
    "jnk3": "EAGER_RELATIVE_AT_CONSTRUCTION",
}


# ---- the wrapper that holds the invariant ----------------------------------


class AssetPinnedOracle:
    """A callable oracle whose pinned assets resolve on every call, from any cwd.

    The wrapped object is any callable taking a SMILES string; in production it is a
    ``tdc.Oracle``.  The asset root is the directory PyTDC's relative
    ``oracle/<name>.pkl`` is relative TO -- i.e. the parent of ``oracle/``.
    """

    def __init__(self, oracle: Callable[[str], float], asset_root: Path, *, name: str) -> None:
        asset_root = Path(asset_root).resolve()
        if not asset_root.is_dir():
            raise FileNotFoundError(f"oracle asset root does not exist: {asset_root}")
        self._oracle = oracle
        self._asset_root = asset_root
        self.name = name
        self.calls = 0

    @property
    def asset_root(self) -> Path:
        return self._asset_root

    def assets_present(self) -> list[str]:
        """Names of the pinned asset files, so a missing capsule fails loudly."""
        directory = self._asset_root / ORACLE_ASSET_SUBDIR
        if not directory.is_dir():
            return []
        return sorted(path.name for path in directory.iterdir() if path.is_file())

    def __call__(self, smiles: str) -> float:
        # The working directory is pinned for the CALL, not merely for construction:
        # the load is lazy, so construction-time pinning resolves nothing.
        with pinned_working_directory(self._asset_root):
            value = float(self._oracle(smiles))
        self.calls += 1
        return value

    def prime(self, smiles: str) -> float:
        """Force the lazy load once inside the pinned window.

        After this the evaluator's module-global cache is populated, so later calls
        do not depend on the working directory at all.  The per-call pinning above
        stays as the guarantee for evaluators that do not cache.
        """
        return self(smiles)


# ---- the fail-closed positive control --------------------------------------


def run_positive_control(evaluate: Callable[[str], float], task: str) -> dict:
    """CALL the oracle on every pinned reference and report agreement.

    Returns a report; never raises for a score mismatch.  Use
    :func:`assert_positive_control` where a failure must block a launch.
    """
    references = POSITIVE_CONTROLS.get(task)
    if not references:
        raise KeyError(f"no pinned positive control for asset-backed task {task!r}")

    rows: list[dict] = []
    for reference in references:
        try:
            observed = float(evaluate(reference.smiles))
            error: str | None = None
        except Exception as failure:  # noqa: BLE001 - reported verbatim, never swallowed
            observed, error = float("nan"), f"{type(failure).__name__}: {failure}"
        delta = abs(observed - reference.expected) if error is None else float("nan")
        rows.append(
            {
                "smiles": reference.smiles,
                "role": reference.role,
                "expected": reference.expected,
                "observed": observed,
                "abs_delta": delta,
                "tolerance": reference.tolerance,
                "agrees": error is None and delta <= reference.tolerance,
                "error": error,
            }
        )

    observed_values = [row["observed"] for row in rows if row["error"] is None]
    # A constant-returning oracle reproduces every 0.0 reference and nothing else;
    # requiring real spread makes that impossible to pass even if the panel were
    # accidentally reduced to inactives.
    distinct = len({round(value, 9) for value in observed_values})
    disagreements = [row for row in rows if not row["agrees"]]
    return {
        "schema_version": "pmo_oracle_positive_control_v1",
        "task": task,
        "reference_status": POSITIVE_CONTROL_STATUS.get(task, "UNKNOWN"),
        "n_references": len(rows),
        "n_agreeing": sum(1 for row in rows if row["agrees"]),
        "n_disagreeing": len(disagreements),
        "distinct_observed_values": distinct,
        "max_abs_delta": max(
            (row["abs_delta"] for row in rows if row["error"] is None), default=float("nan")
        ),
        "all_zero": bool(observed_values) and all(value == 0.0 for value in observed_values),
        "passed": not disagreements and distinct >= 3,
        "rows": rows,
    }


def assert_positive_control(evaluate: Callable[[str], float], task: str) -> dict:
    """Run the positive control and raise unless every reference agrees.

    This is the gate that must sit before the first charged call of any
    asset-backed oracle.  It is fail-closed in both directions: an oracle that
    cannot load its asset returns the swallowed constant and fails the nonzero
    references, and an oracle that loaded the WRONG asset fails the graded ones.
    """
    report = run_positive_control(evaluate, task)
    if not report["passed"]:
        failures = [
            f"{row['role']} expected {row['expected']} got {row['observed']}"
            + (f" ({row['error']})" if row["error"] else "")
            for row in report["rows"]
            if not row["agrees"]
        ]
        detail = "; ".join(failures) or (
            f"only {report['distinct_observed_values']} distinct values across "
            f"{report['n_references']} references"
        )
        raise OraclePositiveControlFailure(
            f"{task}: asset-backed oracle failed its pinned positive control: {detail}"
        )
    return report


def requires_positive_control(task: str) -> bool:
    """Whether scoring `task` is gated on a positive-control call."""
    return task in ASSET_BACKED_PMO_TASKS
