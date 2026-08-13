"""Load MARS's shipped kinase oracles safely, and REFUSE to proceed if they lie.

MARS ships `gsk3b.pkl` and `jnk3.pkl` written with scikit-learn 0.19.0. On a
modern scikit-learn a plain unpickle raises `ModuleNotFoundError`, which is
harmless because it is loud. **The dangerous failure is the quiet one.**

Since scikit-learn 1.3, `tree_.value` stores class *proportions* and
`predict_proba` no longer normalizes; 0.19 stored raw *counts*. A shim that
fixes only the import errors therefore returns "probabilities" as large as
**1169.7** — and does not crash. MARS would sample against meaningless scores
for an entire run that looked completely successful, in a *baseline*, where
nobody would think to check.

So this module does two things and treats them as inseparable:

1. loads the pickles through a compatibility shim, converting counts to
   proportions;
2. runs a **hard preflight gate** against known actives and random ChEMBL, and
   raises :class:`MarsOraclePreflightError` if the numbers are not sane.

Any MARS artifact produced without a passing gate is `INVALID_INSTRUMENT`.

Must run inside the MARS environment (its own scikit-learn), not the project
interpreter.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Reference values measured 2026-08-13 on the shipped actives files. A correct
#: load reproduces these; a count/proportion bug does not come close.
GATE_THRESHOLDS = {
    "gsk3b": {"min_actives_mean": 0.80, "min_actives_frac_above_0.5": 0.90},
    "jnk3": {"min_actives_mean": 0.75, "min_actives_frac_above_0.5": 0.90},
}
#: A random ChEMBL draw must stay far below the actives. If it does not, the
#: oracle is not discriminating and the run is meaningless even if it "works".
MAX_RANDOM_MEAN = 0.35
#: predict_proba must be a probability. The count/proportion bug shows up here
#: first and most brutally -- values up to 1169.7 were measured.
MAX_ANY_PROBABILITY = 1.0 + 1e-6

#: sklearn module paths that moved. Old pickle path -> current path.
_MODULE_RENAMES = {
    "sklearn.ensemble.forest": "sklearn.ensemble._forest",
    "sklearn.tree.tree": "sklearn.tree._classes",
    "sklearn.ensemble.base": "sklearn.ensemble._base",
    "sklearn.tree._tree": "sklearn.tree._tree",
}


class MarsOraclePreflightError(RuntimeError):
    """The MARS oracle did not pass its sanity gate. Do not produce a number."""


class _LegacyUnpickler(pickle.Unpickler):
    """Redirect scikit-learn 0.19 module paths to their modern homes."""

    def find_class(self, module: str, name: str) -> Any:
        module = _MODULE_RENAMES.get(module, module)
        if module == "sklearn.tree._tree" and name == "Tree":
            return _shim_tree_class()
        return super().find_class(module, name)


def _upgrade_node_array(nodes: np.ndarray) -> np.ndarray:
    """Add the ``missing_go_to_left`` field scikit-learn 1.3 added to tree nodes.

    A 0.19 pickle has seven fields; a modern ``Tree.__setstate__`` rejects the
    array outright unless the eighth is present. All historical splits sent
    missing values left, so 0 is the faithful value.
    """
    target = np.dtype(
        [
            ("left_child", "<i8"),
            ("right_child", "<i8"),
            ("feature", "<i8"),
            ("threshold", "<f8"),
            ("impurity", "<f8"),
            ("n_node_samples", "<i8"),
            ("weighted_n_node_samples", "<f8"),
            ("missing_go_to_left", "u1"),
        ]
    )
    if nodes.dtype == target:
        return nodes
    upgraded = np.zeros(nodes.shape, dtype=target)
    for name in nodes.dtype.names:
        upgraded[name] = nodes[name]
    upgraded["missing_go_to_left"] = 0
    return upgraded


def _shim_tree_class():
    """A ``Tree`` subclass whose ``__setstate__`` upgrades the node array.

    ``sklearn.tree._tree.Tree`` is a Cython extension type and is immutable, so
    its ``__setstate__`` cannot be monkeypatched. It *can* be subclassed, and
    the unpickler calls ``__setstate__`` on whatever class ``find_class``
    returns — so the shim is installed by class substitution instead.
    """
    from sklearn.tree._tree import Tree

    class _ShimTree(Tree):
        def __setstate__(self, state):
            if isinstance(state, dict) and "nodes" in state:
                state = dict(state)
                state["nodes"] = _upgrade_node_array(state["nodes"])
            return Tree.__setstate__(self, state)

    return _ShimTree


def _repair_estimator(estimator: Any) -> None:
    """Apply the attribute renames a 0.19 pickle needs on a modern sklearn."""
    if not hasattr(estimator, "n_features_in_") and hasattr(estimator, "n_features_"):
        estimator.n_features_in_ = estimator.n_features_
    if not hasattr(estimator, "estimator") and hasattr(estimator, "base_estimator"):
        estimator.estimator = estimator.base_estimator
    if not hasattr(estimator, "monotonic_cst"):
        estimator.monotonic_cst = None


def _counts_to_proportions(tree: Any) -> None:
    """THE FIX THAT MATTERS. Convert 0.19 raw class counts to proportions.

    Modern ``predict_proba`` averages ``tree_.value`` across the forest without
    normalizing, so leaving counts in place yields values far above 1.
    """
    values = tree.value
    totals = values.sum(axis=2, keepdims=True)
    np.divide(values, np.where(totals == 0, 1.0, totals), out=values)


def load_kinase_oracle(path: str | Path) -> Any:
    """Load a shipped MARS kinase random forest, repaired for modern sklearn."""
    raw = Path(path).read_bytes()
    # latin1: the pickles carry python-2-era byte strings, which the default
    # ascii codec refuses.
    model = _LegacyUnpickler(io.BytesIO(raw), encoding="latin1").load()

    _repair_estimator(model)
    for sub in getattr(model, "estimators_", []):
        _repair_estimator(sub)
        if hasattr(sub, "tree_"):
            _counts_to_proportions(sub.tree_)
    return model


@dataclass
class PreflightResult:
    objective: str
    actives_mean: float
    actives_frac_above_half: float
    random_mean: float
    max_probability: float
    n_actives: int
    n_random: int
    passed: bool
    failures: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "actives_mean": round(self.actives_mean, 4),
            "actives_frac_above_0.5": round(self.actives_frac_above_half, 4),
            "random_mean": round(self.random_mean, 4),
            "max_probability": round(self.max_probability, 4),
            "n_actives": self.n_actives,
            "n_random": self.n_random,
            "passed": self.passed,
            "failures": self.failures,
        }


def fingerprints(smiles_list: list[str]) -> np.ndarray:
    """MARS's OWN fingerprint, not a plausible substitute.

    `MARS/common/chem.py::fingerprints_from_mol` is
    ``GetMorganFingerprintAsBitVect(mol, 2, 1024)``. The kinase forests were
    fitted on exactly that, and a 2048-bit fingerprint raises rather than
    silently mis-scoring -- which is the one place in this file where sklearn
    is helpfully loud.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rows = []
    for smi in smiles_list:
        molecule = Chem.MolFromSmiles(smi)
        if molecule is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, 1024)
        rows.append(np.array(fp, dtype=np.float64))
    if not rows:
        raise MarsOraclePreflightError("no parseable molecules for the preflight")
    return np.vstack(rows)


def run_preflight(
    objective: str,
    model: Any,
    actives: list[str],
    random_molecules: list[str],
) -> PreflightResult:
    """Score known actives and random ChEMBL; decide whether the oracle is sane."""
    active_probs = model.predict_proba(fingerprints(actives))[:, 1]
    random_probs = model.predict_proba(fingerprints(random_molecules))[:, 1]

    thresholds = GATE_THRESHOLDS.get(objective, GATE_THRESHOLDS["gsk3b"])
    actives_mean = float(active_probs.mean())
    frac_above = float((active_probs >= 0.5).mean())
    random_mean = float(random_probs.mean())
    max_prob = float(max(active_probs.max(), random_probs.max()))

    failures: list[str] = []
    if max_prob > MAX_ANY_PROBABILITY:
        failures.append(
            f"predict_proba returned {max_prob:.1f}, which is not a probability. "
            "This is the scikit-learn count/proportion bug: the model loaded but "
            "every score is meaningless."
        )
    if actives_mean < thresholds["min_actives_mean"]:
        failures.append(
            f"known actives mean {actives_mean:.3f} < "
            f"{thresholds['min_actives_mean']}"
        )
    if frac_above < thresholds["min_actives_frac_above_0.5"]:
        failures.append(
            f"only {frac_above:.1%} of known actives clear 0.5, expected "
            f">= {thresholds['min_actives_frac_above_0.5']:.0%}"
        )
    if random_mean > MAX_RANDOM_MEAN:
        failures.append(
            f"random ChEMBL mean {random_mean:.3f} > {MAX_RANDOM_MEAN}; the "
            "oracle is not discriminating"
        )
    if actives_mean <= random_mean:
        failures.append(
            f"actives ({actives_mean:.3f}) do not outscore random "
            f"({random_mean:.3f})"
        )

    return PreflightResult(
        objective=objective,
        actives_mean=actives_mean,
        actives_frac_above_half=frac_above,
        random_mean=random_mean,
        max_probability=max_prob,
        n_actives=len(active_probs),
        n_random=len(random_probs),
        passed=not failures,
        failures=failures,
    )


def require_preflight(
    objective: str,
    model: Any,
    actives: list[str],
    random_molecules: list[str],
) -> PreflightResult:
    """Run the gate and ABORT on failure. This is the enforcement point.

    Callers must use this, not :func:`run_preflight`, anywhere a number will be
    produced. A gate someone remembers to run is not a gate.
    """
    result = run_preflight(objective, model, actives, random_molecules)
    if not result.passed:
        raise MarsOraclePreflightError(
            f"MARS oracle preflight FAILED for {objective}: "
            + "; ".join(result.failures)
            + ". Any artifact produced from this oracle is INVALID_INSTRUMENT."
        )
    return result
