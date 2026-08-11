"""Extract the graph2graph / VJTNN DRD2 SVM into version-independent arrays.

WHY THIS EXISTS
---------------
The classic DRD2 optimisation benchmark scores with an RBF-SVM distributed as a
Python-3.6-era sklearn pickle (`props/clf_py36.pkl` in wengong-jin/iclr19-
graph2graph).  Depending on that pickle loading inside the experiment runtime
would make every DRD2 number contingent on an unpinnable interaction between
sklearn, numpy and the unpickler.

So the pickle is opened EXACTLY ONCE, here, in a throwaway environment.  What
gets frozen is the mathematics: support vectors, dual coefficients, intercept,
gamma, and the Platt scaling coefficients -- plus the fingerprint definition and
the SHA of the upstream source files that define it.  The experiment runtime
then evaluates the SVM in plain numpy and never unpickles anything.

This also runs the ORIGINAL implementation over a fixed molecule panel and
stores its scores, so the numpy reimplementation can be held to numerical
agreement against the real thing rather than against our own reading of it.

CPU only, local, no Modal.  Nothing here touches the corpus or the kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


def _install_legacy_module_shims() -> list[str]:
    """Map py36-era module paths onto their modern homes.

    The pickle names `sklearn.svm.classes` and `numpy.core.multiarray`; both were
    renamed.  Aliasing rather than rewriting the pickle keeps the bytes we hash
    identical to what upstream distributes.
    """

    installed: list[str] = []

    # Import the real packages BEFORE aliasing anything.  numpy 2.x resolves
    # `numpy.core` through a module-level __getattr__, so seeding sys.modules
    # first sends that lookup into unbounded recursion via scipy's importer.
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import sklearn.svm  # noqa: F401

    def alias(old: str, new: str) -> None:
        if old in sys.modules:
            return
        try:
            module = __import__(new, fromlist=["*"])
        except ImportError:
            return
        sys.modules[old] = module
        installed.append(f"{old} -> {new}")

    # numpy 2.x moved the C-extension modules under a private package.  Only the
    # leaves the pickle actually names are aliased; `numpy.core` itself is left
    # to numpy's own compatibility shim.
    for suffix in ("multiarray", "umath", "numeric", "_multiarray_umath"):
        alias(f"numpy.core.{suffix}", f"numpy._core.{suffix}")

    # sklearn privatised its estimator modules in 0.22.
    for old, new in (
        ("sklearn.svm.classes", "sklearn.svm._classes"),
        ("sklearn.svm.base", "sklearn.svm._base"),
        ("sklearn.preprocessing.label", "sklearn.preprocessing._label"),
        ("sklearn.calibration", "sklearn.calibration"),
    ):
        alias(old, new)
    return installed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_counts(smiles: str, size: int = 2048) -> np.ndarray | None:
    """The graph2graph DRD2 fingerprint, reproduced exactly.

    Morgan radius 3, USE COUNTS, USE FEATURES -- i.e. count-based FCFP6 folded
    to `size` by modulo on the raw identifier.  Every part of that matters: bit
    vectors, radius 2, or useFeatures=False all produce a different oracle.
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fingerprint = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    folded = np.zeros((1, size), np.int32)
    for identifier, value in fingerprint.GetNonzeroElements().items():
        folded[0, identifier % size] += int(value)
    return folded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path,
                        help="one SMILES per line; the parity panel")
    parser.add_argument("--panel-size", type=int, default=200)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    shims = _install_legacy_module_shims()
    print(f"module shims installed: {shims}")

    pickle_sha = _sha256(args.pickle)
    print(f"pickle sha256 {pickle_sha}  ({args.pickle.stat().st_size:,} bytes)")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with args.pickle.open("rb") as handle:
            clf = pickle.load(handle)
    print(f"loaded: {type(clf).__module__}.{type(clf).__name__}")

    # ---- the parameters that define the model ---------------------------
    if getattr(clf, "_sparse", False):
        raise SystemExit("sparse SVM: densify before extraction (not expected here)")

    def attribute(*names: str) -> Any:
        """Read the first name that is present in the unpickled instance state.

        sklearn renamed several fitted attributes between the version that wrote
        this pickle and the one reading it (``probA_`` became ``_probA``), and
        the modern property raises rather than falling back.  The instance
        ``__dict__`` holds whatever the pickle actually stored, so it is
        consulted before the descriptors.
        """

        for name in names:
            if name in clf.__dict__:
                return clf.__dict__[name]
        for name in names:
            try:
                return getattr(clf, name)
            except AttributeError:
                continue
        raise SystemExit(f"none of {names} present; stored keys: "
                         f"{sorted(clf.__dict__)}")

    support_vectors = np.ascontiguousarray(
        attribute("support_vectors_"), dtype=np.float64)
    dual_coef = np.ascontiguousarray(
        attribute("_dual_coef_", "dual_coef_"), dtype=np.float64)
    intercept = np.ascontiguousarray(
        np.atleast_1d(attribute("_intercept_", "intercept_")), dtype=np.float64)
    classes = np.asarray(attribute("classes_"))
    prob_a = np.ascontiguousarray(
        np.atleast_1d(attribute("_probA", "probA_")), dtype=np.float64)
    prob_b = np.ascontiguousarray(
        np.atleast_1d(attribute("_probB", "probB_")), dtype=np.float64)

    gamma = attribute("_gamma", "gamma")
    if not isinstance(gamma, float):
        gamma = float(np.asarray(gamma).ravel()[0])

    print(f"  kernel        {clf.kernel}   gamma {gamma!r}")
    print(f"  support_vectors_ {support_vectors.shape}")
    print(f"  dual_coef_       {dual_coef.shape}")
    print(f"  intercept_       {intercept.tolist()}")
    print(f"  classes_         {classes.tolist()}")
    print(f"  probA_/probB_    {prob_a.tolist()} / {prob_b.tolist()}")

    if clf.kernel != "rbf":
        raise SystemExit(f"expected an rbf kernel, found {clf.kernel!r}")
    if dual_coef.shape[0] != 1:
        raise SystemExit("expected a binary SVM with a single dual-coef row")

    # ---- let the ORIGINAL estimator run under the modern class -----------
    # The pickle stores the public fitted names of its era; the modern SVC reads
    # private ones and its properties raise instead of falling back.  Copying
    # across (never overwriting) lets sklearn's own predict_proba execute, which
    # is what makes the parity panel a check against the real implementation
    # rather than against our reading of it.
    print(f"  stored keys: {sorted(clf.__dict__)}")
    for private, public in (("_n_support", "n_support_"),
                            ("_probA", "probA_"),
                            ("_probB", "probB_"),
                            ("_dual_coef_", "dual_coef_"),
                            ("_intercept_", "intercept_"),
                            ("_gamma", "gamma")):
        if private not in clf.__dict__ and public in clf.__dict__:
            clf.__dict__[private] = clf.__dict__[public]
    clf.__dict__.setdefault("_sparse", False)
    clf.__dict__.setdefault("n_features_in_", int(support_vectors.shape[1]))
    if "_n_support" not in clf.__dict__:
        raise SystemExit("no support counts in the pickle; cannot run the original")

    # ---- reference scores from the ORIGINAL implementation ---------------
    panel_smiles = [line.strip() for line in
                    args.panel.read_text().splitlines() if line.strip()]
    panel_smiles = panel_smiles[: args.panel_size]
    reference: list[dict[str, Any]] = []
    skipped = 0
    for smiles in panel_smiles:
        counts = fingerprint_counts(smiles)
        if counts is None:
            skipped += 1
            continue
        probability = float(clf.predict_proba(counts)[:, 1][0])
        decision = float(clf.decision_function(counts)[0])
        reference.append({"smiles": smiles,
                          "predict_proba_class1": probability,
                          "decision_function": decision})
    print(f"scored {len(reference)} panel molecules ({skipped} unparseable)")
    active = sum(1 for r in reference if r["predict_proba_class1"] >= 0.5)
    print(f"  panel actives at 0.5 threshold: {active}/{len(reference)}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "drd2_svm_parameters.npz",
        support_vectors=support_vectors,
        dual_coef=dual_coef,
        intercept=intercept,
        prob_a=prob_a,
        prob_b=prob_b,
        classes=classes.astype(np.float64),
        gamma=np.array([gamma], dtype=np.float64),
    )
    parameters_sha = _sha256(args.out / "drd2_svm_parameters.npz")

    metadata = {
        "schema": "compose.oracle.drd2_svm_v1",
        "provenance": {
            "model": "wengong-jin/iclr19-graph2graph props/clf_py36.pkl",
            "lineage": "Olivecrona et al. REINVENT DRD2 activity SVM; the oracle "
                       "behind the classic similarity-constrained DRD2 benchmark "
                       "and the VJTNN/GrIDDD success numbers",
            "pickle_sha256": pickle_sha,
            "pickle_bytes": args.pickle.stat().st_size,
            "opened_once": "This pickle is unpickled exactly once, by this script, "
                           "in a throwaway environment. The experiment runtime "
                           "consumes only the extracted arrays.",
        },
        "fingerprint": {
            "definition": "AllChem.GetMorganFingerprint(mol, 3, useCounts=True, "
                          "useFeatures=True), folded to 2048 by identifier % 2048, "
                          "int32 counts",
            "radius": 3, "size": 2048, "use_counts": True, "use_features": True,
            "folding": "modulo on the raw Morgan identifier",
        },
        "similarity": {
            "definition": "AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, "
                          "useChirality=False) with Tanimoto",
            "note": "the similarity constraint uses a DIFFERENT fingerprint from the "
                    "oracle: ECFP4 bits, not FCFP6 counts",
        },
        "model": {
            "kernel": "rbf", "gamma": gamma,
            "n_support_vectors": int(support_vectors.shape[0]),
            "n_features": int(support_vectors.shape[1]),
            "intercept": intercept.tolist(),
            "prob_a": prob_a.tolist(), "prob_b": prob_b.tolist(),
            "classes": classes.tolist(),
        },
        "parameters_npz_sha256": parameters_sha,
        "parity_panel": {
            "source": "iclr19-graph2graph data/drd2/test.txt",
            "molecules": len(reference),
            "unparseable": skipped,
        },
        "reference_scores": reference,
    }
    (args.out / "drd2_oracle_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(f"\nwrote {args.out}/drd2_svm_parameters.npz  sha256={parameters_sha[:16]}")
    print(f"wrote {args.out}/drd2_oracle_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
