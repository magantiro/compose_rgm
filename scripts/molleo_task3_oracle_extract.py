#!/usr/bin/env python
"""Freeze the MOLLEO Task 3 oracles once, in an isolated environment.

WHY THIS SCRIPT EXISTS
----------------------
Two of the five objectives (JNK3, GSK3B) are random forests pickled by
scikit-learn 0.21.3.  They cannot be unpickled by a modern scikit-learn -- the
tree node dtype changed in 0.22 -- and even where they can, "the oracle is
whatever this pickle does in whatever sklearn happens to be installed" is not a
benchmark.  So the pickles are opened EXACTLY ONCE, here, in a throwaway
environment pinned to scikit-learn 1.2.2, and their parameters are written to an
npz that the experiment runtime evaluates in plain numpy.  Nothing at runtime
imports sklearn or unpickles anything, so the oracle cannot drift when the
environment does.  This is the same discipline `drd2_oracle_extract.py` applies
to the DRD2 SVM.

WHAT IS CHECKED RATHER THAN ASSUMED
-----------------------------------
* The forests are validated to expect 2048 features.  This is the whole ball
  game: the same two kinase tasks circulate in TWO incompatible versions --
  TDC's (Morgan bits radius 2, **2048**) and HN-GFN/MARS's (radius 2, **1024**).
  MOLLEO calls TDC's.  Feeding 1024-bit fingerprints to a 2048-feature forest
  raises; feeding the RIGHT WIDTH to the WRONG MODEL silently produces a
  plausible, wrong oracle.  `n_features_in_` is therefore asserted, not hoped
  for, and the source sha256 is recorded.
* Every internal split threshold is asserted to be exactly 0.5.  The features
  are bits, so this must hold -- and because it holds, the frozen bundle can
  drop the threshold array.  A silently dropped threshold would corrupt every
  score, so it is verified before it is dropped.
* Reference scores are computed here, from the ORIGINAL estimators and from
  RDKit's own reference SA implementation, and stored in the manifest.  The
  parity script then holds the numpy reimplementation to them.  Orientation
  (which column of `predict_proba` is "active") is pinned by scoring KNOWN
  ACTIVES: a flipped objective fails loudly instead of looking plausible.

RUN IT WITH THE PINNED ENVIRONMENT, NOT THE PROJECT ENVIRONMENT:

    <isolated-venv>/bin/python scripts/molleo_task3_oracle_extract.py \
        --jnk3 .../oracle/jnk3.pkl --gsk3b .../oracle/gsk3b.pkl \
        --fpscores .../fpscores.pkl.gz \
        --drd2 .../drd2/clf_py36.pkl \
        --zinc local_runtime/zinc250k/250k_rndm_zinc_drugs_clean_3.csv \
        --kinase .../kinase_rf/kinase.tsv \
        --out artifacts/oracles/molleo_task3_v1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path

import numpy as np

#: The forests were fitted on 2048-bit Morgan(radius 2) bit vectors. TDC's
#: gsk3b/jnk3 oracles build exactly this; the HN-GFN/MARS variants of the same
#: two tasks use 1024 bits and are a DIFFERENT model.
EXPECTED_FEATURES = 2048
FINGERPRINT_RADIUS = 2
#: Splits are `x[feature] <= 0.5`, i.e. "is this bit off".
SPLIT_THRESHOLD = 0.5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_legacy_pickle(path: Path):
    """Open a pre-0.22 scikit-learn pickle under the module names it expects."""

    # Import the real packages BEFORE aliasing anything.
    import sklearn.ensemble._forest  # noqa: F401
    import sklearn.svm  # noqa: F401
    import sklearn.tree._classes  # noqa: F401

    # 0.22 privatised these estimator modules; the pickles still reference the
    # old names. Aliasing is what makes the artifacts openable at all.
    for old, new in (
        ("sklearn.ensemble.forest", "sklearn.ensemble._forest"),
        ("sklearn.tree.tree", "sklearn.tree._classes"),
        ("sklearn.svm.classes", "sklearn.svm._classes"),
        ("sklearn.svm.base", "sklearn.svm._base"),
    ):
        if old not in sys.modules:
            sys.modules[old] = __import__(new, fromlist=["*"])
    with open(path, "rb") as handle:
        return pickle.load(handle, encoding="iso-8859-1")


def enable_legacy_svc(clf):
    """Let a 0.21-era SVC run under the modern class.

    The pickle stores the public fitted names of its era; the modern SVC reads
    private ones and its properties raise instead of falling back. Copying
    across (never overwriting) lets sklearn's own `predict_proba` execute, so
    the parity panel checks the real implementation rather than our reading of
    it. Same shim as `drd2_oracle_extract.py`.
    """

    for private, public in (("_n_support", "n_support_"), ("_probA", "probA_"),
                            ("_probB", "probB_"), ("_dual_coef_", "dual_coef_"),
                            ("_intercept_", "intercept_"), ("_gamma", "gamma")):
        if private not in clf.__dict__ and public in clf.__dict__:
            clf.__dict__[private] = clf.__dict__[public]
    clf.__dict__.setdefault("_sparse", False)
    clf.__dict__.setdefault("n_features_in_", int(clf.support_vectors_.shape[1]))
    if "_n_support" not in clf.__dict__:
        raise SystemExit("no support counts in the pickle; cannot run the original")
    return clf


def flatten_forest(forest) -> dict[str, np.ndarray]:
    """Pack a RandomForestClassifier into flat arrays a numpy walker can use.

    Trees are concatenated into one global node table so that a batch can walk
    all 100 trees at once: one gather per depth level rather than one per
    (tree, level).  Leaves are made to point at themselves, which turns "have
    we finished" into a single `feature >= 0` test instead of a per-tree
    bookkeeping structure.
    """

    # 0.21 called it `n_features_`; 0.24+ calls it `n_features_in_`. The tree
    # itself is the authority, so it is the one that has to agree.
    n_features = int(forest.estimators_[0].tree_.n_features)
    for attribute in ("n_features_in_", "n_features_"):
        declared = getattr(forest, attribute, None)
        if declared is not None and int(declared) != n_features:
            raise SystemExit(f"forest.{attribute}={declared} disagrees with the "
                             f"tree's own n_features={n_features}")
    if n_features != EXPECTED_FEATURES:
        raise SystemExit(
            f"refusing to freeze: forest expects {n_features} features, not "
            f"{EXPECTED_FEATURES}. A {n_features}-feature model of the same "
            f"task is a DIFFERENT oracle, not a variant of this one.")
    classes = np.asarray(forest.classes_).ravel()
    if classes.shape != (2,) or float(classes[0]) != 0.0 or float(classes[1]) != 1.0:
        raise SystemExit(f"refusing to freeze: unexpected classes_ {classes!r}")

    lefts, rights, feats, leaf_p1 = [], [], [], []
    offsets = [0]
    thresholds_seen: set[float] = set()
    for estimator in forest.estimators_:
        tree = estimator.tree_
        base = offsets[-1]
        left = np.asarray(tree.children_left, dtype=np.int64).copy()
        right = np.asarray(tree.children_right, dtype=np.int64).copy()
        feature = np.asarray(tree.feature, dtype=np.int64).copy()
        internal = left != -1
        thresholds_seen.update(np.unique(tree.threshold[internal]).tolist())

        # value is (n_nodes, 1, 2) class counts; predict_proba normalises per
        # node and then averages over trees, so the per-leaf P(class 1) is all
        # that survives.
        value = np.asarray(tree.value, dtype=np.float64)[:, 0, :]
        total = value.sum(axis=1)
        if np.any(total[~internal] <= 0):
            raise SystemExit("refusing to freeze: a leaf carries zero samples")
        probability = np.zeros(len(value), dtype=np.float64)
        probability[~internal] = value[~internal, 1] / total[~internal]

        # A leaf points at itself, so walking past it is a no-op.
        self_index = np.arange(len(left), dtype=np.int64)
        left = np.where(internal, left, self_index) + base
        right = np.where(internal, right, self_index) + base
        feature = np.where(internal, feature, -1)

        lefts.append(left)
        rights.append(right)
        feats.append(feature)
        leaf_p1.append(probability)
        offsets.append(base + len(left))

    if thresholds_seen != {SPLIT_THRESHOLD}:
        raise SystemExit(
            f"refusing to freeze: internal thresholds are {sorted(thresholds_seen)[:5]}, "
            f"not the single value {SPLIT_THRESHOLD}. The bundle drops the "
            f"threshold array on the strength of this check.")

    total_nodes = offsets[-1]
    if total_nodes >= np.iinfo(np.int32).max:
        raise SystemExit("node table too large for int32 indices")
    return {
        "children_left": np.concatenate(lefts).astype(np.int32),
        "children_right": np.concatenate(rights).astype(np.int32),
        # -1 marks a leaf; 2048 features fit in int16 with room to spare.
        "feature": np.concatenate(feats).astype(np.int16),
        "leaf_p1": np.concatenate(leaf_p1).astype(np.float64),
        "tree_root": np.asarray(offsets[:-1], dtype=np.int32),
        "n_features": np.asarray(EXPECTED_FEATURES, dtype=np.int32),
        "split_threshold": np.asarray(SPLIT_THRESHOLD, dtype=np.float64),
    }


def morgan_bits(smiles: str, n_bits: int = EXPECTED_FEATURES) -> np.ndarray | None:
    """TDC's kinase fingerprint, reproduced exactly.

    `AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)` into a float64
    array -- the same call TDC makes for both gsk3b and jnk3.
    """

    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bits = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=n_bits)
    row = np.zeros((n_bits,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(bits, row)
    return row


def read_zinc(path: Path, count: int, seed: int) -> list[str]:
    """A deterministic sample of ZINC-250k, the benchmark's own init distribution."""

    with open(path, newline="") as handle:
        rows = [row["smiles"].strip() for row in csv.DictReader(handle)]
    rng = np.random.default_rng(seed)
    index = rng.choice(len(rows), size=count, replace=False)
    return [rows[int(i)] for i in sorted(index)]


def read_kinase_actives(path: Path, task: str, count: int, seed: int) -> list[str]:
    """Known actives for `task`.  Agreement on inactives proves nothing: a
    function returning 0.0 everywhere would pass.  Actives are what pin the
    orientation of `predict_proba`."""

    actives: list[str] = []
    with open(path) as handle:
        handle.readline()          # header
        handle.readline()          # blank line in the upstream file
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            target, is_active, _is_train, smiles = parts[0], parts[1], parts[2], parts[3]
            if target == task and is_active == "1":
                actives.append(smiles)
    rng = np.random.default_rng(seed)
    index = rng.choice(len(actives), size=min(count, len(actives)), replace=False)
    return [actives[int(i)] for i in sorted(index)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jnk3", required=True, type=Path)
    parser.add_argument("--gsk3b", required=True, type=Path)
    parser.add_argument("--drd2", required=True, type=Path,
                        help="graph2graph clf_py36.pkl, for an independent DRD2 parity")
    parser.add_argument("--fpscores", required=True, type=Path,
                        help="fpscores.pkl.gz -- the SA fragment table")
    parser.add_argument("--zinc", required=True, type=Path)
    parser.add_argument("--kinase", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--zinc-panel", type=int, default=250)
    parser.add_argument("--actives-panel", type=int, default=75)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import rdkit
    import sklearn
    from rdkit.Chem import QED
    from rdkit import Chem

    # RDKit's own reference SA implementation. TDC's calculateScore is a copy of
    # it; using the upstream original keeps the parity honest rather than
    # comparing our reimplementation against a second copy of itself.
    sys.path.insert(0, str(Path(rdkit.__file__).resolve().parent / "Contrib" / "SA_Score"))
    import sascorer  # noqa: E402

    manifest: dict = {
        "schema": "compose.oracle.molleo_task3_v1",
        "extraction_environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
            "note": ("the pickles are openable only under a pre-1.3 sklearn; "
                     "1.3 changed the tree node dtype"),
        },
        "objectives": {
            "qed": {"direction": "max", "source": "rdkit Chem.QED.qed",
                    "transform": "identity", "range": [0, 1]},
            "jnk3": {"direction": "max", "source": "TDC jnk3 random forest",
                     "transform": "identity", "range": [0, 1]},
            "sa": {"direction": "min", "source": "RDKit Contrib SA_Score sascorer",
                   "transform": "(10 - sa) / 9", "range": [1, 10]},
            "gsk3b": {"direction": "min", "source": "TDC gsk3b random forest",
                      "transform": "1 - gsk3b", "range": [0, 1]},
            "drd2": {"direction": "min", "source": "REINVENT DRD2 RBF-SVM",
                     "transform": "1 - drd2", "range": [0, 1]},
        },
    }

    # ---- panels ---------------------------------------------------------
    zinc = read_zinc(args.zinc, args.zinc_panel, args.seed)
    jnk3_actives = read_kinase_actives(args.kinase, "jnk3", args.actives_panel, args.seed)
    gsk3b_actives = read_kinase_actives(args.kinase, "gsk3b", args.actives_panel, args.seed)
    panel = list(dict.fromkeys(zinc + jnk3_actives + gsk3b_actives))
    groups = ({s: "zinc" for s in zinc} | {s: "jnk3_active" for s in jnk3_actives}
              | {s: "gsk3b_active" for s in gsk3b_actives})
    print(f"panel: {len(panel)} molecules "
          f"({len(zinc)} zinc / {len(jnk3_actives)} jnk3 actives / "
          f"{len(gsk3b_actives)} gsk3b actives)")

    fingerprints = np.vstack([morgan_bits(s) for s in panel])

    # ---- the two kinase forests -----------------------------------------
    forests = {}
    for name, path in (("jnk3", args.jnk3), ("gsk3b", args.gsk3b)):
        digest = sha256_file(path)
        print(f"opening {name}: {path} ({path.stat().st_size:,} bytes)")
        model = load_legacy_pickle(path)
        arrays = flatten_forest(model)
        npz_path = args.out / f"{name}_forest.npz"
        np.savez_compressed(npz_path, **arrays)
        reference = model.predict_proba(fingerprints)[:, 1]
        forests[name] = reference
        manifest[name] = {
            "provenance": {
                "file": path.name,
                "pickle_sha256": digest,
                "pickle_bytes": path.stat().st_size,
                "found_at": str(path),
                "lineage": (
                    "Random forest for the GSK3B/JNK3 multi-property task of "
                    "Jin et al. (RationaleRL); redistributed by TDC as "
                    f"oracle/{name}.pkl and called by MOLLEO through "
                    "tdc.Oracle. This copy was vendored, at TDC's own relative "
                    "path oracle/, in the HN-GFN repository."),
                "identity_evidence": (
                    "2048 input features, classes_ [0., 1.], 100 trees, "
                    "sklearn 0.21.3 -- matching TDC's code path exactly "
                    "(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) "
                    "then predict_proba[:, 1]). NOT the 1024-feature HN-GFN "
                    "variant of the same task."),
                "byte_identity_to_tdc_download": "UNVERIFIED -- host unreachable",
            },
            "fingerprint": {
                "definition": ("AllChem.GetMorganFingerprintAsBitVect(mol, 2, "
                               "nBits=2048) into a float64 array"),
                "radius": FINGERPRINT_RADIUS,
                "n_bits": EXPECTED_FEATURES,
                "counts": False,
                "features": False,
            },
            "model": {
                "type": "RandomForestClassifier",
                "n_estimators": len(model.estimators_),
                "n_features": EXPECTED_FEATURES,
                "n_nodes": int(arrays["children_left"].size),
                "max_depth": int(max(e.tree_.max_depth for e in model.estimators_)),
                "classes": [0.0, 1.0],
                "score": "predict_proba(fp)[:, 1] == P(active)",
                "split_rule": "x[feature] <= 0.5 -> left child",
            },
            "parameters_npz": npz_path.name,
            "parameters_npz_sha256": sha256_file(npz_path),
        }
        print(f"  froze {arrays['children_left'].size:,} nodes -> {npz_path.name} "
              f"({npz_path.stat().st_size:,} bytes)")
        del model

    # ---- DRD2, opened again for an independent parity --------------------
    print(f"opening drd2: {args.drd2}")
    drd2_model = enable_legacy_svc(load_legacy_pickle(args.drd2))
    from compose_v4.drd2_oracle import oracle_fingerprint  # noqa: E402
    drd2_fp = np.vstack([oracle_fingerprint(s) for s in panel])
    drd2_reference = drd2_model.predict_proba(drd2_fp)[:, 1]
    manifest["drd2"] = {
        "provenance": {
            "file": args.drd2.name,
            "pickle_sha256": sha256_file(args.drd2),
            "lineage": ("Olivecrona et al. REINVENT DRD2 activity SVM, "
                        "redistributed by wengong-jin/iclr19-graph2graph as "
                        "props/clf_py36.pkl and by TDC as oracle/drd2.pkl"),
            "byte_identity_to_tdc_download": "UNVERIFIED -- host unreachable",
        },
        "parameters_npz": "../drd2_svm_v1/drd2_svm_parameters.npz",
        "note": ("already frozen by drd2_oracle_extract.py; re-scored here on "
                 "the Task 3 panel so this bundle carries its own parity"),
    }
    del drd2_model

    # ---- SA fragment table ----------------------------------------------
    with gzip.open(args.fpscores, "rb") as handle:
        raw = handle.read()
    rows = pickle.loads(raw)
    table: dict[int, float] = {}
    for row in rows:
        for fragment_id in row[1:]:
            table[int(fragment_id)] = float(row[0])
    keys = np.fromiter(table.keys(), dtype=np.int64, count=len(table))
    order = np.argsort(keys)
    keys = keys[order]
    values = np.fromiter(table.values(), dtype=np.float64, count=len(table))[order]
    sa_path = args.out / "sa_fragment_scores.npz"
    np.savez_compressed(sa_path, fragment_id=keys, score=values)
    manifest["sa"] = {
        "provenance": {
            "table": "fpscores.pkl.gz",
            "gzip_sha256": hashlib.sha256(raw).hexdigest(),
            "entries": len(table),
            "lineage": ("Ertl & Schuffenhauer fragment contributions; the file "
                        "shipped with RDKit Contrib/SA_Score, byte-identical to "
                        "the copy vendored by HN-GFN/MARS and the same table TDC "
                        "downloads as oracle/fpscores.pkl"),
        },
        "algorithm": ("RDKit Contrib/SA_Score sascorer.calculateScore, vendored "
                      "into compose_v4.benchmark.oracles.sa so no pickle is read "
                      "at runtime"),
        "parameters_npz": sa_path.name,
        "parameters_npz_sha256": sha256_file(sa_path),
    }
    print(f"  froze {len(table):,} fragment scores -> {sa_path.name} "
          f"({sa_path.stat().st_size:,} bytes)")

    # ---- reference scores ------------------------------------------------
    reference_rows = []
    for i, smiles in enumerate(panel):
        mol = Chem.MolFromSmiles(smiles)
        reference_rows.append({
            "smiles": smiles,
            "group": groups[smiles],
            "qed": float(QED.qed(mol)),
            "sa": float(sascorer.calculateScore(mol)),
            "jnk3": float(forests["jnk3"][i]),
            "gsk3b": float(forests["gsk3b"][i]),
            "drd2": float(drd2_reference[i]),
        })
    manifest["reference_panel"] = {
        "molecules": len(panel),
        "seed": args.seed,
        "zinc_source": str(args.zinc),
        "kinase_source": str(args.kinase),
        "produced_by": ("original estimators (sklearn "
                        f"{sklearn.__version__}) and RDKit {rdkit.__version__}"),
    }
    manifest["reference_scores"] = reference_rows

    # Orientation, stated as a number rather than as a belief.
    for task, actives_group in (("jnk3", "jnk3_active"), ("gsk3b", "gsk3b_active")):
        active = np.array([r[task] for r in reference_rows if r["group"] == actives_group])
        background = np.array([r[task] for r in reference_rows if r["group"] == "zinc"])
        manifest[task]["orientation_check"] = {
            "known_actives_mean": float(active.mean()),
            "zinc_background_mean": float(background.mean()),
            "separation": float(active.mean() - background.mean()),
            "meaning": ("predict_proba[:, 1] must be HIGHER on known actives. "
                        "If this is negative the objective is inverted."),
        }
        print(f"  {task}: actives {active.mean():.4f} vs zinc {background.mean():.4f}")

    manifest_path = args.out / "molleo_task3_oracle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
