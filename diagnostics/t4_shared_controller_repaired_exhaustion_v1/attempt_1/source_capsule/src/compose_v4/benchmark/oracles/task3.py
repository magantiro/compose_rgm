"""The five MOLLEO Task 3 objectives, assembled from the frozen bundle.

    max QED    max JNK3    min SA    min GSK3B    min DRD2

Minimised objectives are transformed to higher-is-better and all five are
normalised to [0, 1], so the hypervolume reference point is the origin and a
hypervolume is directly a fraction of the unit box.

WHAT IS FROZEN, AND WHAT CANNOT BE
-----------------------------------
JNK3, GSK3B and DRD2 are frozen completely: their parameters live in
checksummed npz files and are evaluated in plain numpy, so no pickle is opened
and no sklearn is imported at runtime.  QED and SA cannot be frozen to the same
degree because both are RDKit computations -- QED is `Chem.QED.qed` and SA needs
RDKit's ring/chirality perception.  What IS pinned for those two is the fragment
table, the arithmetic, and a reference panel: `scripts/molleo_task3_oracle_
parity.py` reports the drift against the extraction environment rather than
leaving it unmeasured.

THE TRANSFORMS ARE PART OF THE BENCHMARK CONTRACT
-------------------------------------------------
`(10 - sa) / 9` is the upstream normalisation, not one of several reasonable
choices.  `1 - p` for the minimised probabilities is the only transform that
keeps them on [0, 1] without introducing a scale.  They are written here once,
in the benchmark, so that no policy can restate them.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compose_v4.benchmark.molleo_task3 import N_OBJECTIVES, OBJECTIVES
from compose_v4.benchmark.oracles.drd2 import BatchInvariantDRD2
from compose_v4.benchmark.oracles.forest import FrozenForest, morgan_bits
from compose_v4.benchmark.oracles.sa import SA_MAX, FragmentScores, sa_from_smiles

#: Where the extraction script writes the bundle. Overridable so a Modal run can
#: point at a mounted volume, but the default is the committed tree -- an oracle
#: that only exists in someone's scratch directory is not frozen.
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BUNDLE_DIR = _REPO_ROOT / "artifacts" / "oracles" / "molleo_task3_v1"
DEFAULT_DRD2_DIR = _REPO_ROOT / "artifacts" / "oracles" / "drd2_svm_v1"
BUNDLE_ENV_VAR = "COMPOSE_MOLLEO_ORACLE_DIR"

#: Objective order. Must match `molleo_task3.OBJECTIVES` exactly -- the meter,
#: the hypervolume and the archive all index by position.
NAMES: tuple[str, ...] = tuple(name for name, _ in OBJECTIVES)


class OracleAccessDuringNavigation(RuntimeError):
    """Raised when anything evaluates objectives while a navigator is running."""


#: Nonzero while a policy is navigating. Module-level ON PURPOSE: a navigator
#: can always construct its own `Task3Objectives`, so a guard living on one
#: instance would be trivially sidestepped. This one lives in the oracle itself,
#: so EVERY instance in the process refuses for the duration.
_NAVIGATION_DEPTH = 0


@contextmanager
def navigation_lockout():
    """Forbid all objective evaluation for the duration of a navigation.

    A navigator may read the archive of molecules ALREADY PAID FOR and learn
    whatever it likes from it. It may not evaluate a NEW molecule, because an
    evaluation that is not charged is exactly the accounting leak this harness
    refuses to copy from the released benchmark.

    Note what this does and does not catch. Importing the oracle is neither
    necessary nor sufficient to violate the rule -- what matters is CALLING it --
    so the check sits at the call and fires no matter who holds the reference or
    how they obtained it.
    """

    global _NAVIGATION_DEPTH
    _NAVIGATION_DEPTH += 1
    try:
        yield
    finally:
        _NAVIGATION_DEPTH -= 1


def navigation_in_progress() -> bool:
    return _NAVIGATION_DEPTH > 0


def canonical(smiles: str) -> str | None:
    """RDKit canonical SMILES, or None if it does not parse.

    Canonicalisation is part of the COUNTING RULE: the released benchmark
    canonicalises before consulting its buffer, so two spellings of one molecule
    must cost one budget unit, not two.
    """

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class RawScores:
    """The five objectives BEFORE any transform, in their native units."""

    qed: float          # [0, 1], higher better
    jnk3: float         # P(active), [0, 1], higher better
    sa: float           # [1, 10],  LOWER better
    gsk3b: float        # P(active), [0, 1], LOWER better
    drd2: float         # P(active), [0, 1], LOWER better

    def normalized(self) -> tuple[float, ...]:
        """Higher-is-better on [0, 1], in `OBJECTIVES` order."""

        return (
            float(np.clip(self.qed, 0.0, 1.0)),
            float(np.clip(self.jnk3, 0.0, 1.0)),
            float(np.clip((10.0 - self.sa) / 9.0, 0.0, 1.0)),
            float(np.clip(1.0 - self.gsk3b, 0.0, 1.0)),
            float(np.clip(1.0 - self.drd2, 0.0, 1.0)),
        )


#: What an unparseable string is worth. Zero in every coordinate is the WORST
#: possible vector after the transforms, so a non-molecule can never enter a
#: Pareto front or add hypervolume.
WORST_VECTOR: tuple[float, ...] = (0.0,) * N_OBJECTIVES


class Task3Objectives:
    """The frozen five. Construct once per process; evaluation is batched."""

    def __init__(self, bundle_dir: Path | None = None,
                 drd2_dir: Path | None = None, *, verify: bool = True):
        if bundle_dir is None:
            bundle_dir = Path(os.environ.get(BUNDLE_ENV_VAR, DEFAULT_BUNDLE_DIR))
        self.bundle_dir = Path(bundle_dir)
        self.drd2_dir = Path(drd2_dir) if drd2_dir is not None else DEFAULT_DRD2_DIR
        manifest_path = self.bundle_dir / "molleo_task3_oracle_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no frozen oracle bundle at {self.bundle_dir}. Run "
                f"scripts/molleo_task3_oracle_extract.py in the pinned "
                f"environment; the benchmark refuses to invent an oracle.")
        self.manifest = json.loads(manifest_path.read_text())
        if verify:
            self._verify_checksums()

        self.jnk3 = FrozenForest(self.bundle_dir / self.manifest["jnk3"]["parameters_npz"])
        self.gsk3b = FrozenForest(self.bundle_dir / self.manifest["gsk3b"]["parameters_npz"])
        self.fragments = FragmentScores(
            self.bundle_dir / self.manifest["sa"]["parameters_npz"])

        # Batch-invariant on purpose: see oracles/drd2.py. A benchmark whose
        # scores depend on batch shape cannot be exactly resumed.
        self.drd2 = BatchInvariantDRD2.from_manifest(
            self.drd2_dir / "drd2_oracle_manifest.json")

    def _verify_checksums(self) -> None:
        """A bundle that does not hash to its manifest is not the frozen oracle."""

        for key in ("jnk3", "gsk3b", "sa"):
            entry = self.manifest[key]
            path = self.bundle_dir / entry["parameters_npz"]
            digest = _sha256(path)
            if digest != entry["parameters_npz_sha256"]:
                raise ValueError(
                    f"{path.name} hashes to {digest[:16]}... but the manifest "
                    f"pins {entry['parameters_npz_sha256'][:16]}.... Refusing to "
                    f"score against an oracle that is not the frozen one.")

    # ---- evaluation ------------------------------------------------------

    def raw_many(self, smiles: list[str]) -> list[RawScores | None]:
        """Raw scores per SMILES; None where the molecule does not parse.

        Every evaluation path in this class funnels through here, which is why
        the navigation lockout is checked here and nowhere else.
        """

        if navigation_in_progress():
            raise OracleAccessDuringNavigation(
                "objectives were evaluated while a navigator was running. A "
                "navigator may learn anything it likes from molecules ALREADY "
                "PAID FOR, but evaluating a new one outside the meter is the "
                "accounting leak this harness exists to refuse. Charge it "
                "through the run instead.")

        from rdkit import Chem
        from rdkit.Chem import QED

        from compose_v4.drd2_oracle import oracle_fingerprint

        smiles = list(smiles)
        out: list[RawScores | None] = [None] * len(smiles)
        keep: list[int] = []
        kinase_rows: list[np.ndarray] = []
        drd2_rows: list[np.ndarray] = []
        qed_values: list[float] = []
        sa_values: list[float] = []

        for position, item in enumerate(smiles):
            mol = Chem.MolFromSmiles(item)
            if mol is None:
                continue
            kinase = morgan_bits(item)
            drd2_fp = oracle_fingerprint(item)
            if kinase is None or drd2_fp is None:
                continue
            keep.append(position)
            kinase_rows.append(kinase)
            drd2_rows.append(drd2_fp)
            qed_values.append(float(QED.qed(mol)))
            sa_values.append(float(sa_from_smiles(item, self.fragments)))

        if not keep:
            return out
        kinase_matrix = np.vstack(kinase_rows)
        # One batched walk per forest and one batched GEMM for the SVM: the
        # per-molecule cost of these three models is dominated by their setup,
        # so evaluating a generation at a time is many times cheaper than
        # evaluating a molecule at a time.
        jnk3 = self.jnk3.probabilities(kinase_matrix)
        gsk3b = self.gsk3b.probabilities(kinase_matrix)
        drd2 = self.drd2.probabilities(np.vstack(drd2_rows))
        for i, position in enumerate(keep):
            out[position] = RawScores(qed=qed_values[i], jnk3=float(jnk3[i]),
                                      sa=sa_values[i], gsk3b=float(gsk3b[i]),
                                      drd2=float(drd2[i]))
        return out

    def raw(self, smiles: str) -> RawScores | None:
        return self.raw_many([smiles])[0]

    def evaluate_many(self, smiles: list[str]) -> list[tuple[float, ...]]:
        """Normalised, higher-is-better vectors; the worst vector for non-molecules."""

        return [WORST_VECTOR if r is None else r.normalized()
                for r in self.raw_many(smiles)]

    def __call__(self, smiles: str) -> tuple[float, ...]:
        """Signature `OracleMeter` expects: one SMILES -> five floats."""

        return self.evaluate_many([smiles])[0]


def worst_raw() -> RawScores:
    """The raw scores that normalise to the worst possible vector."""

    return RawScores(qed=0.0, jnk3=0.0, sa=SA_MAX, gsk3b=1.0, drd2=1.0)
