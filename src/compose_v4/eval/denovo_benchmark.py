"""Published de-novo generation benchmark metrics (GenMol / InVirtuoGen table).

This module implements the four headline metrics those papers report for
unconditional generation, so COMPOSE numbers are computed by the same rules
rather than by a near-miss reimplementation:

``validity``
    fraction of ATTEMPTED generations that parse to an RDKit molecule.  The
    denominator is the number of generations requested, never the number that
    happened to emit a string, so a generation attempt that produces nothing
    counts against validity instead of vanishing from the sample.

``uniqueness``
    distinct canonical SMILES divided by the number of VALID molecules.

``quality``
    the published conjunction -- valid AND unique AND ``QED >= 0.6`` AND
    ``SA <= 4``.  Reported against two explicit denominators because the papers'
    single percentage is ambiguous and the two differ by the uniqueness factor:
    ``quality`` over attempted generations, and ``quality_given_valid_unique``
    over the valid-unique set.  Both are always returned so a reader can never
    silently compare one convention against the other.

``diversity``
    average pairwise Tanimoto DISTANCE (``1 - similarity``) over Morgan
    fingerprints (radius 2, 2048 bits) of the valid-unique set.  ``diversity``
    excludes the ``i == j`` diagonal; ``diversity_moses_intdiv1`` includes it,
    which is the MOSES ``IntDiv1`` convention.  The two agree to O(1/n) and are
    reported separately for the same reason as the quality denominators.

Thresholds are inclusive on both sides (``QED >= 0.6``, ``SA <= 4``) and the
descriptor implementations are the RDKit ones the rest of the repo already
uses -- ``rdkit.Chem.QED.qed`` and the ``SA_Score`` contrib scorer -- imported
from :mod:`compose_v4.eval.molecular_quality` so there is exactly one
definition of each descriptor in the codebase.

Invariants maintained and tested:
  * ``validity`` divides by attempted, so dropping empty emissions cannot
    inflate it;
  * the quality numerator is a subset of the valid-unique set, hence
    ``quality <= uniqueness * validity`` and
    ``quality_given_valid_unique <= 1``;
  * canonicalization happens exactly once, so uniqueness and the quality
    numerator count the same notion of "distinct molecule";
  * diversity is computed over distinct canonical molecules, so duplicate
    generations cannot depress it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from compose_v4.eval.molecular_quality import DESCRIPTORS

# ---- published thresholds ---------------------------------------------------

QED_THRESHOLD = 0.6
SA_THRESHOLD = 4.0
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


# ---- helpers ----------------------------------------------------------------


def _canonicalize(generated: Sequence[str]) -> tuple[list[str], int]:
    """Return canonical SMILES of the parseable entries, and attempted count.

    An empty or unparseable entry is retained in the denominator and dropped
    from the valid list; it is a failed generation, not an absent one.
    """

    attempted = len(generated)
    canonical: list[str] = []
    for text in generated:
        if not text:
            continue
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        canonical.append(Chem.MolToSmiles(mol))
    return canonical, attempted


def _pairwise_tanimoto_distance(smiles: Sequence[str]) -> tuple[float, float]:
    """Mean pairwise Tanimoto distance, off-diagonal and MOSES IntDiv1."""

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    fingerprints = []
    for text in smiles:
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            fingerprints.append(generator.GetFingerprint(mol))

    total = 0.0
    count = 0
    for position, fingerprint in enumerate(fingerprints):
        rest = fingerprints[position + 1 :]
        if rest:
            sims = DataStructs.BulkTanimotoSimilarity(fingerprint, rest)
            total += float(np.sum(sims))
            count += len(sims)

    off_diagonal = 1.0 - (total / count) if count else 0.0
    size = len(fingerprints)
    # IntDiv1 averages the full n x n similarity matrix, whose diagonal is 1.
    moses = 1.0 - ((2.0 * total + size) / (size * size)) if size else 0.0
    return off_diagonal, moses


# ---- benchmark --------------------------------------------------------------


def denovo_benchmark_metrics(
    generated: Sequence[str],
    *,
    attempted: int | None = None,
    train_canonical: frozenset[str] | set[str] | None = None,
    silence_rdkit: bool = True,
) -> dict[str, object]:
    """Compute the published unconditional-generation metrics for one sample.

    Parameters
    ----------
    generated:
        One entry per generation ATTEMPT, in any order.  Empty strings are
        permitted and denote an attempt that emitted no molecule.
    attempted:
        Override for the denominator of ``validity``.  Defaults to
        ``len(generated)``.  Supply it only when failed attempts were dropped
        upstream and their count is known independently.
    train_canonical:
        Canonical SMILES of the training corpus.  When given, novelty is
        reported over the unique valid set (GuacaMol/MOSES convention) and over
        the valid multiset.  Novelty is SECONDARY to the four headline metrics.
    """

    if silence_rdkit:
        RDLogger.DisableLog("rdApp.*")

    canonical, observed_attempts = _canonicalize(generated)
    denominator = observed_attempts if attempted is None else int(attempted)
    if denominator < len(canonical):
        raise ValueError(
            f"attempted ({denominator}) is below the valid count ({len(canonical)})"
        )

    unique = sorted(set(canonical))

    validity = len(canonical) / denominator if denominator else 0.0
    uniqueness = len(unique) / len(canonical) if canonical else 0.0

    # Quality: the published conjunction, evaluated over the DISTINCT molecules
    # so "unique" is part of the predicate rather than a separate filter.
    qed_of = DESCRIPTORS["qed"]
    sa_of = DESCRIPTORS["sa_score"]
    high_quality: list[str] = []
    qed_values: list[float] = []
    sa_values: list[float] = []
    for text in unique:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        qed = qed_of(mol)
        sa = sa_of(mol)
        qed_values.append(qed)
        sa_values.append(sa)
        if qed >= QED_THRESHOLD and sa <= SA_THRESHOLD:
            high_quality.append(text)

    quality = len(high_quality) / denominator if denominator else 0.0
    quality_given_valid_unique = len(high_quality) / len(unique) if unique else 0.0

    diversity, diversity_moses = _pairwise_tanimoto_distance(unique)

    report: dict[str, object] = {
        "attempted": denominator,
        "valid": len(canonical),
        "unique": len(unique),
        "high_quality": len(high_quality),
        "validity": validity,
        "uniqueness": uniqueness,
        "quality": quality,
        "quality_given_valid_unique": quality_given_valid_unique,
        "diversity": diversity,
        "diversity_moses_intdiv1": diversity_moses,
        "qed_threshold": QED_THRESHOLD,
        "sa_threshold": SA_THRESHOLD,
        "mean_qed": float(np.mean(qed_values)) if qed_values else 0.0,
        "mean_sa": float(np.mean(sa_values)) if sa_values else 0.0,
        "fraction_unique_passing_qed": (
            float(np.mean([value >= QED_THRESHOLD for value in qed_values]))
            if qed_values
            else 0.0
        ),
        "fraction_unique_passing_sa": (
            float(np.mean([value <= SA_THRESHOLD for value in sa_values]))
            if sa_values
            else 0.0
        ),
    }

    if train_canonical is not None:
        novel_unique = [text for text in unique if text not in train_canonical]
        report["novelty_unique_set"] = (
            len(novel_unique) / len(unique) if unique else 0.0
        )
        report["novelty_valid_multiset"] = (
            float(np.mean([text not in train_canonical for text in canonical]))
            if canonical
            else 0.0
        )

    return report


def strained_ring_census(smiles: Sequence[str]) -> dict[str, object]:
    """DIAGNOSTIC: how much of the sample carries 3- and 4-membered rings.

    Not a published metric.  It exists because the ``SA <= 4`` half of the
    quality conjunction is dominated by strained small rings, and the de-novo
    base checkpoint has a documented defect of overproducing them
    (aziridine/epoxide).  Reporting the census beside the quality number turns
    "quality is low" into an attributable cause rather than a bare score.

    ``ring_size_histogram`` counts RINGS (SSSR); the fractions count MOLECULES
    containing at least one ring of that size, so a molecule with both a 3- and
    a 4-ring is counted once in ``fraction_with_strained_ring``.
    """

    histogram: dict[int, int] = {}
    with_three = 0
    with_four = 0
    with_either = 0
    total = 0
    for text in smiles:
        if not text:
            continue
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            continue
        total += 1
        sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
        for size in sizes:
            histogram[size] = histogram.get(size, 0) + 1
        distinct = set(sizes)
        if 3 in distinct:
            with_three += 1
        if 4 in distinct:
            with_four += 1
        if 3 in distinct or 4 in distinct:
            with_either += 1

    return {
        "molecules": total,
        "ring_size_histogram": dict(sorted(histogram.items())),
        "fraction_with_3_ring": with_three / total if total else 0.0,
        "fraction_with_4_ring": with_four / total if total else 0.0,
        "fraction_with_strained_ring": with_either / total if total else 0.0,
    }


def aggregate_seed_metrics(per_seed: Sequence[dict[str, object]]) -> dict[str, object]:
    """Mean and sample standard deviation of each metric across seeds.

    The published tables report a single number per metric; this reports the
    across-seed mean alongside its spread so a one-seed result is never quoted
    as if it were the three-seed figure.
    """

    if not per_seed:
        raise ValueError("per_seed must be non-empty")

    keys = [
        "validity",
        "uniqueness",
        "quality",
        "quality_given_valid_unique",
        "diversity",
        "diversity_moses_intdiv1",
        "mean_qed",
        "mean_sa",
    ]
    summary: dict[str, object] = {"seeds": len(per_seed)}
    for key in keys:
        values = [float(entry[key]) for entry in per_seed if key in entry]
        if not values:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        # Sample standard deviation (ddof=1); undefined for a single seed.
        summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    for key in ("novelty_unique_set", "novelty_valid_multiset"):
        values = [float(entry[key]) for entry in per_seed if key in entry]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
    return summary
