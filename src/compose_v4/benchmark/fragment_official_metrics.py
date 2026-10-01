"""Run the OFFICIAL InVirtuoGen fragment-constrained metric on COMPOSE samples.

The reported benchmark numbers (validity / uniqueness / quality / diversity) are
defined by ``in_virtuo_gen.train_utils.metrics.evaluate_smiles``. This module
calls the explicitly supplied, hash-verified evaluator. It never downloads
code or searches the current directory. A tokenizer shim lets the
``already_smiles=True`` branch of the official function receives our strings.

Two invariants make the ``already_smiles=True`` branch semantically identical to
the branch the official model uses, and both are ASSERTED rather than assumed:

``no disconnected sample``
    The ``already_smiles=True`` branch skips ``is_valid_smiles`` and therefore
    skips ``exclude_salts``.  A disconnected ``A.B`` string would be scored
    valid here but invalid on the official model path.  COMPOSE must never emit
    one, so emitting one is a programming error, not a low score.

``every emitted sample is already canonical``
    Official uniqueness de-duplicates the RAW generated string, not its
    canonical form.  Emitting non-canonical strings would inflate uniqueness by
    counting two spellings of one molecule twice.  Emitting canonical strings
    makes official uniqueness exactly ``unique molecules / valid molecules``.

An attempt that produced no chemically committed endpoint is represented by
``FAILED_SAMPLE_PLACEHOLDER``, which RDKit cannot parse. A chemically valid
commit that misses the prompt constraint remains in the *headline* official
metric population and is counted separately as a prompt-fidelity failure.
The caller may additionally score a task-filtered diagnostic population, but
must never relabel that stricter number as published chemical validity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rdkit import Chem

# A deliberately unparseable string: an attempt that produced nothing.
FAILED_SAMPLE_PLACEHOLDER = ""

_OFFICIAL_METRIC_KEYS = ("validity", "uniqueness", "diversity", "quality")


class _IdentitySmilesTokenizer:
    """Shim so the official ``already_smiles=True`` branch receives our strings.

    The official branch computes ``"".join(tokenizer.decode(ids).split())``.
    Returning the SMILES unchanged makes that expression the identity on any
    whitespace-free SMILES.
    """

    @staticmethod
    def decode(ids: Any) -> str:
        return ids if isinstance(ids, str) else str(ids)


def official_unique_valid(samples: Sequence[str]) -> list[str]:
    """The exact set the official metric calls 'unique valid', in emission order.

    Reproduces ``evaluate_smiles``'s own accounting: a sample is valid when
    ``Chem.MolToSmiles(Chem.MolFromSmiles(s))`` is truthy, and uniqueness
    de-duplicates the RAW emitted string.  Deriving the set this way -- rather
    than re-filtering with a rule of our own -- keeps ``distance`` scored over
    the same molecules as ``diversity`` and ``quality``.
    """
    unique: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        mol = Chem.MolFromSmiles(sample) if sample else None
        if mol is None:
            continue
        if not Chem.MolToSmiles(mol, canonical=True):
            continue
        if sample in seen:
            continue
        seen.add(sample)
        unique.append(sample)
    return unique


def official_distance(samples: Sequence[str], reference_smiles: str, *, evaluator) -> float:
    """Mean Tanimoto DISTANCE from ``reference_smiles`` over unique valid samples.

    NOT produced by the released evaluator.  ``evaluate_smiles`` returns no
    ``distance`` key and ``downstream.py`` never calls this path: upstream reads
    the published baselines' distance out of ``references/reference_metrics.csv``
    instead.  What IS upstream is the estimator -- this calls the verified
    ``calculate_average_tanimoto(smiles_list, prompt=...)`` from
    ``in_virtuo_gen/utils/mol.py``, which returns
    ``mean(1 - FingerprintSimilarity(ECFP4-2048(sample), ECFP4-2048(prompt)))``.
    The REFERENCE is ours to choose and changes the number materially, so every
    caller must say which reference it passed.
    """
    unique = official_unique_valid(samples)
    if not unique:
        return float("nan")
    calculate_average_tanimoto = evaluator.average_tanimoto
    return float(calculate_average_tanimoto(unique, prompt=reference_smiles))


def assert_emission_invariants(samples: Sequence[str]) -> None:
    """Fail closed if a sample would change the official metric's semantics."""
    for index, sample in enumerate(samples):
        if sample == FAILED_SAMPLE_PLACEHOLDER:
            continue
        if "." in sample:
            raise ValueError(
                f"sample {index} is disconnected ({sample!r}); COMPOSE must emit a "
                "single connected molecule or the failure placeholder"
            )
        mol = Chem.MolFromSmiles(sample)
        if mol is None:
            raise ValueError(
                f"sample {index} does not parse ({sample!r}); emit the failure "
                "placeholder instead of an unparseable string"
            )
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if sample != canonical:
            raise ValueError(
                f"sample {index} is not canonical ({sample!r} != {canonical!r}); "
                "non-canonical spellings inflate official uniqueness"
            )


def official_prompt_metrics(
    samples: Sequence[str], *, evaluator, expected_samples: int = 100
) -> dict[str, float]:
    """Score one prompt's samples with the official upstream function.

    ``samples`` must have exactly ``expected_samples`` entries: the benchmark
    denominator is the number of ATTEMPTS, so a short list would silently
    inflate validity and quality.
    """
    samples = tuple(samples)
    if len(samples) != expected_samples:
        raise ValueError(f"expected exactly {expected_samples} samples, got {len(samples)}")
    assert_emission_invariants(samples)

    evaluate_smiles = evaluator.evaluate_smiles
    metrics = evaluate_smiles(
        list(samples),
        _IdentitySmilesTokenizer(),
        already_smiles=True,
        return_values=False,
        print_flag=False,
        print_metrics=False,
    )
    return {key: float(metrics[key]) for key in _OFFICIAL_METRIC_KEYS}


__all__ = [
    "FAILED_SAMPLE_PLACEHOLDER",
    "assert_emission_invariants",
    "official_distance",
    "official_prompt_metrics",
    "official_unique_valid",
]
