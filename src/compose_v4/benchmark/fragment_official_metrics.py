"""Run the OFFICIAL InVirtuoGen fragment-constrained metric on COMPOSE samples.

The reported benchmark numbers (validity / uniqueness / quality / diversity) are
defined by ``in_virtuo_gen.train_utils.metrics.evaluate_smiles``.  This module
does not reimplement them: it fetches the pinned upstream bytes (verified by
SHA-256 in ``tools/fetch_official_fragment_evaluator.py``) and calls that
function.  The only adaptation is on COMPOSE's side -- a tokenizer shim so the
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

A failed or constraint-violating attempt is represented by
``FAILED_SAMPLE_PLACEHOLDER``, which RDKit cannot parse, so the official
function counts it as invalid -- which is the intended accounting.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
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


def _official_evaluate_smiles():
    """Import the verified upstream ``evaluate_smiles``."""
    tools = Path(__file__).resolve().parents[3] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from fetch_official_fragment_evaluator import fetch, verify_only

    pkg_root = fetch()
    verify_only()
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    from in_virtuo_gen.train_utils.metrics import evaluate_smiles

    return evaluate_smiles


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
    samples: Sequence[str], *, expected_samples: int = 100
) -> dict[str, float]:
    """Score one prompt's samples with the official upstream function.

    ``samples`` must have exactly ``expected_samples`` entries: the benchmark
    denominator is the number of ATTEMPTS, so a short list would silently
    inflate validity and quality.
    """
    samples = tuple(samples)
    if len(samples) != expected_samples:
        raise ValueError(
            f"expected exactly {expected_samples} samples, got {len(samples)}"
        )
    assert_emission_invariants(samples)

    evaluate_smiles = _official_evaluate_smiles()
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
    "official_prompt_metrics",
]
