"""Frozen oracles for MOLLEO Task 3.

Extract once, checksum, evaluate in plain numpy. The extraction script is
`scripts/molleo_task3_oracle_extract.py` and the parity check that holds this
implementation to the original estimators is
`scripts/molleo_task3_oracle_parity.py`.
"""

from compose_v4.benchmark.oracles.forest import FrozenForest, morgan_bits
from compose_v4.benchmark.oracles.sa import FragmentScores, sa_from_smiles
from compose_v4.benchmark.oracles.task3 import (
    DEFAULT_BUNDLE_DIR,
    NAMES,
    WORST_VECTOR,
    RawScores,
    Task3Objectives,
    canonical,
)

__all__ = [
    "DEFAULT_BUNDLE_DIR",
    "FragmentScores",
    "FrozenForest",
    "NAMES",
    "RawScores",
    "Task3Objectives",
    "WORST_VECTOR",
    "canonical",
    "morgan_bits",
    "sa_from_smiles",
]
