"""SAFE/GenMol fragment-constrained generation benchmark adapter.

The benchmark supplies one or more molecular fragments with open attachment
points (dummy atoms) and asks a generator for 100 completed molecules.  The
published generators enforce the prompt during decoding and therefore do not
report constraint satisfaction separately.  COMPOSE is not a forced-prompt
sequence model, so this adapter makes that hidden assumption explicit: a
candidate must contain every dummy-stripped fragment in a non-overlapping
embedding and must extend every indicated attachment site.

This module is deliberately oracle-free.  QED and synthetic accessibility are
the benchmark's inexpensive ``quality`` diagnostic, not optimization oracles.
"""

from __future__ import annotations

import csv
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, AllChem, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

SCHEMA_VERSION = "compose_fragment_constrained_v1"
DEFAULT_SAMPLES_PER_PROMPT = 100


class FragmentTask(str, Enum):
    LINKER_DESIGN = "linker_design"
    SCAFFOLD_MORPHING = "scaffold_morphing"
    MOTIF_EXTENSION = "motif_extension"
    SCAFFOLD_DECORATION = "scaffold_decoration"
    SUPERSTRUCTURE_GENERATION = "superstructure_generation"


_TASK_COLUMN = {
    FragmentTask.LINKER_DESIGN: "linker_design",
    # GenMol and InVirtuoGen use the linker prompts/results for scaffold morphing.
    FragmentTask.SCAFFOLD_MORPHING: "linker_design",
    FragmentTask.MOTIF_EXTENSION: "motif_extension",
    FragmentTask.SCAFFOLD_DECORATION: "scaffold_decoration",
    FragmentTask.SUPERSTRUCTURE_GENERATION: "superstructure_generation",
}

_REQUIRED_COLUMNS = (
    "name",
    "smiles",
    "linker_design",
    "motif_extension",
    "scaffold_decoration",
    "superstructure_generation",
)


@dataclass(frozen=True)
class FragmentPrompt:
    drug_name: str
    original_smiles: str
    task: FragmentTask
    fragments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.drug_name.strip():
            raise ValueError("fragment prompt has an empty drug name")
        if Chem.MolFromSmiles(self.original_smiles) is None:
            raise ValueError(
                f"invalid original SMILES for {self.drug_name}: {self.original_smiles}"
            )
        if not self.fragments:
            raise ValueError(
                f"fragment prompt has no fragments: {self.drug_name}/{self.task.value}"
            )
        for fragment in self.fragments:
            _fragment_spec(fragment)


@dataclass(frozen=True)
class _FragmentSpec:
    core: Chem.Mol
    attachment_requirements: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ConstraintResult:
    satisfied: bool
    canonical_smiles: str | None
    reason: str | None
    fragment_matches: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class FragmentMetrics:
    expected_samples: int
    chemical_validity: float
    constraint_validity: float
    benchmark_validity: float
    uniqueness: float
    diversity: float
    quality: float
    central_distance: float
    valid_count: int
    constraint_valid_count: int
    unique_constraint_valid_count: int
    quality_count: int


def _fragment_spec(fragment_smiles: str) -> _FragmentSpec:
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        raise ValueError(f"invalid fragment SMILES: {fragment_smiles}")

    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    retained = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0]
    if not retained:
        raise ValueError(f"fragment contains no retained atoms: {fragment_smiles}")
    old_to_new = {old: new for new, old in enumerate(retained)}
    requirements: dict[int, int] = {}
    for idx in dummy_indices:
        atom = mol.GetAtomWithIdx(idx)
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1 or neighbors[0].GetAtomicNum() == 0:
            raise ValueError(
                "each fragment attachment dummy must have exactly one retained neighbor: "
                f"{fragment_smiles}"
            )
        site = old_to_new[neighbors[0].GetIdx()]
        requirements[site] = requirements.get(site, 0) + 1

    # Deleting a dummy from an attachment-bearing aromatic ``n`` can create an
    # unsanitizable standalone molecule even though the retained fragment is a
    # valid substructure query.  Keep the unsanitized retained graph and update
    # only its property cache.  This also preserves atom order.  Converting via
    # MolFragmentToSmarts may reorder atoms and would detach the site indices
    # above from the query atoms they describe.
    editable = Chem.RWMol(mol)
    for idx in sorted(dummy_indices, reverse=True):
        editable.RemoveAtom(idx)
    core = editable.GetMol()
    core.UpdatePropertyCache(strict=False)
    return _FragmentSpec(core, tuple(sorted(requirements.items())))


def _parse_fragments(value: str, task: FragmentTask) -> tuple[str, ...]:
    # The released GenMol asset uses dot-separated fragments only for linker design.
    parts = tuple(part.strip() for part in value.split("."))
    parts = tuple(part for part in parts if part)
    if task in {FragmentTask.LINKER_DESIGN, FragmentTask.SCAFFOLD_MORPHING}:
        if len(parts) != 2:
            raise ValueError(
                f"{task.value} requires exactly two fragments, got {len(parts)}: {value}"
            )
        return parts
    if len(parts) != 1:
        raise ValueError(f"{task.value} requires exactly one fragment, got {len(parts)}: {value}")
    return parts


def load_genmol_prompts(path: str | Path) -> tuple[FragmentPrompt, ...]:
    """Load the released ten-drug prompt asset into 50 explicit task prompts."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _REQUIRED_COLUMNS:
            raise ValueError(
                f"unexpected fragment benchmark columns in {path}: {reader.fieldnames}; "
                f"expected {_REQUIRED_COLUMNS}"
            )
        rows = list(reader)
    names = [row["name"] for row in rows]
    if len(rows) != 10 or len(set(names)) != 10:
        raise ValueError(f"expected ten uniquely named SAFE-drug rows in {path}, got {len(rows)}")

    prompts = []
    for row in rows:
        for task, column in _TASK_COLUMN.items():
            prompts.append(
                FragmentPrompt(
                    drug_name=row["name"],
                    original_smiles=row["smiles"],
                    task=task,
                    fragments=_parse_fragments(row[column], task),
                )
            )
    return tuple(prompts)


def _attachment_sites_are_extended(
    candidate: Chem.Mol,
    match: tuple[int, ...],
    spec: _FragmentSpec,
) -> bool:
    matched = set(match)
    for core_index, required_count in spec.attachment_requirements:
        candidate_index = match[core_index]
        external_heavy = sum(
            neighbor.GetAtomicNum() > 1 and neighbor.GetIdx() not in matched
            for neighbor in candidate.GetAtomWithIdx(candidate_index).GetNeighbors()
        )
        if external_heavy < required_count:
            return False
    return True


def check_fragment_constraint(prompt: FragmentPrompt, candidate_smiles: str) -> ConstraintResult:
    """Check containment, open-site completion, and non-overlap of prompt fragments."""

    candidate = Chem.MolFromSmiles(candidate_smiles) if candidate_smiles else None
    if candidate is None:
        return ConstraintResult(False, None, "unparseable")
    canonical = Chem.MolToSmiles(candidate, isomericSmiles=True)
    if len(Chem.GetMolFrags(candidate)) != 1:
        return ConstraintResult(False, canonical, "disconnected")

    specs = tuple(_fragment_spec(fragment) for fragment in prompt.fragments)
    options: list[tuple[tuple[int, ...], ...]] = []
    for spec in specs:
        matches = tuple(
            match
            for match in candidate.GetSubstructMatches(
                # Dummy removal can make the retained query symmetric while the
                # declared attachment site is not.  Keep automorphic embeddings
                # or a valid site-specific completion can be silently missed.
                spec.core,
                uniquify=False,
                useChirality=False,
                maxMatches=1000,
            )
            if _attachment_sites_are_extended(candidate, match, spec)
        )
        if not matches:
            return ConstraintResult(False, canonical, "missing_fragment_or_attachment")
        options.append(matches)

    chosen: list[tuple[int, ...]] = []

    def bind(fragment_index: int, used: frozenset[int]) -> bool:
        if fragment_index == len(options):
            return True
        for match in options[fragment_index]:
            atoms = frozenset(match)
            if atoms.isdisjoint(used):
                chosen.append(match)
                if bind(fragment_index + 1, used | atoms):
                    return True
                chosen.pop()
        return False

    if not bind(0, frozenset()):
        return ConstraintResult(False, canonical, "fragment_embeddings_overlap")
    return ConstraintResult(True, canonical, None, tuple(chosen))


def _fingerprint(mol: Chem.Mol, *, bits: int) -> DataStructs.ExplicitBitVect:
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=bits, useChirality=False)


def _diversity(molecules: Sequence[Chem.Mol]) -> float:
    if len(molecules) < 2:
        return 0.0
    fps = [_fingerprint(mol, bits=2048) for mol in molecules]
    distances = [
        1.0 - DataStructs.TanimotoSimilarity(left, right)
        for index, left in enumerate(fps)
        for right in fps[index + 1 :]
    ]
    return float(np.mean(distances))


def evaluate_prompt(
    prompt: FragmentPrompt,
    samples: Iterable[str],
    *,
    expected_samples: int = DEFAULT_SAMPLES_PER_PROMPT,
) -> FragmentMetrics:
    """Evaluate one prompt with the published denominators and explicit constraints.

    The official generators report chemical validity because their decoder is
    assumed to preserve the prompt.  ``benchmark_validity`` is COMPOSE's stricter
    counterpart: valid, connected, and fragment compliant.  Quality follows the
    GenMol denominator, unique qualifying molecules divided by all attempted
    samples, with QED >= 0.6 and SA <= 4.
    """

    samples = tuple(samples)
    if len(samples) != expected_samples:
        raise ValueError(f"expected exactly {expected_samples} samples, got {len(samples)}")

    valid_canonical = []
    compliant_canonical = []
    for sample in samples:
        mol = Chem.MolFromSmiles(sample) if sample else None
        if mol is not None and len(Chem.GetMolFrags(mol)) == 1:
            valid_canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        result = check_fragment_constraint(prompt, sample)
        if result.satisfied and result.canonical_smiles is not None:
            compliant_canonical.append(result.canonical_smiles)

    unique = tuple(sorted(set(compliant_canonical)))
    molecules = tuple(Chem.MolFromSmiles(smiles) for smiles in unique)
    quality_count = sum(
        QED.qed(mol) >= 0.6 and sascorer.calculateScore(mol) <= 4.0 for mol in molecules
    )

    original = Chem.MolFromSmiles(prompt.original_smiles)
    original_fp = _fingerprint(original, bits=1024)
    central_distance = (
        float(
            np.mean(
                [
                    1.0 - DataStructs.TanimotoSimilarity(original_fp, _fingerprint(mol, bits=1024))
                    for mol in molecules
                ]
            )
        )
        if molecules
        else 0.0
    )
    valid_count = len(valid_canonical)
    compliant_count = len(compliant_canonical)
    return FragmentMetrics(
        expected_samples=expected_samples,
        chemical_validity=valid_count / expected_samples,
        constraint_validity=compliant_count / max(valid_count, 1),
        benchmark_validity=compliant_count / expected_samples,
        uniqueness=len(unique) / max(compliant_count, 1),
        diversity=_diversity(molecules),
        quality=quality_count / expected_samples,
        central_distance=central_distance,
        valid_count=valid_count,
        constraint_valid_count=compliant_count,
        unique_constraint_valid_count=len(unique),
        quality_count=quality_count,
    )


__all__ = [
    "DEFAULT_SAMPLES_PER_PROMPT",
    "SCHEMA_VERSION",
    "ConstraintResult",
    "FragmentMetrics",
    "FragmentPrompt",
    "FragmentTask",
    "check_fragment_constraint",
    "evaluate_prompt",
    "load_genmol_prompts",
]
