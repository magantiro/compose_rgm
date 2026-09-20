"""Neutral connected C/N/O/F corpus preparation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from compose_v4.chem.molecular_graph import (
    IDX_TO_ELEMENT,
    MolecularGraphError,
    is_element,
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import is_connected_or_null


@dataclass(frozen=True)
class CNOFCorpusSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    scanned_lines: int
    eligible_molecules: int = 0
    split_strategy: str = "random"
    scaffold_overlap_count: int = 0


def load_cnof_corpus_split(
    path: Path,
    *,
    train_size: int = 96,
    validation_size: int = 16,
    test_size: int = 16,
    max_atoms: int = 12,
    seed: int = 20260714,
    split_strategy: str = "random",
    scan_all: bool = False,
    workers: int = 0,
) -> CNOFCorpusSplit:
    """Load, canonicalize, deduplicate, filter, and deterministically split."""

    requested = train_size + validation_size + test_size
    if requested <= 0 or min(train_size, validation_size, test_size) < 0:
        raise ValueError("split sizes must be non-negative with positive total")
    if max_atoms <= 0:
        raise ValueError("max_atoms must be positive")
    if split_strategy not in {"random", "scaffold"}:
        raise ValueError("split_strategy must be 'random' or 'scaffold'")
    if workers < 0:
        raise ValueError("corpus workers must be non-negative")
    if split_strategy == "scaffold":
        scan_all = True

    if scan_all and workers > 1:
        with path.open() as handle:
            fields = tuple(_first_field(line) for line in handle)
        scanned = len(fields)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
            ) as executor:
                keys = executor.map(
                    _canonical_cnof_smiles,
                    ((text, max_atoms) for text in fields),
                    chunksize=128,
                )
                accepted = {
                    key: None
                    for key in keys
                    if key is not None
                }
        except (OSError, PermissionError):
            accepted = {
                key: None
                for text in fields
                if (key := _canonical_cnof_smiles((text, max_atoms))) is not None
            }
    else:
        accepted = {}
        scanned = 0
        with path.open() as handle:
            for line in handle:
                scanned += 1
                fields = line.strip().split()
                if not fields:
                    continue
                key = _canonical_cnof_smiles((fields[0], max_atoms))
                if key is not None:
                    accepted.setdefault(key, None)
                if not scan_all and len(accepted) >= requested:
                    break
    if len(accepted) < requested:
        raise ValueError(
            f"only found {len(accepted)} eligible molecules; requested {requested}"
        )

    rng = np.random.default_rng(seed)
    molecules = np.asarray(tuple(accepted), dtype=object)
    rng.shuffle(molecules)
    molecules = molecules[:requested]
    if split_strategy == "random":
        train_end = train_size
        validation_end = train_end + validation_size
        train = tuple(str(item) for item in molecules[:train_end])
        validation = tuple(str(item) for item in molecules[train_end:validation_end])
        test = tuple(str(item) for item in molecules[validation_end:requested])
    else:
        train, validation, test = _scaffold_partition(
            tuple(str(item) for item in molecules),
            targets=(train_size, validation_size, test_size),
            rng=rng,
        )
    overlap = _scaffold_overlap_count(train, validation, test)
    return CNOFCorpusSplit(
        train=train,
        validation=validation,
        test=test,
        scanned_lines=scanned,
        eligible_molecules=len(accepted),
        split_strategy=split_strategy,
        scaffold_overlap_count=overlap,
    )


def murcko_scaffold_key(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES for scaffold extraction: {smiles!r}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )
    return scaffold or "<ACYCLIC>"


def _scaffold_partition(
    molecules: tuple[str, ...],
    *,
    targets: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for smiles in molecules:
        groups.setdefault(murcko_scaffold_key(smiles), []).append(smiles)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)

    partitions: list[list[str]] = [[], [], []]
    for group in grouped:
        costs = []
        for candidate in range(3):
            counts = [len(items) for items in partitions]
            counts[candidate] += len(group)
            normalized_error = sum(
                abs(count - target) / max(target, 1)
                for count, target in zip(counts, targets)
            )
            overflow = max(counts[candidate] - targets[candidate], 0)
            costs.append((normalized_error, overflow, candidate))
        destination = min(costs)[2]
        partitions[destination].extend(group)

    for partition in partitions:
        rng.shuffle(partition)
    return tuple(partitions[0]), tuple(partitions[1]), tuple(partitions[2])


def _scaffold_overlap_count(
    train: tuple[str, ...],
    validation: tuple[str, ...],
    test: tuple[str, ...],
) -> int:
    scaffold_sets = [
        {murcko_scaffold_key(smiles) for smiles in partition}
        for partition in (train, validation, test)
    ]
    return len(
        (scaffold_sets[0] & scaffold_sets[1])
        | (scaffold_sets[0] & scaffold_sets[2])
        | (scaffold_sets[1] & scaffold_sets[2])
    )


def _in_scope(graph, *, max_atoms: int) -> bool:
    if graph.n_real_atoms == 0 or graph.n_real_atoms > max_atoms:
        return False
    if not is_connected_or_null(graph):
        return False
    real = is_element(graph.atom_types)
    if np.any(graph.formal_charges[real] != 0):
        return False
    return all(
        IDX_TO_ELEMENT[int(atom_type)] in {"C", "N", "O", "F"}
        for atom_type in graph.atom_types[real]
    )


def _canonical_cnof_smiles(task: tuple[str, int]) -> str | None:
    text, max_atoms = task
    if not text:
        return None
    try:
        graph = smiles_to_molecular_graph(text)
    except (MolecularGraphError, ValueError):
        return None
    if not _in_scope(graph, max_atoms=max_atoms):
        return None
    return molecular_graph_to_smiles(graph)


def _first_field(line: str) -> str:
    fields = line.strip().split()
    return fields[0] if fields else ""
