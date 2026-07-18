"""Randomized program fuzzing for the validity-closed rewrite runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, smiles_to_molecular_graph
from compose_v4.chem.state import is_valid_state, pad_molecular_graph
from compose_v4.rewrite.compiler import compile_null_to_target
from compose_v4.rewrite.kernel import canonical_state_key, default_rewrite_system
from compose_v4.rewrite.trace import execute_trace, invert_trace


DEFAULT_FUZZ_SMILES = (
    "CC(=O)NCCO",
    "N#CCO",
    "C[N+](C)(C)C",
    "c1ccccc1",
    "c1ccc2ccccc2c1",
    "C1CC2CCC1C2",
    "C1CCC2(CC1)CCCC2",
    "c1ncc2ccccc2n1",
)


@dataclass(frozen=True)
class FuzzReport:
    seed: int
    programs: int
    committed_rewrites: int
    exact_forward: int
    exact_reverse: int
    canonical_permutation_matches: int


def _same_arrays(left: MolecularGraph, right: MolecularGraph) -> bool:
    return bool(
        np.array_equal(left.atom_types, right.atom_types)
        and np.array_equal(left.formal_charges, right.formal_charges)
        and np.array_equal(left.implicit_h_counts, right.implicit_h_counts)
        and np.array_equal(left.bonds, right.bonds)
    )


def _permute_slots(target: MolecularGraph, rng: np.random.Generator) -> MolecularGraph:
    permutation = rng.permutation(target.n_atoms)
    return MolecularGraph(
        atom_types=target.atom_types[permutation],
        formal_charges=target.formal_charges[permutation],
        implicit_h_counts=target.implicit_h_counts[permutation],
        bonds=target.bonds[np.ix_(permutation, permutation)],
    )


def run_compiled_trace_fuzz(
    *,
    target_commits: int = 10_000,
    seed: int = 0,
    smiles: tuple[str, ...] = DEFAULT_FUZZ_SMILES,
) -> FuzzReport:
    """Replay random slot-permuted forward/inverse programs until the gate is met."""

    if target_commits <= 0:
        raise ValueError("target_commits must be positive")
    rng = np.random.default_rng(seed)
    runtime = default_rewrite_system()
    parsed = tuple(smiles_to_molecular_graph(smi) for smi in smiles)
    programs = committed = exact_forward = exact_reverse = canonical_matches = 0

    while committed < target_commits:
        base = parsed[int(rng.integers(0, len(parsed)))]
        target = pad_molecular_graph(base, base.n_atoms + int(rng.integers(1, 9)))
        original_key = canonical_state_key(target)
        target = _permute_slots(target, rng)
        if canonical_state_key(target) != original_key:
            raise AssertionError("slot permutation changed the chemical state")
        canonical_matches += 1

        trace = compile_null_to_target(target, system=runtime)
        reconstructed, forward_states = execute_trace(
            trace.source, trace.steps, system=runtime, return_states=True
        )
        reverse = invert_trace(trace.source, trace.steps, system=runtime)
        restored, reverse_states = execute_trace(
            reconstructed, reverse, system=runtime, return_states=True
        )
        if not _same_arrays(reconstructed, target):
            raise AssertionError("forward trace failed exact reconstruction")
        if not _same_arrays(restored, trace.source):
            raise AssertionError("inverse trace failed exact reconstruction")
        if not all(is_valid_state(state) for state in (*forward_states, *reverse_states)):
            raise AssertionError("a committed intermediate state was invalid")

        exact_forward += 1
        exact_reverse += 1
        programs += 1
        committed += len(trace.steps) + len(reverse)

    return FuzzReport(
        seed=seed,
        programs=programs,
        committed_rewrites=committed,
        exact_forward=exact_forward,
        exact_reverse=exact_reverse,
        canonical_permutation_matches=canonical_matches,
    )
