"""Training-derived, stochastic two-boundary proposals with exact core assembly.

The joint structural prior is fitted elsewhere once from the accepted training
molecules. This module only binds and samples its cells, compiles full programs,
and delegates learned selection to the shared frozen-model scorer. It does not
implement a beam, property guidance, a learned model, or a benchmark launcher.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rdkit import Chem

from compose_v4.benchmark.fragment_constrained import FragmentPrompt
from compose_v4.benchmark.fragment_constrained_runner import ProposalLimits, _capped_core
from compose_v4.benchmark.fragment_linker_assembly import (
    _connector,
    _specs,
    assemble_linker_program,
)
from compose_v4.benchmark.fragment_program_adapter import (
    CompleteProgram,
    learned_program_scores,
    select_learned_program,
    select_learned_program_novelty4,
    select_learned_program_strict_unseen,
)
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior
from compose_v4.benchmark.training_attachment_fragments import atom_context, physical_sha256


@dataclass(frozen=True)
class LinkerCatalog:
    catalog_sha256: str
    by_context: dict[tuple[str, str], tuple[dict[str, Any], ...]]
    two_boundary_entries: int


@dataclass(frozen=True)
class LinkerPanelResult:
    selected: CompleteProgram | None
    receipt: dict[str, Any]


class LinkerProposalAbstention(ValueError):
    def __init__(self, reason: str, receipt: dict[str, Any]):
        super().__init__(reason)
        self.receipt = receipt


CellAllocation = Literal["frozen", "sqrt_train_mass"]


def linker_cell_probabilities(
    prior: JointCompletionPrior,
    cells: tuple[tuple[int, int], ...],
    *,
    allocation: CellAllocation = "frozen",
) -> np.ndarray:
    """Temper training-cell mass without changing its reachable support."""
    probabilities = prior.cell_probabilities(cells)
    if allocation == "frozen":
        return probabilities
    if allocation != "sqrt_train_mass":
        raise ValueError(f"unknown linker cell allocation: {allocation}")
    weights = np.sqrt(probabilities)
    return weights / weights.sum()


def prepare_linker_catalog(catalog: dict, *, catalog_sha256: str) -> LinkerCatalog:
    """Index only the frozen split-first entries; do not extract or fit content."""
    if catalog.get("schema") != "split_first_training_region_catalog_v1" or catalog.get(
        "split"
    ) != {
        "algorithm": "murcko+carbonized-wl3",
        "version": 2,
        "salt": "ringcore-v1",
        "ratios": [0.9, 0.05, 0.05],
        "partition": "train",
    }:
        raise ValueError("linker catalog must use the frozen split-first training partition")
    admitted = catalog["accepted_source_rows"]
    if (
        len(admitted) != catalog["training_molecules"]
        or len(set(admitted)) != len(admitted)
        or any(type(row) is not int or row < 1 for row in admitted)
    ):
        raise ValueError("linker catalog accepted training-row identity is malformed")
    admitted = frozenset(admitted)
    grouped, count = defaultdict(list), 0
    for index, entry in enumerate(catalog["entries"]):
        contexts = tuple(entry["contexts"])
        if len(contexts) != 2:
            continue
        if (
            any(not isinstance(key, str) or not key for key in contexts)
            or type(entry["heavy_atoms"]) is not int
            or entry["heavy_atoms"] < 1
            or type(entry["ring_count"]) is not int
            or entry["ring_count"] < 0
            or type(entry["occurrences"]) is not int
            or entry["occurrences"] < 1
            or len(entry["source_rows"]) != entry["occurrences"]
            or not set(entry["source_rows"]).issubset(admitted)
        ):
            raise ValueError(f"malformed/non-training two-boundary catalog entry {index}")
        # Each content entry has unit identity even if both bindings are possible.
        # Orientation is sampled separately, so it cannot double occurrence mass.
        for key in sorted({contexts, contexts[::-1]}):
            grouped[key].append(entry)
        count += 1
    return LinkerCatalog(catalog_sha256, {key: tuple(v) for key, v in grouped.items()}, count)


def load_linker_catalog(path: Path, *, expected_sha256: str) -> LinkerCatalog:
    if physical_sha256(path) != expected_sha256:
        raise ValueError(f"linker catalog physical hash mismatch: {path}")
    return prepare_linker_catalog(json.loads(path.read_text()), catalog_sha256=expected_sha256)


def compatible_linker_cells(
    prompt: FragmentPrompt, catalog: LinkerCatalog, *, limits: ProposalLimits | None = None
):
    """Enumerate context/atom-capacity support, not compiler/model support.

    Joining distinct cores through exactly one contact each adds no cross-core
    ring, so final RDKit ring count is the sum of the three connected parts.
    Exact execution and the final count independently verify every selected row.
    Primitive/block budget failures remain visible after full compilation.
    """
    limits = limits or ProposalLimits()
    specs = _specs(prompt)
    cores = tuple(_capped_core(text) for text in prompt.fragments)
    contexts = tuple(
        atom_context(core.GetAtomWithIdx(spec.attachment_requirements[0][0]))
        for core, spec in zip(cores, specs, strict=True)
    )
    atoms = sum(core.GetNumHeavyAtoms() for core in cores)
    rings = sum(core.GetRingInfo().NumRings() for core in cores)
    grouped = defaultdict(list)
    entries = catalog.by_context.get(contexts, ())
    for entry in entries:
        if atoms + entry["heavy_atoms"] <= limits.max_active_atoms:
            grouped[(atoms + entry["heavy_atoms"], rings + entry["ring_count"])].append(entry)
    receipt = {
        "catalog_sha256": catalog.catalog_sha256,
        "boundary_contexts": contexts,
        "core_heavy_atoms": atoms,
        "core_ring_count": rings,
        "context_compatible_entries": len(entries),
        "atom_capacity_compatible_entries": sum(map(len, grouped.values())),
        "atom_budget_excluded_entries": len(entries) - sum(map(len, grouped.values())),
        "reachable_structural_cells": len(grouped),
        "support_semantics": "context/atom-capacity only; exact compiler and model checked later",
    }
    return {cell: tuple(grouped[cell]) for cell in sorted(grouped)}, receipt


def propose_linker_completion(
    prompt: FragmentPrompt,
    catalog: LinkerCatalog,
    prior: JointCompletionPrior,
    rng: np.random.Generator,
    *,
    limits: ProposalLimits | None = None,
    cell_allocation: CellAllocation = "frozen",
) -> CompleteProgram:
    """Draw a joint structural cell, observed connector and relative orientation."""
    if cell_allocation not in ("frozen", "sqrt_train_mass"):
        raise ValueError(f"unknown linker cell allocation: {cell_allocation}")
    grouped, receipt = compatible_linker_cells(prompt, catalog, limits=limits)
    if not grouped:
        raise LinkerProposalAbstention("no context/atom-capacity compatible connector", receipt)
    cells = tuple(grouped)
    probabilities = linker_cell_probabilities(prior, cells, allocation=cell_allocation)
    cell_index = int(rng.choice(len(cells), p=probabilities))
    cell = cells[cell_index]
    entries = grouped[cell]
    weights = np.sqrt([entry["occurrences"] for entry in entries])
    weights /= weights.sum()
    entry_index = int(rng.choice(len(entries), p=weights))
    entry = entries[entry_index]
    contexts = tuple(receipt["boundary_contexts"])
    orientations = tuple(
        reverse
        for reverse in (False, True)
        if tuple(entry["contexts"])[:: -1 if reverse else 1] == contexts
    )
    reverse = orientations[int(rng.integers(len(orientations)))]
    connector = Chem.MolFromSmiles(entry["rooted_smiles"])
    if connector is None:
        raise ValueError("selected catalog connector is unparseable")
    _connector(entry["rooted_smiles"])
    if (
        connector.GetNumHeavyAtoms() != entry["heavy_atoms"]
        or connector.GetRingInfo().NumRings() != entry["ring_count"]
    ):
        raise ValueError("selected catalog connector size/ring metadata changed")
    if reverse:
        for atom in connector.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(3 - atom.GetIsotope())
    rooted = Chem.MolToSmiles(connector)
    receipt.update(
        planned_final_cell=list(cell),
        cell_probability=float(probabilities[cell_index]),
        connector_probability_given_cell=float(weights[entry_index]),
        orientation_probability=1 / len(orientations),
        reversed_boundary_roles=reverse,
        observed_rooted_smiles=entry["rooted_smiles"],
        bound_rooted_smiles=rooted,
        source_rows=entry["source_rows"],
        training_occurrences=entry["occurrences"],
        source_stereo_annotations=entry.get("source_stereo_annotations", 0),
        cell_allocation=cell_allocation,
        sampling_law=(
            "joint_cell_then_sqrt_occurrence_then_uniform_compatible_orientation"
            if cell_allocation == "frozen"
            else "sqrt_joint_cell_then_sqrt_occurrence_then_uniform_compatible_orientation"
        ),
    )
    try:
        candidate = assemble_linker_program(prompt, rooted, limits=limits)
    except ValueError as error:
        raise LinkerProposalAbstention(str(error), receipt) from error
    molecule = Chem.MolFromSmiles(candidate.smiles)
    final_cell = (molecule.GetNumHeavyAtoms(), molecule.GetRingInfo().NumRings())
    if final_cell != cell:
        # RDKit's symmetrized perceived-ring count is not additive for every
        # bridged connector, even though graph cycle rank is. The joint prior
        # sampled the declared cell, so a different exact cell is an offered
        # proposal refusal, never an unrecorded crash or a silently relabelled
        # accepted candidate.
        raise LinkerProposalAbstention(
            "exact connector endpoint differs from its sampled perceived-ring cell",
            {**receipt, "actual_final_cell": list(final_cell)},
        )
    return CompleteProgram(
        candidate.endpoint,
        candidate.trace,
        {**candidate.provenance, "training_connector": receipt, "final_cell": list(final_cell)},
    )


def sample_linker_panel(
    prompt: FragmentPrompt,
    catalog: LinkerCatalog,
    prior: JointCompletionPrior,
    model,
    rng: np.random.Generator,
    *,
    limits: ProposalLimits | None = None,
    cell_allocation: CellAllocation = "frozen",
    prior_emitted: frozenset[str] | None = None,
    archive_selection: Literal["novelty4", "strict_unseen"] = "novelty4",
) -> LinkerPanelResult:
    """Exactly eight draws, exact endpoint deduplication, shared learned softmax.

    Every refusal consumes its offered draw. A panel with no finite supported
    candidate yields no output; it never substitutes a deterministic connector
    or an unlearned fallback. Callers must hash-bind the catalog, prior artifact,
    checkpoint and code before any diagnostic or evaluation launch.
    """
    if cell_allocation not in ("frozen", "sqrt_train_mass"):
        raise ValueError(f"unknown linker cell allocation: {cell_allocation}")
    if archive_selection not in ("novelty4", "strict_unseen"):
        raise ValueError(f"unknown linker archive selection: {archive_selection}")
    if prior_emitted is None and archive_selection == "strict_unseen":
        raise ValueError("strict-unseen selection requires a prior-emission archive")
    before = copy.deepcopy(rng.bit_generator.state)
    offered, candidates, scores, draw_indices, seen = [], [], [], [], set()
    for draw in range(8):
        record = {"draw": draw}
        try:
            candidate = propose_linker_completion(
                prompt, catalog, prior, rng, limits=limits, cell_allocation=cell_allocation
            )
        except ValueError as error:
            record.update(status="compiler_or_constraint_abstention", reason=str(error))
            if isinstance(error, LinkerProposalAbstention):
                record["proposal_receipt"] = error.receipt
        else:
            record.update(
                endpoint=candidate.smiles,
                provenance=candidate.provenance,
                trace=candidate.trace,
            )
            if candidate.smiles in seen:
                record["status"] = "duplicate_endpoint"
            else:
                seen.add(candidate.smiles)
                try:
                    [score] = learned_program_scores(model, (candidate,))
                except (ValueError, KeyError) as error:
                    record.update(status="model_support_abstention", reason=str(error))
                else:
                    if not np.isfinite(score):
                        record.update(
                            status="model_support_abstention", reason="nonfinite_native_probability"
                        )
                    else:
                        record.update(status="model_supported", mean_log_mark=score)
                        candidates.append(candidate)
                        scores.append(score)
                        draw_indices.append(draw)
        offered.append(record)
    selected, selection = None, None
    if candidates:
        if prior_emitted is None:
            selected, selection = select_learned_program(tuple(candidates), tuple(scores), rng)
        elif archive_selection == "strict_unseen":
            selected, selection = select_learned_program_strict_unseen(
                tuple(candidates), tuple(scores), rng, prior_emitted
            )
        else:
            selected, selection = select_learned_program_novelty4(
                tuple(candidates), tuple(scores), rng, prior_emitted
            )
        selection["selected_draw"] = draw_indices[selection["selected_index"]]
    return LinkerPanelResult(
        selected,
        {
            "schema": "fragment_training_linker_panel_v1",
            "rng_state_before": before,
            "rng_state_after": copy.deepcopy(rng.bit_generator.state),
            "offered": offered,
            "selected_smiles": selected.smiles if selected else None,
            "selection": selection,
            "output_count": int(selected is not None),
            "offered_count": len(offered),
            "exact_compiled_count": sum("trace" in row for row in offered),
            "unique_compiled_endpoints": len(seen),
            "model_supported_count": len(candidates),
            "qed_sa_guidance": False,
            "reference_used_for_proposal": False,
            "beam_search": False,
        },
    )
