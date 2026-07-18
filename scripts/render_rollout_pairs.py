"""Render exact source-prior and generated-molecule pairs from a rollout cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
import torch

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles


def _trajectory_seeds(seed: int, samples: int) -> tuple[int, ...]:
    sequence = np.random.SeedSequence(seed)
    return tuple(
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in sequence.spawn(samples)
    )


def _topology_label(molecule: Chem.Mol) -> str:
    rings = tuple(frozenset(ring) for ring in molecule.GetRingInfo().AtomRings())
    if rdMolDescriptors.CalcNumSpiroAtoms(molecule) > 0:
        return "spiro"
    if rdMolDescriptors.CalcNumBridgeheadAtoms(molecule) > 0:
        return "bridged"
    if any(
        len(left & right) >= 2
        for index, left in enumerate(rings)
        for right in rings[index + 1 :]
    ):
        return "fused"
    if any(
        all(molecule.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring)
        for ring in rings
    ):
        return "aromatic ring"
    if rings:
        return "saturated/nonaromatic ring"
    return "acyclic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=8)
    args = parser.parse_args()
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")

    payload = torch.load(args.rollout_cache, map_location="cpu", weights_only=False)
    rollouts = tuple(payload["rollouts"])
    prior = payload["tree_source_prior"]
    if prior is None:
        raise ValueError("rollout cache does not contain a tree source prior")
    count = min(args.pairs, len(rollouts))
    seeds = _trajectory_seeds(int(payload["seed"]), len(rollouts))
    n_slots = int(payload["n_slots"])

    molecules: list[Chem.Mol] = []
    legends: list[str] = []
    for index, (rollout, trajectory_seed) in enumerate(zip(rollouts, seeds)):
        if index >= count:
            break
        source = prior.sample(
            np.random.default_rng(trajectory_seed),
            n_slots=n_slots,
        )
        source_smiles = molecular_graph_to_smiles(source)
        generated_smiles = molecular_graph_to_smiles(rollout.final_state)
        if source_smiles is None or generated_smiles is None:
            raise ValueError(f"rollout {index} cannot be converted to SMILES")
        source_molecule = Chem.MolFromSmiles(source_smiles)
        generated_molecule = Chem.MolFromSmiles(generated_smiles)
        if source_molecule is None or generated_molecule is None:
            raise ValueError(f"rollout {index} has an invalid SMILES rendering")
        molecules.extend((source_molecule, generated_molecule))
        legends.extend(
            (
                f"#{index} SOURCE: carbon-tree prior\n"
                f"heavy atoms={source_molecule.GetNumHeavyAtoms()}",
                f"#{index} GENERATED: {_topology_label(generated_molecule)}\n"
                f"heavy atoms={generated_molecule.GetNumHeavyAtoms()}, "
                f"rings={generated_molecule.GetRingInfo().NumRings()}, "
                f"events={len(rollout.event_times)}",
            )
        )

    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=2,
        subImgSize=(520, 340),
        legends=legends,
        useSVG=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
