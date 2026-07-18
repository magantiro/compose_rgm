"""Render representative COMPOSE preview trajectories from a rollout cache.

The released cache stores exact source RNG seeds, terminal states, event times,
and event-family labels, but not every intermediate state.  The atom-count curve
tracks primitive atom insertions and deletions.  In this preview it is exact:
for every trajectory, source + insert - delete equals the terminal atom count,
so the sampled ring-system macros had zero net atom-count change.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
import torch

from compose_v4.chem.molecular_graph import (
    NULL_IDX,
    molecular_graph_to_smiles,
)


SELECTED = (
    (0, "Typical preview trajectory"),
    (32, "Highest Graft count"),
    (75, "Most insert/delete activity"),
    (50, "Largest net shrink"),
    (23, "Small-ring failure"),
    (2, "Cleaner fused-ring outcome"),
)

COLORS = {
    "bond_reroute": "#4C78A8",
    "atom_insert": "#59A14F",
    "atom_delete": "#E15759",
    "atom_restate": "#B279A2",
    "bond_reorder": "#F28E2B",
    "ring_system_grow": "#76B7B2",
}
DISPLAY_NAMES = {
    "bond_reroute": "Graft",
    "atom_insert": "Insert",
    "atom_delete": "Delete",
    "atom_restate": "Retype",
    "bond_reorder": "Bond order",
    "ring_system_grow": "Ring grow",
}


def trajectory_seeds(seed: int, samples: int) -> tuple[int, ...]:
    sequence = np.random.SeedSequence(seed)
    return tuple(
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in sequence.spawn(samples)
    )


def active_atoms(graph) -> int:
    return int(np.sum(graph.atom_types != NULL_IDX))


def topology_label(molecule: Chem.Mol) -> str:
    rings = tuple(frozenset(ring) for ring in molecule.GetRingInfo().AtomRings())
    sizes = Counter(len(ring) for ring in rings)
    fused = any(
        len(left & right) >= 2
        for index, left in enumerate(rings)
        for right in rings[index + 1 :]
    )
    features = []
    if fused:
        features.append("fused")
    if rdMolDescriptors.CalcNumBridgeheadAtoms(molecule):
        features.append("bridged")
    if rdMolDescriptors.CalcNumSpiroAtoms(molecule):
        features.append("spiro")
    if sizes:
        features.append("rings " + ", ".join(f"{size}×{count}" for size, count in sorted(sizes.items())))
    else:
        features.append("acyclic")
    return "; ".join(features)


def molecule_image(molecule: Chem.Mol, size: tuple[int, int] = (520, 310)) -> np.ndarray:
    image = Draw.MolToImage(molecule, size=size, kekulize=True)
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.rollout_cache, map_location="cpu", weights_only=False)
    rollouts = tuple(payload["rollouts"])
    prior = payload["tree_source_prior"]
    seeds = trajectory_seeds(int(payload["seed"]), len(rollouts))
    n_slots = int(payload["n_slots"])

    figure = plt.figure(figsize=(19, 22), facecolor="white")
    grid = figure.add_gridspec(
        len(SELECTED),
        3,
        width_ratios=(1.05, 2.3, 1.05),
        hspace=0.78,
        wspace=0.08,
    )

    for row, (index, selection_label) in enumerate(SELECTED):
        rollout = rollouts[index]
        source = prior.sample(
            np.random.default_rng(seeds[index]),
            n_slots=n_slots,
        )
        source_smiles = molecular_graph_to_smiles(source)
        final_smiles = molecular_graph_to_smiles(rollout.final_state)
        if source_smiles is None or final_smiles is None:
            raise RuntimeError(f"trajectory {index} is not renderable")
        source_molecule = Chem.MolFromSmiles(source_smiles)
        final_molecule = Chem.MolFromSmiles(final_smiles)
        if source_molecule is None or final_molecule is None:
            raise RuntimeError(f"trajectory {index} contains invalid SMILES")

        source_axis = figure.add_subplot(grid[row, 0])
        source_axis.imshow(molecule_image(source_molecule))
        source_axis.axis("off")
        source_axis.set_title(
            f"SOURCE  |  random carbon tree\n{source_molecule.GetNumHeavyAtoms()} heavy atoms",
            fontsize=10,
            fontweight="semibold",
        )

        final_axis = figure.add_subplot(grid[row, 2])
        final_axis.imshow(molecule_image(final_molecule))
        final_axis.axis("off")
        final_axis.set_title(
            f"GENERATED  |  {topology_label(final_molecule)}\n"
            f"{final_molecule.GetNumHeavyAtoms()} heavy atoms",
            fontsize=10,
            fontweight="semibold",
        )

        trajectory_axis = figure.add_subplot(grid[row, 1])
        rules = tuple(rollout.event_rules)
        counts = Counter(rules)
        positions = np.arange(1, len(rules) + 1)
        for event_position, rule in zip(positions, rules):
            trajectory_axis.axvline(
                event_position,
                ymin=0.05,
                ymax=0.42,
                color=COLORS.get(rule, "#777777"),
                linewidth=2.1,
                alpha=0.95,
            )

        lower_bound = [active_atoms(source)]
        for rule in rules:
            change = int(rule == "atom_insert") - int(rule == "atom_delete")
            lower_bound.append(lower_bound[-1] + change)
        lower_bound = np.asarray(lower_bound)
        trajectory_axis.plot(
            np.arange(len(lower_bound)),
            lower_bound,
            color="black",
            linewidth=1.8,
            marker="o",
            markersize=1.8,
            zorder=3,
        )
        trajectory_axis.axhline(1, color="#888888", linestyle="--", linewidth=0.9)
        trajectory_axis.set_xlim(0, max(len(rules), 1) + 1)
        low = min(0, int(lower_bound.min()) - 2)
        high = max(int(lower_bound.max()) + 3, active_atoms(source) + 3)
        trajectory_axis.set_ylim(low, high)
        if row == len(SELECTED) - 1:
            trajectory_axis.set_xlabel("Fired-event index", fontsize=9)
        trajectory_axis.set_ylabel("Heavy atoms", fontsize=9)
        trajectory_axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        trajectory_axis.spines[["top", "right"]].set_visible(False)
        trajectory_axis.tick_params(labelsize=8)
        count_text = "  ·  ".join(
            f"{DISPLAY_NAMES[name]} {counts[name]}"
            for name in (
                "bond_reroute",
                "atom_insert",
                "atom_delete",
                "atom_restate",
                "bond_reorder",
                "ring_system_grow",
            )
            if counts[name]
        )
        trajectory_axis.set_title(
            f"#{index}: {selection_label}  |  {len(rules)} events\n{count_text}\n"
            f"minimum = {int(lower_bound.min())} atoms; "
            f"event budget exhausted = {rollout.exhausted_event_budget}",
            fontsize=10,
            fontweight="semibold",
            pad=8,
        )

    handles = [
        Line2D([0], [0], color=color, lw=5, label=DISPLAY_NAMES[rule])
        for rule, color in COLORS.items()
    ]
    handles.append(Line2D([0], [0], color="black", lw=2, marker="o", label="Atom count"))
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.975),
    )
    figure.suptitle(
        "COMPOSE early-preview trajectories: carbon-tree prior → ancestral CTMC sample",
        fontsize=17,
        fontweight="bold",
        y=0.997,
    )
    figure.text(
        0.5,
        0.978,
        "Exact event-family sequence from the selected step-4,750 checkpoint preview. "
        "Here the black atom-count curve is exact: sampled ring macros had zero net size change.",
        ha="center",
        va="top",
        fontsize=10,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
