"""Audit rewrite trajectories and molecular distribution mismatch.

The audit is intentionally independent of the training script.  It consumes a
saved rollout cache and a reference SMILES file, then writes machine-readable
summaries, per-trajectory data, descriptor comparisons, and a stratified image
grid.  This makes changes to the rewrite teacher or sampler directly
comparable across experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdMolDescriptors
from scipy.stats import wasserstein_distance
import torch

from compose_v4.chem.molecular_graph import molecular_graph_to_smiles


CONSTRUCTIVE_RULES = frozenset(
    {"atom_insert", "cycle_insert", "cycle_attach", "ring_ear_insert"}
)


@dataclass(frozen=True)
class MoleculeRecord:
    smiles: str
    molecule: Chem.Mol
    descriptors: dict[str, float]
    ring_sizes: tuple[int, ...]


def _rings(molecule: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(atom) for atom in ring) for ring in Chem.GetSymmSSSR(molecule))


def _aromatic_ring_count(
    molecule: Chem.Mol,
    rings: Sequence[Sequence[int]],
) -> int:
    return sum(
        all(molecule.GetAtomWithIdx(atom).GetIsAromatic() for atom in ring)
        for ring in rings
    )


def _fused_ring_system_count(rings: Sequence[Sequence[int]]) -> int:
    ring_sets = tuple(frozenset(ring) for ring in rings)
    if len(ring_sets) < 2:
        return 0
    adjacency = {
        index: {
            other
            for other in range(len(ring_sets))
            if other != index and len(ring_sets[index] & ring_sets[other]) >= 2
        }
        for index in range(len(ring_sets))
    }
    visited: set[int] = set()
    fused_systems = 0
    for root, neighbors in adjacency.items():
        if root in visited or not neighbors:
            continue
        fused_systems += 1
        stack = [root]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adjacency[current] - visited)
    return fused_systems


def molecular_descriptors(molecule: Chem.Mol) -> tuple[dict[str, float], tuple[int, ...]]:
    """Return descriptors that expose topology, composition, and drug likeness."""

    rings = _rings(molecule)
    ring_sizes = tuple(len(ring) for ring in rings)
    aromatic_rings = _aromatic_ring_count(molecule, rings)
    atoms = tuple(molecule.GetAtoms())
    heavy_atoms = molecule.GetNumHeavyAtoms()
    heteroatoms = sum(atom.GetAtomicNum() not in {1, 6} for atom in atoms)
    element_counts = Counter(atom.GetSymbol() for atom in atoms)
    edge_count = molecule.GetNumBonds()
    connected_components = len(Chem.GetMolFrags(molecule))
    cycle_rank = edge_count - heavy_atoms + connected_components
    bridgeheads = int(rdMolDescriptors.CalcNumBridgeheadAtoms(molecule))
    spiro = int(rdMolDescriptors.CalcNumSpiroAtoms(molecule))
    fused_systems = _fused_ring_system_count(rings)
    descriptors = {
        "heavy_atoms": float(heavy_atoms),
        "bonds": float(edge_count),
        "cycle_rank": float(cycle_rank),
        "rings": float(len(rings)),
        "aromatic_rings": float(aromatic_rings),
        "nonaromatic_rings": float(len(rings) - aromatic_rings),
        "aromatic_ring_fraction": float(aromatic_rings / max(len(rings), 1)),
        "has_ring": float(bool(rings)),
        "has_aromatic_ring": float(aromatic_rings > 0),
        "has_3_ring": float(3 in ring_sizes),
        "has_4_ring": float(4 in ring_sizes),
        "has_small_ring": float(any(size in {3, 4} for size in ring_sizes)),
        "has_macrocycle": float(any(size >= 9 for size in ring_sizes)),
        "bridgehead_atoms": float(bridgeheads),
        "spiro_atoms": float(spiro),
        "has_bridgehead": float(bridgeheads > 0),
        "has_spiro": float(spiro > 0),
        "fused_ring_systems": float(fused_systems),
        "has_fused_ring_system": float(fused_systems > 0),
        "heteroatoms": float(heteroatoms),
        "heteroatom_fraction": float(heteroatoms / max(heavy_atoms, 1)),
        "carbon": float(element_counts["C"]),
        "nitrogen": float(element_counts["N"]),
        "oxygen": float(element_counts["O"]),
        "fluorine": float(element_counts["F"]),
        "chlorine": float(element_counts["Cl"]),
        "formal_charge": float(sum(atom.GetFormalCharge() for atom in atoms)),
        "absolute_formal_charge": float(
            sum(abs(atom.GetFormalCharge()) for atom in atoms)
        ),
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "logp": float(Crippen.MolLogP(molecule)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "hbd": float(Lipinski.NumHDonors(molecule)),
        "hba": float(Lipinski.NumHAcceptors(molecule)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(molecule)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
    }
    return descriptors, ring_sizes


def molecule_record(smiles: str) -> MoleculeRecord | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    descriptors, ring_sizes = molecular_descriptors(molecule)
    return MoleculeRecord(canonical, molecule, descriptors, ring_sizes)


def _read_reference(path: Path) -> tuple[MoleculeRecord, ...]:
    records = []
    with path.open() as handle:
        for line in handle:
            token = line.strip().split()[0] if line.strip() else ""
            if not token:
                continue
            record = molecule_record(token)
            if record is not None:
                records.append(record)
    if not records:
        raise ValueError(f"reference file has no valid molecules: {path}")
    return tuple(records)


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _integer_tv(left: np.ndarray, right: np.ndarray) -> float:
    lo = int(min(left.min(), right.min()))
    hi = int(max(left.max(), right.max()))
    support = np.arange(lo, hi + 1)
    left_hist = np.array([(left == value).mean() for value in support])
    right_hist = np.array([(right == value).mean() for value in support])
    return float(0.5 * np.abs(left_hist - right_hist).sum())


def descriptor_comparison(
    generated: Sequence[MoleculeRecord],
    reference: Sequence[MoleculeRecord],
) -> list[dict[str, float | str]]:
    names = tuple(generated[0].descriptors)
    comparisons = []
    for name in names:
        generated_values = np.array([record.descriptors[name] for record in generated])
        reference_values = np.array([record.descriptors[name] for record in reference])
        ref_std = float(reference_values.std())
        comparisons.append(
            {
                "descriptor": name,
                "generated_mean": float(generated_values.mean()),
                "reference_mean": float(reference_values.mean()),
                "mean_delta": float(generated_values.mean() - reference_values.mean()),
                "wasserstein": float(
                    wasserstein_distance(generated_values, reference_values)
                ),
                "standardized_wasserstein": float(
                    wasserstein_distance(generated_values, reference_values)
                    / max(ref_std, 1e-8)
                ),
                "integer_tv": _integer_tv(generated_values, reference_values),
            }
        )
    return comparisons


def _safe_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.std() == 0.0 or right_array.std() == 0.0:
        return 0.0
    return float(np.corrcoef(left_array, right_array)[0, 1])


def _trajectory_rows(rollouts: Sequence[object], records: Sequence[MoleculeRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (rollout, record) in enumerate(zip(rollouts, records)):
        rules = tuple(str(rule) for rule in rollout.event_rules)
        times = tuple(float(time) for time in rollout.event_times)
        rule_counts = Counter(rules)
        first_constructive = next(
            (position for position, rule in enumerate(rules) if rule in CONSTRUCTIVE_RULES),
            len(rules),
        )
        initial_delete_run = next(
            (position for position, rule in enumerate(rules) if rule != "atom_delete"),
            len(rules),
        )
        row: dict[str, object] = {
            "index": index,
            "smiles": record.smiles,
            "events": len(rules),
            "initial_delete_run": initial_delete_run,
            "deletes_before_first_construction": sum(
                rule == "atom_delete" for rule in rules[:first_constructive]
            ),
            "first_constructive_event_fraction": float(
                first_constructive / max(len(rules), 1)
            ),
            "last_event_time": float(times[-1]) if times else 0.0,
            "exhausted_event_budget": bool(rollout.exhausted_event_budget),
        }
        for rule in sorted(set(rules)):
            row[f"events/{rule}"] = int(rule_counts[rule])
        row.update(record.descriptors)
        rows.append(row)
    return rows


def event_summary(rows: Sequence[dict[str, object]], rollouts: Sequence[object]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    positions: defaultdict[str, list[float]] = defaultdict(list)
    times: defaultdict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        event_count = len(rollout.event_rules)
        for position, (rule, time) in enumerate(
            zip(rollout.event_rules, rollout.event_times)
        ):
            rule = str(rule)
            counts[rule] += 1
            positions[rule].append((position + 0.5) / max(event_count, 1))
            times[rule].append(float(time))
    total = sum(counts.values())
    outcomes = (
        "has_bridgehead",
        "has_spiro",
        "has_small_ring",
        "aromatic_ring_fraction",
        "heavy_atoms",
    )
    correlations: dict[str, dict[str, float]] = {}
    for rule in sorted(counts):
        event_values = [float(row.get(f"events/{rule}", 0)) for row in rows]
        correlations[rule] = {
            outcome: _safe_correlation(
                event_values,
                [float(row[outcome]) for row in rows],
            )
            for outcome in outcomes
        }
    return {
        "total_events": total,
        "mean_events_per_trajectory": float(total / max(len(rows), 1)),
        "families": {
            rule: {
                "count": count,
                "fraction": float(count / max(total, 1)),
                "median_sequence_fraction": float(np.median(positions[rule])),
                "median_operational_time": float(np.median(times[rule])),
            }
            for rule, count in sorted(counts.items())
        },
        "initial_delete_run": _distribution_summary(
            np.array([float(row["initial_delete_run"]) for row in rows])
        ),
        "deletes_before_first_construction": _distribution_summary(
            np.array(
                [float(row["deletes_before_first_construction"]) for row in rows]
            )
        ),
        "event_outcome_correlations": correlations,
    }


def _ring_size_summary(records: Sequence[MoleculeRecord]) -> dict[str, object]:
    sizes = Counter(size for record in records for size in record.ring_sizes)
    total = sum(sizes.values())
    return {
        "total_rings": total,
        "counts": {str(size): count for size, count in sorted(sizes.items())},
        "fractions": {
            str(size): float(count / max(total, 1))
            for size, count in sorted(sizes.items())
        },
    }


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _select_stratified(rows: Sequence[dict[str, object]], per_cohort: int) -> list[tuple[str, int]]:
    cohorts = {
        "random": list(range(len(rows))),
        "bridgehead": [i for i, row in enumerate(rows) if row["has_bridgehead"]],
        "spiro": [i for i, row in enumerate(rows) if row["has_spiro"]],
        "small ring (3/4)": [i for i, row in enumerate(rows) if row["has_small_ring"]],
        "low aromatic / >=2 rings": [
            i
            for i, row in enumerate(rows)
            if float(row["rings"]) >= 2 and float(row["aromatic_ring_fraction"]) == 0
        ],
        "aromatic": [i for i, row in enumerate(rows) if row["has_aromatic_ring"]],
    }
    rng = np.random.default_rng(0)
    selected: list[tuple[str, int]] = []
    used_by_cohort: set[tuple[str, int]] = set()
    for name, candidates in cohorts.items():
        if not candidates:
            continue
        for index in rng.choice(candidates, size=min(per_cohort, len(candidates)), replace=False):
            key = (name, int(index))
            if key not in used_by_cohort:
                selected.append(key)
                used_by_cohort.add(key)
    return selected


def _render_grid(
    output: Path,
    records: Sequence[MoleculeRecord],
    rows: Sequence[dict[str, object]],
    *,
    per_cohort: int,
) -> None:
    selected = _select_stratified(rows, per_cohort)
    molecules = [records[index].molecule for _, index in selected]
    legends = []
    for cohort, index in selected:
        row = rows[index]
        legends.append(
            f"#{index} | {cohort}\n"
            f"HA={int(float(row['heavy_atoms']))}, rings={int(float(row['rings']))}, "
            f"arom={int(float(row['aromatic_rings']))}, events={int(row['events'])}"
        )
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=per_cohort,
        subImgSize=(360, 260),
        legends=legends,
        useSVG=False,
    )
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_cache", type=Path)
    parser.add_argument("reference_smiles", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--grid-per-cohort", type=int, default=6)
    args = parser.parse_args()
    if args.grid_per_cohort <= 0:
        raise ValueError("--grid-per-cohort must be positive")

    payload = torch.load(args.rollout_cache, map_location="cpu", weights_only=False)
    rollouts = tuple(payload["rollouts"])
    generated = []
    invalid_indices = []
    valid_rollouts = []
    for index, rollout in enumerate(rollouts):
        smiles = molecular_graph_to_smiles(rollout.final_state)
        record = molecule_record(smiles) if smiles is not None else None
        if record is None:
            invalid_indices.append(index)
            continue
        generated.append(record)
        valid_rollouts.append(rollout)
    if not generated:
        raise ValueError("rollout cache has no valid final molecules")
    reference = _read_reference(args.reference_smiles)
    rows = _trajectory_rows(valid_rollouts, generated)
    comparisons = descriptor_comparison(generated, reference)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "trajectories.csv", rows)
    _write_csv(args.output_dir / "descriptor_comparison.csv", comparisons)
    _render_grid(
        args.output_dir / "stratified_generated_molecules.png",
        generated,
        rows,
        per_cohort=args.grid_per_cohort,
    )

    generated_by_descriptor = {
        name: _distribution_summary(
            np.array([record.descriptors[name] for record in generated])
        )
        for name in generated[0].descriptors
    }
    reference_by_descriptor = {
        name: _distribution_summary(
            np.array([record.descriptors[name] for record in reference])
        )
        for name in reference[0].descriptors
    }
    worst = sorted(
        comparisons,
        key=lambda row: float(row["standardized_wasserstein"]),
        reverse=True,
    )[:15]
    summary = {
        "generated_molecules": len(generated),
        "reference_molecules": len(reference),
        "invalid_rollout_indices": invalid_indices,
        "events": event_summary(rows, valid_rollouts),
        "generated_descriptors": generated_by_descriptor,
        "reference_descriptors": reference_by_descriptor,
        "generated_ring_sizes": _ring_size_summary(generated),
        "reference_ring_sizes": _ring_size_summary(reference),
        "worst_standardized_descriptor_mismatches": worst,
    }
    with (args.output_dir / "audit.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
