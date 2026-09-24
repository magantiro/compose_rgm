"""Analytic old-law content allocation census, with no molecule generation or quality scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

from rdkit import Chem, rdBase
from run_fragment_attachment_library_pilot import _atomic_json

from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    _fragment_spec,
    load_genmol_prompts,
)
from compose_v4.benchmark.training_attachment_fragments import atom_context


def allocation_law(contexts, cells, room):
    """Integrate sequential sampling over uniform random boundary permutations."""
    keys = tuple(sorted(set(contexts)))
    minima = tuple(min(a for a, _ in cells[key]) for key in keys)
    counts = tuple(contexts.count(key) for key in keys)
    if sum(n * minimum for n, minimum in zip(counts, minima, strict=True)) > room:
        raise ValueError("minimum observed regions do not fit")

    @cache
    def recurse(remaining, capacity):
        if not any(remaining):
            return {(0, 0): 1.0}
        result = defaultdict(float)
        for index, count in enumerate(remaining):
            if not count:
                continue
            rest = tuple(n - (i == index) for i, n in enumerate(remaining))
            reserve = sum(n * minimum for n, minimum in zip(rest, minima, strict=True))
            options = {
                cell: mass
                for cell, mass in cells[keys[index]].items()
                if cell[0] <= capacity - reserve
            }
            denominator = sum(options.values())
            for (atoms, rings), mass in options.items():
                weight = count / sum(remaining) * mass / denominator
                for (later_atoms, later_rings), probability in recurse(
                    rest, capacity - atoms
                ).items():
                    result[(atoms + later_atoms, rings + later_rings)] += weight * probability
        return dict(result)

    result = recurse(counts, room)
    if abs(sum(result.values()) - 1) > 1e-12:
        raise ValueError("allocation mass does not sum to one")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    prompts_path = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")
    paths = [
        args.catalog,
        prompts_path,
        Path(__file__),
        Path("src/compose_v4/benchmark/fragment_program_adapter.py"),
        Path("src/compose_v4/benchmark/training_attachment_fragments.py"),
        Path("src/compose_v4/benchmark/fragment_constrained.py"),
    ]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    catalog = json.loads(args.catalog.read_text())
    cells = defaultdict(lambda: defaultdict(float))
    entry_counts, occurrences = Counter(), Counter()
    for entry in catalog["entries"]:
        if len(entry["contexts"]) != 1:
            continue
        [context] = entry["contexts"]
        cells[context][(entry["heavy_atoms"], entry["ring_count"])] += math.sqrt(
            entry["occurrences"]
        )
        entry_counts[context] += 1
        occurrences[context] += entry["occurrences"]
    context_rows = []
    for key, mass in sorted(cells.items()):
        total = sum(mass.values())
        context_rows.append(
            {
                "context": key,
                "unique_regions": entry_counts[key],
                "occurrences": occurrences[key],
                "minimum_atoms": min(a for a, _ in mass),
                "sqrt_occurrence_mass": total,
                "mean_atoms": sum(a * p for (a, _), p in mass.items()) / total,
                "mean_rings": sum(r * p for (_, r), p in mass.items()) / total,
                "cells": [
                    {"atoms": a, "rings": r, "mass": p, "probability": p / total}
                    for (a, r), p in sorted(mass.items())
                ],
            }
        )
    prompt_rows = []
    for prompt in load_genmol_prompts(prompts_path):
        if prompt.task not in (FragmentTask.MOTIF_EXTENSION, FragmentTask.SCAFFOLD_DECORATION):
            continue
        spec = _fragment_spec(prompt.fragments[0])
        Chem.GetSymmSSSR(spec.core)
        contexts = [
            atom_context(spec.core.GetAtomWithIdx(site))
            for site, count in spec.attachment_requirements
            for _ in range(count)
        ]
        atoms = spec.core.GetNumHeavyAtoms()
        rings = spec.core.GetRingInfo().NumRings()
        law = allocation_law(contexts, cells, 40 - atoms)
        prompt_rows.append(
            {
                "task": prompt.task.value,
                "drug": prompt.drug_name,
                "core_atoms": atoms,
                "core_rings": rings,
                "contexts": contexts,
                "room": 40 - atoms,
                "minimum_reserved_atoms": sum(min(a for a, _ in cells[c]) for c in contexts),
                "mean_added_atoms": sum(a * p for (a, _), p in law.items()),
                "mean_added_rings": sum(r * p for (_, r), p in law.items()),
                "mean_total_atoms": atoms + sum(a * p for (a, _), p in law.items()),
                "mean_total_rings": rings + sum(r * p for (_, r), p in law.items()),
                "probability_at_40_atoms": sum(p for (a, _), p in law.items() if a + atoms == 40),
                "probability_at_least_38_atoms": sum(
                    p for (a, _), p in law.items() if a + atoms >= 38
                ),
                "cells": [
                    {"added_atoms": a, "added_rings": r, "probability": p}
                    for (a, r), p in sorted(law.items())
                ],
            }
        )
    if hashes != {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:
        raise ValueError("input changed during census")
    output = {
        "schema": "fragment_old_region_allocation_census_v1",
        "role": "exact numerical distribution of saved-catalog allocation before compilation; no generated molecules, checkpoint loads, quality scores or runtime change",
        "scope": "one-boundary content; exact uniform boundary permutations and per-entry sqrt-occurrence masses with minimum-size reservations; chemistry/compiler/model rejection is NOT included",
        "inputs_sha256": hashes,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "configuration": {
            "maximum_total_heavy_atoms": 40,
            "seed": "none; exact deterministic dynamic programming",
            "device": "cpu",
            "workers": 1,
            "precision": "float64",
            "split": catalog["split"],
            "training_molecules": catalog["training_molecules"],
        },
        "contexts": context_rows,
        "prompts": prompt_rows,
    }
    args.output_dir.mkdir(parents=True)
    _atomic_json(args.output_dir / "allocation.json", output)
    print(json.dumps({"prompts": len(prompt_rows), "contexts": len(context_rows)}))


if __name__ == "__main__":
    main()
