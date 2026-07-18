"""Correct and re-render preserved COMPOSE trajectory JSON artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_dir", type=Path)
    args = parser.parse_args()

    summaries: list[dict[str, object]] = []
    paths = sorted(args.trajectory_dir.glob("full_trajectory_step*_index*.json"))
    if not paths:
        raise FileNotFoundError("no preserved full-trajectory JSON files found")

    for json_path in paths:
        report = json.loads(json_path.read_text())
        rows = report["states"]
        keys = [str(row["canonical_state_key"]) for row in rows]
        self_transitions = {
            index
            for index in range(1, len(keys))
            if keys[index] == keys[index - 1]
        }
        strict_two_cycles = {
            index
            for index in range(2, len(keys))
            if keys[index] == keys[index - 2] and keys[index] != keys[index - 1]
        }

        molecules = []
        legends = []
        for index, row in enumerate(rows):
            molecule = Chem.MolFromSmiles(str(row["smiles"]))
            if molecule is None:
                raise ValueError(f"invalid preserved SMILES at state {index}: {row['smiles']}")
            molecules.append(molecule)
            row["canonical_self_transition"] = index in self_transitions
            # Keep this legacy key, but give it the strict A -> B -> A meaning.
            row["immediate_reversal"] = index in strict_two_cycles
            if index == 0:
                legends.append(
                    f"STATE 0 | SOURCE\n"
                    f"atoms={molecule.GetNumHeavyAtoms()}, rings={molecule.GetRingInfo().NumRings()}"
                )
                continue
            note = ""
            if index in self_transitions:
                note = " | molecular self-transition"
            elif index in strict_two_cycles:
                note = " | strict two-cycle"
            legends.append(
                f"STATE {index} | {row['rule']}{note}\n"
                f"t={float(row['event_time']):.3f}, atoms={molecule.GetNumHeavyAtoms()}, "
                f"rings={molecule.GetRingInfo().NumRings()}"
            )

        png_path = Path(report["png"])
        if not png_path.is_absolute():
            png_path = json_path.parent / png_path
        Draw.MolsToGridImage(
            molecules,
            molsPerRow=5,
            subImgSize=(360, 270),
            legends=legends,
            useSVG=False,
        ).save(png_path)

        page_paths = []
        page_size = 30
        for page_index, start in enumerate(range(0, len(molecules), page_size), start=1):
            page_path = png_path.with_name(
                f"{png_path.stem}_page{page_index:02d}.png"
            )
            Draw.MolsToGridImage(
                molecules[start : start + page_size],
                molsPerRow=5,
                subImgSize=(420, 315),
                legends=legends[start : start + page_size],
                useSVG=False,
            ).save(page_path)
            page_paths.append(page_path.name)

        report["unique_states"] = len(set(keys))
        report["state_revisits"] = len(keys) - len(set(keys))
        report["canonical_self_transitions"] = len(self_transitions)
        report["immediate_reversals"] = len(strict_two_cycles)
        report["png_pages"] = page_paths
        report["png"] = png_path.name
        json_path.write_text(json.dumps(report, indent=2) + "\n")
        summaries.append({key: value for key, value in report.items() if key != "states"})
        print(
            json.dumps(
                {
                    "json": str(json_path),
                    "canonical_self_transitions": len(self_transitions),
                    "strict_two_cycles": len(strict_two_cycles),
                },
                sort_keys=True,
            )
        )

    (args.trajectory_dir / "full_trajectory_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
