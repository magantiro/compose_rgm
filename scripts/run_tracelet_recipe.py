"""Run a frozen tracelet-GM recipe against explicit data files."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from compose_v4.experiments.recipe import (
    build_tracelet_recipe_argv,
    load_tracelet_recipe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", type=Path)
    parser.add_argument("smiles_file", type=Path)
    parser.add_argument("--quality-reference-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    recipe = load_tracelet_recipe(args.recipe)
    trainer = Path(__file__).with_name("train_tracelet_cnof_gate.py")
    command = (
        sys.executable,
        str(trainer),
        *build_tracelet_recipe_argv(
            recipe,
            smiles_file=args.smiles_file,
            quality_reference_file=args.quality_reference_file,
        ),
    )
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
