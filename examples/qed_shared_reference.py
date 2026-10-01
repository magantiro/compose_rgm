"""Sample one QED-editing transition from the shared molecular reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compose_v4.chem.molecular_graph import (
    molecular_graph_to_smiles,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.qed_shared_reference import (
    QEDSharedReference,
    SharedReferenceConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("local_assets/fragments"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/fragments/assets.json"))
    parser.add_argument("--source", default="CCO")
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    checkpoint = manifest["assets"]["checkpoint"]
    catalog = manifest["assets"]["catalog"]
    config = SharedReferenceConfig(
        checkpoint=args.assets / checkpoint["path"],
        checkpoint_sha256=checkpoint["sha256"],
        catalog_fingerprint=manifest["catalog_fingerprint"],
        time=args.time,
        catalog_path=args.assets / catalog["path"],
        catalog_sha256=catalog["sha256"],
    )
    process = QEDSharedReference.load(config)
    source = pad_molecular_graph(smiles_to_molecular_graph(args.source), config.persistent_slots)
    successors = process.successors(source)
    selected = process.sample(source, np.random.default_rng(args.seed))
    print(
        json.dumps(
            {
                "schema": "compose.qed.shared_reference_smoke.v1",
                "reference": process.identity(),
                "source": molecular_graph_to_smiles(source),
                "seed": args.seed,
                "successor_count": len(successors),
                "selected": None if selected is None else molecular_graph_to_smiles(selected),
                "note": "One reference transition. No value head, QED benchmark, or oracle call.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
