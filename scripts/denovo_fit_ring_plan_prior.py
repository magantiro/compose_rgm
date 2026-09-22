"""Fit the corpus ring-system plan prior, and the reference census beside it.

Both outputs come from the SAME pass over the SAME corpus file, so the law the
arms draw from and the law they are scored against cannot drift apart.  The
census is the decisive comparison target: "did arm C reproduce the TRAINING
ring distribution" is answered by applying ``ring_signature_census`` to the
corpus here and to each arm's molecules there.

Usage::

    python3 scripts/denovo_fit_ring_plan_prior.py \
        --smiles <guacamol_subset_500000_seed0.smiles> \
        --prior diagnostics/denovo_ring_marginal_v1/plan_prior_v1.json \
        --census diagnostics/denovo_ring_marginal_v1/corpus_ring_census_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger

from compose_v4.eval.denovo_ring_marginal import (
    DEFAULT_HEAVY_ATOM_BIN_EDGES,
    RingSystemPlanPrior,
    heavy_atom_bin,
    ring_signature_census,
    ring_system_signature,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_corpus(path: Path, limit: int | None = None) -> list[tuple[int, tuple]]:
    """``(heavy_atoms, ring signature)`` for every parseable corpus molecule."""

    RDLogger.DisableLog("rdApp.*")
    rows: list[tuple[int, tuple]] = []
    unparseable = 0
    with path.open() as handle:
        for line in handle:
            text = line.strip().split()[0] if line.strip() else ""
            if not text:
                continue
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                unparseable += 1
                continue
            rows.append((int(mol.GetNumHeavyAtoms()), ring_system_signature(mol)))
            if limit is not None and len(rows) >= limit:
                break
    if unparseable:
        print(f"skipped {unparseable} unparseable corpus lines", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--census", required=True)
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()

    source = Path(arguments.smiles)
    rows = scan_corpus(source, arguments.limit)
    prior = RingSystemPlanPrior.fit(rows, bin_edges=DEFAULT_HEAVY_ATOM_BIN_EDGES)
    census = ring_signature_census(signature for _heavy, signature in rows)
    census["source"] = source.name
    census["source_sha256"] = _sha256(source)
    census["heavy_atom_bin_edges"] = list(DEFAULT_HEAVY_ATOM_BIN_EDGES)
    census["by_heavy_atom_bin"] = {
        label: ring_signature_census(
            signature
            for heavy, signature in rows
            if heavy_atom_bin(heavy, edges=DEFAULT_HEAVY_ATOM_BIN_EDGES) == label
        )
        for label in sorted({
            heavy_atom_bin(heavy, edges=DEFAULT_HEAVY_ATOM_BIN_EDGES)
            for heavy, _signature in rows
        })
    }

    prior_path = Path(arguments.prior)
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior.write(prior_path)
    census_path = Path(arguments.census)
    census_path.parent.mkdir(parents=True, exist_ok=True)
    census_path.write_text(json.dumps(census, indent=1))
    print(
        json.dumps(
            {
                "molecules": len(rows),
                "prior_bins": sorted(prior.tables),
                "prior_bytes": prior_path.stat().st_size,
                "ring_size_fraction": census["ring_size_fraction"],
                "ring_systems_per_molecule": census["ring_systems_per_molecule"],
                "strained_ring_fraction": census["strained_ring_fraction"],
                "fraction_with_strained_ring": census["fraction_with_strained_ring"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
