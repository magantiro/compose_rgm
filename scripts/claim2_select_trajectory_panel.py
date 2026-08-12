"""Freeze a Claim-2 trajectory panel BEFORE any rollout runs.

Selection is outcome-independent and mechanical.  No property, no rollout, no
model, and nothing that could be tuned after seeing a result enters this file:
the panel is a deterministic function of (pool, seed, size band, support band).

Two pools, one code path
------------------------
``--pool training``  the held-in development panel.  Free to use, free to redo.
``--pool reserve``   the source-disjoint matched-reserve confirmatory panel.

The reserve pool is HELD OUT.  Materializing it is the act that opens it, so
the reserve path is gated behind an explicit flag *and* a written authorization
string that is recorded in the artifact.  A convention in a document does not
stop an accidental ``--pool reserve``; a required argument does.

The support band
----------------
Scaffold support is the number of DISTINCT retained held-in training source
molecules sharing this molecule's RDKit Murcko scaffold, banded
``0 / 1-4 / 5-24 / 25+``.  That is the definition used by the matched-validation
carve (``scripts/editing_v2_carve_matched_validation.py``), and it is
recomputed here from the committed reserve-ids file rather than copied, so the
two cannot drift apart silently.  A held-in molecule does not support itself,
so its own count is decremented -- the same rule the carve applies.

Acyclic molecules have an EMPTY Murcko scaffold and form one explicit class
rather than being treated as missing.  They therefore concentrate in the
densest support band; the artifact records ring status per source so that
confound is visible rather than buried.

The heavy-atom window
---------------------
Sources are restricted to ``[12, 34]`` heavy atoms.  The upper bound is not
cosmetic: the frozen process scope caps active atoms at 40, so a 38-atom source
would hit the process ceiling inside a six-edit horizon and its suppressed
growth would be indistinguishable from a learned preference against growing.
Six edits of headroom removes that confound from the mobility measurement.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any

#: Frozen before any rollout. See the module docstring for why 34, not 38.
MIN_HEAVY_ATOMS = 12
MAX_HEAVY_ATOMS = 34

#: Heavy-atom size bands, balanced within each support band.
SIZE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("small", 12, 19),
    ("medium", 20, 27),
    ("large", 28, 34),
)

#: Scaffold-support bands, identical to the matched-validation carve.
SUPPORT_BANDS: tuple[tuple[int, int], ...] = ((0, 0), (1, 4), (5, 24), (25, 10**9))
SUPPORT_BAND_ORDER: tuple[str, ...] = ("0", "1-4", "5-24", "25+")

CANONICAL_SLOTS = 48
DEFAULT_SEED = 20260812

RESERVE_IDS = Path("diagnostics/editing_v2_matched_validation_reserve_ids.json.gz")

#: Reserve sources already consumed by a committed panel. Both SOURCE and
#: TARGET endpoints are burned: the sealed67 amendment dropped two pairs
#: precisely because a target had become another panel's source.
BURNED_SOURCE_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("diagnostics/editing_v2_controller_panel_seal.json", ("development", "sealed")),
    ("diagnostics/editing_v2_experiment_c0_planning_signal.json", ("per_source",)),
)


def band_label(support: int) -> str:
    for (low, high), label in zip(SUPPORT_BANDS, SUPPORT_BAND_ORDER):
        if low <= support <= high:
            return label
    raise ValueError(f"support {support} fell outside every band")


def size_band(heavy_atoms: int) -> str | None:
    for label, low, high in SIZE_BANDS:
        if low <= heavy_atoms <= high:
            return label
    return None


def load_burned_endpoints(repo_root: Path) -> set[str]:
    """Every reserve endpoint already spent by a committed panel.

    Collected by walking the committed artifacts rather than by copying a list,
    so a panel added later cannot be forgotten here while its file still exists.
    """
    burned: set[str] = set()
    for relative, sections in BURNED_SOURCE_FILES:
        path = repo_root / relative
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for section in sections:
            for row in payload.get(section, []) or []:
                if isinstance(row, str):
                    burned.add(row)
                    continue
                for field in ("source", "target", "source_key", "target_key"):
                    value = row.get(field)
                    if isinstance(value, str):
                        burned.add(value)
    return burned


def scaffold_support_histogram(held_in: list[str]) -> tuple[dict[str, str], dict[str, int]]:
    """Murcko scaffold per held-in molecule, and how many molecules share each."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    scaffold_of: dict[str, str] = {}
    counts: collections.Counter[str] = collections.Counter()
    for smiles in held_in:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        except Exception:  # noqa: BLE001 - an unscaffoldable source is simply skipped
            continue
        scaffold_of[smiles] = scaffold
        counts[scaffold] += 1
    return scaffold_of, dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=("training", "reserve"), default="training")
    parser.add_argument("--count", type=int, default=40, help="total sources")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--i-am-authorized-to-open-the-matched-reserve",
        default="",
        metavar="AUTHORIZATION",
        help=(
            "Required for --pool reserve. Free text naming who authorized opening "
            "the held-out panel; recorded verbatim in the artifact."
        ),
    )
    args = parser.parse_args()

    authorization = args.i_am_authorized_to_open_the_matched_reserve.strip()
    if args.pool == "reserve" and not authorization:
        parser.error(
            "--pool reserve materializes the HELD-OUT confirmatory panel. Pass "
            "--i-am-authorized-to-open-the-matched-reserve '<who authorized this>' "
            "or run with --pool training."
        )
    if args.pool == "training" and authorization:
        parser.error("an authorization string is meaningless for the held-in pool")

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    reserve = json.load(gzip.open(args.repo_root / RESERVE_IDS, "rt"))
    held_in = list(reserve["training_source_keys"])
    print(f"held-in universe: {len(held_in):,} sources")

    scaffold_of, scaffold_counts = scaffold_support_histogram(held_in)
    print(f"scaffolds: {len(scaffold_counts):,} distinct over {len(scaffold_of):,} molecules")

    if args.pool == "training":
        pool = sorted(held_in)
        burned: set[str] = set()
    else:
        pool = sorted(reserve["reserve_source_keys"])
        burned = load_burned_endpoints(args.repo_root)
        before = len(pool)
        pool = [s for s in pool if s not in burned]
        print(f"reserve: {before:,} sources, {before - len(pool)} burned by committed panels")

    random.Random(args.seed).shuffle(pool)

    per_band = max(args.count // len(SUPPORT_BAND_ORDER), 1)
    per_cell = max(per_band // len(SIZE_BANDS), 1)
    wanted = {
        (support, size): per_cell
        for support in SUPPORT_BAND_ORDER
        for size, _low, _high in SIZE_BANDS
    }
    chosen: list[dict[str, Any]] = []
    taken: collections.Counter[tuple[str, str]] = collections.Counter()
    rejected = {"unparseable": 0, "size_window": 0, "cell_full": 0, "no_scaffold": 0}

    for smiles in pool:
        if len(chosen) >= args.count:
            break
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            rejected["unparseable"] += 1
            continue
        heavy = int(molecule.GetNumHeavyAtoms())
        size = size_band(heavy)
        if size is None:
            rejected["size_window"] += 1
            continue
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold

            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        except Exception:  # noqa: BLE001
            rejected["no_scaffold"] += 1
            continue
        support = scaffold_counts.get(scaffold, 0)
        if args.pool == "training":
            # A training molecule does not support itself. Same rule as the carve.
            support = max(support - 1, 0)
        band = band_label(support)
        cell = (band, size)
        if taken[cell] >= wanted[cell]:
            rejected["cell_full"] += 1
            continue
        taken[cell] += 1
        chosen.append(
            {
                "index": len(chosen),
                "source": smiles,
                "slots": CANONICAL_SLOTS,
                "heavy_atoms": heavy,
                "size_band": size,
                "scaffold_support": support,
                "support_band": band,
                "has_ring": bool(molecule.GetRingInfo().NumRings() > 0),
            }
        )

    chosen.sort(key=lambda row: (row["support_band"], row["size_band"], row["source"]))
    for index, row in enumerate(chosen):
        row["index"] = index

    # The panel's identity is WHICH MOLECULES it contains, so the digest is taken
    # over the sorted source set. Hashing presentation order instead would make
    # a cosmetic reordering look like a different panel.
    digest = hashlib.sha256(
        json.dumps(sorted(row["source"] for row in chosen), sort_keys=True).encode()
    ).hexdigest()

    composition = collections.Counter(
        (row["support_band"], row["size_band"]) for row in chosen
    )
    print(f"\nselected {len(chosen)} sources, panel sha256 {digest[:16]}")
    for band in SUPPORT_BAND_ORDER:
        row = [f"{size}={composition[(band, size)]}" for size, _l, _h in SIZE_BANDS]
        print(f"  band {band:>5}: " + "  ".join(row))
    print(f"  ring-containing: {sum(r['has_ring'] for r in chosen)}/{len(chosen)}")
    for reason, count in rejected.items():
        print(f"  rejected {reason}: {count:,}")

    payload = {
        "schema": "compose.claim2.trajectory_panel",
        "status": (
            "SMOKE_HELD_IN" if args.pool == "training" else "CONFIRMATORY_HELD_OUT"
        ),
        "pool": (
            "held-in training sources only"
            if args.pool == "training"
            else "matched validation reserve, source-disjoint from every committed panel"
        ),
        "held_out_opened": args.pool == "reserve",
        "authorization": authorization or None,
        "criteria": {
            "heavy_atoms": [MIN_HEAVY_ATOMS, MAX_HEAVY_ATOMS],
            "size_bands": [[label, low, high] for label, low, high in SIZE_BANDS],
            "support_bands": list(SUPPORT_BAND_ORDER),
            "support_definition": (
                "distinct retained held-in training source molecules sharing the "
                "RDKit Murcko scaffold; a held-in molecule does not support itself"
            ),
            "headroom_rule": (
                "upper heavy-atom bound leaves 6 atoms below the frozen 40-atom "
                "process ceiling so the ceiling cannot masquerade as a learned "
                "preference against growth"
            ),
        },
        "seed": args.seed,
        "requested": args.count,
        "rejected": rejected,
        "burned_excluded": sorted(burned) if args.pool == "reserve" else [],
        "burned_source_files": [relative for relative, _s in BURNED_SOURCE_FILES],
        "composition": {f"{band}|{size}": count for (band, size), count in sorted(composition.items())},
        "panel_sha256": digest,
        "sources": chosen,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
