"""Decompose de-novo quality into ring-strain and molecule-size contributions.

The de-novo base checkpoint (Lineage B, step 1,000) overproduces 3- and 4-membered
rings, and its published-metric quality deficit sits entirely in the ``SA <= 4``
half of the conjunction.  A first conditional split (n=50) found strained molecules
worse on BOTH axes -- but also ~4 heavy atoms LARGER, and SA rises with size
independently of ring strain.  So that split establishes ASSOCIATION, not attribution.

This module separates the two.  It reports, for one sample of generated molecules:

``ring_size_distribution``
    The size histogram over RINGS, normalised per ring rather than per molecule.
    This is the closure-policy shape ``P(size | a ring was formed)``.  It is
    invariant to how many rings a molecule carries and to how large molecules are,
    so it does NOT move merely because training shrank the molecules -- which is
    exactly the confound the per-molecule prevalence cannot exclude.

``strain_size_strata``
    Mean SA and QED within each (heavy-atom bin x strained-ring) cell.  Comparing
    strained against clean WITHIN a size bin holds size fixed by construction.

``strain_size_regression``
    Ordinary least squares ``SA ~ 1 + strained + heavy_atoms`` and the same for QED,
    with standard errors, so the strain coefficient is read with size partialled out.

``ring_event_census``
    Ring-forming and ring-removing EVENT counts taken from each trajectory's
    recorded ``event_rules``.  This is the model's own transition statistic rather
    than an endpoint census.

Invariants maintained here:

* Heavy-atom bin edges are FIXED constants, never derived from the sample.  A
  data-dependent binning is not comparable across checkpoints, which is the only
  comparison this module exists to support.
* Every reported denominator is explicit.  ``molecules`` counts parsed molecules;
  ``rings`` counts SSSR rings; strata carry their own ``n``.
* A cell or regression with too few points reports ``None`` rather than a number,
  so an empty stratum can never be read as a measured zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from rdkit import Chem, RDLogger

from compose_v4.eval.molecular_quality import DESCRIPTORS

# ---- Constants ----

# Fixed so strata are comparable across checkpoints.  The de-novo sampler is capped
# at MAX_ATOMS = 40 heavy atoms, so the final bin is closed at 41.
HEAVY_ATOM_BIN_EDGES: tuple[int, ...] = (0, 15, 20, 25, 30, 35, 41)

# Ring sizes counted as strained.  Matches the census in denovo_benchmark.
STRAINED_RING_SIZES: frozenset[int] = frozenset({3, 4})

# Rewrite families that create or remove whole ring systems in the de-novo path.
RING_FORMING_RULES: frozenset[str] = frozenset({"ring_system_grow", "cycle_insert", "cycle_attach"})
RING_REMOVING_RULES: frozenset[str] = frozenset({"ring_system_delete"})

# A stratum below this many molecules reports None instead of a mean.
MINIMUM_STRATUM = 3


# ---- Per-molecule features ----


def molecule_features(smiles: str) -> dict[str, object] | None:
    """Heavy-atom count, strain indicator, QED and SA for one molecule.

    Returns ``None`` when the string is empty or does not parse, so callers
    distinguish "no molecule" from a molecule with extreme values.
    """

    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
    return {
        "smiles": smiles,
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "ring_sizes": sizes,
        "strained": bool(STRAINED_RING_SIZES & set(sizes)),
        "qed": float(DESCRIPTORS["qed"](mol)),
        "sa_score": float(DESCRIPTORS["sa_score"](mol)),
    }


def _feature_table(smiles: Sequence[str], *, silence_rdkit: bool = True) -> list[dict[str, object]]:
    if silence_rdkit:
        RDLogger.DisableLog("rdApp.*")
    rows = []
    for text in smiles:
        row = molecule_features(text)
        if row is not None:
            rows.append(row)
    return rows


# ---- Mechanism proxy: the closure-policy shape ----


def ring_size_distribution(smiles: Sequence[str]) -> dict[str, object]:
    """Size histogram over RINGS, plus the per-ring strained fraction.

    ``strained_ring_fraction`` is ``P(size in {3,4} | a ring exists)``.  Because it
    is normalised per ring, shrinking the molecules does not move it; only a change
    in which ring sizes the model closes does.  Report it beside the per-molecule
    prevalence, never instead of it -- they answer different questions.
    """

    rows = _feature_table(smiles)
    histogram: dict[int, int] = {}
    for row in rows:
        for size in row["ring_sizes"]:  # type: ignore[index]
            histogram[size] = histogram.get(size, 0) + 1
    total_rings = sum(histogram.values())
    strained_rings = sum(count for size, count in histogram.items() if size in STRAINED_RING_SIZES)
    molecules = len(rows)
    return {
        "molecules": molecules,
        "rings": total_rings,
        "ring_size_histogram": dict(sorted(histogram.items())),
        "ring_size_fraction": {
            size: count / total_rings for size, count in sorted(histogram.items())
        }
        if total_rings
        else {},
        "strained_ring_fraction": strained_rings / total_rings if total_rings else None,
        "rings_per_molecule": total_rings / molecules if molecules else None,
        "fraction_with_strained_ring": (
            sum(1 for row in rows if row["strained"]) / molecules if molecules else None
        ),
        "mean_heavy_atoms": (
            float(np.mean([row["heavy_atoms"] for row in rows])) if molecules else None
        ),
    }


# ---- Size-controlled strata ----


def _bin_label(heavy: int) -> str:
    for low, high in pairwise(HEAVY_ATOM_BIN_EDGES):
        if low < heavy <= high:
            return f"{low + 1}-{high}"
    return f">{HEAVY_ATOM_BIN_EDGES[-1]}"


def strain_size_strata(smiles: Sequence[str]) -> dict[str, object]:
    """Mean SA and QED in each (heavy-atom bin x strained) cell.

    Holding the size bin fixed is what turns the strained-vs-clean contrast from an
    association into a size-controlled comparison.  Cells below ``MINIMUM_STRATUM``
    report ``None`` so a thin cell is never read as a measured value.
    """

    rows = _feature_table(smiles)
    cells: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        label = _bin_label(int(row["heavy_atoms"]))  # type: ignore[arg-type]
        arm = "strained" if row["strained"] else "clean"
        cells.setdefault(label, {}).setdefault(arm, []).append(row)

    def _lower_edge(label: str) -> int:
        head = label.lstrip(">").split("-")[0]
        return int(head) if head.isdigit() else 10**6

    table: dict[str, object] = {}
    for label in sorted(cells, key=_lower_edge):
        entry: dict[str, object] = {}
        for arm in ("strained", "clean"):
            members = cells[label].get(arm, [])
            if len(members) < MINIMUM_STRATUM:
                entry[arm] = {"n": len(members), "mean_sa": None, "mean_qed": None}
                continue
            entry[arm] = {
                "n": len(members),
                "mean_sa": float(np.mean([row["sa_score"] for row in members])),
                "mean_qed": float(np.mean([row["qed"] for row in members])),
                "mean_heavy_atoms": float(np.mean([row["heavy_atoms"] for row in members])),
            }
        strained = entry["strained"]
        clean = entry["clean"]
        if isinstance(strained, dict) and isinstance(clean, dict):
            if strained["mean_sa"] is not None and clean["mean_sa"] is not None:
                entry["within_bin_sa_difference"] = float(strained["mean_sa"] - clean["mean_sa"])
                entry["within_bin_qed_difference"] = float(strained["mean_qed"] - clean["mean_qed"])
            else:
                entry["within_bin_sa_difference"] = None
                entry["within_bin_qed_difference"] = None
        table[label] = entry
    return {"bin_edges": list(HEAVY_ATOM_BIN_EDGES), "strata": table}


# ---- Size-partialled regression ----


def _ordinary_least_squares(
    design: np.ndarray, response: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Coefficients and their standard errors, or None when underdetermined."""

    n_rows, n_columns = design.shape
    if n_rows <= n_columns:
        return None
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    degrees = n_rows - n_columns
    sigma_squared = float(residual @ residual) / degrees
    try:
        covariance = sigma_squared * np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return None
    return coefficients, np.sqrt(np.diag(covariance))


def strain_size_regression(smiles: Sequence[str]) -> dict[str, object]:
    """Fit ``SA ~ 1 + strained + heavy_atoms`` and ``QED ~ 1 + strained + heavy_atoms``.

    The ``strained`` coefficient is the strain effect with molecule size held fixed.
    If the raw strained-vs-clean gap is mostly size, this coefficient collapses
    toward zero while ``heavy_atoms`` carries the signal.  Standard errors are
    reported so the coefficient is read with its uncertainty, never alone.
    """

    rows = _feature_table(smiles)
    if len(rows) < 4:
        return {"n": len(rows), "sa": None, "qed": None}
    strained = np.array([1.0 if row["strained"] else 0.0 for row in rows])
    heavy = np.array([float(row["heavy_atoms"]) for row in rows])  # type: ignore[arg-type]
    design = np.column_stack([np.ones(len(rows)), strained, heavy])

    report: dict[str, object] = {
        "n": len(rows),
        "strained_count": int(strained.sum()),
        "terms": ["intercept", "strained", "heavy_atoms"],
    }
    for name, key in (("sa", "sa_score"), ("qed", "qed")):
        response = np.array([float(row[key]) for row in rows])  # type: ignore[arg-type]
        fitted = _ordinary_least_squares(design, response)
        if fitted is None:
            report[name] = None
            continue
        coefficients, errors = fitted
        report[name] = {
            "intercept": float(coefficients[0]),
            "strained_coefficient": float(coefficients[1]),
            "strained_stderr": float(errors[1]),
            "heavy_atom_coefficient": float(coefficients[2]),
            "heavy_atom_stderr": float(errors[2]),
            "raw_strained_difference": (
                float(response[strained == 1].mean() - response[strained == 0].mean())
                if strained.sum() and (strained == 0).sum()
                else None
            ),
        }
    return report


# ---- Transition-level census ----


def ring_event_census(records: Sequence[dict]) -> dict[str, object]:
    """Ring-forming and ring-removing EVENT counts from recorded ``event_rules``.

    This reads the model's own transitions rather than the endpoint.  The endpoint
    ring census can fall simply because molecules got smaller while the closure
    policy stayed unchanged; the events-per-molecule rate separates those.

    ``ring_removals`` is reported so the reader can check the assumption that lets
    the endpoint ring-size histogram stand in for the size distribution of ring
    CREATION events: when removals are ~0, no ring is created and then destroyed, so
    every endpoint ring corresponds to a creation event.
    """

    families: dict[str, int] = {}
    ring_forming = 0
    ring_removing = 0
    trajectories = 0
    total_events = 0
    for record in records:
        rules = record.get("event_rules")
        if rules is None:
            continue
        trajectories += 1
        for rule in rules:
            families[rule] = families.get(rule, 0) + 1
            total_events += 1
            if rule in RING_FORMING_RULES:
                ring_forming += 1
            elif rule in RING_REMOVING_RULES:
                ring_removing += 1
    return {
        "trajectories": trajectories,
        "events": total_events,
        "event_family_histogram": dict(sorted(families.items())),
        "ring_forming_events": ring_forming,
        "ring_removing_events": ring_removing,
        "ring_forming_events_per_trajectory": (
            ring_forming / trajectories if trajectories else None
        ),
        "ring_removing_events_per_trajectory": (
            ring_removing / trajectories if trajectories else None
        ),
        "endpoint_census_tracks_creation": (ring_removing == 0) if trajectories else None,
    }


# ---- Report ----


def ring_decomposition_report(
    smiles: Sequence[str], records: Sequence[dict] | None = None
) -> dict[str, object]:
    """The full decomposition for one checkpoint's sample."""

    report: dict[str, object] = {
        "ring_size_distribution": ring_size_distribution(smiles),
        "strain_size_strata": strain_size_strata(smiles),
        "strain_size_regression": strain_size_regression(smiles),
    }
    if records is not None:
        report["ring_event_census"] = ring_event_census(records)
    return report
