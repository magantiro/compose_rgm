"""The global ring-system marginal: what a molecule's ring skeleton IS, and
what the corpus law over it looks like.

The de-novo process builds through a CHANGING legal fiber.  A ring system is
installed by ``ring_system_grow`` onto a host that ``_eligible_grow_host_graph``
restricts to acyclic, neutral, single-bonded carbon, so every committed ring
permanently removes its atoms from the host that the NEXT ring decision is made
against.  The support therefore distorts the realized ring marginal before the
learned rates ever choose, and the distortion is measurable as an ordinal
cascade.

This module supplies the two objects a global ring plan needs, and nothing more:

``ring_system_signature``
    The ring skeleton of one molecule, as a sorted multiset of per-ring-system
    minimum-cycle-basis size tuples -- e.g. benzene ``((6,),)``, indole
    ``((5, 6),)``, a biphenyl ``((6,), (6,))``.  It is computed by the SAME
    ``nx.minimum_cycle_basis`` call that ``ring_template_cycle_sizes`` applies
    to a catalog template, so a corpus signature and a template signature are
    comparable by construction rather than by convention.

``RingSystemPlanPrior``
    The empirical corpus law over those multisets, conditioned on a heavy-atom
    bin.  Conditioning is deliberate: ring content correlates strongly with
    molecule size, and the de-novo initial law already fixes the size, so an
    unconditioned draw would decorrelate the two and manufacture infeasible
    plans.  The prior is nonparametric -- it draws an observed multiset -- which
    is the same move InVirtuoGen makes for sequence length (``p(n)`` fitted
    empirically on ZINC250k) and GenMol makes for mask count.

Nothing here reads a model, a checkpoint or an oracle.  The census functions are
used both to fit the prior and to score an arm's output, so "did arm C reproduce
the training ring distribution" is answered by one function applied twice.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem

# A ring system's signature is the sorted tuple of its minimum-cycle-basis ring
# sizes; a molecule's signature is the sorted tuple of its systems' signatures.
RingSystemSignature = tuple[int, ...]
MoleculeRingSignature = tuple[RingSystemSignature, ...]

# Heavy-atom bin edges for the conditional prior.  Upper-exclusive, with an
# open first and last bin.  Chosen to be wide enough that every bin of the
# 500k GuacaMol subset carries thousands of molecules.
DEFAULT_HEAVY_ATOM_BIN_EDGES: tuple[int, ...] = (15, 19, 23, 27, 31, 35)

SMALL_RING_MAXIMUM = 4


# ---- Signatures -------------------------------------------------------------


def ring_system_signature(mol: Chem.Mol) -> MoleculeRingSignature:
    """The molecule's ring skeleton, as a sorted multiset of system signatures.

    Ring systems are the connected components of the RING-BOND subgraph, which
    is what makes a fused bicyclic ONE system rather than two, matching the
    catalog's notion of a ring-system template exactly.
    """

    ring_bonds: set[int] = set()
    for ring in mol.GetRingInfo().AtomRings():
        for position in range(len(ring)):
            bond = mol.GetBondBetweenAtoms(
                int(ring[position]), int(ring[(position + 1) % len(ring)])
            )
            if bond is not None:
                ring_bonds.add(int(bond.GetIdx()))
    graph = nx.Graph()
    for index in ring_bonds:
        bond = mol.GetBondWithIdx(index)
        graph.add_edge(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))
    systems = [
        tuple(
            sorted(
                len(cycle)
                for cycle in nx.minimum_cycle_basis(graph.subgraph(component))
            )
        )
        for component in nx.connected_components(graph)
    ]
    return tuple(sorted(systems))


def ring_system_signature_of_smiles(smiles: str) -> MoleculeRingSignature | None:
    """The ring skeleton of a SMILES, or ``None`` when it does not parse."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return ring_system_signature(mol)


def heavy_atom_bin(
    heavy_atoms: int,
    *,
    edges: Sequence[int] = DEFAULT_HEAVY_ATOM_BIN_EDGES,
) -> str:
    """The label of the heavy-atom bin a molecule falls in.

    Labels are strings so a fitted prior round-trips through JSON without a
    key-type conversion that could silently reorder or collide bins.
    """

    if any(right <= left for left, right in pairwise(edges)):
        raise ValueError("heavy-atom bin edges must be strictly increasing")
    if heavy_atoms < edges[0]:
        return f"<{edges[0]}"
    for left, right in pairwise(edges):
        if heavy_atoms < right:
            return f"{left}-{right - 1}"
    return f">={edges[-1]}"


# ---- Census -----------------------------------------------------------------


def ring_signature_census(
    signatures: Iterable[MoleculeRingSignature],
    *,
    small_ring_maximum: int = SMALL_RING_MAXIMUM,
) -> dict:
    """The distribution facts a ring-marginal claim rests on.

    Reports the per-RING size histogram, the per-SYSTEM signature distribution
    and the per-MOLECULE multiset distribution separately, because they answer
    different questions and only the per-ring one is comparable to an endpoint
    strain prevalence.  ``strained_ring_fraction`` is the per-RING rate and
    ``fraction_with_strained_ring`` the per-MOLECULE one; the two diverge, and
    a previous de-novo reading was inverted by quoting only the molecule-level
    number.
    """

    rows = list(signatures)
    if not rows:
        raise ValueError("a ring-signature census needs at least one molecule")
    ring_sizes: dict[int, int] = {}
    system_counts: dict[RingSystemSignature, int] = {}
    multiset_counts: dict[MoleculeRingSignature, int] = {}
    system_total = 0
    ring_total = 0
    strained_rings = 0
    strained_molecules = 0
    count_histogram: dict[int, int] = {}
    for signature in rows:
        multiset_counts[signature] = multiset_counts.get(signature, 0) + 1
        count_histogram[len(signature)] = count_histogram.get(len(signature), 0) + 1
        molecule_has_small = False
        for system in signature:
            system_total += 1
            system_counts[system] = system_counts.get(system, 0) + 1
            for size in system:
                ring_total += 1
                ring_sizes[size] = ring_sizes.get(size, 0) + 1
                if size <= small_ring_maximum:
                    strained_rings += 1
                    molecule_has_small = True
        strained_molecules += int(molecule_has_small)
    return {
        "molecules": len(rows),
        "rings": ring_total,
        "ring_systems": system_total,
        "rings_per_molecule": ring_total / len(rows),
        "ring_systems_per_molecule": system_total / len(rows),
        "ring_size_histogram": {str(k): v for k, v in sorted(ring_sizes.items())},
        "ring_size_fraction": {
            str(k): v / ring_total for k, v in sorted(ring_sizes.items())
        }
        if ring_total
        else {},
        "strained_ring_fraction": (strained_rings / ring_total) if ring_total else 0.0,
        "fraction_with_strained_ring": strained_molecules / len(rows),
        "ring_system_count_fraction": {
            str(k): v / len(rows) for k, v in sorted(count_histogram.items())
        },
        "system_signature_fraction": {
            repr(k): v / system_total
            for k, v in sorted(system_counts.items(), key=lambda kv: -kv[1])
        }
        if system_total
        else {},
        "distinct_system_signatures": len(system_counts),
        "distinct_molecule_multisets": len(multiset_counts),
    }


def ring_size_total_variation(left: dict, right: dict) -> float:
    """Total variation between two ``ring_size_fraction`` tables.

    The single number the decisive check reduces to: how far an arm's realized
    per-ring size law sits from the training corpus's.  It is symmetric, bounded
    in [0, 1], and unlike a small-ring prevalence it cannot be gamed by trading
    one off-distribution size for another.
    """

    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys
    )


# ---- The prior --------------------------------------------------------------


@dataclass(frozen=True)
class RingSystemPlanPrior:
    """Empirical ``p(R | heavy-atom bin)`` over ring-system multisets.

    ``tables`` maps a bin label to the observed multisets and their counts.  A
    draw returns one observed multiset, so the prior can only ever propose a
    ring skeleton some real molecule of that size actually had -- there is no
    factorization assumption and therefore no way for it to invent a skeleton.
    """

    bin_edges: tuple[int, ...]
    tables: dict[str, tuple[tuple[MoleculeRingSignature, int], ...]]
    fitted_molecules: int

    def __post_init__(self) -> None:
        if not self.tables:
            raise ValueError("a ring-system plan prior needs at least one bin")
        for label, rows in self.tables.items():
            if not rows:
                raise ValueError(f"bin {label!r} has no observations")
            if any(count <= 0 for _signature, count in rows):
                raise ValueError(f"bin {label!r} has a non-positive count")

    # -- fitting --

    @classmethod
    def fit(
        cls,
        observations: Iterable[tuple[int, MoleculeRingSignature]],
        *,
        bin_edges: Sequence[int] = DEFAULT_HEAVY_ATOM_BIN_EDGES,
        minimum_bin_observations: int = 200,
    ) -> RingSystemPlanPrior:
        """Fit from ``(heavy_atoms, signature)`` pairs.

        A bin thinner than ``minimum_bin_observations`` is refused rather than
        silently pooled: a plan prior that quietly falls back to a different
        size regime would break exactly the size/ring correlation it exists to
        preserve, and would do so invisibly.
        """

        buckets: dict[str, dict[MoleculeRingSignature, int]] = {}
        total = 0
        for heavy_atoms, signature in observations:
            label = heavy_atom_bin(int(heavy_atoms), edges=bin_edges)
            buckets.setdefault(label, {})
            buckets[label][signature] = buckets[label].get(signature, 0) + 1
            total += 1
        if not buckets:
            raise ValueError("no observations to fit a ring-system plan prior")
        thin = {
            label: sum(rows.values())
            for label, rows in buckets.items()
            if sum(rows.values()) < minimum_bin_observations
        }
        if thin:
            raise ValueError(
                "ring-system plan prior bins are too thin to fit: "
                + ", ".join(f"{label}={count}" for label, count in sorted(thin.items()))
            )
        return cls(
            bin_edges=tuple(int(edge) for edge in bin_edges),
            tables={
                label: tuple(sorted(rows.items()))
                for label, rows in sorted(buckets.items())
            },
            fitted_molecules=total,
        )

    # -- use --

    def bin_for(self, heavy_atoms: int) -> str:
        """The bin a host of this size draws from, nearest-populated if empty.

        The fallback is by bin ORDER, not by label string: a size regime the
        corpus never covers must still yield a plan, and the nearest covered
        regime is the only defensible stand-in.  It is reported by
        ``sample_with_provenance`` so a fallback draw is never invisible.
        """

        label = heavy_atom_bin(int(heavy_atoms), edges=self.bin_edges)
        if label in self.tables:
            return label
        ordered = self.ordered_bin_labels()
        if label not in ordered:
            raise ValueError(f"bin {label!r} is not expressible under these edges")
        position = ordered.index(label)
        present = [index for index, name in enumerate(ordered) if name in self.tables]
        nearest = min(present, key=lambda index: (abs(index - position), index))
        return ordered[nearest]

    def ordered_bin_labels(self) -> tuple[str, ...]:
        """Every bin label these edges can produce, in increasing size order."""

        edges = self.bin_edges
        labels = [f"<{edges[0]}"]
        labels.extend(f"{left}-{right - 1}" for left, right in pairwise(edges))
        labels.append(f">={edges[-1]}")
        return tuple(labels)

    def sample_with_provenance(
        self, rng: np.random.Generator, heavy_atoms: int
    ) -> tuple[MoleculeRingSignature, dict]:
        """Draw one ring skeleton, and say which bin it came from."""

        requested = heavy_atom_bin(int(heavy_atoms), edges=self.bin_edges)
        label = self.bin_for(heavy_atoms)
        rows = self.tables[label]
        weights = np.asarray([count for _signature, count in rows], dtype=np.float64)
        weights /= weights.sum()
        index = int(rng.choice(len(rows), p=weights))
        return rows[index][0], {
            "requested_bin": requested,
            "drawn_bin": label,
            "bin_fallback": requested != label,
            "bin_observations": int(weights.size),
        }

    def sample(
        self, rng: np.random.Generator, heavy_atoms: int
    ) -> MoleculeRingSignature:
        return self.sample_with_provenance(rng, heavy_atoms)[0]

    # -- serialization --

    def to_json(self) -> dict:
        return {
            "schema": "compose.denovo.ring_system_plan_prior",
            "schema_version": 1,
            "bin_edges": list(self.bin_edges),
            "fitted_molecules": self.fitted_molecules,
            "tables": {
                label: [
                    [[list(system) for system in signature], count]
                    for signature, count in rows
                ]
                for label, rows in self.tables.items()
            },
        }

    @classmethod
    def from_json(cls, payload: dict) -> RingSystemPlanPrior:
        if payload.get("schema") != "compose.denovo.ring_system_plan_prior":
            raise ValueError("payload is not a ring-system plan prior")
        return cls(
            bin_edges=tuple(int(edge) for edge in payload["bin_edges"]),
            tables={
                label: tuple(
                    (
                        tuple(tuple(int(size) for size in system) for system in signature),
                        int(count),
                    )
                    for signature, count in rows
                )
                for label, rows in payload["tables"].items()
            },
            fitted_molecules=int(payload["fitted_molecules"]),
        )

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json()))

    @classmethod
    def read(cls, path: str | Path) -> RingSystemPlanPrior:
        return cls.from_json(json.loads(Path(path).read_text()))
