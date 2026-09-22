"""Global graph delta between two molecules, computed without any primitive sequence.

The unit is ``omega = (retain region, edit regions, operation family, scale,
completion constraints)``: what a complete COMPOSE program would have to
accomplish, treated as ONE macro option rather than a sequence of primitives.

DATA STRUCTURE
--------------
``GlobalDelta`` is a flat record over one ordered pair ``(source, target)``.
The retained core is a maximum common substructure; every atom outside it on
the source side is *excised* and every atom outside it on the target side is
*installed*.  Both sets are partitioned into connected components in their own
molecule's graph, so "how many separated regions change" is a count of
components rather than a count of primitives.

INVARIANTS maintained and tested
--------------------------------
* ``retained_atoms + excised_atoms == source_heavy`` and
  ``retained_atoms + installed_atoms == target_heavy``.
* ``delta_heavy == target_heavy - source_heavy == installed - excised``.
* every excised/installed component is non-empty and connected in its parent
  graph; component sizes sum to the corresponding total.
* an anchor of a changed component is a RETAINED atom adjacent to it, so
  anchors are always a subset of the core.
* the operation family is a total function of the (excised, installed,
  ring-delta, region-count) tuple, so every pair receives exactly one label.

The MCS is computed with ``ringMatchesRingOnly=False`` and bond comparison
``CompareAny``, so a bond-order change inside the core counts as RETAINED
structure and is reported separately as ``core_bond_changes``.  That keeps a
saturation/aromatisation change from being mis-read as an excision.

RING MATCHING IS DELIBERATELY OFF.  With ``ringMatchesRingOnly=True`` a plain
cyclization (``CCCCCC -> C1CCCCC1``) admits no common substructure at all and
is scored as a six-atom replacement, which would inflate the replacement
template with transformations that change no atom.  Atom-level retention and
ring-topology change are therefore kept as two INDEPENDENT axes: the core
records which atoms survive, and ``ring_count_delta`` / ``ring_topology_changed``
record what happened to the cycles over those same atoms.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import networkx as nx
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# ---- Configuration ----

MCS_TIMEOUT_SECONDS = 20
RING_MATCHES_RING_ONLY = False
DISTANT_ANCHOR_HOPS = 4  # core-graph distance at or above which two edits are "distant"


# ---- Records ----


@dataclass
class GlobalDelta:
    """One ordered (source, target) pair characterised globally."""

    source_smiles: str
    target_smiles: str
    status: str = "ok"

    source_heavy: int = 0
    target_heavy: int = 0
    delta_heavy: int = 0

    retained_atoms: int = 0
    retained_fraction_source: float = 0.0
    retained_fraction_target: float = 0.0
    mcs_canonical_smarts: str = ""
    mcs_timed_out: bool = False

    excised_atoms: int = 0
    installed_atoms: int = 0
    excised_region_sizes: list[int] = field(default_factory=list)
    installed_region_sizes: list[int] = field(default_factory=list)
    n_excised_regions: int = 0
    n_installed_regions: int = 0
    n_changed_regions: int = 0
    largest_changed_region: int = 0
    total_changed_atoms: int = 0

    n_attachment_boundaries: int = 0
    n_anchor_atoms: int = 0
    n_anchor_groups: int = 0
    n_replacement_sites: int = 0
    attachment_moved: bool = False
    substituent_replaced: bool = False

    multi_region: bool = False
    distant_coupled_edits: bool = False
    max_anchor_separation: int = 0

    core_bond_changes: int = 0
    ring_count_source: int = 0
    ring_count_target: int = 0
    ring_count_delta: int = 0
    ring_system_count_source: int = 0
    ring_system_count_target: int = 0
    ring_topology_changed: bool = False
    ring_sizes_source: list[int] = field(default_factory=list)
    ring_sizes_target: list[int] = field(default_factory=list)

    installed_ring_atoms: int = 0
    installed_region_has_ring: bool = False
    installed_heteroatoms: int = 0
    installed_branch_points: int = 0
    largest_installed_is_linear_chain: bool = False
    installed_expressible_as_grow_chain: bool = False
    excised_ring_atoms: int = 0
    excised_region_has_ring: bool = False

    heteroatom_changes: int = 0
    element_delta: dict[str, int] = field(default_factory=dict)
    charge_source: int = 0
    charge_target: int = 0
    charge_delta: int = 0

    scaffold_source: str = ""
    scaffold_target: str = ""
    scaffold_retained: bool = False

    operation_family: str = "unclassified"

    def as_dict(self) -> dict:
        return asdict(self)


# ---- Helpers ----


def _mol(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


def _graph(mol: Chem.Mol) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(mol.GetNumAtoms()))
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return graph


def _components(graph: nx.Graph, nodes: Iterable[int]) -> list[set[int]]:
    nodes = set(nodes)
    if not nodes:
        return []
    return [set(component) for component in nx.connected_components(graph.subgraph(nodes))]


def _ring_systems(mol: Chem.Mol) -> list[set[int]]:
    """Fused ring systems: rings sharing at least one atom are one system."""
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    systems: list[set[int]] = []
    for ring in rings:
        merged = [ring]
        remaining = []
        for system in systems:
            if system & ring:
                merged.append(system)
            else:
                remaining.append(system)
        combined: set[int] = set()
        for part in merged:
            combined |= part
        systems = remaining + [combined]
    return systems


def _element_counts(mol: Chem.Mol) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    return counts


def _net_charge(mol: Chem.Mol) -> int:
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def _scaffold(mol: Chem.Mol) -> str:
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:  # noqa: BLE001 - scaffold perception is best effort
        return ""
    if core is None or core.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(core)


def stable_seed(*parts: str) -> int:
    """Digest-derived seed. Never ``hash()`` -- that is PYTHONHASHSEED-salted."""
    digest = hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


# ---- Core computation ----


def _classify(delta: GlobalDelta) -> str:
    """Total function from the measured shape to one operation family label."""
    excised = delta.excised_atoms
    installed = delta.installed_atoms
    ring_moved = delta.ring_count_delta != 0 or delta.ring_topology_changed

    if excised == 0 and installed == 0:
        if delta.core_bond_changes > 0 or ring_moved:
            return "ring_rewire_only"
        return "identity"
    if excised == 0:
        base = "grow"
    elif installed == 0:
        base = "delete"
    else:
        base = "replace"
    if delta.multi_region:
        base = f"multi_region_{base}"
    if ring_moved:
        base = f"{base}+ring_rewire"
    return base


def compute_global_delta(
    source_smiles: str,
    target_smiles: str,
    *,
    mcs_timeout_seconds: int = MCS_TIMEOUT_SECONDS,
    distant_hops: int = DISTANT_ANCHOR_HOPS,
    ring_matches_ring_only: bool = RING_MATCHES_RING_ONLY,
) -> GlobalDelta:
    """Characterise one ordered pair without ever enumerating a primitive."""
    delta = GlobalDelta(source_smiles=source_smiles, target_smiles=target_smiles)
    source = _mol(source_smiles)
    target = _mol(target_smiles)
    if source is None or target is None:
        delta.status = "unparseable"
        return delta

    delta.source_heavy = source.GetNumAtoms()
    delta.target_heavy = target.GetNumAtoms()
    delta.delta_heavy = delta.target_heavy - delta.source_heavy

    result = rdFMCS.FindMCS(
        [source, target],
        ringMatchesRingOnly=ring_matches_ring_only,
        completeRingsOnly=False,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False,
        timeout=mcs_timeout_seconds,
    )
    delta.mcs_timed_out = bool(result.canceled)
    delta.mcs_canonical_smarts = result.smartsString or ""

    if not result.smartsString or result.numAtoms == 0:
        source_match: tuple[int, ...] = ()
        target_match: tuple[int, ...] = ()
    else:
        query = Chem.MolFromSmarts(result.smartsString)
        if query is None:
            delta.status = "mcs_unparseable"
            return delta
        source_hits = source.GetSubstructMatch(query)
        target_hits = target.GetSubstructMatch(query)
        if not source_hits or not target_hits:
            delta.status = "mcs_unmatched"
            return delta
        source_match, target_match = source_hits, target_hits

    core_source = set(source_match)
    core_target = set(target_match)
    source_to_target = dict(zip(source_match, target_match, strict=False))

    delta.retained_atoms = len(core_source)
    delta.retained_fraction_source = (
        len(core_source) / delta.source_heavy if delta.source_heavy else 0.0
    )
    delta.retained_fraction_target = (
        len(core_source) / delta.target_heavy if delta.target_heavy else 0.0
    )

    source_graph = _graph(source)
    target_graph = _graph(target)
    excised_nodes = set(range(delta.source_heavy)) - core_source
    installed_nodes = set(range(delta.target_heavy)) - core_target
    excised_regions = _components(source_graph, excised_nodes)
    installed_regions = _components(target_graph, installed_nodes)

    delta.excised_atoms = len(excised_nodes)
    delta.installed_atoms = len(installed_nodes)
    delta.excised_region_sizes = sorted((len(r) for r in excised_regions), reverse=True)
    delta.installed_region_sizes = sorted((len(r) for r in installed_regions), reverse=True)
    delta.n_excised_regions = len(excised_regions)
    delta.n_installed_regions = len(installed_regions)
    delta.total_changed_atoms = delta.excised_atoms + delta.installed_atoms
    delta.largest_changed_region = max(
        delta.excised_region_sizes + delta.installed_region_sizes, default=0
    )

    # Anchors: retained atoms adjacent to a changed region, expressed in SOURCE indices.
    target_to_source = {v: k for k, v in source_to_target.items()}
    excised_anchors: list[set[int]] = []
    for region in excised_regions:
        anchors = {
            neighbour
            for node in region
            for neighbour in source_graph.neighbors(node)
            if neighbour in core_source
        }
        excised_anchors.append(anchors)
    installed_anchors: list[set[int]] = []
    for region in installed_regions:
        anchors = {
            target_to_source[neighbour]
            for node in region
            for neighbour in target_graph.neighbors(node)
            if neighbour in core_target
        }
        installed_anchors.append(anchors)

    boundary_edges = sum(
        1
        for node in excised_nodes
        for neighbour in source_graph.neighbors(node)
        if neighbour in core_source
    ) + sum(
        1
        for node in installed_nodes
        for neighbour in target_graph.neighbors(node)
        if neighbour in core_target
    )
    delta.n_attachment_boundaries = boundary_edges

    all_anchor_atoms: set[int] = set()
    for anchors in excised_anchors + installed_anchors:
        all_anchor_atoms |= anchors
    delta.n_anchor_atoms = len(all_anchor_atoms)

    excised_anchor_union: set[int] = set()
    for anchors in excised_anchors:
        excised_anchor_union |= anchors
    installed_anchor_union: set[int] = set()
    for anchors in installed_anchors:
        installed_anchor_union |= anchors

    delta.n_replacement_sites = len(excised_anchor_union & installed_anchor_union)
    delta.substituent_replaced = delta.n_replacement_sites > 0
    # An attachment MOVED when both sides changed but at disjoint anchor sets:
    # structure left one core position and arrived at a different one.
    delta.attachment_moved = bool(
        excised_anchor_union
        and installed_anchor_union
        and not (excised_anchor_union & installed_anchor_union)
    )

    delta.n_changed_regions = delta.n_excised_regions + delta.n_installed_regions
    # Two changed regions sharing an anchor are ONE site, not two.
    anchor_groups: list[set[int]] = []
    for anchors in excised_anchors + installed_anchors:
        merged = [anchors]
        remaining = []
        for group in anchor_groups:
            if group & anchors:
                merged.append(group)
            else:
                remaining.append(group)
        combined: set[int] = set()
        for part in merged:
            combined |= part
        anchor_groups = remaining + [combined]
    delta.n_anchor_groups = len(anchor_groups)
    delta.multi_region = len(anchor_groups) >= 2

    if len(anchor_groups) >= 2 and core_source:
        core_graph = source_graph.subgraph(core_source)
        worst = 0
        for i in range(len(anchor_groups)):
            for j in range(i + 1, len(anchor_groups)):
                best = None
                for a in anchor_groups[i]:
                    for b in anchor_groups[j]:
                        try:
                            hops = nx.shortest_path_length(core_graph, a, b)
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
                        best = hops if best is None else min(best, hops)
                if best is not None:
                    worst = max(worst, best)
        delta.max_anchor_separation = worst
        delta.distant_coupled_edits = worst >= distant_hops

    # What the installed region CONTAINS.  ``_grow_actions`` in the production
    # proposal path can only build a linear single-bonded C/N/O chain, so
    # ring content, branching and non-CNO elements are the axes on which an
    # installed region can be outside that channel's reach.
    target_rings = target.GetRingInfo()
    source_rings_info = source.GetRingInfo()
    delta.installed_ring_atoms = sum(
        1 for node in installed_nodes if target_rings.NumAtomRings(node) > 0
    )
    delta.installed_region_has_ring = delta.installed_ring_atoms > 0
    delta.installed_heteroatoms = sum(
        1 for node in installed_nodes if target.GetAtomWithIdx(node).GetSymbol() != "C"
    )
    delta.excised_ring_atoms = sum(
        1 for node in excised_nodes if source_rings_info.NumAtomRings(node) > 0
    )
    delta.excised_region_has_ring = delta.excised_ring_atoms > 0

    if installed_regions:
        largest = max(installed_regions, key=len)
        sub = target_graph.subgraph(largest)
        delta.installed_branch_points = sum(1 for n in sub.nodes if sub.degree(n) >= 3)
        is_path = (
            sub.number_of_edges() == sub.number_of_nodes() - 1
            and all(sub.degree(n) <= 2 for n in sub.nodes)
        )
        all_single = all(
            target.GetBondBetweenAtoms(u, v).GetBondType() == Chem.BondType.SINGLE
            for u, v in sub.edges
        )
        only_cno = all(
            target.GetAtomWithIdx(n).GetSymbol() in ("C", "N", "O") for n in largest
        )
        # One anchor bond only: ``_grow_actions`` grows a chain off a single anchor.
        anchor_bonds = sum(
            1
            for node in largest
            for neighbour in target_graph.neighbors(node)
            if neighbour in core_target
        )
        delta.largest_installed_is_linear_chain = bool(is_path and all_single)
        delta.installed_expressible_as_grow_chain = bool(
            is_path and all_single and only_cno and anchor_bonds == 1 and len(largest) <= 8
        )

    # Bond-order changes strictly inside the retained core.
    changes = 0
    for bond in source.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in core_source and b in core_source:
            ta, tb = source_to_target.get(a), source_to_target.get(b)
            if ta is None or tb is None:
                continue
            counterpart = target.GetBondBetweenAtoms(ta, tb)
            if counterpart is None or counterpart.GetBondType() != bond.GetBondType():
                changes += 1
    delta.core_bond_changes = changes

    delta.ring_count_source = source.GetRingInfo().NumRings()
    delta.ring_count_target = target.GetRingInfo().NumRings()
    delta.ring_count_delta = delta.ring_count_target - delta.ring_count_source
    source_systems = _ring_systems(source)
    target_systems = _ring_systems(target)
    delta.ring_system_count_source = len(source_systems)
    delta.ring_system_count_target = len(target_systems)
    delta.ring_sizes_source = sorted(len(r) for r in source.GetRingInfo().AtomRings())
    delta.ring_sizes_target = sorted(len(r) for r in target.GetRingInfo().AtomRings())
    delta.ring_topology_changed = (
        delta.ring_sizes_source != delta.ring_sizes_target
        or delta.ring_system_count_source != delta.ring_system_count_target
    )

    source_elements = _element_counts(source)
    target_elements = _element_counts(target)
    element_delta = {}
    for symbol in set(source_elements) | set(target_elements):
        difference = target_elements.get(symbol, 0) - source_elements.get(symbol, 0)
        if difference:
            element_delta[symbol] = difference
    delta.element_delta = element_delta
    delta.heteroatom_changes = sum(
        abs(v) for k, v in element_delta.items() if k not in ("C",)
    )

    delta.charge_source = _net_charge(source)
    delta.charge_target = _net_charge(target)
    delta.charge_delta = delta.charge_target - delta.charge_source

    delta.scaffold_source = _scaffold(source)
    delta.scaffold_target = _scaffold(target)
    delta.scaffold_retained = bool(
        delta.scaffold_source and delta.scaffold_source == delta.scaffold_target
    )

    delta.operation_family = _classify(delta)
    return delta


# ---- Macro-template assignment ----

#: Generic, task-independent macro edit templates.  Each is a predicate over a
#: ``GlobalDelta`` only -- no target identity, no task name, no fragment content.
def assign_template(delta: GlobalDelta, *, small: int = 2, large: int = 6) -> str:
    """Assign one generic macro template.  Ordered; first match wins."""
    if delta.status != "ok":
        return "unclassified"
    excised = delta.excised_atoms
    installed = delta.installed_atoms
    changed = delta.total_changed_atoms
    ring_moved = delta.ring_count_delta != 0 or delta.ring_topology_changed

    if changed == 0:
        return "T0_core_restate" if delta.core_bond_changes or ring_moved else "T0_identity"
    if delta.multi_region:
        return "T5_multi_region_edit"
    if excised >= small and installed >= small and changed >= large:
        return "T1_large_replacement"
    if excised == 0 and installed >= large:
        return "T2_large_growth"
    if installed == 0 and excised >= large:
        return "T3_large_excision"
    if ring_moved and changed < large:
        return "T4_ring_rewire"
    return "T6_small_local_edit"


TEMPLATES = (
    "T0_identity",
    "T0_core_restate",
    "T1_large_replacement",
    "T2_large_growth",
    "T3_large_excision",
    "T4_ring_rewire",
    "T5_multi_region_edit",
    "T6_small_local_edit",
    "unclassified",
)


def census(pairs: Sequence[tuple[str, str]], **kwargs) -> list[GlobalDelta]:
    return [compute_global_delta(a, b, **kwargs) for a, b in pairs]
