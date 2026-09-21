"""A parent-conditioned prior over COMPLETE structural programs, from teacher routes.

WHAT THIS LEARNS, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------
The architecture underneath PMO is a product of two terms::

    pi_route(Z | G)        offline, teacher-trained structural SKILL
    pi_task(Z, G | A_t)    online adaptation from the scored archive

This module is the FIRST term only.  It never sees a docking score, an oracle
value, a task identity or an archive.  It is fitted from exact teacher routes --
ordered primitive rewrites that carry a drug-like source molecule to a
drug-like endpoint -- and it answers one question: *given this molecule, what
does a coherent complete structural program look like here?*

It does NOT predict value.  A previous future-value head fitted from
best-witnessed teacher routes predicted eventual score worse than the current
score did, because routes are a statement about structure and not about the
search policy that will consume them.  Nothing in this module regresses an
outcome.

THE OBJECT: Z = (R, H, alpha, D) AND A STOP
-------------------------------------------
``AGENTS.md`` fixes the representation as ``g = (R, H, alpha, D)`` -- a released
source region, a target patch, its attachment to surviving context, and the
dependency structure -- and warns that the audit "does not authorize independent
marginal heads for those coupled variables".  This module therefore factorizes
as an explicitly COUPLED chain, not as independent marginals::

    pi_route(Z | G) = p(K | G) * prod_i p(F_i | G) * p(R | G)

``K`` is the module count -- the STOP decision.  ``F_i`` is the generic module
family at position ``i``.  ``R`` is the region draw, conditioned on the parent
only because that is what the production seam exposes:
``_delete_pendant_fragment`` reaches the law through ``substituent_delete`` and
``segment_replace`` alike and does not tell it which.  ``H``, ``alpha`` and
``D`` are then realized by the production synthesizer against the CURRENT
parent -- which is the whole point, see below.

WHY THE FAMILY TERM IS A PROJECTION AND NOT A HEAD
--------------------------------------------------
A teacher route does NOT decompose into production modules.  Measured over the
T4 teacher corpus, its 160 routes carry 238 dependency-connected components at
a median of 12 primitives each, and the overwhelming majority mix several
executor rules in one component -- ``atom_delete + atom_insert + cycle_close``
alone accounts for 76 of them.  Only 65 of 238 components match the emission
signature of a single production family.  Fitting p(F | G) directly on that
labelling would throw away three quarters of the corpus and would invent a
module boundary the teacher never drew.

So the family term is not fitted; it is PROJECTED.  Two measured objects meet:

* ``p(rule | G)`` -- how the teacher's primitive-rule mass is distributed on
  molecules of this class, estimated over every primitive in the corpus, which
  is the well-powered quantity;
* ``E[f]`` -- the emission profile of each production family, MEASURED by
  compiling all thirteen families on real drug-like sources and recording the
  executor rules they actually emit (``scripts/route_prior_family_signatures.py``).

A family is then weighted by the geometric mean of teacher rule probability
under its own emissions.  Nothing here is an independent marginal head over a
coupled coordinate: the emission matrix is what couples the family name to the
primitive vocabulary the teacher actually speaks.

A CAVEAT THAT BELONGS BESIDE EVERY NUMBER THIS PRODUCES
-------------------------------------------------------
The teacher routes are COMPILED witnesses: a search reconstructed a reported
endpoint from a benchmark source.  That compiler decomposes a transformation
into a demolish-then-rebuild schedule, so the corpus rule mix is partly a
property of the compiler and not only of the chemistry.  The repository has
already paid for missing this once, when progress conditioning on a compiled
path lane turned out to encode the compiler's tie-breaking convention.  Yield
measurements are unaffected -- a program either compiles and replays or it does
not -- but any claim that this prior has learned *chemistry* rather than
*route-compiler habit* is not established by yield alone, and is not claimed
here.

SEMANTIC ROLES, NOT SOURCE FINGERPRINTS
---------------------------------------
This is the measured failure this module exists to avoid.  The v1 PMO plan
representation pinned each operand as a complete elemental-neighbourhood
fingerprint of the molecule the plan was mined from; 97.7% of refusals on
production parents were at depth 0 with
``no_atom_carries_a_required_operand_descriptor``, and only 18.0% of
(parent x excision plan) pairs passed the root certificate.  Route knowledge
stored that way does not transfer.

Every quantity this prior stores is therefore a ROLE computed on whatever
molecule is being proposed for, never an address or a fingerprint of a teacher
molecule:

* parent features are global graph statistics (size, cycle rank, bridge count,
  heteroatom fraction, capacity headroom);
* a region is described by its role ON ITS OWN PARENT -- relative size, whether
  it carries ring atoms, whether its anchor is in a ring, the order of the cut
  bond, whether it carries heteroatoms;
* a family is one of the thirteen production module names.

There is no atom identity, no neighbour histogram, no slot index and no teacher
SMILES anywhere in a fitted payload.  :func:`assert_payload_is_address_free`
enforces that, and a test mutates a stored table to prove the check bites.

WHY A CONDITIONAL LOGIT FOR THE REGION
--------------------------------------
A teacher event is not "this region is good in the abstract"; it is "out of the
regions THIS parent offered, the teacher took this one".  Modelling the marginal
frequency of a role would confound the teacher's preference with how often that
role simply occurs in the support -- the repository's standing rule that a rare
cell whose corpus share matches its substrate is faithful, not defective.  A
conditional logit over each event's own realized support divides the substrate
frequency out by construction, and costs a handful of parameters, which is the
right scale for a corpus of this size.

THE LAW INTERFACE
-----------------
:class:`RouteProgramPrior` implements ``order(graph, rng)``, so it is accepted
wherever ``dynamic_program_synthesis`` takes ``region_law=``.  It returns the
FULL support in weighted order -- never a truncation -- and every weight is
floored, so like :class:`~compose_v4.control.bridge_region_law.BridgeRegionLaw`
it is a RE-RANKING and never a filter.  A learned law that deleted part of the
support could not be compared against the uniform control on equal support.

Unlike ``free_gate_margin_v1`` this law needs no similarity reference and no
``delta``.  That is deliberate: PMO has neither, so a law defined by a free
similarity gate cannot serve it, and a law defined by learned structural roles
can.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from compose_v4.chem.molecular_graph import MolecularGraph, is_element
from compose_v4.control.bridge_region_law import BridgeRegion, bridge_separated_regions
from compose_v4.control.dependency_region_program import (
    DependencyRegionConfig,
    dependency_region_program,
    trace_structure,
)

SCHEMA_VERSION = "route_program_prior_v1"

#: Minimum relative draw weight. Mirrors ``BridgeRegionLaw.SUPPORT_FLOOR``: it
#: is what makes a learned tilt a re-ranking of the support rather than a
#: filter on it.
SUPPORT_FLOOR = 0.05

#: Additive smoothing for every categorical table. Small relative to the
#: corpus, large enough that an unobserved (K, position, family) cell keeps
#: positive mass -- a zero there would be a silent filter on the family
#: vocabulary.
CATEGORICAL_ALPHA = 0.5

#: L2 penalty on the region logit. The corpus is a few hundred events over
#: fifteen source molecules; an unpenalized logit would fit the fold.
REGION_L2 = 1.0

#: Largest module count the prior will propose. Matches
#: ``dynamic_program_synthesis.MAX_GENERIC_MODULES``.
MAX_MODULE_COUNT = 8


class RouteProgramPriorError(ValueError):
    """The corpus, the payload or the parent cannot carry this prior."""


# ---- Parent features -----------------------------------------------------


def _adjacency(graph: MolecularGraph) -> dict[int, set[int]]:
    real = {int(i) for i in np.flatnonzero(is_element(graph.atom_types))}
    return {
        slot: {int(j) for j in np.flatnonzero(graph.bonds[slot]) if int(j) in real}
        for slot in real
    }


def _ring_atoms(graph: MolecularGraph) -> set[int]:
    """Slots lying on a cycle, by iterated leaf peeling.

    Peeling degree-one vertices from a graph leaves exactly the union of its
    cycles and the paths between them; repeating until nothing peels leaves the
    2-core, whose vertices are precisely the ring atoms of a connected
    molecular graph. No RDKit call, so this is kernel-independent and cheap.
    """

    adjacency = {slot: set(nbrs) for slot, nbrs in _adjacency(graph).items()}
    changed = True
    while changed:
        changed = False
        for slot in [s for s, nbrs in adjacency.items() if len(nbrs) <= 1]:
            for other in adjacency[slot]:
                adjacency[other].discard(slot)
            del adjacency[slot]
            changed = True
    return set(adjacency)


def parent_features(graph: MolecularGraph) -> dict[str, float]:
    """Global graph statistics of the molecule being proposed for.

    Every entry is a property of THIS molecule. None of them can carry an atom
    identity, a slot address or anything traceable to a teacher molecule.
    """

    adjacency = _adjacency(graph)
    heavy = len(adjacency)
    if heavy == 0:
        raise RouteProgramPriorError("a parent with no real atoms has no features")
    bonds = sum(len(nbrs) for nbrs in adjacency.values()) // 2
    ring = _ring_atoms(graph)
    types = graph.atom_types
    carbon = sum(1 for slot in adjacency if int(types[slot]) == 2)
    return {
        "heavy_atoms": float(heavy),
        "cycle_rank": float(bonds - heavy + 1),
        "ring_atom_fraction": len(ring) / heavy,
        "heteroatom_fraction": (heavy - carbon) / heavy,
        "mean_degree": (2.0 * bonds) / heavy,
        "capacity_headroom": float(max(0, 40 - heavy)),
    }


#: Coarse parent classes. Fifteen training source molecules cannot support a
#: fine conditioning grid, and a context that never occurs in training would
#: back off to the pooled table anyway -- so the grid is deliberately two-way.
def parent_context(graph: MolecularGraph) -> tuple[str, str]:
    """The conditioning class of a parent: (size band, capacity band)."""

    features = parent_features(graph)
    heavy = features["heavy_atoms"]
    if heavy < 24:
        size = "small"
    elif heavy < 33:
        size = "medium"
    else:
        size = "large"
    capacity = "tight" if features["capacity_headroom"] < 5 else "open"
    return size, capacity


# ---- Region roles --------------------------------------------------------

#: Names of the region logit's features, in the order
#: :func:`region_role_features` emits them. Stored in the payload so a fitted
#: coefficient vector can never be silently reinterpreted against a different
#: feature order.
REGION_FEATURE_NAMES = (
    "relative_size",
    "log_size",
    "fragment_has_ring",
    "anchor_in_ring",
    "cut_bond_is_single",
    "fragment_has_heteroatom",
    "fragment_is_single_atom",
    "fragment_all_carbon",
)


def region_role_features(
    graph: MolecularGraph,
    region: BridgeRegion,
    *,
    ring: set[int] | None = None,
    heavy: int | None = None,
) -> np.ndarray:
    """The role a region plays ON ITS OWN PARENT, as a feature vector.

    Deliberately scale-relative and label-free: ``relative_size`` is a fraction
    of this parent, not an absolute atom count copied from a teacher molecule,
    so the same role transfers between molecules of different size.
    """

    if ring is None:
        ring = _ring_atoms(graph)
    if heavy is None:
        heavy = int(np.count_nonzero(is_element(graph.atom_types)))
    fragment = region.fragment
    size = len(fragment)
    types = graph.atom_types
    hetero = any(int(types[slot]) != 2 for slot in fragment)
    return np.array(
        [
            size / float(heavy),
            math.log1p(size),
            1.0 if any(slot in ring for slot in fragment) else 0.0,
            1.0 if int(region.anchor) in ring else 0.0,
            1.0 if int(region.bond_order) == 1 else 0.0,
            1.0 if hetero else 0.0,
            1.0 if size == 1 else 0.0,
            0.0 if hetero else 1.0,
        ],
        dtype=float,
    )


# ---- Teacher extraction --------------------------------------------------


@dataclass(frozen=True)
class RegionChoice:
    """One teacher decision: out of this parent's support, it took this one."""

    source_id: str
    context: tuple[str, str]
    features: np.ndarray
    support: np.ndarray
    chosen: int

    def __post_init__(self) -> None:
        if not 0 <= self.chosen < len(self.support):
            raise RouteProgramPriorError("chosen region index is outside the support")


@dataclass(frozen=True)
class ProgramShape:
    """One teacher route's module-level shape: how many, and of what kind."""

    source_id: str
    context: tuple[str, str]
    module_count: int
    rule_counts: dict[str, int]
    primitive_count: int


#: Executor rules the production module families can emit. Fixed here so a
#: payload that names anything else is refused rather than silently ignored.
RULE_VOCABULARY = (
    "atom_delete",
    "atom_insert",
    "atom_restate_semantic",
    "bond_reorder",
    "bond_reroute",
    "cycle_close",
    "cycle_open",
    "ring_system_restate",
)


def normalize_emission_profile(profile: dict[str, dict[str, float]]) -> dict:
    """Validate and row-normalize a MEASURED family emission matrix."""

    out: dict[str, dict[str, float]] = {}
    for family, counts in profile.items():
        unknown = set(counts) - set(RULE_VOCABULARY)
        if unknown:
            raise RouteProgramPriorError(
                f"emission profile for {family!r} names unknown rules {sorted(unknown)}"
            )
        total = float(sum(float(v) for v in counts.values()))
        if total <= 0.0:
            raise RouteProgramPriorError(
                f"emission profile for {family!r} records no emitted rule"
            )
        out[str(family)] = {
            rule: float(counts.get(rule, 0.0)) / total for rule in RULE_VOCABULARY
        }
    if not out:
        raise RouteProgramPriorError("an emission profile cannot be empty")
    return out


def route_events(
    states: tuple[dict, ...],
    actions: tuple[dict, ...],
    *,
    source_id: str,
) -> tuple[tuple[RegionChoice, ...], ProgramShape]:
    """Decompose one exact teacher route into its supervisable decisions.

    Returns the region choices the route makes and the module-level shape of
    the whole program. ``states``/``actions`` are an exact primitive trace; the
    decomposition into dependency-connected components is the production one,
    so a component here is the same object the runtime calls a module.
    """

    program = dependency_region_program(states, actions, DependencyRegionConfig())
    structure = trace_structure(states, actions)
    graphs = structure["graphs"]
    choices: list[RegionChoice] = []
    rules: Counter = Counter()
    for component in program["components"]:
        indices = component["primitive_indices"]
        start, stop = int(indices[0]), int(indices[-1])
        before_graph = graphs[start]
        before = {int(i) for i in np.flatnonzero(is_element(before_graph.atom_types))}
        after = {int(i) for i in np.flatnonzero(is_element(graphs[stop + 1].atom_types))}
        released = before - after
        for rule, count in (component.get("rule_counts") or {}).items():
            if str(rule) in RULE_VOCABULARY:
                rules[str(rule)] += int(count)
        if not released:
            continue
        support = bridge_separated_regions(before_graph, maximum=None)
        if not support:
            continue
        chosen = _containing_region(support, released)
        if chosen is None:
            continue
        context = parent_context(before_graph)
        ring = _ring_atoms(before_graph)
        heavy = len(before)
        features = np.stack(
            [
                region_role_features(before_graph, region, ring=ring, heavy=heavy)
                for region in support
            ]
        )
        choices.append(
            RegionChoice(
                source_id=source_id,
                context=context,
                features=features[chosen],
                support=features,
                chosen=chosen,
            )
        )
    shape = ProgramShape(
        source_id=source_id,
        context=parent_context(graphs[0]),
        module_count=int(program["component_count"]),
        rule_counts=dict(rules),
        primitive_count=int(program["primitive_transitions"]),
    )
    return tuple(choices), shape


def _containing_region(
    support: tuple[BridgeRegion, ...], released: set[int]
) -> int | None:
    """Index of the SMALLEST bridge region containing every released atom.

    Teacher releases are bridge-coherent but not always whole: measured over the
    T4 teacher corpus, 72 of 214 releasing components remove a bridge region
    exactly and a further 141 remove a strict subset of one, so 213 of 214 are
    contained in a single region. The containing region is therefore the right
    supervision target for a law that chooses WHERE -- how much of it to remove
    is the module's own parameter, not the law's.
    """

    best: int | None = None
    best_size = None
    for position, region in enumerate(support):
        fragment = set(region.fragment)
        if not released <= fragment:
            continue
        if best_size is None or len(fragment) < best_size:
            best, best_size = position, len(fragment)
    return best


# ---- Fitting -------------------------------------------------------------


def _conditional_logit(
    choices: list[RegionChoice], *, l2: float, iterations: int = 2000
) -> tuple[np.ndarray, dict]:
    """Maximum-likelihood coefficients of a softmax over each event's support.

    Each observation contributes ``log p(chosen) = theta.phi_chosen -
    logsumexp(theta.phi_support)``. The gradient is the chosen feature vector
    minus the model's expected feature vector under that event's own support,
    which is why substrate frequency divides out: a role that dominates the
    support raises the expectation exactly as much as it raises the observation.

    The objective is the SUM of log-likelihoods less ``l2 * ||theta||^2``, so
    the penalty scales against the corpus rather than against a per-event mean;
    an earlier mean-normalized form let ``l2`` dominate and returned
    coefficients two orders of magnitude too small.
    """

    if not choices:
        raise RouteProgramPriorError("a region logit needs at least one choice")
    width = len(REGION_FEATURE_NAMES)
    supports = [choice.support for choice in choices]
    chosen = np.stack([choice.features for choice in choices])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        value = -l2 * float(theta @ theta)
        gradient = -2.0 * l2 * theta
        for position, support in enumerate(supports):
            scores = support @ theta
            shift = scores.max()
            exponentials = np.exp(scores - shift)
            total = exponentials.sum()
            value += float(scores[choices[position].chosen] - (shift + math.log(total)))
            probabilities = exponentials / total
            gradient += chosen[position] - probabilities @ support
        return value, gradient

    theta = np.zeros(width, dtype=float)
    value, gradient = objective(theta)
    step = 1.0 / max(1.0, len(choices))
    for _ in range(iterations):
        candidate = theta + step * gradient
        candidate_value, candidate_gradient = objective(candidate)
        if candidate_value < value:
            step *= 0.5
            if step < 1e-14:
                break
            continue
        theta, value, gradient = candidate, candidate_value, candidate_gradient
        step *= 1.2
        if float(np.linalg.norm(gradient)) < 1e-7:
            break
    return theta, {
        "log_likelihood": float(value),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "events": len(choices),
    }


def _smoothed(counts: Counter, vocabulary: tuple[str, ...], alpha: float) -> dict:
    total = sum(counts.values()) + alpha * len(vocabulary)
    return {name: (counts.get(name, 0) + alpha) / total for name in vocabulary}


@dataclass(frozen=True)
class RouteProgramPrior:
    """The fitted prior. Implements the production region-law ``order``."""

    region_coefficients: np.ndarray
    region_feature_names: tuple[str, ...]
    module_count: dict[str, list[float]]
    rule_table: dict[str, dict[str, float]]
    emission_profile: dict[str, dict[str, float]]
    training_sources: tuple[str, ...]
    training_identity: str
    floor: float = SUPPORT_FLOOR
    telemetry: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if tuple(self.region_feature_names) != REGION_FEATURE_NAMES:
            raise RouteProgramPriorError(
                "fitted region coefficients were built against a different "
                "feature order and cannot be scored against this one"
            )
        if not 0.0 < self.floor <= 1.0:
            raise RouteProgramPriorError("support floor must lie in (0, 1]")
        if "*" not in self.rule_table:
            raise RouteProgramPriorError("rule table has no pooled fallback")
        if "*" not in self.module_count:
            raise RouteProgramPriorError("module-count table has no pooled fallback")

    @property
    def family_vocabulary(self) -> tuple[str, ...]:
        return tuple(sorted(self.emission_profile))

    # -- the coordinates ---------------------------------------------------

    def module_count_probabilities(self, graph: MolecularGraph) -> np.ndarray:
        """p(K | G) over ``1..len(table)``; the learned STOP decision."""

        key = _context_key(parent_context(graph))
        table = self.module_count.get(key) or self.module_count["*"]
        return np.asarray(table, dtype=float)

    def sample_module_count(self, graph: MolecularGraph, rng) -> int:
        probabilities = self.module_count_probabilities(graph)
        return int(rng.choice(np.arange(1, len(probabilities) + 1), p=probabilities))

    def rule_probabilities(self, graph: MolecularGraph) -> dict[str, float]:
        """p(rule | G) -- the teacher's primitive vocabulary on this class."""

        key = _context_key(parent_context(graph))
        return self.rule_table.get(key) or self.rule_table["*"]

    def family_weights(self, graph: MolecularGraph) -> dict[str, float]:
        """Production-family weights, PROJECTED through the emission matrix.

        ``w_f = exp(sum_r E[f](r) * log p(r | G))`` -- the geometric mean of
        teacher rule probability under family ``f``'s own measured emissions.
        A family whose rules the teacher never uses on molecules of this class
        is down-weighted; it is never removed, because the emission rows and the
        rule table are both smoothed and strictly positive.
        """

        rules = self.rule_probabilities(graph)
        scores = {}
        for family, emissions in self.emission_profile.items():
            scores[family] = sum(
                weight * math.log(max(rules.get(rule, 0.0), 1e-12))
                for rule, weight in emissions.items()
                if weight > 0.0
            )
        best = max(scores.values())
        weights = {family: math.exp(score - best) for family, score in scores.items()}
        total = sum(weights.values())
        return {family: weight / total for family, weight in weights.items()}

    def sample_family_sequence(
        self, graph: MolecularGraph, rng, *, module_count: int
    ) -> tuple[str, ...]:
        """``module_count`` families drawn WITH replacement from the projection.

        With replacement, because the production synthesizer accepts a repeated
        family and the capacity-bootstrap pair it names explicitly
        (``substituent_delete`` twice) is one such sequence.
        """

        weights = self.family_weights(graph)
        names = np.asarray(sorted(weights))
        probabilities = np.array([weights[name] for name in names], dtype=float)
        probabilities /= probabilities.sum()
        return tuple(
            str(name) for name in rng.choice(names, size=module_count, p=probabilities)
        )

    # -- the law interface -------------------------------------------------

    def regions(self, graph: MolecularGraph) -> tuple[BridgeRegion, ...]:
        return bridge_separated_regions(graph, maximum=None)

    def region_weights(self, graph: MolecularGraph, regions) -> np.ndarray:
        """Strictly positive relative draw weight for every region offered."""

        if len(regions) == 0:
            return np.zeros(0, dtype=float)
        ring = _ring_atoms(graph)
        heavy = int(np.count_nonzero(is_element(graph.atom_types)))
        features = np.stack(
            [
                region_role_features(graph, region, ring=ring, heavy=heavy)
                for region in regions
            ]
        )
        scores = features @ self.region_coefficients
        scores -= scores.max()
        return np.maximum(self.floor, np.exp(scores))

    def weights(self, graph: MolecularGraph, regions) -> np.ndarray:
        return self.region_weights(graph, regions)

    def order(self, graph: MolecularGraph, rng) -> list[BridgeRegion]:
        """The full support in weighted draw order, sampled without replacement.

        Efraimidis-Spirakis, exactly as ``BridgeRegionLaw.order``: the head is
        one draw from the law and the tail is the law conditioned on that draw
        having been rejected by the executor. Returning the full support -- not
        a top-k -- is what keeps this a re-ranking, so the learned and uniform
        arms are compared on identical support.
        """

        regions = self.regions(graph)
        if not regions:
            return []
        weights = self.region_weights(graph, regions)
        keys = -np.log(np.clip(rng.random(len(regions)), 1e-300, 1.0)) / weights
        return [regions[int(at)] for at in np.argsort(keys, kind="stable")]

    # -- serialization -----------------------------------------------------

    def to_payload(self) -> dict:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "region_coefficients": [float(x) for x in self.region_coefficients],
            "region_feature_names": list(self.region_feature_names),
            "module_count": {
                k: [float(x) for x in v] for k, v in sorted(self.module_count.items())
            },
            "rule_table": {
                k: {n: float(p) for n, p in sorted(v.items())}
                for k, v in sorted(self.rule_table.items())
            },
            "emission_profile": {
                k: {n: float(p) for n, p in sorted(v.items())}
                for k, v in sorted(self.emission_profile.items())
            },
            "training_sources": sorted(self.training_sources),
            "training_identity": self.training_identity,
            "floor": float(self.floor),
            "telemetry": dict(self.telemetry),
        }
        assert_payload_is_address_free(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> RouteProgramPrior:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RouteProgramPriorError(
                f"payload schema {payload.get('schema_version')!r} is not {SCHEMA_VERSION!r}"
            )
        assert_payload_is_address_free(payload)
        return cls(
            region_coefficients=np.asarray(payload["region_coefficients"], dtype=float),
            region_feature_names=tuple(payload["region_feature_names"]),
            module_count={k: list(v) for k, v in payload["module_count"].items()},
            rule_table={k: dict(v) for k, v in payload["rule_table"].items()},
            emission_profile={k: dict(v) for k, v in payload["emission_profile"].items()},
            training_sources=tuple(payload["training_sources"]),
            training_identity=str(payload["training_identity"]),
            floor=float(payload["floor"]),
            telemetry=dict(payload.get("telemetry") or {}),
        )


def _context_key(context: tuple[str, str]) -> str:
    return "/".join(context)


_CONTEXT_ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz0123456789/|*")


def _assert_context_key(key: str) -> None:
    if key == "*":
        return
    if not set(str(key)) <= _CONTEXT_ALPHABET:
        raise RouteProgramPriorError(
            f"context key {key!r} carries characters a parent class cannot"
        )


def _is_opaque_identity(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def assert_payload_is_address_free(payload: dict) -> None:
    """Refuse a payload that stores anything traceable to a teacher molecule.

    The v1 PMO plan representation failed precisely by storing source-side
    descriptors -- 97.7% of its refusals on production parents were at depth 0
    for want of an atom carrying a required operand descriptor -- so "this prior
    stores only roles" has to be an ENFORCED property and not a comment. A SMILES
    string, an atom index or an undeclared key is a defect, not a curiosity.
    """

    for name in payload.get("region_feature_names", []):
        if name not in REGION_FEATURE_NAMES:
            raise RouteProgramPriorError(f"undeclared region feature {name!r}")
    for key, table in (payload.get("rule_table") or {}).items():
        _assert_context_key(key)
        unknown = set(table) - set(RULE_VOCABULARY)
        if unknown:
            raise RouteProgramPriorError(
                f"rule table cell {key!r} names non-rules {sorted(unknown)}"
            )
    for family, emissions in (payload.get("emission_profile") or {}).items():
        if not set(str(family)) <= frozenset(
            "abcdefghijklmnopqrstuvwxyz_0123456789"
        ):
            raise RouteProgramPriorError(f"emission profile key {family!r} is not a family name")
        unknown = set(emissions) - set(RULE_VOCABULARY)
        if unknown:
            raise RouteProgramPriorError(
                f"emission profile {family!r} names non-rules {sorted(unknown)}"
            )
    for key in payload.get("module_count") or {}:
        _assert_context_key(key)
    for source in payload.get("training_sources") or ():
        if not _is_opaque_identity(str(source)):
            raise RouteProgramPriorError(
                "training_sources must be opaque identities, not molecules; "
                f"got {str(source)[:40]!r}"
            )


def fit_route_program_prior(
    choices: list[RegionChoice],
    shapes: list[ProgramShape],
    *,
    training_identity: str,
    emission_profile: dict[str, dict[str, float]],
    alpha: float = CATEGORICAL_ALPHA,
    l2: float = REGION_L2,
    floor: float = SUPPORT_FLOOR,
    max_module_count: int = MAX_MODULE_COUNT,
) -> RouteProgramPrior:
    """Fit p(K|G), p(rule|G) and the region logit from teacher decisions."""

    if not shapes:
        raise RouteProgramPriorError("a program prior needs at least one route")
    emissions = normalize_emission_profile(emission_profile)
    sources = sorted(
        {shape.source_id for shape in shapes} | {choice.source_id for choice in choices}
    )

    module_buckets: dict[str, Counter] = defaultdict(Counter)
    rule_buckets: dict[str, Counter] = defaultdict(Counter)
    for shape in shapes:
        key = _context_key(shape.context)
        capped = min(int(shape.module_count), max_module_count)
        module_buckets[key][capped] += 1
        module_buckets["*"][capped] += 1
        for rule, count in shape.rule_counts.items():
            if rule in RULE_VOCABULARY:
                rule_buckets[key][rule] += int(count)
                rule_buckets["*"][rule] += int(count)

    module_count = {}
    for key, counts in module_buckets.items():
        total = sum(counts.values()) + alpha * max_module_count
        module_count[key] = [
            (counts.get(k, 0) + alpha) / total for k in range(1, max_module_count + 1)
        ]
    rule_table = {
        key: _smoothed(counts, RULE_VOCABULARY, alpha)
        for key, counts in rule_buckets.items()
    }

    if choices:
        coefficients, fit_telemetry = _conditional_logit(choices, l2=l2)
    else:
        coefficients = np.zeros(len(REGION_FEATURE_NAMES))
        fit_telemetry = {"log_likelihood": 0.0, "gradient_norm": 0.0, "events": 0}

    telemetry = {
        "region_choices": len(choices),
        "program_shapes": len(shapes),
        "training_source_count": len(sources),
        "primitive_observations": int(sum(rule_buckets["*"].values())),
        "region_logit": fit_telemetry,
    }
    return RouteProgramPrior(
        region_coefficients=coefficients,
        region_feature_names=REGION_FEATURE_NAMES,
        module_count=module_count,
        rule_table=rule_table,
        emission_profile=emissions,
        training_sources=tuple(sources),
        training_identity=training_identity,
        floor=floor,
        telemetry=telemetry,
    )
