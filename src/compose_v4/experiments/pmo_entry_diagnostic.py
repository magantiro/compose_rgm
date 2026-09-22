"""Teacher-atlas ENTRY diagnostic: a hidden, zero-oracle evaluator of global search.

Test A measured that COMPOSE can CONSTRUCT the atlas destinations, Test B that a
frozen local controller started AT one is productive, and Test C that blind
search never reaches one.  This module asks the question those three leave open,
and asks it without spending a single charged call:

    starting from the actual parents a BLIND run visited, does the general
    proposal mechanism ever generate a molecule inside a known productive
    region, and with what probability as a function of the number of proposals?

Data structures
---------------
``ProductiveRegion``
    The reference molecules of one task.  ``primary`` is the single molecule
    Test B measured as productive (the spine anchor, which equals the task's
    declared destination); ``wide`` adds every recorded route endpoint, whose
    productivity was NOT measured and which is therefore a sensitivity arm only.

``EntryPredicate``
    The committed structural criterion.  A molecule ENTERS a region when its
    Morgan(r=2, 2048-bit, no chirality) Tanimoto similarity to some reference
    reaches ``delta``.  The predicate is frozen in
    ``diagnostics/pmo_entry_diagnostic_v1/predicate_v1.json`` and this module
    refuses to score against a file whose payload hash has moved.

``ProposalArm``
    One registered proposal source: ``(source_state, rng, draws) -> Proposal``
    stream.  Adding arm 2/3/4 is a REGISTRATION; nothing in the measurement,
    the predicate or the reporting changes.

Invariants maintained and tested
--------------------------------
* Parent states are decoded from the blind run's 48-slot payloads, never
  re-parsed from SMILES -- a SMILES round trip yields a tight graph and
  silently deletes the whole ``atom_insert`` family from the legal support.
* Every seed is a BLAKE2b digest of declared strings; ``hash()`` is never used.
* A refused draw COUNTS as a proposal.  ``P(entry by N)`` is over draws the
  mechanism made, not over draws that happened to execute; the executed-only
  curve is reported beside it rather than in place of it.
* Similarity is recorded per proposal, so a null at the committed ``delta``
  stays informative: the whole threshold sweep is a reduction of stored data,
  never a re-measurement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

# ---- Frozen predicate ----

#: Morgan fingerprint used everywhere in this module.  ``useChirality`` is left
#: at its default False, matching the PMO oracles (measured: stereochemistry
#: costs exactly zero on the similarity oracles) and matching
#: ``pmo_atlas_discovery._fingerprint``, which a test pins.
FINGERPRINT = {"kind": "morgan", "radius": 2, "n_bits": 2048, "use_chirality": False}

#: The committed entry threshold.  Justification, all computed BEFORE any arm
#: was measured and reproducible from ``calibration`` in the sealed predicate:
#:   * 0 of 1,100 objective-blind init-bank x answer pairs and 0 of 55
#:     cross-answer pairs reach 0.30 (measured maxima 0.2159 and 0.2034), so a
#:     random drug-like molecule does not satisfy it;
#:   * 0.30 is at or below the MEDIAN similarity (0.330) of the ``near_anchor``
#:     spine position to its own anchor, and Test B measured that position as
#:     productive -- so the predicate is no stricter at the median than a
#:     position already shown to work.
ENTRY_DELTA = 0.30

#: Reported beside the committed value so the headline never rests on one
#: number.  These are reductions of stored per-proposal similarities.
DELTA_SWEEP = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.85)

#: The graded ladder a proposal can climb.  Every threshold is anchored to a
#: measured quantity rather than chosen for roundness:
#:   above_chance      0.25  -- above the maximum of 1,100 objective-blind
#:                             init-bank x answer pairs (0.2159) and of 55
#:                             cross-answer pairs (0.2034)
#:   entry             0.30  -- the committed predicate; at or below the median
#:                             near_anchor->anchor similarity (0.330), and Test
#:                             B measured near_anchor as productive
#:   near_anchor_band  0.40  -- above that median, below its maximum (0.4906)
#:   anchor_band       0.60  -- twice the entry threshold; no atlas ladder
#:                             position outside the anchor itself reaches it
#:   anchor            0.85  -- effectively the destination
RUNGS: tuple[tuple[str, float], ...] = (
    ("above_chance", 0.25),
    ("entry", 0.30),
    ("near_anchor_band", 0.40),
    ("anchor_band", 0.60),
    ("anchor", 0.85),
)

#: Matched proposal prefixes every arm is reported at.  A production 250-call
#: blind run issued ~1,091 proposals, so 1,000 is one run's worth of proposal
#: work and the three points span it.
MATCHED_PROPOSALS = (100, 500, 1000)

#: Predeclared, written into the sealed predicate BEFORE any arm number exists.
PASS_CRITERION = {
    "decides": (
        "whether the global-search work (arms 2-4, collectively C) is scored. "
        "A null here is decisive and cheap: it says the mechanism does not do "
        "the thing it was built to do."
    ),
    "primary_tasks": ("celecoxib_rediscovery", "albuterol_similarity", "jnk3"),
    "control_task": "qed",
    "rule": (
        "C PASSES when, on at least 2 of the 3 primary tasks, C's best rung at "
        "1,000 matched proposals is at least ONE named rung above B's best rung "
        "at 1,000 matched proposals, AND qed is not degraded."
    ),
    "qed_not_degraded": (
        "C's best rung on qed at 1,000 proposals is not below B's, and C's entry "
        "rate on qed at 1,000 proposals is at least B's."
    ),
    "rungs": [list(rung) for rung in RUNGS],
    "matched_proposals": list(MATCHED_PROPOSALS),
    "why_not_narrow_versus_broad": (
        "Test C measured the discovery gap as UNIVERSAL: jnk3 reached 38.2% of "
        "anchor and perindopril_mpo 38.4%, both broad-objective, sitting with "
        "narrow celecoxib at 37.2%. The primary set therefore mixes two "
        "rediscovery tasks with one broad-objective task, and no task other "
        "than qed is treated as expected-easy. qed is the control precisely "
        "because it was already 0.9016 at INITIALIZATION."
    ),
}

#: The five tasks this gate is run on.
GATE_TASKS = (
    "celecoxib_rediscovery",
    "albuterol_similarity",
    "jnk3",
    "perindopril_mpo",
    "qed",
)

PREDICATE_SCHEMA = "pmo_entry_predicate_v1"
DIAGNOSTIC_SCHEMA = "pmo_entry_diagnostic_v1"

#: This instrument reads answer-known destinations.  It is offline, structural
#: and never supplies anything to an optimizer.
REGIME = "DEVELOPMENT_INFORMED_DIAGNOSTIC"
REGIME_STATEMENT = (
    "Offline structural evaluator. The atlas destinations are published task "
    "answers; they are read only to SCORE proposals after the fact and are "
    "never passed to the proposal mechanism, an initialization, a library or a "
    "prior. Zero charged oracle calls."
)


def payload_sha256(payload: Any) -> str:
    """Stable content hash of a JSON payload."""

    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def stable_seed(*parts: str) -> int:
    """A reproducible 63-bit seed from declared strings.

    ``hash()`` is PYTHONHASHSEED-salted and has already made one table in this
    repository irreproducible; BLAKE2b is stable across processes and machines.
    """

    digest = hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


# ---- Chemistry helpers ----


def fingerprint(smiles: str):
    """Morgan bit vector, or None when RDKit refuses the string."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        molecule, FINGERPRINT["radius"], nBits=FINGERPRINT["n_bits"]
    )


def canonical(smiles: str) -> str | None:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else Chem.MolToSmiles(molecule)


def drug_likeness(smiles: str) -> dict[str, Any]:
    """QED and heavy-atom count of one molecule.

    The manifold column.  Blind PMO search leaves the drug-like manifold on 10
    of 11 tasks (``diagnostics/pmo_atlas_v1/manifold_drift_v1.json``) while the
    productive regions sit on it, so an arm that lifts entry while its own
    population keeps drifting has probably found a different kind of region.

    DIAGNOSTIC ONLY.  This is computed AFTER the fact, on proposals nothing
    selected with it, and is never returned to a proposal mechanism.  On the
    ``qed`` task it coincides with the task objective and must not be read there
    as an axis independent of the score.
    """

    from rdkit import Chem
    from rdkit.Chem import QED

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"qed": None, "heavy_atoms": None}
    try:
        value = float(QED.qed(molecule))
    except Exception:  # noqa: BLE001 - a descriptor failure is data, not a crash
        value = None
    return {"qed": value, "heavy_atoms": molecule.GetNumHeavyAtoms()}


#: Median QED of the objective-blind initialization, measured in the drift
#: artifact. Every trajectory in this harness starts from a blind parent whose
#: own QED is reported beside it.
INIT_MEDIAN_QED = 0.716


def murcko_scaffold(smiles: str) -> str | None:
    """Bemis-Murcko scaffold SMILES, or None when the molecule has no ring."""

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold is None or scaffold.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold)


# ---- Productive regions ----


@dataclass(frozen=True)
class ProductiveRegion:
    """The reference molecules of one task, split by what was MEASURED.

    ``primary`` is the spine anchor: the exact molecule Test B seeded and
    measured.  ``wide`` adds sibling route endpoints, which were never probed,
    so a ``wide`` number is a sensitivity and never the headline.
    """

    task: str
    primary: tuple[str, ...]
    wide: tuple[str, ...]
    anchor_top_ten: float | None = None
    _prints: dict[str, list] = field(default_factory=dict, repr=False, compare=False)

    def references(self, scope: str) -> tuple[str, ...]:
        if scope == "primary":
            return self.primary
        if scope == "wide":
            return self.wide
        raise ValueError(f"unknown region scope: {scope}")

    def prints(self, scope: str) -> list:
        if scope not in self._prints:
            built = []
            for smiles in self.references(scope):
                bits = fingerprint(smiles)
                if bits is None:
                    raise ValueError(f"unreadable atlas reference for {self.task}: {smiles}")
                built.append(bits)
            self._prints[scope] = built
        return self._prints[scope]

    def approach(self, smiles: str, scope: str = "primary") -> tuple[float, str | None]:
        """Maximum Tanimoto to the region, with the nearest reference."""

        from rdkit import DataStructs

        bits = fingerprint(smiles)
        if bits is None:
            return 0.0, None
        similarities = DataStructs.BulkTanimotoSimilarity(bits, self.prints(scope))
        best = max(similarities)
        return float(best), self.references(scope)[similarities.index(best)]


def load_atlas_payload(repo_root: Path | str) -> dict[str, Any]:
    """Read ``atlas.json`` and refuse it if its own recorded hash has moved."""

    path = Path(repo_root) / "diagnostics/pmo_atlas_v1/atlas.json"
    document = json.loads(path.read_text())
    payload = document["payload"]
    recorded = document.get("payload_sha256")
    if recorded is not None and payload_sha256(payload) != recorded:
        raise ValueError("the atlas payload hash has moved; refusing to score against it")
    return payload


def load_productive_regions(
    repo_root: Path | str, test_b_payload: dict[str, Any] | None = None
) -> dict[str, ProductiveRegion]:
    """Build one region per task from the sealed atlas.

    ``primary`` is the SPINE anchor, which the atlas records as identical to the
    task's declared destination and which Test B used as its anchor seed.  A
    task whose spines disagree contributes every distinct spine anchor.
    """

    payload = load_atlas_payload(repo_root)
    spine_anchors: dict[str, set[str]] = {}
    endpoints: dict[str, set[str]] = {}
    for route in payload["routes"]:
        task = route["task"]
        anchors = [c["smiles"] for c in route["checkpoints"] if c["label"] == "anchor"]
        endpoints.setdefault(task, set()).update(
            s for s in (*anchors, route.get("recorded_endpoint_smiles")) if s
        )
        if route.get("is_spine"):
            spine_anchors.setdefault(task, set()).update(anchors)
    lift: dict[str, float] = {}
    if test_b_payload is not None:
        for run in test_b_payload["runs"]:
            if run["checkpoint_label"] == "anchor":
                lift[run["task"]] = float(run["top_ten_new_mean"])
    regions: dict[str, ProductiveRegion] = {}
    for task in sorted(spine_anchors):
        primary = tuple(sorted(spine_anchors[task]))
        if not primary:
            raise ValueError(f"{task} has no spine anchor")
        regions[task] = ProductiveRegion(
            task=task,
            primary=primary,
            wide=tuple(sorted(endpoints[task] | set(primary))),
            anchor_top_ten=lift.get(task),
        )
    return regions


# ---- Parent selection ----


@dataclass(frozen=True)
class Parent:
    task: str
    endpoint: str
    score: float
    role: str
    index: int
    heavy_atoms: int | None
    score_stratum: str
    state: dict[str, Any]

    def graph(self):
        from compose_v4.rewrite.trace_shard import decode_state

        return decode_state(self.state)


def _stratum(rank_fraction: float) -> str:
    if rank_fraction < 0.25:
        return "q1_worst"
    if rank_fraction < 0.50:
        return "q2"
    if rank_fraction < 0.75:
        return "q3"
    return "q4_best"


def select_parents(rows: Sequence[Any], task: str, *, per_stratum: int) -> tuple[Parent, ...]:
    """Stratified parent sample, one deterministic draw per score quartile.

    Slice order in this repository is lane- and family-sorted and taking the
    first N has produced at least three bad measurements, so parents are drawn
    inside declared score strata with a BLAKE2b seed and the realized strata are
    reported by the caller.  The single best-scoring molecule is always
    included: it is the one parent a score-greedy controller is most likely to
    expand, so omitting it would understate the mechanism.
    """

    usable = [row for row in rows if row.state is not None]
    if not usable:
        raise ValueError(f"{task} has no blind row carrying a 48-slot state")
    ordered = sorted(usable, key=lambda row: (row.score, row.index))
    buckets: dict[str, list[Any]] = {}
    for rank, row in enumerate(ordered):
        buckets.setdefault(_stratum(rank / len(ordered)), []).append(row)
    chosen: dict[str, Any] = {}
    best = ordered[-1]
    chosen[best.endpoint] = (best, "q4_best")
    for name in sorted(buckets):
        pool = buckets[name]
        rng = np.random.default_rng(stable_seed(DIAGNOSTIC_SCHEMA, "parents", task, name))
        take = min(per_stratum, len(pool))
        for at in rng.choice(len(pool), size=take, replace=False):
            row = pool[int(at)]
            chosen.setdefault(row.endpoint, (row, name))
    parents = []
    for row, name in chosen.values():
        parents.append(
            Parent(
                task=task,
                endpoint=row.endpoint,
                score=float(row.score),
                role=row.role,
                index=int(row.index),
                heavy_atoms=row.heavy_atoms,
                score_stratum=name,
                state=row.state,
            )
        )
    parents.sort(key=lambda parent: parent.index)
    return tuple(parents)


# ---- Proposals ----


@dataclass(frozen=True)
class Proposal:
    draw: int
    status: str
    channel: str
    endpoint: str | None = None
    primitive_count: int | None = None
    heavy_atoms: int | None = None
    reason: str | None = None
    seconds: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "draw": self.draw,
            "status": self.status,
            "channel": self.channel,
            "endpoint": self.endpoint,
            "primitive_count": self.primitive_count,
            "heavy_atoms": self.heavy_atoms,
            "reason": self.reason,
            "seconds": round(self.seconds, 4),
        }


ArmFunction = Callable[[Any, np.random.Generator, int], Iterator[Proposal]]

ARM_REGISTRY: dict[str, dict[str, Any]] = {}


def register_arm(name: str, function: ArmFunction, *, describe: dict[str, Any]) -> None:
    """Register a proposal source. Adding an arm must never edit the harness."""

    if name in ARM_REGISTRY:
        raise ValueError(f"proposal arm already registered: {name}")
    ARM_REGISTRY[name] = {"name": name, "function": function, "describe": describe}


def arm(name: str) -> dict[str, Any]:
    if name not in ARM_REGISTRY:
        known = ", ".join(sorted(ARM_REGISTRY)) or "(none)"
        raise ValueError(f"unknown proposal arm {name!r}; registered: {known}")
    return ARM_REGISTRY[name]


def _execute(source, program, binding):
    from compose_v4.control.edit_program_graph import (
        compile_program_graph,
        execute_program_graph,
    )

    _, trace = execute_program_graph(
        source, compile_program_graph(program), binding, max_primitives=32, max_blocks=8
    )
    return trace


def baseline_b_proposals(source, rng, draws: int) -> Iterator[Proposal]:
    """Arm 1: the production generic proposal mechanism, parent-first.

    The two generic channels are exactly what ``_generate_channel_pool`` drives:
    ``synthesize_dynamic_program`` (shallow) and ``synthesize_structured_program``
    (structured), each bound to the supplied exact 48-slot parent state.  They
    alternate, matching production's equal per-batch attempt allocation.

    DECLARED SCOPE.  The jump lane is excluded: it needs an archive and a plan
    library that a single parent does not have, and it is measured dead in
    production (3 executions from 1,091 proposals).  The program-mutation and
    recombination sub-paths of the warm shallow channel are excluded for the
    same structural reason -- they mutate a parent's PROGRAM, which a molecule
    visited by a blind run does not carry here.
    """

    from compose_v4.control.dynamic_program_synthesis import synthesize_dynamic_program
    from compose_v4.control.dynamic_program_synthesis_v2 import (
        synthesize_structured_program,
    )

    panel_cache: dict = {}
    for index in range(1, draws + 1):
        channel = "shallow" if index % 2 else "structured"
        began = perf_counter()
        try:
            if channel == "shallow":
                built = synthesize_dynamic_program(
                    source, rng, max_modules=3, max_primitives=32, max_blocks=8
                )
            else:
                built = synthesize_structured_program(
                    source,
                    rng,
                    max_modules=3,
                    max_primitives=32,
                    max_blocks=8,
                    panel_cache=panel_cache,
                )
            trace = _execute(built[0], built[1], built[2])
        except (ValueError, RuntimeError) as error:
            yield Proposal(
                draw=index,
                status="refused",
                channel=channel,
                reason=f"{type(error).__name__}: {error}",
                seconds=perf_counter() - began,
            )
            continue
        endpoint = trace["endpoint"]
        yield Proposal(
            draw=index,
            status="executed",
            channel=channel,
            endpoint=endpoint,
            primitive_count=len(trace["actions"]),
            heavy_atoms=_heavy(endpoint),
            seconds=perf_counter() - began,
        )


def _heavy(smiles: str) -> int | None:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    return None if molecule is None else molecule.GetNumHeavyAtoms()


#: Elements the production growth modules insert.  Reused, not re-chosen: an
#: invented vocabulary is how a probe ends up asking the prior about phosphorus.
CONTROL_INSERT_ELEMENTS = ("C", "N", "O", "F")

#: Primitive walk length of the negative control, drawn uniformly.  It is
#: declared here rather than fitted to arm 1, so the control cannot be tuned
#: after the fact; the realized lengths of BOTH sources are reported and the
#: per-length control rate is published so a matched comparison is a lookup.
CONTROL_WALK_LENGTHS = (1, 2, 3, 4, 5, 6, 7, 8)


def _legal_primitive_actions(graph) -> list[tuple[str, Any]]:
    """Every legal single primitive at ``graph``, via production enumerators.

    Covers seven of the eight Active8 executor rules.  ``bond_reorder`` is
    absent because PMO's module vocabulary can never emit it; atom birth and
    death are enumerated the way the production growth and shrink modules build
    them, not by an invented parameterisation.
    """

    import numpy as _np

    from compose_v4.chem.molecular_graph import (
        ALLOWED_VALENCES,
        ELEMENT_TO_IDX,
        is_element,
    )
    from compose_v4.control.current_state_edits import ENUMERATORS
    from compose_v4.experiments.whole_ring_plan import fresh_slot
    from compose_v4.rewrite.operators import AtomDelete, AtomInsert

    found: list[tuple[str, Any]] = []
    for family, enumerate_actions in ENUMERATORS.items():
        for action in enumerate_actions(graph):
            found.append((family, action))
    real = [int(i) for i in _np.flatnonzero(is_element(graph.atom_types))]
    for slot in real:
        found.append(("atom_delete", AtomDelete(slot)))
    if graph.n_real_atoms < 40:
        for slot in real:
            if int(graph.implicit_h_counts[slot]) < 1:
                continue
            for element in CONTROL_INSERT_ELEMENTS:
                valences = ALLOWED_VALENCES[element]
                if len(valences) != 1:
                    continue
                found.append(
                    (
                        "atom_insert",
                        AtomInsert(
                            fresh_slot(graph),
                            ELEMENT_TO_IDX[element],
                            0,
                            valences[0] - 1,
                            ((slot, 1),),
                        ),
                    )
                )
    return found


def uniform_legal_edit_proposals(source, rng, draws: int) -> Iterator[Proposal]:
    """NEGATIVE CONTROL: a uniform random walk over the legal primitive fiber.

    Each draw takes a walk length uniformly from ``CONTROL_WALK_LENGTHS`` and
    then, at each step, one action uniformly at random from EVERY legal single
    primitive at the current state -- no module vocabulary, no region law, no
    completion law, no weighting.  A source that enters a productive region at
    this rate would show the entry predicate is measuring nothing.
    """

    from compose_v4.control.docking_value import identity
    from compose_v4.experiments.whole_ring_plan import execute_program
    from compose_v4.rewrite.action_codec_v4 import encode_action
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.trace_shard import encode_state

    # Enumerating the whole fiber costs ~2.4 s per state on a drug-like parent,
    # dominated by the semantic restate and cycle-close admission masks, and the
    # first step of every walk revisits the same parent.  The memo caches the
    # COMPUTATION, never the choice: the draw law is unchanged.
    memo: dict[str, list[tuple[str, Any]]] = {}

    def legal(graph):
        key = identity(encode_state(graph))
        if key not in memo:
            memo[key] = _legal_primitive_actions(graph)
        return memo[key]

    lengths = np.asarray(CONTROL_WALK_LENGTHS)
    for index in range(1, draws + 1):
        began = perf_counter()
        length = int(lengths[int(rng.integers(len(lengths)))])
        current, applied = source, 0
        reason = None
        for _ in range(length):
            actions = legal(current)
            if not actions:
                reason = "no legal primitive at the current state"
                break
            order = rng.permutation(len(actions))
            advanced = False
            for at in order[: min(len(order), 8)]:
                family, action = actions[int(at)]
                try:
                    current, _ = execute_program(
                        current, [encode_action(family, action)]
                    )
                except (ValueError, RuntimeError):
                    continue
                applied += 1
                advanced = True
                break
            if not advanced:
                reason = "every sampled legal primitive was refused by the executor"
                break
        if applied == 0:
            yield Proposal(
                draw=index,
                status="refused",
                channel="uniform_legal_walk",
                reason=reason or "walk applied no primitive",
                seconds=perf_counter() - began,
            )
            continue
        endpoint = canonical_state_key(current)
        yield Proposal(
            draw=index,
            status="executed",
            channel="uniform_legal_walk",
            endpoint=endpoint,
            primitive_count=applied,
            heavy_atoms=_heavy(endpoint),
            seconds=perf_counter() - began,
        )


register_arm(
    "arm1_baseline_b",
    baseline_b_proposals,
    describe={
        "role": "arm",
        "label": "current B -- production generic proposal mechanism",
        "channels": ("shallow", "structured"),
        "provenance_class": {
            "shallow": "local_search",
            "structured": "broad_exploration",
        },
        "functions": (
            "compose_v4.control.dynamic_program_synthesis.synthesize_dynamic_program",
            "compose_v4.control.dynamic_program_synthesis_v2.synthesize_structured_program",
        ),
        "excluded": (
            (
                "jump lane (needs an archive and a plan library; measured dead: "
                "3 executions from 1,091 production proposals)"
            ),
            (
                "program mutation and recombination (mutate a parent PROGRAM, "
                "which a blind-run molecule does not carry in this harness)"
            ),
        ),
    },
)

register_arm(
    "nc1_uniform_legal_edits",
    uniform_legal_edit_proposals,
    describe={
        "role": "negative_control",
        "label": "uniform random legal primitive walk",
        "channels": ("uniform_legal_walk",),
        "provenance_class": {"uniform_legal_walk": "uniform_random_legal_edit"},
        "walk_lengths": CONTROL_WALK_LENGTHS,
        "insert_elements": CONTROL_INSERT_ELEMENTS,
        "covers_executor_rules": (
            "atom_delete",
            "atom_insert",
            "atom_restate_semantic",
            "bond_reroute",
            "cycle_close",
            "cycle_open",
            "ring_system_restate",
        ),
        "excluded": ("bond_reorder: PMO's module vocabulary can never emit it",),
    },
)


#: The DEPLOYED controller's own proposal pool, read from a completed blind
#: run rather than regenerated.  It is not a registered generator -- it cannot
#: be drawn from -- but it is described here so the report has one place to
#: look up an arm, and its provenance tags are the controller's OWN
#: ``planner_channel`` strings, never re-derived.
PRODUCTION_ARM = "arm0_production_blind_run"

OBSERVED_ARMS: dict[str, dict[str, Any]] = {
    PRODUCTION_ARM: {
        "role": "observed_production",
        "label": "the deployed controller's own proposals, from the Test C blind runs",
        "channels": (
            "shallow_program_channel",
            "structured_program_channel",
            "joint_dependency_region_jump",
        ),
        "provenance_class": {
            "shallow_program_channel": "local_search",
            "structured_program_channel": "broad_exploration",
            "joint_dependency_region_jump": "donor_transport",
        },
        "source": (
            "~/compose_pmo_atlas_runs/test_c_blind/<task>__blind_250/campaign/"
            "round_*/pending.json"
        ),
        "why": (
            "The regenerated arm 1 is a reconstruction of the proposal mechanism "
            "from stored parents. This arm is the run itself: every attempt the "
            "controller actually made, in round order, with the charged calls "
            "marked, so a first CALL index exists here and nowhere else in this "
            "harness."
        ),
        "excluded": (),
    }
}


def arm_describe(name: str) -> dict[str, Any] | None:
    if name in ARM_REGISTRY:
        return ARM_REGISTRY[name]["describe"]
    return OBSERVED_ARMS.get(name)


# ---- Entry curves ----


def entry_rows(
    proposals: Sequence[Proposal], region: ProductiveRegion, *, scope: str = "primary"
) -> list[dict[str, Any]]:
    """Attach the structural approach of every executed proposal."""

    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        row = proposal.as_row()
        if proposal.endpoint is None:
            row["similarity"] = None
            row["nearest_reference"] = None
            row["scaffold_match"] = None
        else:
            best, nearest = region.approach(proposal.endpoint, scope)
            row["similarity"] = round(best, 6)
            row["nearest_reference"] = nearest
            row["scaffold_match"] = bool(
                murcko_scaffold(proposal.endpoint) is not None
                and murcko_scaffold(proposal.endpoint)
                in {
                    murcko_scaffold(reference)
                    for reference in region.references(scope)
                    if murcko_scaffold(reference) is not None
                }
            )
        rows.append(row)
    return rows


def first_entry_draw(rows: Sequence[dict[str, Any]], delta: float) -> int | None:
    """Draw ordinal of the first proposal that enters, counting refusals."""

    for row in rows:
        similarity = row.get("similarity")
        if similarity is not None and similarity >= delta:
            return int(row["draw"])
    return None


def entry_curve(
    per_parent: Sequence[Sequence[dict[str, Any]]], delta: float, draws: int
) -> list[dict[str, Any]]:
    """``P(entry by N proposals)`` over parents, as a curve in N.

    A parent is a trial: it enters by N when any of its first N draws enters.
    """

    firsts = [first_entry_draw(rows, delta) for rows in per_parent]
    total = len(firsts)
    curve = []
    for at in _curve_points(draws):
        entered = sum(1 for first in firsts if first is not None and first <= at)
        curve.append(
            {
                "proposals": at,
                "parents_entered": entered,
                "parents": total,
                "probability": (entered / total) if total else None,
            }
        )
    return curve


def _curve_points(draws: int) -> list[int]:
    points = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    kept = [at for at in points if at <= draws]
    if not kept or kept[-1] != draws:
        kept.append(draws)
    return kept


def best_rung(similarity: float | None) -> str | None:
    """Highest named rung a similarity reaches, or None below the ladder."""

    reached = None
    for name, threshold in RUNGS:
        if similarity is not None and similarity >= threshold:
            reached = name
    return reached


def rung_index(name: str | None) -> int:
    """Position on the ladder; -1 for "below the ladder", so rungs compare."""

    if name is None:
        return -1
    for at, (rung, _) in enumerate(RUNGS):
        if rung == name:
            return at
    raise ValueError(f"unknown rung: {name}")


def rate_bound(entries: int, draws: int, *, confidence: float = 0.95) -> float:
    """Upper bound on the per-draw entry rate. At 0 entries this is the rule of three.

    A zero is only informative with the budget beside it, so every null this
    harness reports carries this number.
    """

    if draws <= 0:
        raise ValueError("a rate bound needs a positive number of draws")
    if entries == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / draws)
    return min(1.0, (entries + 2.0 * (entries**0.5) + 3.0) / draws)


# ---- Sealed predicate ----


def predicate_payload(calibration: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PREDICATE_SCHEMA,
        "information_regime": REGIME,
        "information_regime_statement": REGIME_STATEMENT,
        "statement": (
            "A proposal ENTERS the productive region of task T when the maximum "
            "Tanimoto similarity of its Morgan(radius 2, 2048 bits, no chirality) "
            "fingerprint to the region's reference molecules is at least delta. "
            "The primary region is the SPINE ANCHOR -- the single molecule Test B "
            "seeded and measured productive, identical to the task's declared "
            "destination. The wide region adds sibling route endpoints whose "
            "productivity was never measured and is a sensitivity only."
        ),
        "fingerprint": FINGERPRINT,
        "delta": ENTRY_DELTA,
        "delta_sweep": list(DELTA_SWEEP),
        "rungs": [list(rung) for rung in RUNGS],
        "matched_proposals": list(MATCHED_PROPOSALS),
        "pass_criterion": PASS_CRITERION,
        "gate_tasks": list(GATE_TASKS),
        "diversity_rule": (
            "Distinct productive BASINS entered is the number of distinct "
            "nearest references, under the WIDE region, among entering "
            "proposals; distinct entrant Murcko scaffolds is reported beside it."
        ),
        "descendant_rule": (
            "For every entering proposal, the shallow local channel is drawn "
            "from that entrant and the best rung its descendants reach is "
            "compared with the entrant's own. With no entrant the field is "
            "reported as not evaluable, never as a zero."
        ),
        "disclosure": (
            "A 20-draw, single-parent smoke test of both registered sources ran "
            "before this seal, to size the measurement. Its maximum similarity "
            "was 0.086 (arm 1) and 0.1205 (the control), both far below every "
            "rung, and delta was already fixed at 0.30 in the module source "
            "before it ran. No entry number existed when the ladder was set."
        ),
        "secondary_predicate": "Bemis-Murcko scaffold identity with a region reference",
        "calibration": calibration,
        "sealed_before_any_arm_or_control_measurement": True,
    }


def assert_predicate_sealed(repo_root: Path | str) -> dict[str, Any]:
    """Refuse to score unless the committed predicate matches this module."""

    path = (
        Path(repo_root)
        / "diagnostics/pmo_entry_diagnostic_v1/predicate_v1.json"
    )
    if not path.is_file():
        raise ValueError(
            "the entry predicate must be sealed and committed before any arm runs"
        )
    document = json.loads(path.read_text())
    payload = document["payload"]
    if payload_sha256(payload) != document.get("payload_sha256"):
        raise ValueError("the sealed entry predicate payload hash has moved")
    if payload["delta"] != ENTRY_DELTA or payload["fingerprint"] != FINGERPRINT:
        raise ValueError(
            "the sealed entry predicate disagrees with the module constants; "
            "a threshold may not move after measurement"
        )
    return payload
