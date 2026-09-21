"""Answer-known PMO teacher-route atlas: loading, exact replay and regime guards.

DATA STRUCTURE
--------------
An :class:`AtlasRoute` is one recorded teacher route through the Editing-V2
rewrite system: a 48-slot source state, an ordered tuple of schema-4 executor
action records, the recorded intermediate states, and the destination the route
was compiled toward.  Routes are read from six frozen development artifacts
under ``diagnostics/``; every artifact's file SHA-256 is verified against
:data:`ATLAS_ARTIFACTS` before a single route is returned.

Routes group into *lineages*.  A lineage is one compiled spine (a ``base_route``
or a perindopril root/target pair); the sibling programs of a lineage share that
spine's prefix verbatim and append a short suffix.  The dossier therefore holds
far fewer independent transformations than it holds programs, and
:func:`lineage_census` is the honest denominator for any breadth claim.

INFORMATION REGIME
------------------
Every route in this dossier is **answer-known, panel-informed or
winner-informed development evidence**.  The destinations are published PMO
answers or transcriptions of a competitor's plotted winners.  This atlas is a
CHALLENGE SET and a diagnostic instrument.  It is never a training distribution,
a prior, a library or an initialization for a scored no-prescreen run.
:func:`assert_not_scored_consumable` exists so that a caller which tries has to
delete a guard rather than forget one.

INVARIANTS MAINTAINED AND TESTED
--------------------------------
1. ``load_atlas`` refuses an artifact whose bytes moved (``AtlasIntegrityError``).
2. Every route carries :data:`DEVELOPMENT_INFORMED_LABEL`; a payload built from
   atlas material and lacking it is refused by
   :func:`assert_development_informed`.
3. :func:`replay_route` re-executes the recorded actions through the production
   ``editing_v2_semantic_rewrite_system`` and compares *every* intermediate
   state to the recorded one by canonical state key.  ``exact`` is True only
   when all K+1 states agree and the endpoint matches.
4. :func:`route_checkpoints` returns committed states only: each checkpoint is a
   complete, valid, connected molecule inside the 1..40 active-atom support of
   the 48-slot PMO representation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---- Information regime ----

DEVELOPMENT_INFORMED_LABEL = "DEVELOPMENT_INFORMED_DIAGNOSTIC"

REGIME_STATEMENT = (
    "Answer-known, panel-informed and winner-informed PMO development evidence. "
    "Destinations are published task answers or transcribed competitor winners. "
    "This is a challenge set and a diagnostic instrument, never a training "
    "distribution, prior, library or initialization for a scored no-prescreen run."
)

# PMO representation constants, re-asserted here so a probe cannot drift.
PMO_SLOTS = 48
PMO_MAX_ACTIVE_ATOMS = 40


class AtlasIntegrityError(RuntimeError):
    """Raised when a frozen atlas input does not match its recorded identity."""


class AtlasRegimeError(RuntimeError):
    """Raised when atlas material is routed toward a prohibited consumer."""


# ---- Frozen inputs ----


@dataclass(frozen=True)
class AtlasArtifact:
    relative_path: str
    file_sha256: str
    kind: str


#: The six frozen artifacts the dossier is assembled from.  The hashes are the
#: values the frozen ``pmo_route_distillation`` export recorded in its
#: ``source_checks`` block, independently re-verified at load time.
ATLAS_ARTIFACTS: tuple[AtlasArtifact, ...] = (
    AtlasArtifact(
        "diagnostics/pmo_target_program_wave/curriculum.json",
        "97ac684ff9fda25992adb9f552a545283232e5620eb5deb467b3a723112ff9b0",
        "task_wave",
    ),
    AtlasArtifact(
        "diagnostics/pmo_property_program_wave/curriculum_v2.json",
        "2f3a11b5fd5f818358e9de132f60ec4c6ba2882b5cd71b66aaad5c407069bed9",
        "task_wave",
    ),
    AtlasArtifact(
        "diagnostics/pmo_property_panel_refinement/curriculum.json",
        "e0325aab91b7f5be332af9d9d9e6546aac3a734b7537412f77091b97f7337b8a",
        "task_wave",
    ),
    AtlasArtifact(
        "diagnostics/pmo_formula_median_panel_wave/curriculum_v2.json",
        "224b7459307496d649a8310baad7bd47ed38bee762ffbfb37e216bfadbbc918c",
        "task_wave",
    ),
    AtlasArtifact(
        "diagnostics/pmo_winner_program_curriculum/curriculum.json",
        "99a68d75e79d4284e1831dcdd9101eb92f54ffa8417007b16b2b918111b54316",
        "winner_curriculum",
    ),
    AtlasArtifact(
        "diagnostics/pmo_public_winner_recovery/route_audit.json",
        "fa7efcb02651b1400fa295701f22f528b1fc6c63b08819e80b6dd15f8f7d5918",
        "winner_route_audit",
    ),
)

#: The frozen distillation export that first assembled these six inputs.  It is
#: read only for its census, never for route content.
ATLAS_DISTILLATION_RESULT = "diagnostics/pmo_route_distillation/attempt_1/result.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha256(payload: Any) -> str:
    """The repository's canonical payload identity: sorted, compact JSON."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---- Route records ----


@dataclass(frozen=True)
class AtlasRoute:
    """One recorded teacher route from a source state to a recorded endpoint."""

    task: str
    artifact: str
    lineage_id: str
    program_id: str
    role: str
    source_smiles: str
    source_state: dict[str, Any]
    destination_smiles: str | None
    destination_role: str
    recorded_endpoint_smiles: str
    actions: tuple[dict[str, Any], ...]
    recorded_states: tuple[dict[str, Any], ...]
    is_spine: bool
    regime: str = DEVELOPMENT_INFORMED_LABEL

    @property
    def primitive_steps(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class AtlasDossier:
    routes: tuple[AtlasRoute, ...]
    artifact_hashes: dict[str, str]
    regime: str = DEVELOPMENT_INFORMED_LABEL
    regime_statement: str = REGIME_STATEMENT

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted({route.task for route in self.routes}))

    def for_task(self, task: str) -> tuple[AtlasRoute, ...]:
        return tuple(route for route in self.routes if route.task == task)

    def spines(self) -> tuple[AtlasRoute, ...]:
        return tuple(route for route in self.routes if route.is_spine)


# ---- Loading ----


def _verify_artifacts(repo_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for artifact in ATLAS_ARTIFACTS:
        path = repo_root / artifact.relative_path
        if not path.is_file():
            raise AtlasIntegrityError(f"missing atlas input: {artifact.relative_path}")
        digest = file_sha256(path)
        if digest != artifact.file_sha256:
            raise AtlasIntegrityError(
                f"{artifact.relative_path} moved: expected {artifact.file_sha256}, "
                f"observed {digest}"
            )
        observed[artifact.relative_path] = digest
    return observed


def _load_payload(repo_root: Path, relative_path: str) -> dict[str, Any]:
    document = json.loads((repo_root / relative_path).read_text())
    payload = document["payload"]
    recorded = document.get("payload_sha256")
    if recorded is not None and payload_sha256(payload) != recorded:
        raise AtlasIntegrityError(f"{relative_path}: payload hash disagrees with its content")
    return payload


def _wave_routes(payload: dict[str, Any], relative_path: str) -> list[AtlasRoute]:
    """Read one of the four task-wave curricula.

    Each task section carries a ``base_route`` spine plus sibling ``programs``
    whose receipts hold the complete action list from the shared source.
    """

    routes: list[AtlasRoute] = []
    shared_source = payload.get("source")
    per_task_sources = payload.get("sources") or {}
    for task, section in sorted(payload["tasks"].items()):
        source = per_task_sources.get(task, shared_source)
        if source is None:
            raise AtlasIntegrityError(f"{relative_path}:{task} has no recorded source")
        destination = section.get("target")
        destination_role = "target"
        if destination is None:
            destination = section.get("anchor")
            destination_role = "anchor"
        if destination is None:
            destination_role = "absent"
        lineage_id = f"{relative_path}::{task}"
        base = section["base_route"]
        routes.append(
            AtlasRoute(
                task=task,
                artifact=relative_path,
                lineage_id=lineage_id,
                program_id=f"{lineage_id}::base_route",
                role="spine",
                source_smiles=source["smiles"],
                source_state=source["state"],
                destination_smiles=destination,
                destination_role=destination_role,
                recorded_endpoint_smiles=destination if destination else "",
                actions=tuple(base["actions"]),
                recorded_states=tuple(base["states"]),
                is_spine=True,
            )
        )
        for program in section["programs"]:
            receipt = program["receipt"]
            role = program.get("operation") or program.get("compilation_mode") or "program"
            routes.append(
                AtlasRoute(
                    task=task,
                    artifact=relative_path,
                    lineage_id=lineage_id,
                    program_id=f"{lineage_id}::{program['candidate_id']}",
                    role=str(role),
                    source_smiles=source["smiles"],
                    source_state=source["state"],
                    destination_smiles=destination,
                    destination_role=destination_role,
                    recorded_endpoint_smiles=program["smiles"],
                    actions=tuple(receipt["actions"]),
                    recorded_states=tuple(receipt["states"]),
                    is_spine=False,
                )
            )
    return routes


def _winner_routes(payload: dict[str, Any], relative_path: str) -> list[AtlasRoute]:
    """Read the perindopril winner curriculum.

    The readable distillation index carries no source or endpoint SMILES for
    this task.  Both are present here: sources in ``roots[*]`` and the compiled
    destination in ``programs[*].target.canonical_smiles``.
    """

    roots = {root["root_id"]: root for root in payload["roots"]}
    programs = list(payload["programs"])
    # The spine is the program whose action list IS the shared prefix, i.e. the
    # shortest one.  Choosing it by variant name would pin a label rather than a
    # structural fact, and the perindopril variants are not ordered by length.
    spine_steps = min(len(program["receipt"]["actions"]) for program in programs)
    routes: list[AtlasRoute] = []
    for program in programs:
        receipt = program["receipt"]
        source_id = receipt.get("source_id") or program.get("source_id")
        if source_id is None:
            # The receipt's first state is authoritative; match it to a root.
            first_state = receipt["states"][0]
            source_id = next(
                (rid for rid, root in roots.items() if root["state"] == first_state),
                None,
            )
        if source_id is None or source_id not in roots:
            raise AtlasIntegrityError(
                f"{relative_path}: perindopril program {program['variant_id']} "
                "cannot be attributed to a recorded root"
            )
        root = roots[source_id]
        destination = program["target"]["canonical_smiles"]
        routes.append(
            AtlasRoute(
                task="perindopril_mpo",
                artifact=relative_path,
                lineage_id=f"{relative_path}::perindopril_mpo::{source_id}",
                program_id=f"{relative_path}::perindopril_mpo::{program['variant_id']}",
                role=program.get("evidence_role", "program"),
                source_smiles=root["endpoint"],
                source_state=root["state"],
                destination_smiles=destination,
                destination_role="target",
                recorded_endpoint_smiles=receipt.get("endpoint", destination),
                actions=tuple(receipt["actions"]),
                recorded_states=tuple(receipt["states"]),
                is_spine=len(receipt["actions"]) == spine_steps,
            )
        )
    return routes


def load_atlas(repo_root: Path | str, *, verify: bool = True) -> AtlasDossier:
    """Load every recorded teacher route, refusing any input whose bytes moved."""

    repo_root = Path(repo_root)
    hashes = _verify_artifacts(repo_root) if verify else {}
    routes: list[AtlasRoute] = []
    for artifact in ATLAS_ARTIFACTS:
        if artifact.kind == "winner_route_audit":
            # Descriptive audit only: it carries coverage and step counts, not
            # action records, so it contributes provenance rather than routes.
            continue
        payload = _load_payload(repo_root, artifact.relative_path)
        if artifact.kind == "task_wave":
            routes.extend(_wave_routes(payload, artifact.relative_path))
        elif artifact.kind == "winner_curriculum":
            routes.extend(_winner_routes(payload, artifact.relative_path))
        else:  # pragma: no cover - the tuple above is exhaustive
            raise AtlasIntegrityError(f"unknown artifact kind: {artifact.kind}")
    return AtlasDossier(routes=tuple(routes), artifact_hashes=hashes)


# ---- Replay ----


@dataclass(frozen=True)
class RouteReplay:
    program_id: str
    task: str
    exact: bool
    steps_executed: int
    steps_total: int
    endpoint_smiles: str | None
    endpoint_matches_record: bool
    endpoint_matches_destination: bool
    intermediate_key_mismatches: tuple[int, ...]
    out_of_support_indices: tuple[int, ...]
    failure: str | None


def _rewrite_system():
    from compose_v4.rewrite.kernel import editing_v2_semantic_rewrite_system

    return editing_v2_semantic_rewrite_system()


def replay_route(route: AtlasRoute, system: Any | None = None) -> RouteReplay:
    """Re-execute a recorded route through the production Editing-V2 executor.

    Every intermediate state is compared to the recorded one by canonical state
    key, so a route that reaches the right endpoint by a different interior path
    is reported as inexact rather than as a pass.
    """

    from compose_v4.rewrite.action_codec_v4 import decode_action
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.trace_shard import decode_state

    system = system if system is not None else _rewrite_system()
    graph = decode_state(route.source_state)
    mismatches: list[int] = []
    out_of_support: list[int] = []
    failure: str | None = None
    executed = 0

    def _check(index: int, current: Any) -> None:
        if not 1 <= current.n_real_atoms <= PMO_MAX_ACTIVE_ATOMS:
            out_of_support.append(index)
        if index < len(route.recorded_states):
            recorded = decode_state(route.recorded_states[index])
            if canonical_state_key(current) != canonical_state_key(recorded):
                mismatches.append(index)

    _check(0, graph)
    for index, record in enumerate(route.actions):
        try:
            family, action = decode_action(record)
            graph = system.apply(graph, family, action)
        except (ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            # InvalidRewrite and the codec errors are ValueError subclasses; the
            # RDKit canonicaliser raises RuntimeError on states it refuses.
            failure = f"step {index} {type(exc).__name__}: {exc}"
            break
        executed += 1
        _check(index + 1, graph)

    endpoint = canonical_state_key(graph) if failure is None else None
    matches_record = endpoint == route.recorded_endpoint_smiles if endpoint else False
    matches_destination = (
        endpoint == route.destination_smiles
        if endpoint and route.destination_smiles
        else False
    )
    exact = (
        failure is None
        and executed == len(route.actions)
        and not mismatches
        and not out_of_support
        and matches_record
    )
    return RouteReplay(
        program_id=route.program_id,
        task=route.task,
        exact=exact,
        steps_executed=executed,
        steps_total=len(route.actions),
        endpoint_smiles=endpoint,
        endpoint_matches_record=matches_record,
        endpoint_matches_destination=matches_destination,
        intermediate_key_mismatches=tuple(mismatches),
        out_of_support_indices=tuple(out_of_support),
        failure=failure,
    )


# ---- Structural checkpoints ----


@dataclass(frozen=True)
class Checkpoint:
    program_id: str
    task: str
    label: str
    step_index: int
    fraction: float
    remaining_steps: int
    smiles: str
    state: dict[str, Any]
    heavy_atoms: int


#: Named positions along a route, expressed as a fraction of its length.
#: ``origin`` is the untouched source, ``anchor`` the compiled destination.
DEFAULT_CHECKPOINT_POSITIONS: tuple[tuple[str, float], ...] = (
    ("early", 0.25),
    ("midway", 0.55),
    ("near_anchor", 0.85),
    ("anchor", 1.0),
)


def route_checkpoints(
    route: AtlasRoute,
    positions: Sequence[tuple[str, float]] = DEFAULT_CHECKPOINT_POSITIONS,
    *,
    system: Any | None = None,
) -> tuple[Checkpoint, ...]:
    """Materialise committed intermediate states at named positions on a route.

    The states are produced by re-executing the route, never by re-parsing a
    recorded SMILES: a SMILES round trip yields a tight graph, which silently
    deletes the ``atom_insert`` family from the legal support.
    """

    from compose_v4.rewrite.action_codec_v4 import decode_action
    from compose_v4.rewrite.kernel import canonical_state_key
    from compose_v4.rewrite.trace_shard import decode_state, encode_state

    system = system if system is not None else _rewrite_system()
    total = len(route.actions)
    if total == 0:
        raise ValueError(f"{route.program_id} has no actions to checkpoint")

    wanted: dict[int, str] = {}
    for label, fraction in positions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"checkpoint fraction out of range: {label}={fraction}")
        index = max(1, min(total, round(fraction * total)))
        wanted.setdefault(index, label)

    graph = decode_state(route.source_state)
    found: list[Checkpoint] = []
    for index, record in enumerate(route.actions, start=1):
        family, action = decode_action(record)
        graph = system.apply(graph, family, action)
        if index in wanted:
            if not 1 <= graph.n_real_atoms <= PMO_MAX_ACTIVE_ATOMS:
                raise AtlasIntegrityError(
                    f"{route.program_id} step {index} leaves the 1..40 active-atom support"
                )
            found.append(
                Checkpoint(
                    program_id=route.program_id,
                    task=route.task,
                    label=wanted[index],
                    step_index=index,
                    fraction=index / total,
                    remaining_steps=total - index,
                    smiles=canonical_state_key(graph),
                    state=encode_state(graph),
                    heavy_atoms=int(graph.n_real_atoms),
                )
            )
    return tuple(found)


# ---- Census ----


def lineage_census(dossier: AtlasDossier) -> dict[str, dict[str, Any]]:
    """Per-task counts that keep programs and independent transformations apart."""

    census: dict[str, dict[str, Any]] = {}
    for task in dossier.tasks:
        routes = dossier.for_task(task)
        census[task] = {
            "programs": len(routes),
            "lineages": len({route.lineage_id for route in routes}),
            "distinct_sources": len({route.source_smiles for route in routes}),
            "distinct_destinations": len(
                {route.destination_smiles for route in routes if route.destination_smiles}
            ),
            "distinct_recorded_endpoints": len(
                {route.recorded_endpoint_smiles for route in routes}
            ),
            "primitive_steps_min": min(route.primitive_steps for route in routes),
            "primitive_steps_max": max(route.primitive_steps for route in routes),
        }
    return census


# ---- Regime guards ----

#: Consumers that may never receive atlas-derived material.
PROHIBITED_CONSUMERS: frozenset[str] = frozenset(
    {
        "scored_no_prescreen_run",
        "pmo_scored_contract",
        "route_prior_fit",
        "controller_training",
        "proposal_library",
        "initialization_bank",
    }
)


def assert_development_informed(payload: dict[str, Any]) -> None:
    """Refuse a payload built from atlas material that omits its regime label."""

    if payload.get("information_regime") != DEVELOPMENT_INFORMED_LABEL:
        raise AtlasRegimeError(
            "atlas-derived payloads must carry "
            f"information_regime == {DEVELOPMENT_INFORMED_LABEL!r}"
        )


def assert_not_scored_consumable(consumer: str) -> None:
    """Refuse routing atlas material into a prohibited consumer."""

    if consumer in PROHIBITED_CONSUMERS:
        raise AtlasRegimeError(
            f"{consumer!r} may not consume answer-known atlas material: "
            + REGIME_STATEMENT
        )
