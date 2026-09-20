"""PMO route -> complete dependency-aware program corpus.

This module turns successful PMO routes into the *same* representation the T4
route-distilled expert consumes: an address-free
``StructuralDeltaTemplate`` multiset per program, plus the region count that
carries the dependency structure ``D``.

Representation, and why it is the T4 one
----------------------------------------
``extract_structural_goal(states, actions)`` is the single decomposition
primitive.  It runs ``dependency_region_program`` over one primitive trace and
returns a ``StructuralGoal`` whose subgoals are the independent dependency
regions.  ``minimize_subgoal`` then drops unchanged context roles, leaving the
transferable patch:

    retained/released region R  -> ``input_atoms`` / ``target_atoms`` (``None`` = released)
    replacement topology     H  -> ``output_atoms`` + ``target_bonds``
    attachment              alpha -> the input-role x output-role block of ``target_bonds``
    dependency graph         D  -> the region partition (one subgoal per region)

No second decomposer is written here.  This module only *selects sub-traces*
and delegates.

Windowing
---------
``extract_structural_goal`` abstains above 32 primitives or 4 dependency
components.  Measured on the local PMO curricula, whole base routes are 25-42
primitives and only 3 of 10 extract at those limits.  A route is therefore cut
into contiguous windows of ``window`` primitives, and each window is extracted
independently.  Every window is a complete program over its own sub-route, so
the representation is unchanged; only the unit of supervision is smaller.
Windowing is recorded in the census and in every row, because it is a modelling
choice, not a property of the data.

Zero oracle calls.  Every input is a local artifact; nothing is scored here.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from compose_v4.control.docking_value import identity
from compose_v4.control.structural_subgoal import extract_structural_goal
from compose_v4.control.structural_subgoal_policy import (
    StructuralDeltaTemplate,
    minimize_subgoal,
    structural_rewrite_event_count,
)
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_route_program_corpus_v1"

# T4 bands, renamed to the PMO scale vocabulary.  Thresholds are T4's
# ``proposal_rewrite_scale`` thresholds and are deliberately not re-tuned.
SCALE_BANDS = ("small", "medium", "large")
T4_BAND_TO_SCALE = {"local": "small", "medium": "medium", "large": "large"}
MODES = ("grow", "prune", "replace", "remodel")

DEFAULT_WINDOW = 4

# ---- Local route sources -------------------------------------------------
#
# Each entry is a curriculum artifact that carries, per PMO task, one
# ``base_route`` of exact primitive (states, actions).  ``success_basis``
# records WHY the route counts as successful; it is never inferred.

# The dependency-region training corpus is the PRIMARY source: it already
# collects every curriculum route plus the winner curriculum and the public
# perindopril witnesses, and it labels each route with its PMO task, its route
# lineage, its task family and the frozen task-family test fold.
REGION_CORPUS_PATH = (
    "diagnostics/pmo_dependency_region_program_v2/attempt_1/"
    "training_dependency_region_corpus.json.gz"
)

CURRICULUM_SOURCES: tuple[dict, ...] = (
    {
        "path": "diagnostics/pmo_target_program_wave/curriculum.json",
        "success_basis": "exact replayed route to the published PMO task target",
    },
    {
        "path": "diagnostics/pmo_property_program_wave/curriculum_v2.json",
        "success_basis": "exact replayed route to the published PMO task target",
    },
    {
        "path": "diagnostics/pmo_formula_median_panel_wave/curriculum_v2.json",
        "success_basis": "exact replayed route to the panel endpoint",
    },
    {
        "path": "diagnostics/pmo_property_panel_refinement/curriculum.json",
        "success_basis": "exact replayed route to the panel endpoint",
    },
)


@dataclass(frozen=True)
class PmoRoute:
    """One successful PMO route with its exact primitive trace."""

    task: str
    route_id: str
    source_group: str
    source_smiles: str | None
    states: tuple[dict, ...]
    actions: tuple[dict, ...]
    success_basis: str
    artifact: str
    lineage: str = ""
    task_family: str = ""
    evidence_role: str = ""
    test_fold: int | None = None
    collection: str = ""

    def __post_init__(self) -> None:
        if len(self.states) != len(self.actions) + 1 or not self.actions:
            raise ValueError(f"route {self.route_id!r} is not one complete trace")


@dataclass(frozen=True)
class ProgramRow:
    """One complete dependency-aware program extracted from a route window."""

    task: str
    route_id: str
    source_group: str
    lineage: str
    task_family: str
    window_index: int
    window_primitives: int
    templates: tuple[StructuralDeltaTemplate, ...]
    region_count: int
    rewrite_events: int
    scale: str
    mode: str
    created_atoms: int
    released_atoms: int
    restated_atoms: int
    artifact: str
    success_basis: str
    endpoint_key: str

    @property
    def template_ids(self) -> tuple[str, ...]:
        return tuple(row.template_id for row in self.templates)

    def census_payload(self) -> dict:
        """Structure only.  No molecular content and no route identifier."""
        return {
            "task": self.task,
            "task_family": self.task_family,
            "lineage": self.lineage,
            "source_group": self.source_group,
            "window_index": self.window_index,
            "window_primitives": self.window_primitives,
            "region_count": self.region_count,
            "rewrite_events": self.rewrite_events,
            "scale": self.scale,
            "mode": self.mode,
            "created_atoms": self.created_atoms,
            "released_atoms": self.released_atoms,
            "restated_atoms": self.restated_atoms,
            "template_ids": list(self.template_ids),
            "endpoint_key": self.endpoint_key,
        }


def template_shape(template: StructuralDeltaTemplate) -> dict:
    """Atom-level created/released/restated counts for one patch."""
    released = sum(row is None for row in template.target_atoms)
    restated = sum(
        after is not None and before[:3] != after[:3]
        for before, after in zip(template.input_atoms, template.target_atoms, strict=True)
    )
    return {
        "created": len(template.output_atoms),
        "released": int(released),
        "restated": int(restated),
    }


def program_scale(rewrite_events: int) -> str:
    """T4's schedule-free band thresholds under the PMO scale names."""
    if rewrite_events <= 3:
        return "small"
    if rewrite_events <= 11:
        return "medium"
    return "large"


def program_mode(created: int, released: int) -> str:
    """Coarse edit mode of a complete program, from its atom-count deltas."""
    if created and released:
        return "remodel"
    if created:
        return "grow"
    if released:
        return "prune"
    return "replace"


def _window_bounds(n_actions: int, window: int) -> list[tuple[int, int]]:
    if window < 1:
        raise ValueError("window must be positive")
    return [(start, min(start + window, n_actions)) for start in range(0, n_actions, window)]


def route_programs(route: PmoRoute, *, window: int = DEFAULT_WINDOW) -> tuple[list[ProgramRow], dict]:
    """Extract one complete program per contiguous window of the route."""
    rows: list[ProgramRow] = []
    telemetry: Counter = Counter()
    for index, (start, stop) in enumerate(_window_bounds(len(route.actions), window)):
        actions = tuple(route.actions[start:stop])
        states = tuple(route.states[start : stop + 1])
        telemetry["windows"] += 1
        try:
            goal, _bindings, _regions = extract_structural_goal(states, actions)
        except ValueError as exc:
            reason = str(exc)
            key = "component_budget" if "component_budget" in reason else (
                "primitive_budget" if "primitive_budget" in reason else "other_value_error"
            )
            telemetry[f"abstained_{key}"] += 1
            continue
        except TypeError:
            # Pre-existing defect in the pinned extractor: the component sort key
            # compares a released role's ``None`` signature against a tuple.
            telemetry["abstained_extractor_typeerror"] += 1
            continue
        try:
            templates = tuple(minimize_subgoal(subgoal)[0] for subgoal in goal.subgoals)
        except ValueError:
            telemetry["abstained_no_transferable_delta"] += 1
            continue
        shapes = [template_shape(row) for row in templates]
        created = sum(row["created"] for row in shapes)
        released = sum(row["released"] for row in shapes)
        restated = sum(row["restated"] for row in shapes)
        events = sum(structural_rewrite_event_count(row) for row in templates)
        rows.append(
            ProgramRow(
                task=route.task,
                route_id=route.route_id,
                source_group=route.source_group,
                lineage=route.lineage,
                task_family=route.task_family,
                window_index=index,
                window_primitives=len(actions),
                templates=templates,
                region_count=len(templates),
                rewrite_events=events,
                scale=program_scale(events),
                mode=program_mode(created, released),
                created_atoms=created,
                released_atoms=released,
                restated_atoms=restated,
                artifact=route.artifact,
                success_basis=route.success_basis,
                # The window's realized final state IS the teacher endpoint of
                # this program: what an exactly-rebound proposal must reach.
                endpoint_key=canonical_state_key(decode_state(states[-1])),
            )
        )
        telemetry["programs"] += 1
    return rows, dict(sorted(telemetry.items()))


def iter_region_corpus_routes(repo_root: Path) -> Iterator[PmoRoute]:
    """Yield every task-labelled route in the dependency-region training corpus.

    ``members`` carries the PMO task.  A route with more than one member is
    the same trace reached from more than one curriculum; the task is taken
    from the first member in sorted order so the assignment is deterministic,
    and the member count is preserved in the route id.
    """
    path = repo_root / REGION_CORPUS_PATH
    if not path.exists():
        return
    with gzip.open(path, "rt") as handle:
        payload = json.loads(handle.read())["payload"]
    for index, route in enumerate(payload["routes"]):
        members = sorted(route["members"], key=lambda row: (row["task"], row["member_id"]))
        if not members:
            continue
        primary = members[0]
        states = tuple(route["states"])
        group = identity({"schema": "pmo_route_source_group_v1", "state": states[0]})
        yield PmoRoute(
            task=primary["task"],
            route_id=f"{REGION_CORPUS_PATH}::route_{index:04d}::{route['trace_identity'][:12]}",
            source_group=group,
            source_smiles=None,
            states=states,
            actions=tuple(route["actions"]),
            success_basis=(
                f"{primary['evidence_role']} route in the frozen dependency-region "
                f"training corpus; terminal endpoint replayed exactly"
            ),
            artifact=REGION_CORPUS_PATH,
            lineage=route["lineage_identity"],
            task_family=route.get("task_family", ""),
            evidence_role=primary.get("evidence_role", ""),
            test_fold=route.get("test_fold"),
            collection=primary.get("collection", ""),
        )


def iter_curriculum_routes(repo_root: Path) -> Iterator[PmoRoute]:
    """Yield every base route in the local PMO curriculum artifacts."""
    for entry in CURRICULUM_SOURCES:
        path = repo_root / entry["path"]
        if not path.exists():
            continue
        payload = json.loads(path.read_text())["payload"]
        source = payload.get("source")
        source_smiles = source.get("smiles") if isinstance(source, dict) else None
        for task, block in sorted(payload["tasks"].items()):
            base = block.get("base_route")
            if not base or not base.get("actions"):
                continue
            states = tuple(base["states"])
            # The source molecule is the route's own first state.  Routes that
            # share a first state share a source group, which is what the
            # source-balanced marginal needs.
            group = identity({"schema": "pmo_route_source_group_v1", "state": states[0]})
            yield PmoRoute(
                task=task,
                route_id=f"{entry['path']}::{task}::base_route",
                source_group=group,
                source_smiles=source_smiles,
                states=states,
                actions=tuple(base["actions"]),
                success_basis=entry["success_basis"],
                artifact=entry["path"],
            )


@dataclass
class RouteCorpus:
    rows: list[ProgramRow] = field(default_factory=list)
    routes: list[PmoRoute] = field(default_factory=list)
    telemetry: dict = field(default_factory=dict)

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(sorted({row.task for row in self.rows}))

    def source_graph(self, task: str):
        """Padded source graph of the first route for one task."""
        for route in self.routes:
            if route.task == task:
                return decode_state(route.states[0])
        raise KeyError(task)

    def census(self) -> dict:
        by_task = Counter(row.task for row in self.rows)
        return {
            "schema_version": SCHEMA,
            "routes": len(self.routes),
            "route_tasks": sorted({route.task for route in self.routes}),
            "programs": len(self.rows),
            "program_tasks": list(self.tasks),
            "distinct_source_groups": len({route.source_group for route in self.routes}),
            "distinct_lineages": len({route.lineage for route in self.routes if route.lineage}),
            "lineages_per_task": {
                task: len({r.lineage for r in self.routes if r.task == task and r.lineage})
                for task in sorted({r.task for r in self.routes})
            },
            "routes_per_task": dict(
                sorted(Counter(route.task for route in self.routes).items())
            ),
            "evidence_roles": dict(
                sorted(Counter(route.evidence_role for route in self.routes).items())
            ),
            "task_families": dict(
                sorted(Counter(route.task_family for route in self.routes).items())
            ),
            "distinct_templates": len({name for row in self.rows for name in row.template_ids}),
            "programs_per_task": dict(sorted(by_task.items())),
            "scale_distribution": dict(sorted(Counter(row.scale for row in self.rows).items())),
            "mode_distribution": dict(sorted(Counter(row.mode for row in self.rows).items())),
            "region_count_distribution": dict(
                sorted(Counter(row.region_count for row in self.rows).items())
            ),
            "rewrite_events_five_number": _five_number(
                sorted(row.rewrite_events for row in self.rows)
            ),
            "extraction_telemetry": self.telemetry,
        }


def _five_number(values: list[int]) -> dict:
    if not values:
        return {}
    n = len(values)

    def q(fraction: float) -> int:
        return values[min(n - 1, max(0, round(fraction * (n - 1))))]

    return {"min": values[0], "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "max": values[-1]}


def build_corpus(repo_root: Path, *, window: int = DEFAULT_WINDOW) -> RouteCorpus:
    """Build the complete local PMO route-program corpus."""
    corpus = RouteCorpus()
    telemetry: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    routes = list(iter_region_corpus_routes(repo_root))
    if routes:
        telemetry["source"] = "dependency_region_training_corpus"
    else:
        routes = list(iter_curriculum_routes(repo_root))
        telemetry["source"] = "curriculum_base_routes"
    for route in routes:
        # The same base route is serialized into more than one curriculum
        # artifact.  A duplicate serialization is not a second observation, so
        # it is dropped rather than allowed to carry double weight.
        trace_key = (
            route.task,
            identity({"schema": "pmo_route_trace_v1", "actions": list(route.actions)}),
        )
        if trace_key in seen:
            telemetry["duplicate_route_serializations"] += 1
            continue
        seen.add(trace_key)
        corpus.routes.append(route)
        rows, route_telemetry = route_programs(route, window=window)
        corpus.rows.extend(rows)
        telemetry.update(route_telemetry)
    telemetry["window"] = window
    corpus.telemetry = dict(sorted(telemetry.items()))
    return corpus
