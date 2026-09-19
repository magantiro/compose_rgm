"""Route-supervised, target-free proposal policy for Dynamic COMPOSE.

The fitted checkpoint contains one shared neural option actor and aggregate
module-count statistics.  It never contains an executable teacher program,
endpoint, target label, seed label, or source atom address.  At inference the
actor scores generic structural options from the current molecule, maps those
options to the existing Dynamic-v1 module vocabulary, and compiles every
sampled module against the actual current exact state.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from compose_v4.chem.molecular_graph import ELEMENT_TO_IDX, is_element
from compose_v4.control.dynamic_program_synthesis_v1 import (
    GENERIC_MODULES,
    compile_generic_module_v1,
)
from compose_v4.control.edit_program import extract_program
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
)
from compose_v4.control.option_demonstrations import descriptor_menu
from compose_v4.control.option_features import structural_option_features
from compose_v4.control.option_policy import AdvantageWeightedOptionActor
from compose_v4.control.option_selector import balanced_option_prior
from compose_v4.control.ring_program import ring_spec
from compose_v4.control.trajectory_value import molecule_features
from compose_v4.rewrite.kernel import canonical_state_key

SCHEMA = "route_distilled_program_policy_v1"
CHECKPOINT_SCHEMA = "route_distilled_program_actor_v1"


STAGE_DESCRIPTOR_NAMES = (
    "atom_insert_fraction",
    "atom_delete_fraction",
    "atom_restate_fraction",
    "cycle_close_fraction",
    "cycle_open_fraction",
    "bond_change_fraction",
    "ring_restate_fraction",
    "existing_touched_fraction",
    "mean_existing_degree",
    "mean_existing_hydrogens",
    "net_atom_delta",
    "created_dependency_fraction",
    "inserted_carbon_fraction",
    "inserted_nitrogen_fraction",
    "inserted_oxygen_fraction",
    "inserted_fluorine_fraction",
    "inserted_other_fraction",
    "primitive_fraction",
)


def option_family_weights(option: str) -> dict[str, float]:
    """Map one learned structural option to executable generic modules."""
    spec = ring_spec(option)
    if spec is not None:
        return {"append_ring" if spec.topology == "pendant" else "fuse_ring": 1.0}
    mapping = {
        "add_carbonyl": {"carbonyl_insert": 1.0},
        "insert_ring_carbonyl": {"carbonyl_insert": 0.7, "ring_system_restate": 0.3},
        "local": {"heteroatom_substitute": 0.7, "ring_system_restate": 0.3},
        "grow": {"segment_grow": 1.0},
        "append": {"functionalize": 0.6, "segment_grow": 0.4},
        "scaffold_extend": {"segment_grow": 1.0},
        "decorate": {"functionalize": 1.0},
        "cyclize": {"cycle_close": 1.0},
        "append_system": {"append_ring": 1.0},
        "annulate": {"fuse_ring": 1.0},
        "small_ring": {"cycle_close": 1.0},
        "aromatize": {"ring_system_restate": 1.0},
        "restate": {"ring_system_restate": 1.0},
        "open": {"cycle_open": 1.0},
        "rebuild": {"bond_reroute": 1.0},
        "shrink": {"substituent_delete": 0.65, "segment_shrink": 0.35},
    }
    if option == "generic":
        return {family: 1 / len(GENERIC_MODULES) for family in GENERIC_MODULES}
    if option not in mapping:
        raise KeyError(f"distilled option has no generic module mapping: {option!r}")
    return mapping[option]


def _record_slots(record: dict) -> tuple[int, ...]:
    rule, payload = record["executor_rule"], record["payload"]
    if rule == "atom_insert":
        return (int(payload["slot"]), *(int(row[0]) for row in payload["neighbors"]))
    if rule in ("atom_delete", "atom_restate_semantic"):
        return (int(payload["v"]),)
    if rule in ("cycle_close", "cycle_open", "bond_reorder"):
        return (int(payload["a"]), int(payload["b"]))
    if rule == "bond_reroute":
        return tuple(int(payload[name]) for name in ("a", "b", "u", "v"))
    if rule == "ring_system_restate":
        return tuple(
            int(change[name]) for change in payload["changes"] for name in ("a", "b")
        )
    raise ValueError(f"unsupported distilled-stage executor rule: {rule!r}")


def stage_descriptor(graph, records) -> np.ndarray:
    """Describe a realized edit through chemistry and relative structural roles."""
    actions = tuple(records)
    if not actions:
        raise ValueError("a distilled stage descriptor requires actions")
    real = {int(slot) for slot in np.flatnonzero(is_element(graph.atom_types))}
    touched, created, dependent_references = set(), set(), 0
    inserted = Counter()
    rules = Counter()
    total_references = 0
    for record in actions:
        rule = record["executor_rule"]
        rules[rule] += 1
        slots = _record_slots(record)
        total_references += len(slots)
        dependent_references += sum(slot in created for slot in slots)
        touched.update(slot for slot in slots if slot in real)
        if rule == "atom_insert":
            slot = int(record["payload"]["slot"])
            created.add(slot)
            atom_type = int(record["payload"]["atom_type"])
            inserted[atom_type] += 1
    n_actions = len(actions)
    degrees = [int(np.count_nonzero(graph.bonds[slot])) for slot in touched]
    hydrogens = [int(graph.implicit_h_counts[slot]) for slot in touched]
    bond_change = rules["bond_reorder"] + rules["bond_reroute"]
    inserted_total = rules["atom_insert"]
    values = (
        rules["atom_insert"] / n_actions,
        rules["atom_delete"] / n_actions,
        rules["atom_restate_semantic"] / n_actions,
        rules["cycle_close"] / n_actions,
        rules["cycle_open"] / n_actions,
        bond_change / n_actions,
        rules["ring_system_restate"] / n_actions,
        len(touched) / max(1, len(real)),
        (sum(degrees) / len(degrees) / 4) if degrees else 0.0,
        (sum(hydrogens) / len(hydrogens) / 4) if hydrogens else 0.0,
        (rules["atom_insert"] - rules["atom_delete"]) / 8,
        dependent_references / max(1, total_references),
        inserted[ELEMENT_TO_IDX["C"]] / max(1, inserted_total),
        inserted[ELEMENT_TO_IDX["N"]] / max(1, inserted_total),
        inserted[ELEMENT_TO_IDX["O"]] / max(1, inserted_total),
        inserted[ELEMENT_TO_IDX["F"]] / max(1, inserted_total),
        (
            inserted_total
            - sum(inserted[ELEMENT_TO_IDX[name]] for name in ("C", "N", "O", "F"))
        )
        / max(1, inserted_total),
        n_actions / 32,
    )
    result = np.asarray(values, dtype=float)
    if result.shape != (len(STAGE_DESCRIPTOR_NAMES),) or not np.isfinite(result).all():
        raise RuntimeError("distilled stage descriptor is invalid")
    return result


def actor_from_checkpoint(checkpoint: dict) -> AdvantageWeightedOptionActor:
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("route-distilled checkpoint schema mismatch")
    actor = AdvantageWeightedOptionActor(
        int(checkpoint["state_dim"]),
        int(checkpoint["option_dim"]),
        int(checkpoint["hidden"]),
    )
    expected = actor.state_dict()
    parameters = checkpoint.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != set(expected):
        raise ValueError("route-distilled actor parameter names mismatch")
    state = {}
    for name, template in expected.items():
        value = torch.tensor(parameters[name], dtype=template.dtype)
        if value.shape != template.shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid route-distilled actor tensor: {name}")
        state[name] = value
    actor.load_state_dict(state, strict=True)
    return actor.eval()


@dataclass(frozen=True)
class RouteDistilledProgramPolicy:
    actor: AdvantageWeightedOptionActor
    options: tuple[str, ...]
    reference: tuple[float, ...]
    module_count_probabilities: tuple[float, float, float]
    exploration_floor: float
    training_identity: str
    binding_prototypes: tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...]

    def __post_init__(self):
        reference = np.asarray(self.reference, dtype=float)
        counts = np.asarray(self.module_count_probabilities, dtype=float)
        if (
            self.options != descriptor_menu()
            or reference.shape != (len(self.options),)
            or np.any(reference <= 0)
            or not np.isclose(reference.sum(), 1.0)
            or counts.shape != (3,)
            or np.any(counts <= 0)
            or not np.isclose(counts.sum(), 1.0)
            or not 0 < self.exploration_floor <= 1
            or not self.training_identity
        ):
            raise ValueError("invalid route-distilled policy contract")
        if self.actor.option_dim != len(structural_option_features(self.options[0])):
            raise ValueError("route-distilled option feature dimension mismatch")
        names = [row[0] for row in self.binding_prototypes]
        if names != sorted(set(names)) or set(names) != set(GENERIC_MODULES):
            raise ValueError("route-distilled binding prototypes are incomplete")
        for _, mean, scale in self.binding_prototypes:
            if (
                len(mean) != len(STAGE_DESCRIPTOR_NAMES)
                or len(scale) != len(STAGE_DESCRIPTOR_NAMES)
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or np.any(np.asarray(scale) <= 0)
            ):
                raise ValueError("invalid route-distilled binding prototype")

    @classmethod
    def from_checkpoint(cls, checkpoint: dict):
        forbidden = {
            "target",
            "protein",
            "seed_identity",
            "route_id",
            "endpoint",
            "source_atom",
            "program",
        }
        if forbidden & set(checkpoint):
            raise ValueError("runtime checkpoint contains a forbidden teacher field")
        actor = actor_from_checkpoint(checkpoint)
        policy = cls(
            actor=actor,
            options=tuple(checkpoint["options"]),
            reference=tuple(map(float, checkpoint["reference"])),
            module_count_probabilities=tuple(
                map(float, checkpoint["module_count_probabilities"])
            ),
            exploration_floor=float(checkpoint["exploration_floor"]),
            training_identity=str(checkpoint["training_identity"]),
            binding_prototypes=tuple(
                (
                    str(row["family"]),
                    tuple(map(float, row["mean"])),
                    tuple(map(float, row["scale"])),
                )
                for row in checkpoint["binding_prototypes"]
            ),
        )
        if actor.state_dim != int(checkpoint["state_dim"]):
            raise ValueError("route-distilled state feature dimension mismatch")
        return policy

    def option_probabilities(self, graph) -> np.ndarray:
        features = np.asarray(
            molecule_features(canonical_state_key(graph)), dtype=np.float32
        )
        if features.shape != (self.actor.state_dim,) or not np.isfinite(features).all():
            raise ValueError("current molecule has invalid distilled-policy features")
        option_features = torch.from_numpy(
            np.stack([structural_option_features(name) for name in self.options])
        )
        with torch.inference_mode():
            scores = self.actor(torch.from_numpy(features), option_features).numpy()
        reference = np.asarray(self.reference, dtype=float)
        shifted = scores - float(np.max(scores))
        learned = reference * np.exp(shifted)
        learned /= learned.sum()
        result = (
            self.exploration_floor * reference + (1 - self.exploration_floor) * learned
        )
        if np.any(result <= 0) or not np.isclose(result.sum(), 1.0):
            raise RuntimeError("route-distilled option probabilities are invalid")
        return result

    def family_probabilities(self, graph) -> dict[str, float]:
        totals = Counter()
        for option, probability in zip(
            self.options, self.option_probabilities(graph), strict=True
        ):
            for family, conditional in option_family_weights(option).items():
                totals[family] += float(probability) * conditional
        missing = set(GENERIC_MODULES) - set(totals)
        if missing:
            raise RuntimeError(
                f"distilled mapping lost generic families: {sorted(missing)}"
            )
        total = sum(totals.values())
        return {family: totals[family] / total for family in GENERIC_MODULES}

    def stage_score(self, family: str, graph, stage: dict) -> float:
        prototypes = {
            name: (mean, scale) for name, mean, scale in self.binding_prototypes
        }
        if family not in prototypes:
            raise KeyError(f"missing binding prototype for {family!r}")
        mean, scale = map(np.asarray, prototypes[family])
        descriptor = stage_descriptor(graph, stage["actions"])
        return -float(np.square((descriptor - mean) / scale).mean())


def _compile_ranked_module(
    source,
    rng,
    family: str,
    policy: RouteDistilledProgramPolicy,
    *,
    preferred_anchors: frozenset[int],
    panel_cache: dict | None,
    alternatives: int = 8,
):
    candidates, seen, failures = [], set(), Counter()
    for _ in range(alternatives):
        try:
            product, stage = compile_generic_module_v1(
                source,
                rng,
                family,
                preferred_anchors=preferred_anchors,
                panel_cache=panel_cache,
            )
        except ValueError as error:
            failures[str(error)] += 1
            continue
        endpoint = canonical_state_key(product)
        if endpoint in seen:
            continue
        seen.add(endpoint)
        candidates.append(
            (policy.stage_score(family, source, stage), endpoint, product, stage)
        )
    if not candidates:
        raise ValueError(f"no distilled binding candidate: {dict(failures)}")
    candidates.sort(key=lambda row: (-row[0], row[1]))
    if len(candidates) > 1 and rng.random() < policy.exploration_floor:
        selected = int(rng.integers(len(candidates)))
    else:
        selected = 0
    score, _, product, stage = candidates[selected]
    stage = {
        **stage,
        "parameters": {
            **stage["parameters"],
            "route_distilled_binding": {
                "alternatives": len(candidates),
                "selected_rank": selected,
                "teacher_prototype_score": score,
                "exploration_floor": policy.exploration_floor,
            },
        },
    }
    return product, stage


def synthesize_route_distilled_program(
    source,
    rng,
    policy: RouteDistilledProgramPolicy,
    *,
    max_modules: int = 3,
    max_primitives: int = 32,
    max_blocks: int = 8,
    panel_cache: dict | None = None,
):
    """Sample and compile one complete route from the learned generic policy."""
    if not 1 <= max_modules <= 3:
        raise ValueError("route-distilled synthesis supports one to three modules")
    count_probabilities = np.asarray(
        policy.module_count_probabilities[:max_modules], dtype=float
    )
    count_probabilities /= count_probabilities.sum()
    requested = int(rng.choice(np.arange(1, max_modules + 1), p=count_probabilities))
    current, stages, selected, failures = source, [], [], Counter()
    preferred = frozenset()
    for module_index in range(requested):
        probabilities = policy.family_probabilities(current)
        remaining = list(GENERIC_MODULES)
        accepted = None
        while remaining:
            weights = np.asarray(
                [probabilities[name] for name in remaining], dtype=float
            )
            family = remaining.pop(
                int(rng.choice(len(remaining), p=weights / weights.sum()))
            )
            try:
                product, stage = _compile_ranked_module(
                    current,
                    rng,
                    family,
                    policy,
                    preferred_anchors=preferred,
                    panel_cache=panel_cache,
                )
                program, assignment = extract_program(source, [*stages, stage])
            except ValueError as error:
                failures[f"{family}:{error!s}"] += 1
                continue
            if len(program.marks) > max_primitives or len(program.blocks) > max_blocks:
                failures[f"{family}:work_limit"] += 1
                continue
            accepted = family, product, stage, program, assignment
            break
        if accepted is None:
            if not stages:
                raise ValueError(f"no distilled module executed: {dict(failures)}")
            break
        family, current, stage, program, assignment = accepted
        stages.append(stage)
        selected.append(
            {
                "index": module_index,
                "family": family,
                "parameters": stage["parameters"],
                "primitive_edits": len(stage["actions"]),
                "intermediate_endpoint": stage["endpoint"],
            }
        )
        parameters = stage["parameters"]
        roots = set(parameters.get("retained_boundaries", ()))
        roots.update(parameters.get("ring_slots", ()))
        roots.update(parameters.get("branch_slots", ()))
        if "retained_anchor" in parameters:
            roots.add(parameters["retained_anchor"])
        if "anchor" in parameters:
            roots.add(parameters["anchor"])
        preferred = frozenset(roots)
    program, assignment = extract_program(source, stages)
    product, trace = execute_program_graph(
        source,
        compile_program_graph(program),
        assignment,
        max_primitives=max_primitives,
        max_blocks=max_blocks,
    )
    if canonical_state_key(product) != canonical_state_key(current):
        raise RuntimeError("route-distilled program changed on exact replay")
    return (
        source,
        program,
        assignment,
        trace,
        {
            "schema_version": SCHEMA,
            "training_identity": policy.training_identity,
            "initial_stored_complete_routes": 0,
            "source_library_rows_loaded": 0,
            "requested_module_count": requested,
            "completed_module_count": len(stages),
            "modules": selected,
            "module_failure_counts": dict(failures),
            "intermediate_task_evaluations": 0,
            "runtime_teacher_lookup": False,
        },
    )


def default_reference() -> tuple[float, ...]:
    return tuple(map(float, balanced_option_prior(descriptor_menu())))


def normalized_module_count_probabilities(counts) -> tuple[float, float, float]:
    values = np.ones(3, dtype=float)
    for count, weight in counts:
        if not 1 <= int(count) <= 3 or not math.isfinite(float(weight)) or weight <= 0:
            raise ValueError("invalid distilled module-count observation")
        values[int(count) - 1] += float(weight)
    values /= values.sum()
    return tuple(map(float, values))
