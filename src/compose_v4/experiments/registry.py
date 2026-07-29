"""Load and validate the frozen experiment registry.

The registry is the pre-commitment device for the whole experimental program: objectives, splits, budgets,
seeds, normalizations, hypervolume reference points and success definitions are declared before any result
is inspected. A loader that silently tolerated a missing or altered block would defeat that, so every check
here fails loudly instead of defaulting.

Three properties are enforced:

* **Version.** A v1 registry is REJECTED rather than partially read. v2 renamed ``primary_metric``/
  ``metrics``/``panel`` to ``primary_metrics``/``secondary_metrics``/``panel_ids``; a v1 file read under v2
  expectations would look like it had no primary metrics at all.
* **Metric hierarchy.** Each experiment declares 1-3 PRIMARY metrics. Too many coequal metrics invite
  assembling a story after the fact, which is the failure the freeze exists to prevent.
* **Freeze integrity.** ``protocol.protocol_freeze.content_hash`` is recomputed from the protocol block on
  load. An edit to any frozen protocol field changes the hash, so post-hoc protocol drift is detectable
  rather than invisible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REGISTRY_SCHEMA = "compose.experiments.registry"
REGISTRY_SCHEMA_VERSION = 2

# Result artifacts carry a SEMANTIC panel id, never a figure number, so the manuscript layout can be
# rearranged (or exact and dynamic panels merged) without touching experiment code.
SEMANTIC_PANEL_IDS = frozenset({
    "GEN_UNCONDITIONAL",
    "TRANSPORT_LEARNED_VS_UNIFORM",
    "CARDINALITY_ADAPTATION",
    "TOPOLOGY_ADAPTATION",
    "EXACT_TERMINAL_TILT",
    "QUOTIENT_INVARIANCE",
    "PARETO_SAME_BASE",
    "DYNAMIC_SWITCH",
    "PARETO_FAN",
    "PATHWISE_CONSTRAINTS",
})

_REQUIRED_PROTOCOL_BLOCKS = (
    "seeds",
    "successor_kernel",
    "editing_budget_convention",
    "held_out_policy",
    "similarity",
    "normalization_policy",
    "objectives",
    "multi_objective",
    "dynamic",
    "compute_accounting",
    "task_generation",
    "hard_gates",
    "metric_definitions",
    "protocol_freeze",
)

_REQUIRED_TOP_LEVEL = (
    "schema",
    "schema_version",
    "provenance",
    "protocol",
    "experiments",
    "exact_sizing",
    "controllers",
    "learned_controller",
    "post_training_queue",
)

_REQUIRED_COMPUTE_FIELDS = frozenset({
    "oracle_calls",
    "forward_passes",
    "successors_scored",
    "particles_propagated",
    "accepted_edits",
    "wall_clock_seconds",
    "gpu_hours",
})

MAX_PRIMARY_METRICS = 3


class RegistryViolation(ValueError):
    """The registry is missing, malformed, stale, or has drifted from its frozen content hash."""


def protocol_content_hash(protocol: dict[str, Any]) -> str:
    """Hash of the frozen protocol, excluding the freeze record that carries the hash itself."""
    payload = {key: value for key, value in protocol.items() if key != "protocol_freeze"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]


def load_registry(path: str | Path, *, verify_freeze: bool = True) -> dict[str, Any]:
    """Load, validate and return the registry. Raises ``RegistryViolation`` on any problem."""
    path = Path(path)
    if not path.is_file():
        raise RegistryViolation(f"experiment registry not found: {path}")
    try:
        registry = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise RegistryViolation(f"experiment registry is not valid YAML: {error}") from error
    if not isinstance(registry, dict):
        raise RegistryViolation("experiment registry must be a mapping")

    if registry.get("schema") != REGISTRY_SCHEMA:
        raise RegistryViolation(f"unexpected registry schema: {registry.get('schema')!r}")
    version = registry.get("schema_version")
    if version != REGISTRY_SCHEMA_VERSION:
        raise RegistryViolation(
            f"registry schema_version {version!r} != {REGISTRY_SCHEMA_VERSION}. v2 renamed "
            "primary_metric/metrics/panel to primary_metrics/secondary_metrics/panel_ids; reading a v1 "
            "file here would silently report no primary metrics. Re-run the migration."
        )

    missing_top = [key for key in _REQUIRED_TOP_LEVEL if key not in registry]
    if missing_top:
        raise RegistryViolation(f"registry is missing required top-level blocks: {missing_top}")

    protocol = registry["protocol"]
    if not isinstance(protocol, dict):
        raise RegistryViolation("registry protocol must be a mapping")
    missing_protocol = [key for key in _REQUIRED_PROTOCOL_BLOCKS if key not in protocol]
    if missing_protocol:
        raise RegistryViolation(
            f"frozen protocol is missing required blocks: {missing_protocol}. These must be declared "
            "BEFORE any experiment result is inspected."
        )

    declared_compute = set(protocol["compute_accounting"].get("required_per_arm", ()))
    missing_compute = sorted(_REQUIRED_COMPUTE_FIELDS - declared_compute)
    if missing_compute:
        raise RegistryViolation(
            f"compute_accounting must require every cost field, missing: {missing_compute}. Counting "
            "committed states for one arm while ignoring particles or reranked candidates for another "
            "makes the controller comparison unfair."
        )

    experiments = registry["experiments"]
    if not isinstance(experiments, dict) or not experiments:
        raise RegistryViolation("registry must declare at least one experiment")
    for key, experiment in sorted(experiments.items()):
        _validate_experiment(key, experiment)

    if verify_freeze:
        verify_protocol_freeze(registry)
    return registry


def _validate_experiment(key: str, experiment: Any) -> None:
    if not isinstance(experiment, dict):
        raise RegistryViolation(f"experiment {key} must be a mapping")
    for field in ("question", "checkpoint", "primary_metrics", "panel_ids", "task_artifacts"):
        if field not in experiment:
            raise RegistryViolation(f"experiment {key} is missing required field {field!r}")

    primary = experiment["primary_metrics"]
    if not isinstance(primary, list) or not primary:
        raise RegistryViolation(f"experiment {key} must declare at least one primary metric")
    if len(primary) > MAX_PRIMARY_METRICS:
        raise RegistryViolation(
            f"experiment {key} declares {len(primary)} primary metrics (max {MAX_PRIMARY_METRICS}). "
            "Too many coequal primaries let a result be assembled after the fact; demote the rest to "
            "secondary_metrics."
        )
    if len(set(primary)) != len(primary):
        raise RegistryViolation(f"experiment {key} repeats a primary metric: {primary}")

    overlap = set(primary) & set(experiment.get("secondary_metrics", ()) or ())
    if overlap:
        raise RegistryViolation(
            f"experiment {key} lists {sorted(overlap)} as both primary and secondary"
        )

    panels = experiment["panel_ids"]
    if not isinstance(panels, list) or not panels:
        raise RegistryViolation(f"experiment {key} must declare at least one panel id")
    unknown = sorted(set(panels) - SEMANTIC_PANEL_IDS)
    if unknown:
        raise RegistryViolation(
            f"experiment {key} uses unknown panel ids {unknown}; panel ids must be semantic and drawn "
            f"from {sorted(SEMANTIC_PANEL_IDS)} -- never a figure number, which would hard-code layout."
        )

    artifacts = experiment["task_artifacts"]
    if not isinstance(artifacts, dict) or {"development", "final"} - set(artifacts):
        raise RegistryViolation(
            f"experiment {key} must declare BOTH development and final task artifacts: development "
            "tasks are for debugging and tuning, final tasks are generated once from the frozen "
            "algorithm and used once"
        )


def verify_protocol_freeze(registry: dict[str, Any]) -> None:
    """Recompute the protocol content hash and refuse a drifted registry."""
    protocol = registry["protocol"]
    freeze = protocol.get("protocol_freeze")
    if not isinstance(freeze, dict):
        raise RegistryViolation("protocol.protocol_freeze is missing or malformed")
    recorded = freeze.get("content_hash")
    actual = protocol_content_hash(protocol)
    if recorded != actual:
        raise RegistryViolation(
            f"frozen protocol has DRIFTED: recorded content_hash {recorded!r} != recomputed {actual!r}. "
            "A protocol field changed after the freeze. If the change is intended it must be recorded as "
            "an explicit contract update with a reason, not absorbed silently."
        )


def experiment(registry: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    """Fetch one experiment, failing loudly on an unknown id."""
    try:
        return registry["experiments"][experiment_id]
    except KeyError:
        known = sorted(registry.get("experiments", {}))
        raise RegistryViolation(
            f"unknown experiment id {experiment_id!r}; the registry declares {known}"
        ) from None
