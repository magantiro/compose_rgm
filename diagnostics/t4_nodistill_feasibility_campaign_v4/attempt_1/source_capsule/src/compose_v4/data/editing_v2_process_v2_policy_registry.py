"""Neutral versioned registry for the settled editing policy both chains share.

WHY THIS EXISTS
---------------
``editing_v2_process_v2_contract_chain`` used to carry roughly five hundred lines
of policy transcribed by hand out of the frozen ``configs/editing_v2_semantic_*
_v1.json`` contracts: cell definitions, family contexts, evidence strata,
thresholds, panel cardinalities, optimizer laws.  A hand copy of a settled
definition is the duplication ``AGENTS.md`` forbids, and it had already failed in
the same module for a smaller value: a transcribed schema literal drifted from
the constant its owning adapter declares, and no test could see it because the
test restated the same literal.

This registry replaces the transcription with a **narrow allowlisted projection
from an immutable frozen source**.  Every block names

* the frozen source file,
* that file's pinned physical SHA-256 and pinned self-hash, and
* the exact JSON path inside it that may be projected.

Reading a block re-verifies both pinned identities before returning anything, so
the projection cannot drift from the frozen artifact and the frozen artifact
cannot be edited without this registry refusing.  Nothing here is derived
dynamically, nothing is recomputed from a measured artifact, and no V1 loader is
imported or modified.

WHAT "NEUTRAL" MEANS
--------------------
The registry is authority only for values *proven shared*.  It publishes settled
scientific policy -- what a capability cell is, which contexts a family has, what
a panel's cardinality is -- and states nothing about which process identity a
consumer binds.  A consumer that needs a Process-V2 delta declares that delta
itself; the registry never encodes one.  The blocks below are the exact set a
consumer proved identical between the two chains, and a test asserts that
equality against the frozen source rather than against this module.

WHAT THIS IS NOT
----------------
Not an artifact, not evidence, not authority.  It reads bytes and returns a deep
copy.  It grants no permission of any kind, it does not know the Process-V2
identity, and it never writes.

INVARIANTS MAINTAINED (and tested)
----------------------------------
* every declared block projects successfully from its pinned frozen source;
* a modified frozen source -- one byte, anywhere -- makes every block reading it
  refuse with a message naming the file and which identity moved;
* a projection is a deep copy, so a caller mutating the result cannot poison the
  next read;
* an undeclared block name and an allowlisted path that does not resolve are
  both errors, never a silent empty projection;
* the projected value is byte-identical, under the one canonical encoding, to the
  subtree of the frozen source it names.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compose_v4.data.editing_v2_process_v2_schema import canonical_sha256

# ---- Schema identity ----

POLICY_REGISTRY_SCHEMA = "compose.editing_v2.shared_policy_registry"
POLICY_REGISTRY_SCHEMA_VERSION = 1
POLICY_REGISTRY_STATUS = "SETTLED_SHARED_POLICY_PROJECTION_NO_AUTHORITY"


class PolicyRegistryError(ValueError):
    """A frozen policy source moved, or a block was requested that is not declared."""


# ---- Frozen sources ----


@dataclass(frozen=True)
class _FrozenPolicySource:
    """One immutable file this registry may project, pinned two ways.

    ``file_sha256`` catches any byte change.  ``self_sha256`` catches the case a
    physical pin alone cannot distinguish from an intentional reseal: a body
    edited and correctly re-sealed under a new self-hash.  Checking both means a
    reader learns *which* identity moved.
    """

    relative_path: str
    file_sha256: str
    self_hash_field: str
    self_sha256: str


#: ``source key -> frozen source``.  Every path here is a frozen prospective
#: contract.  None is a measured artifact: a measured result must never become
#: scientific authority merely because it is convenient input lineage.
FROZEN_POLICY_SOURCES: Mapping[str, _FrozenPolicySource] = {
    "active8_decision_runtime": _FrozenPolicySource(
        relative_path="configs/editing_v2_semantic_active8_decision_runtime_v1.json",
        file_sha256="7ec62c76ceab7cda107d02db16a6f656abb89081937b3c89be7103d4697f6c1d",
        self_hash_field="runtime_contract_sha256",
        self_sha256="1332de2fa54c1275c21f0b4d58ab804474f412934c2d20cc8c9468d07d63e270",
    ),
    "capability_cells": _FrozenPolicySource(
        relative_path="configs/editing_v2_semantic_capability_cells_v1.json",
        file_sha256="7ce113eeea5149cb4c8102afcd119e1b461e5388fbdfbfa555beb5fccd34d1e3",
        self_hash_field="registry_sha256",
        self_sha256="68c9a7f0b9339d4c7b62fba8bae33abdf78ff469b07d0cf393cc38c54b7f7ae0",
    ),
    "development_cell_roles": _FrozenPolicySource(
        relative_path="configs/editing_v2_semantic_development_cell_roles_v1.json",
        file_sha256="49a7b174a6b1c0cdf9d8efd0955841d4b95f5f12de855a1e85cbe4ee0254f861",
        self_hash_field="policy_sha256",
        self_sha256="ecf673f5420bc9546571c236739c1f474c98b962a194e5c59665c8f56c18bd10",
    ),
    "gate_zero_structural": _FrozenPolicySource(
        relative_path="configs/editing_v2_semantic_gate_zero_structural_v1.json",
        file_sha256="f19988f39b27c5555094e808b62689ede736fa594857a24ef49b0431b6f901f2",
        self_hash_field="contract_sha256",
        self_sha256="4ee761da80d7e632c3dadc7c78e4355a2a0e4fb17188739178a9a4b0409aa2c6",
    ),
    "t1_panel_policy": _FrozenPolicySource(
        relative_path="configs/editing_v2_semantic_t1_panel_policy_v1.json",
        file_sha256="81cf3154e26358347a9c0267b5520ee6670b322e504bd49466b0195125476ef3",
        self_hash_field="policy_sha256",
        self_sha256="7902d10e2f02785e7d5ddb4ea769d598a95d7655259e24a8546fe85a887bcf34",
    ),
    "t1_capacity_policy": _FrozenPolicySource(
        relative_path="configs/editing_v2_semantic_t1_capacity_policy_v1.json",
        file_sha256="feacddf927597075a5fc0e180c3841b4c4273df602b11a0919396fee1144ed53",
        self_hash_field="policy_sha256",
        self_sha256="08c886739e584c75ce12df7846043890f00ea6fe8821db97f933bbd1784e2e8d",
    ),
    "p50_recipe_policy": _FrozenPolicySource(
        relative_path="configs/editing_v2_semantic_p50_recipe_policy_v1.json",
        file_sha256="6367d4de95f101dc4b8bda0a4b5f33bd274ac50611dcde74071f56cec7d9fb43",
        self_hash_field="policy_sha256",
        self_sha256="295a3155a2e1fc038e9fd6f297bc24a537e3062e3d495ba51b3fd4e88eff35ed",
    ),
}


# ---- The allowlist ----


@dataclass(frozen=True)
class _PolicyBlock:
    """One projectable block: a frozen source plus the exact path allowed out.

    ``path`` is a tuple of object keys, never a pattern and never a wildcard.  An
    allowlist that could match more than one subtree is not an allowlist.
    """

    source: str
    path: tuple[str, ...]


#: ``block name -> allowlisted projection``.  The names are neutral: they say
#: what the value *is*, not which chain reads it.
SHARED_POLICY_BLOCKS: Mapping[str, _PolicyBlock] = {
    # Decision-runtime facts.
    "runtime_model": _PolicyBlock("active8_decision_runtime", ("model",)),
    "runtime_software": _PolicyBlock("active8_decision_runtime", ("software",)),
    # Capability-cell registry facts.
    "cell_scope": _PolicyBlock("capability_cells", ("scope",)),
    "cell_identity_policy": _PolicyBlock("capability_cells", ("cell_identity_policy",)),
    "family_contexts": _PolicyBlock("capability_cells", ("family_contexts",)),
    "exact_evidence_strata": _PolicyBlock("capability_cells", ("exact_evidence_strata",)),
    "fail_closed_policy": _PolicyBlock("capability_cells", ("fail_closed_policy",)),
    "data_lanes": _PolicyBlock("capability_cells", ("bindings", "data_lanes")),
    "partition_roles": _PolicyBlock("capability_cells", ("bindings", "partition_roles")),
    "action_codec_schema_version": _PolicyBlock(
        "capability_cells", ("bindings", "action_codec_schema_version")
    ),
    # Development cell-role facts.
    "required_cell_ids": _PolicyBlock("development_cell_roles", ("required_cell_ids",)),
    "conditional_cell_ids": _PolicyBlock("development_cell_roles", ("conditional_cell_ids",)),
    "separate_lane_cell_ids": _PolicyBlock(
        "development_cell_roles", ("separate_lane_cell_ids",)
    ),
    "conditional_policy": _PolicyBlock("development_cell_roles", ("conditional_policy",)),
    "separate_lane_policy": _PolicyBlock("development_cell_roles", ("separate_lane_policy",)),
    "cell_role_partition_policy": _PolicyBlock(
        "development_cell_roles", ("partition_policy",)
    ),
    "cell_role_scope": _PolicyBlock("development_cell_roles", ("scope",)),
    # Gate-0 structural facts.
    "required_architecture": _PolicyBlock("gate_zero_structural", ("required_architecture",)),
    "structural_checks": _PolicyBlock("gate_zero_structural", ("structural_checks",)),
    "gate_zero_decision_policy": _PolicyBlock("gate_zero_structural", ("decision_policy",)),
}

#: ``policy body name -> (frozen source, allowlisted top-level field names)``.
#:
#: The panel, capacity and recipe policies are projected as a set of top-level
#: fields rather than one subtree, because their bodies live at the root of the
#: frozen artifact alongside its own schema and authority envelope.  Listing the
#: fields explicitly keeps the envelope out: a projection that swept the whole
#: root would carry the frozen artifact's ``schema``, ``status`` and authority
#: flags into a different artifact's body.
SHARED_POLICY_BODIES: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "t1_panel_body": (
        "t1_panel_policy",
        (
            "cache_handoff",
            "empirical_multiplicity_receipts_included",
            "gate_thresholds_included",
            "hazard_included",
            "maximum_entries_by_family",
            "minimum_entries_by_family",
            "objective_unit",
            "optimizer_policy_included",
            "p50_policy_included",
            "panel_kind",
            "repeated_state_panel_included",
            "successor_fiber_cache_compiled",
            "support_time_hex",
        ),
    ),
    "t1_capacity_body": (
        "t1_capacity_policy",
        (
            "empirical_repeated_state_gate",
            "hazard_included",
            "objective_unit",
            "optimization",
            "panel_cardinality",
            "panel_kind",
            "required_families",
            "sampling_law",
            "support_time_hex",
            "thresholds",
        ),
    ),
    "p50_recipe_body": (
        "p50_recipe_policy",
        (
            "active_families",
            "cache",
            "objective",
            "optimization",
            "p500_authorized",
            "required_physical_binding_purposes",
            "sampling",
            "scientific_scope",
            "thresholds",
            "time_derivation",
        ),
    ),
}


# ---- Reading ----


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_frozen_source(key: str, *, repo_root: Path) -> dict[str, Any]:
    """Read one frozen source after proving both pinned identities still hold."""

    try:
        source = FROZEN_POLICY_SOURCES[key]
    except KeyError:
        raise PolicyRegistryError(
            f"{key!r} is not a declared frozen policy source; expected one of "
            f"{sorted(FROZEN_POLICY_SOURCES)}"
        ) from None
    path = Path(repo_root) / source.relative_path
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PolicyRegistryError(
            f"the frozen policy source {source.relative_path} is not readable"
        ) from error
    physical = _sha256_bytes(raw)
    if physical != source.file_sha256:
        raise PolicyRegistryError(
            f"the frozen policy source {source.relative_path} has physical hash "
            f"{physical}, not the pinned {source.file_sha256}; a frozen source may "
            "not be edited, and this registry projects nothing from a moved one"
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PolicyRegistryError(
            f"the frozen policy source {source.relative_path} is not valid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise PolicyRegistryError(
            f"the frozen policy source {source.relative_path} is not a JSON object"
        )
    declared = payload.get(source.self_hash_field)
    body = {key_: value for key_, value in payload.items() if key_ != source.self_hash_field}
    if declared != source.self_sha256 or canonical_sha256(body) != source.self_sha256:
        raise PolicyRegistryError(
            f"the frozen policy source {source.relative_path} declares "
            f"{source.self_hash_field}={declared!r} over a body hashing to "
            f"{canonical_sha256(body)}; the pinned self-hash is {source.self_sha256}"
        )
    return payload


def _resolve(payload: Mapping[str, Any], path: Sequence[str], *, label: str) -> Any:
    node: Any = payload
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise PolicyRegistryError(
                f"the allowlisted path {'.'.join(path)} does not resolve in {label}"
            )
        node = node[key]
    return node


def project_shared_policy(block: str, *, repo_root: Path) -> Any:
    """Return a deep copy of one allowlisted block from its frozen source.

    Raises:
        PolicyRegistryError: if the block is not declared, its frozen source
            moved, or the allowlisted path does not resolve.
    """

    try:
        declared = SHARED_POLICY_BLOCKS[block]
    except KeyError:
        raise PolicyRegistryError(
            f"{block!r} is not a declared shared policy block; expected one of "
            f"{sorted(SHARED_POLICY_BLOCKS)}"
        ) from None
    payload = _read_frozen_source(declared.source, repo_root=repo_root)
    label = FROZEN_POLICY_SOURCES[declared.source].relative_path
    return copy.deepcopy(_resolve(payload, declared.path, label=label))


def project_shared_policy_body(body: str, *, repo_root: Path) -> dict[str, Any]:
    """Return a deep copy of one allowlisted top-level field set.

    Raises:
        PolicyRegistryError: if the body is not declared, its frozen source
            moved, or an allowlisted field is absent.
    """

    try:
        source_key, fields = SHARED_POLICY_BODIES[body]
    except KeyError:
        raise PolicyRegistryError(
            f"{body!r} is not a declared shared policy body; expected one of "
            f"{sorted(SHARED_POLICY_BODIES)}"
        ) from None
    payload = _read_frozen_source(source_key, repo_root=repo_root)
    label = FROZEN_POLICY_SOURCES[source_key].relative_path
    return {
        field: copy.deepcopy(_resolve(payload, (field,), label=label)) for field in fields
    }


def policy_registry_identity(*, repo_root: Path) -> dict[str, Any]:
    """The deterministic descriptor a consumer records as provenance.

    It names every frozen source and its verified identities, so an artifact
    built from this registry records exactly which immutable bytes it projected.
    """

    sources = {}
    for key in sorted(FROZEN_POLICY_SOURCES):
        source = FROZEN_POLICY_SOURCES[key]
        _read_frozen_source(key, repo_root=repo_root)
        sources[key] = {
            "file_sha256": source.file_sha256,
            "path": source.relative_path,
            "self_hash_field": source.self_hash_field,
            "semantic_sha256": source.self_sha256,
        }
    body = {
        "blocks": sorted(SHARED_POLICY_BLOCKS),
        "bodies": sorted(SHARED_POLICY_BODIES),
        "frozen_sources": sources,
        "schema": POLICY_REGISTRY_SCHEMA,
        "schema_version": POLICY_REGISTRY_SCHEMA_VERSION,
        "status": POLICY_REGISTRY_STATUS,
    }
    return {**body, "registry_identity_sha256": canonical_sha256(body)}


__all__ = [
    "FROZEN_POLICY_SOURCES",
    "POLICY_REGISTRY_SCHEMA",
    "POLICY_REGISTRY_SCHEMA_VERSION",
    "POLICY_REGISTRY_STATUS",
    "SHARED_POLICY_BLOCKS",
    "SHARED_POLICY_BODIES",
    "PolicyRegistryError",
    "policy_registry_identity",
    "project_shared_policy",
    "project_shared_policy_body",
]
