"""Deterministic builder, validator and transactional publisher for the Process-V2 chain.

WHAT THIS IS
------------
Editing-V2 froze a seven-artifact prospective contract chain under the V1 semantic
process identity (``configs/editing_v2_semantic_*_v1.json``).  Process-V2 changed
``atom_delete`` admission, which moved ``editing_process_v2_identity()``, so every
one of those artifacts describes a process that is no longer the live one.  Because
the V1 artifacts are frozen and must never be relabelled, the Process-V2 chain is a
SEPARATE set of seven configs that MIRRORS the V1 policy while binding the live
Process-V2 identity:

===  ==================================================================  ==========
 #   ``configs/editing_v2_process_v2_<kind>.json``                        mirrors
===  ==================================================================  ==========
 1   ``..._active8_decision_runtime.json``                                the V1 CPU decision runtime
 2   ``..._capability_cells.json``                                        the V1 capability-cell registry
 3   ``..._development_cell_roles.json``                                  the V1 development cell roles
 4   ``..._gate_zero_structural.json``                                    the V1 Gate-0 structural contract
 5   ``..._t1_panel_policy.json``                                         the V1 T1 panel policy
 6   ``..._t1_capacity_policy.json``                                      the V1 T1 capacity policy
 7   ``..._p50_recipe_policy.json``                                       the V1 P50 recipe policy
===  ==================================================================  ==========

MIRROR, WITH ONE VERSIONED OPERATIONAL DELTA
--------------------------------------------
No scientific policy value moves.  Cell definitions, family contexts, data lanes,
partition roles, the ``editing_v2_active8_v1`` cell namespace, the 17 required / 3
conditional / 2 separate-lane counts, panel cardinalities, every threshold, the
optimizer and sampling laws are **projected** from the frozen V1 contracts through
``compose_v4.data.editing_v2_process_v2_policy_registry`` rather than transcribed
here.  Process-V2 T1 operational-semantics version 1 evaluates only at step zero
and the frozen report points, and makes step 10 the earliest eligible threshold
stop.  The validator proves every other T1 value is still identical to V1.  The
registry pins each frozen source by physical hash and self-hash, so a projection
cannot drift and an edited frozen source makes the build refuse.  This replaces
roughly five hundred lines of hand-copied policy; the projection was proved
byte-identical to every constant it replaced before those constants were deleted.

THE DEPENDENCY GRAPH IS EXPLICIT
--------------------------------
Tuple order is not a dependency edge.  Every edge is a **declared typed pointer**
carrying its kind, provider, target schema, identity role and hash algorithm, so a
verifier discovers it structurally instead of guessing that a string which happens
to name an existing file is an edge.  A declared pointer whose target is deleted or
misspelled cannot vanish from the graph, which is what made the previous verifier's
missing-target diagnostic unreachable.

Schema version 2 adds the edges Gate 0 -> T1 panel -> T1 capacity -> P50 that
version 1 left implicit:

* the T1 panel policy binds the Process-V2 Gate-0 structural contract and the
  frozen corpus contract that states its source requirements;
* the T1 capacity policy binds the T1 panel policy and Gate 0;
* the P50 recipe policy binds T1 capacity, T1 panel, Gate 0, and every other exact
  policy it consumes: the cell roles it round-robins over, the capability-cell
  registry that defines those cells, the decision runtime it binds as trainer
  runtime, and the corpus contract naming its source population.

No earlier artifact points at a later one.  Measured Gate-0, T1 and P50 decisions
bind these immutable policies from their own side.

MEASURED EVIDENCE LIVES ELSEWHERE
---------------------------------
Version 1 carried an ``admitted_source`` block with null hashes, advertised as
later fillable.  It was not: the builder always emits null, the validator rebuilds
the null body and requires equality, so a correctly resealed body with measured
hashes fails.  Confirmed against this code before the slot was removed.  Measured
provenance now lives in ``editing_v2_process_v2_evidence_binding``, a separate
versioned artifact that cites a contract and pins what a run measured.  Each
contract only *declares* which schema will cite it, which is a constant and never a
hash, so the contract stays deterministic and never addresses a later result.

SEMANTIC HASH RULE
------------------
A pointer states its own algorithm rather than leaving it to convention.
``self_hash_field_v1`` means the target declares a unique ``*_sha256`` field equal
to the canonical hash of its body minus that field.  ``whole_canonical_body_v1``
means the target declares no such field, so its whole canonical body is hashed
(this is what ``configs/editing_corpus_v2_contract.json`` needs).  A non-JSON target
has no separable semantic body and is therefore pinned by physical hash only,
rather than by a "semantic" hash that is secretly the physical one.

WHAT THIS DOES NOT AUTHORIZE
----------------------------
Nothing.  These are prospective binding contracts, not evidence and not permission.
Every artifact carries the complete frozen authority vocabulary, every field
``false``, and every ``status`` ends in ``_NO_DOWNSTREAM_AUTHORITY``.  Building,
validating or publishing this chain does not run Gate 0, T1 or P50, does not
authorize a Modal launch or any training, and does not revalidate evidence produced
under a superseded identity: such evidence stays invalid regardless.

INVARIANTS MAINTAINED (and tested)
----------------------------------
* build -> serialize -> validate -> load round-trips for all seven artifacts;
* the whole chain is built parents-first through an in-memory overlay and is
  byte-stable across repeated runs;
* publication is transactional: artifacts are written under one content-addressed
  generation directory, the complete graph is validated there, and the
  ``COMMITTED.json`` marker is written last.  A reader ignores any generation
  without a valid marker, so an interruption cannot expose a partial generation;
* every declared pointer resolves to the target's live physical or semantic hash;
* a V1 identity pin, a ``_v1`` parent path, a changed Active8 order, a non-``False``
  authority flag, a disagreeing self-hash, a changed field set, and a missing
  dependency edge each raise :class:`ProcessV2ChainError` with a message naming
  that specific defect.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from compose_v4.data.editing_corpus_contract import ACTIVE8_FAMILIES
from compose_v4.data.editing_v2_process_v2_policy_registry import (
    PolicyRegistryError,
    policy_registry_identity,
    project_shared_policy,
    project_shared_policy_body,
)
from compose_v4.data.editing_v2_process_v2_schema import (
    AUTHORITY_FIELDS,
    GENERATION_COMMITTED_MARKER,
    GENERATION_COMMITTED_SCHEMA,
    GENERATION_COMMITTED_SCHEMA_VERSION,
    RESOLVED_EVIDENCE_BINDING_SCHEMA,
    RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION,
    IdentityRole,
    PointerKind,
    ProcessV2SchemaError,
    authority_false_block,
    canonical_sha256,
    require_authority_false,
    require_no_granted_authority,
    self_hashed,
    typed_pointer,
    validate_typed_pointer,
    verify_self_hash,
)
from compose_v4.rewrite.editing_v2_process_identity import (
    editing_process_v2_identity,
    editing_v2_process_identity,
)

# ---- Errors ----


class ProcessV2ChainError(ValueError):
    """A Process-V2 chain artifact is unbuildable, mixed with V1, or inconsistent."""


# ---- Chain membership ----

ACTIVE8_DECISION_RUNTIME = "configs/editing_v2_process_v2_active8_decision_runtime.json"
CAPABILITY_CELLS = "configs/editing_v2_process_v2_capability_cells.json"
DEVELOPMENT_CELL_ROLES = "configs/editing_v2_process_v2_development_cell_roles.json"
GATE_ZERO_STRUCTURAL = "configs/editing_v2_process_v2_gate_zero_structural.json"
T1_PANEL_POLICY = "configs/editing_v2_process_v2_t1_panel_policy.json"
T1_CAPACITY_POLICY = "configs/editing_v2_process_v2_t1_capacity_policy.json"
P50_RECIPE_POLICY = "configs/editing_v2_process_v2_p50_recipe_policy.json"

#: The seven chain artifacts in dependency order; parents always precede children.
PROCESS_V2_CHAIN_ARTIFACTS: tuple[str, ...] = (
    ACTIVE8_DECISION_RUNTIME,
    CAPABILITY_CELLS,
    DEVELOPMENT_CELL_ROLES,
    GATE_ZERO_STRUCTURAL,
    T1_PANEL_POLICY,
    T1_CAPACITY_POLICY,
    P50_RECIPE_POLICY,
)

# Frozen external parents.  None is a V1-named artifact.
GATE_ZERO_MODEL_PROCESS_V2 = "configs/editing_gate_zero_semantic_model_process_v2.json"
SEMANTIC_PROCESS_V2 = "configs/editing_v2_semantic_process_v2.json"
EDITING_CORPUS_V2_CONTRACT = "configs/editing_corpus_v2_contract.json"
CAPABILITY_CELL_CLASSIFIER = "src/compose_v4/data/editing_v2_semantic_capability_cells.py"

#: The single self-hash field name every chain artifact carries.
SELF_HASH_FIELD = "contract_sha256"

#: Version 1 mirrored the V1 policy and carried an unfillable measured-evidence
#: slot.  Version 2 made the dependency graph explicit and removed that slot.
#: Version 3 declares resolved-evidence-binding schema 2 -- the binding now embeds
#: the admitted-source descriptor verbatim instead of projecting a field list that
#: named three hashes the adapter never emitted -- and carries the whole design
#: lineage rather than only the immediately preceding generation.  Version 4
#: makes the Process-V2 Gate-0 policy successor-directed: whole-corpus evidence
#: proves exact teacher support plus a productive nonself successor, while full
#: quotient size and alias multiplicity move to the completion-bound sentinel.
#: Version 5 prospectively repairs the bounded T1 protocol after its first
#: completed NO_GO: it samples the already family-balanced 512-entry panel
#: uniformly by unique entry and adds a low-rate stabilization phase.  The panel,
#: objective, thresholds, model, seed, batch size, and hazard exclusion do not
#: move.
#: No generation is overwritten or resealed; each is preserved as lineage.
CHAIN_SCHEMA_VERSION = 5

#: The named successor generation, following this repository's ``contract_revision``
#: convention (see ``configs/editing_v2_semantic_process_v2.json``).
CHAIN_CONTRACT_REVISION = "process_v2_unique_entry_two_stage_t1_capacity"

#: Versioned Process-V2-only recovery protocol for the T1 capacity run.  The
#: shared V1 policy remains frozen.  Process V2 keeps its exact panel, objective,
#: thresholds, model, seed, batch size, optimizer, and hazard exclusion, while
#: replacing the discovered hierarchical exposure imbalance with equal unique-
#: entry sampling and adding one bounded low-rate stabilization phase.  Recovery
#: restarts from the exact scratch state; no historical optimizer state is reused.
PROCESS_V2_T1_OPERATIONAL_SEMANTICS_VERSION = 2
PROCESS_V2_T1_TRAJECTORY_EVALUATION = "step_zero_and_report_points_only"
PROCESS_V2_T1_EARLY_STOP_RULE = (
    "all_required_family_nonempty_cell_and_entry_thresholds_pass_at_one_"
    "evaluated_report_point_after_minimum_optimizer_steps"
)
PROCESS_V2_T1_MINIMUM_STEPS_BEFORE_EARLY_STOP = 10
PROCESS_V2_T1_SAMPLING_LAW = {
    "importance_correction": "none",
    "order": ["unique_panel_entry"],
    "probability_within_each_level": "uniform_over_unique_panel_entries",
    "target_coefficient": 1.0,
}
PROCESS_V2_T1_LEARNING_RATE_SCHEDULE = [
    {
        "first_optimizer_step": 1,
        "last_optimizer_step": 500,
        "learning_rate": 0.001,
    },
    {
        "first_optimizer_step": 501,
        "last_optimizer_step": 750,
        "learning_rate": 0.0001,
    },
]
PROCESS_V2_T1_REPORT_POINTS = [1, 10, 50, 100, 250, 500, 600, 700, 750]
PROCESS_V2_T1_MAXIMUM_OPTIMIZER_STEPS = 750
PROCESS_V2_T1_SCHEDULER = "piecewise_constant_by_optimizer_step"

#: The revision the version-1 bodies were sealed under.  Kept as a named constant
#: because it is the oldest generation and several tests address it by name.
SUPERSEDED_CHAIN_CONTRACT_REVISION = (
    "process_v2_mirrored_chain_with_unfillable_admitted_source_slot"
)

#: The revision the version-2 bodies were sealed under.
SUPERSEDED_CHAIN_CONTRACT_REVISION_V2 = "process_v2_explicit_dependency_graph"

#: The revision the version-3 bodies were sealed under.
SUPERSEDED_CHAIN_CONTRACT_REVISION_V3 = (
    "process_v2_binding_v2_and_cumulative_design_lineage"
)

#: The revision the version-4 bodies were sealed under.
SUPERSEDED_CHAIN_CONTRACT_REVISION_V4 = "process_v2_gate_zero_productive_nonself"

STATUS_SUFFIX = "_NO_DOWNSTREAM_AUTHORITY"

PROCESS_IDENTITY_PIN_FIELD = "process_identity_sha256"
PROCESS_V2_IDENTITY_PROVIDER = "editing_process_v2_identity"
PROCESS_V2_IDENTITY_SCHEMA = "compose.editing.semantic_process_v2_identity"
PROCESS_IDENTITY_MODULE = "src/compose_v4/rewrite/editing_v2_process_identity.py"

#: The exact field set a declared process-identity edge carries.  A verifier
#: resolves it CONTEXTUALLY -- from provider, identity schema, process semantics
#: and value together -- to exactly one identity node, instead of accepting either
#: live identity and reporting the pin as addressing both.
PROCESS_IDENTITY_EDGE_FIELDS: tuple[str, ...] = (
    "identity_schema",
    "identity_schema_version",
    "module",
    PROCESS_IDENTITY_PIN_FIELD,
    "process_semantics",
    "provider",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_V1_STEM = re.compile(r"_v1(?:$|_)")

# ---- Semantic hash algorithms ----

SELF_HASH_FIELD_ALGORITHM = "self_hash_field_v1"
WHOLE_CANONICAL_BODY_ALGORITHM = "whole_canonical_body_v1"


# ---- Per-artifact envelope metadata ----

_ENVELOPE: dict[str, dict[str, str]] = {
    ACTIVE8_DECISION_RUNTIME: {
        "schema": "compose.editing_v2.process_v2_active8_decision_runtime",
        "contract_id": "editing_v2_process_v2_active8_decision_runtime",
        "status": "FROZEN_PROCESS_V2_CPU_DECISION_RUNTIME" + STATUS_SUFFIX,
    },
    CAPABILITY_CELLS: {
        "schema": "compose.editing_v2.process_v2_capability_cells",
        "contract_id": "editing_v2_process_v2_capability_cells",
        "status": "FROZEN_PROCESS_V2_SEMANTIC_CAPABILITY_CELLS" + STATUS_SUFFIX,
    },
    DEVELOPMENT_CELL_ROLES: {
        "schema": "compose.editing_v2.process_v2_development_cell_roles",
        "contract_id": "editing_v2_process_v2_development_cell_roles",
        "status": "FROZEN_PROCESS_V2_DEVELOPMENT_CELL_ROLES" + STATUS_SUFFIX,
    },
    GATE_ZERO_STRUCTURAL: {
        "schema": "compose.editing_v2.process_v2_gate_zero_structural",
        "contract_id": "editing_v2_process_v2_gate_zero_structural",
        "status": "FROZEN_PROCESS_V2_GATE_ZERO_STRUCTURAL" + STATUS_SUFFIX,
    },
    T1_PANEL_POLICY: {
        "schema": "compose.editing_v2.process_v2_t1_panel_policy",
        "contract_id": "editing_v2_process_v2_t1_panel_policy",
        "status": "FROZEN_PROCESS_V2_T1_PANEL_POLICY" + STATUS_SUFFIX,
    },
    T1_CAPACITY_POLICY: {
        "schema": "compose.editing_v2.process_v2_t1_capacity_policy",
        "contract_id": "editing_v2_process_v2_t1_capacity_policy",
        "status": "FROZEN_PROCESS_V2_T1_CAPACITY_POLICY" + STATUS_SUFFIX,
    },
    P50_RECIPE_POLICY: {
        "schema": "compose.editing_v2.process_v2_p50_recipe_policy",
        "contract_id": "editing_v2_process_v2_p50_recipe_policy",
        "status": "FROZEN_PROCESS_V2_P50_RECIPE_POLICY" + STATUS_SUFFIX,
    },
}

#: ``target -> the producer of that target's bytes``.  Declared, because "who
#: computed this value" is not recoverable from a path.
_PROVIDERS: dict[str, str] = {
    GATE_ZERO_MODEL_PROCESS_V2: "frozen_repository_artifact",
    SEMANTIC_PROCESS_V2: "editing_process_v2_contract",
    EDITING_CORPUS_V2_CONTRACT: "frozen_repository_artifact",
    CAPABILITY_CELL_CLASSIFIER: "repository_source_file",
}
_CHAIN_PROVIDER = "editing_v2_process_v2_contract_chain"

#: ``artifact -> {edge role: target relative path}``.  No value is a V1 config.
#:
#: Schema version 2 added the T1 and P50 edges marked below.  Version 1 encoded
#: only the cell-role edge for each of the three policies, which left the
#: Gate 0 -> T1 -> P50 order to display convention rather than to the graph.
_PARENTS: dict[str, dict[str, str]] = {
    ACTIVE8_DECISION_RUNTIME: {
        "semantic_model_process": GATE_ZERO_MODEL_PROCESS_V2,
    },
    CAPABILITY_CELLS: {
        "classifier_implementation": CAPABILITY_CELL_CLASSIFIER,
        "editing_corpus_contract": EDITING_CORPUS_V2_CONTRACT,
    },
    DEVELOPMENT_CELL_ROLES: {
        "capability_cell_registry": CAPABILITY_CELLS,
    },
    GATE_ZERO_STRUCTURAL: {
        "capability_cell_registry": CAPABILITY_CELLS,
        "decision_runtime": ACTIVE8_DECISION_RUNTIME,
        "development_cell_roles": DEVELOPMENT_CELL_ROLES,
        "semantic_model_process": GATE_ZERO_MODEL_PROCESS_V2,
        "semantic_process": SEMANTIC_PROCESS_V2,
    },
    T1_PANEL_POLICY: {
        "development_cell_roles": DEVELOPMENT_CELL_ROLES,
        "gate_zero_structural": GATE_ZERO_STRUCTURAL,  # added in schema version 2
        "source_requirements": EDITING_CORPUS_V2_CONTRACT,  # added in schema version 2
    },
    T1_CAPACITY_POLICY: {
        "development_cell_roles": DEVELOPMENT_CELL_ROLES,
        "gate_zero_structural": GATE_ZERO_STRUCTURAL,  # added in schema version 2
        "t1_panel_policy": T1_PANEL_POLICY,
    },
    P50_RECIPE_POLICY: {
        "capability_cell_registry": CAPABILITY_CELLS,  # added in schema version 2
        "decision_runtime": ACTIVE8_DECISION_RUNTIME,  # added in schema version 2
        "development_cell_roles": DEVELOPMENT_CELL_ROLES,
        "gate_zero_structural": GATE_ZERO_STRUCTURAL,  # added in schema version 2
        "source_requirements": EDITING_CORPUS_V2_CONTRACT,  # added in schema version 2
        "t1_capacity_policy": T1_CAPACITY_POLICY,  # added in schema version 2
        "t1_panel_policy": T1_PANEL_POLICY,  # added in schema version 2
    },
}

#: The edges schema version 2 introduced, kept explicit so a reviewer can see the
#: delta without diffing two generations of JSON.
DEPENDENCY_EDGES_ADDED_IN_SCHEMA_VERSION_2: Mapping[str, tuple[str, ...]] = {
    T1_PANEL_POLICY: ("gate_zero_structural", "source_requirements"),
    T1_CAPACITY_POLICY: ("gate_zero_structural",),
    P50_RECIPE_POLICY: (
        "capability_cell_registry",
        "decision_runtime",
        "gate_zero_structural",
        "source_requirements",
        "t1_capacity_policy",
        "t1_panel_policy",
    ),
}

#: Every superseded generation of each artifact, oldest first, preserved as
#: design lineage rather than discarded.  A generation is appended, never
#: replaced: version 2 recorded only version 1, so republishing under version 3
#: would have dropped version 1 entirely and left the chain unable to say what it
#: had been before its immediately preceding shape.
#:
#: The version-1 values are those sealed at ``3f3258e``.  The version-2 values are
#: those sealed at ``b39420c`` and inherited byte-for-byte at this base, captured
#: from the committed configs before the version-3 rebuild replaced them.
#:
#: None of these carries a currency claim and none may equal a live value.
_SUPERSEDED_GENERATIONS: tuple[tuple[int, str, dict[str, dict[str, str]]], ...] = (
    (
        1,
        SUPERSEDED_CHAIN_CONTRACT_REVISION,
        {
            ACTIVE8_DECISION_RUNTIME: {
                "contract_sha256": (
                    "45f3de02da92991802c4cc192d00ceac47291e5ac3d177d19630799808df1f8d"
                ),
                "file_sha256": (
                    "93151f51dac9962237e9d98fa9b0d790602a948ec65b60da209222be9ae016c5"
                ),
            },
            CAPABILITY_CELLS: {
                "contract_sha256": (
                    "d37105d41599a129f527f152ea0d3789c624addd76cbb538a285308a11ce6d99"
                ),
                "file_sha256": (
                    "10d399f65362db588ace32b03c8b8930a641a044c1b6504f389f8109a0f1a592"
                ),
            },
            DEVELOPMENT_CELL_ROLES: {
                "contract_sha256": (
                    "55348f85c6694dac362528d378162e02d48cef464348c2b0d8c0d0e5381f1221"
                ),
                "file_sha256": (
                    "4d9ee72310e6ab191bf12e15b65f9d8b650931c4858f491e277c184d62f9b017"
                ),
            },
            GATE_ZERO_STRUCTURAL: {
                "contract_sha256": (
                    "c2b469fb3f91f712f8fa9bc6fb8d1ca761fb648ee712c4fc11614c9d33e3bb25"
                ),
                "file_sha256": (
                    "386e37012807da508f6b060a37e7f8f2b97b561d7e8e4c95f58ed51584dac4e8"
                ),
            },
            T1_PANEL_POLICY: {
                "contract_sha256": (
                    "d181ab3946439e374d9844cb6d1c864f925b6a6dfa7197eacf2e04576ef3b184"
                ),
                "file_sha256": (
                    "67aef9d024b762c980833c5b5aaf327e8cfb22be85b3aa241a9983fe058199aa"
                ),
            },
            T1_CAPACITY_POLICY: {
                "contract_sha256": (
                    "7fa4862d844034d2cea6c998edc78883fdfceda497c6c39df26779149907bcf6"
                ),
                "file_sha256": (
                    "3465b0378894ba4ebe2545942eafbf5d2391bf6dc9620c9e4cb9e4648f1b700f"
                ),
            },
            P50_RECIPE_POLICY: {
                "contract_sha256": (
                    "1a6de7e76bed2aa2a3c52ec5b68fb78a1484a924d2c6f17a712384a27b47bef8"
                ),
                "file_sha256": (
                    "97024a28e161763f2f7e123e8e85eb6ff28a9ac8f5eda71343696bc976bcd716"
                ),
            },
        },
    ),
    (
        2,
        SUPERSEDED_CHAIN_CONTRACT_REVISION_V2,
        {
            ACTIVE8_DECISION_RUNTIME: {
                "contract_sha256": (
                    "8b829cb311837205ca786dc15fe797bd4fd7926e30467f43a539150d37bbe656"
                ),
                "file_sha256": (
                    "673f733a8b3e9619794e5a408e0500f42e74e8e8e18e4ffe711023302736c891"
                ),
            },
            CAPABILITY_CELLS: {
                "contract_sha256": (
                    "b9ef14f32d83eaaae9c4ee3ac90121056b97a551199c8b96cbee6fd614d44068"
                ),
                "file_sha256": (
                    "42793cea4483e1ad9ebbd2e95315623dab2237b8f6354372c7a607c2409d8772"
                ),
            },
            DEVELOPMENT_CELL_ROLES: {
                "contract_sha256": (
                    "4c8f9d5ca412b0ed8a39fe940bf534e2f3ee18d883a07b2a3cf704985e34b900"
                ),
                "file_sha256": (
                    "507ae7eefe80a1c5502571610fcd4689c21831929b26b23898bd19040b14d03a"
                ),
            },
            GATE_ZERO_STRUCTURAL: {
                "contract_sha256": (
                    "06ac042507427ae622c5c8dc22f4ab47687800aa08e466a3c5f999557e364380"
                ),
                "file_sha256": (
                    "d6b8fa91ec2ec821d5b6010b6d57e46ae392579f95fb778b55b8547414607d8d"
                ),
            },
            T1_PANEL_POLICY: {
                "contract_sha256": (
                    "8cc3c39d47e55cc55df68900cbcc7c4f6a1beec1d9c02f0299a4aa14cafcb12b"
                ),
                "file_sha256": (
                    "9c5f2540686994a2d86e4d62a47a37b2986fbac1234bc89f08d01177a6f749e7"
                ),
            },
            T1_CAPACITY_POLICY: {
                "contract_sha256": (
                    "17569116220ab424d5dbd4d9b3607b97f7156f16e1cfb9e08e9debf7e2f5b919"
                ),
                "file_sha256": (
                    "157820865031bde57e4e796fa7d07070d11f3e54c3979f166f455abe9c572ef1"
                ),
            },
            P50_RECIPE_POLICY: {
                "contract_sha256": (
                    "b27e28a087e1c333882e6f97b5737a9fd2713acfee071045dabc808d1cbd4408"
                ),
                "file_sha256": (
                    "4355d360baba35927fe20dbedc898e08db5b6211a5e302fa4a7f33b238ff620d"
                ),
            },
        },
    ),
    (
        3,
        SUPERSEDED_CHAIN_CONTRACT_REVISION_V3,
        {
            ACTIVE8_DECISION_RUNTIME: {
                "contract_sha256": (
                    "fc0bc10625b9d80464f2e4098fa6085aa17c91878dba52bc24339d04e4a43766"
                ),
                "file_sha256": (
                    "53f8fea7d0165d229d674725fc2aaab5eeb61a138d4d5b45def03c969fde146f"
                ),
            },
            CAPABILITY_CELLS: {
                "contract_sha256": (
                    "3891411db00387edebcc4d0770a4f5e4624535c1d58fa51e562dfe0441c12962"
                ),
                "file_sha256": (
                    "f8919ff530cb22823f6c9e4c806550fa4d4ba65f963a4e94a672ef962fdc6c57"
                ),
            },
            DEVELOPMENT_CELL_ROLES: {
                "contract_sha256": (
                    "a88086e6762bd55aa1a888752d7cb6dc87615055693e94c374d2e99c45ee8977"
                ),
                "file_sha256": (
                    "344175f30a3c0ae13cb5d491ac0729b76c7ecfb6f4b82ccb15d3a41c8c900cd8"
                ),
            },
            GATE_ZERO_STRUCTURAL: {
                "contract_sha256": (
                    "b7d663ebd30d31051a5f01e08db15817b49f86f7a2ba0d7c4e7a3efc631fc042"
                ),
                "file_sha256": (
                    "c14fd34b649a7581e93fcec317d8905a066017bb1817ae058c0148941be76658"
                ),
            },
            T1_PANEL_POLICY: {
                "contract_sha256": (
                    "f6a305f2f652e3f411a8b0b8188203942c83e2fa279e56ef8b57c5febe47ba78"
                ),
                "file_sha256": (
                    "7f81ce3313ccca8c1b2e9fa5c4ebd9039c17655f72c199a24a7df424c0dc5d64"
                ),
            },
            T1_CAPACITY_POLICY: {
                "contract_sha256": (
                    "913b9bd86deebce726b3e2e866cf54fff664847faa25c351528bd0fb70e29625"
                ),
                "file_sha256": (
                    "5911bf120297ebb5b88b89f2daa5e596622fef8c12df0594562fb933b563a6f4"
                ),
            },
            P50_RECIPE_POLICY: {
                "contract_sha256": (
                    "885f2b13bac4ea7b80df35b286fad8933e33a5a7757f587abb6fb5b5ba25c69b"
                ),
                "file_sha256": (
                    "4534cc62b10aecc3d201ffa774d9a08fdf82f27205bb774fa38504a3c7a2f7a9"
                ),
            },
        },
    ),
    (
        4,
        SUPERSEDED_CHAIN_CONTRACT_REVISION_V4,
        {
            ACTIVE8_DECISION_RUNTIME: {
                "contract_sha256": (
                    "799291b6ab17a689d34e08cc849e3a0b510e7c07967fc1d3382d3805a7870bdf"
                ),
                "file_sha256": (
                    "997f2f596cd7a766d1f1a37271602dc3bec6fbcf982e44664c5d20daa11f155f"
                ),
            },
            CAPABILITY_CELLS: {
                "contract_sha256": (
                    "96a3418bb3a364b1135c46917b97902bcec75e93653add21f1ba583499824fe4"
                ),
                "file_sha256": (
                    "968e4343b0051dd82a893bb75c78219f1cd587e7b17a395f122ac9fd0eb740b2"
                ),
            },
            DEVELOPMENT_CELL_ROLES: {
                "contract_sha256": (
                    "23ec60562cbc7de5d39be7505d6928c89ea34f1eb1e4c20bbe7b66a619e71259"
                ),
                "file_sha256": (
                    "319dc27ecb9ee42ecb088953a8339740de3aab868dc83a106ede3d51c3d97aa0"
                ),
            },
            GATE_ZERO_STRUCTURAL: {
                "contract_sha256": (
                    "ad88dc9712f2f9f7ddd172ce2886b0d46d928d29f71bf4e5c6aa3799bb76980a"
                ),
                "file_sha256": (
                    "c6d4a41f695371633967246deb48ba2c2c7c1bd7fa27b9b653cca9d477e3905b"
                ),
            },
            T1_PANEL_POLICY: {
                "contract_sha256": (
                    "8a57487b9181b0e35b1d28895f18278a7c527554bdfab2730300651d33e4e2fb"
                ),
                "file_sha256": (
                    "46137eed0cf063b649eb77d855876ebca950c5a9bb3e4be6788a9e595d98a769"
                ),
            },
            T1_CAPACITY_POLICY: {
                "contract_sha256": (
                    "e2d0e0e0bc708c6f821ec92912170516c9c579c52f792127fcd652a0ea672968"
                ),
                "file_sha256": (
                    "73a6cfbcfff1b9711a2bb8d69f73e9a4dc9bd662688c8c383e980c1bfbb98df3"
                ),
            },
            P50_RECIPE_POLICY: {
                "contract_sha256": (
                    "365e2fa675154353500189f7460d0cb64c65cdb847d5ceb4f7216299d3f14b6a"
                ),
                "file_sha256": (
                    "3f0cf8b5eb840c20f1f58b3647fdb63af130621065676a70a765239cc3f5639b"
                ),
            },
        },
    ),
)

#: The exact field set of one lineage generation entry.
LINEAGE_GENERATION_FIELDS: tuple[str, ...] = (
    "contract_revision",
    "physical",
    "schema_version",
    "semantic",
)


# ---- Canonical hashing ----


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def process_v2_chain_self_hash(payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 of ``payload`` MINUS its ``contract_sha256``.

    This is the one self-hash equation the whole chain uses, and it is the same
    equation the generic verifier uses to DISCOVER a self-hash field, so a chain
    artifact self-identifies to that verifier without the verifier knowing this
    module exists.
    """

    return canonical_sha256(
        {key: value for key, value in payload.items() if key != SELF_HASH_FIELD}
    )


def declared_self_hash(payload: Mapping[str, Any], *, label: str) -> str | None:
    """Return the target's own self-hash, found by the equation, not by name.

    Raises:
        ProcessV2ChainError: if more than one top-level field satisfies it, which
            would make a semantic pin ambiguous.
    """

    found = [
        key
        for key, value in payload.items()
        if key.endswith("_sha256")
        and isinstance(value, str)
        and _HEX64.match(value)
        and value == canonical_sha256({k: v for k, v in payload.items() if k != key})
    ]
    if len(found) > 1:
        raise ProcessV2ChainError(
            f"{label} declares {len(found)} self-hash fields {sorted(found)}; a "
            "semantic pin would be ambiguous"
        )
    return payload[found[0]] if found else None


def semantic_pointer_algorithm(relative_path: str, raw: bytes) -> str | None:
    """Which canonical-body rule a semantic pin on this target is computed under.

    ``None`` means the target has no separable semantic body, so it must be pinned
    physically rather than by a "semantic" hash that is secretly the physical one.
    """

    if not relative_path.endswith(".json"):
        return None
    payload = _json_object(relative_path, raw)
    if declared_self_hash(payload, label=relative_path) is not None:
        return SELF_HASH_FIELD_ALGORITHM
    return WHOLE_CANONICAL_BODY_ALGORITHM


def semantic_sha256_of(relative_path: str, raw: bytes) -> str:
    """The semantic hash of a pin target under its declared algorithm.

    Raises:
        ProcessV2ChainError: if the target has no separable semantic body.
    """

    algorithm = semantic_pointer_algorithm(relative_path, raw)
    if algorithm is None:
        raise ProcessV2ChainError(
            f"{relative_path} is not JSON, so it has no separable semantic body and "
            "must be pinned by its physical hash"
        )
    payload = _json_object(relative_path, raw)
    if algorithm == SELF_HASH_FIELD_ALGORITHM:
        declared = declared_self_hash(payload, label=relative_path)
        assert declared is not None  # narrowed by the algorithm above
        return declared
    return canonical_sha256(payload)


def _json_object(relative_path: str, raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProcessV2ChainError(f"{relative_path} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ProcessV2ChainError(f"{relative_path} is not a JSON object")
    return payload


# ---- Pointer construction ----


def _read_target_bytes(
    repo_root: Path, relative_path: str, sealed: Mapping[str, bytes]
) -> bytes:
    """Read a pin target, preferring bytes already sealed in this generation.

    A child pins its parent's FINAL bytes.  During a generation the parent may not
    be on disk yet, so the in-memory overlay is authoritative; outside one, disk
    is.  Neither means the parent may be skipped.
    """

    if relative_path in sealed:
        return sealed[relative_path]
    try:
        return (Path(repo_root) / relative_path).read_bytes()
    except OSError as error:
        raise ProcessV2ChainError(
            f"parent {relative_path} is not readable; build the chain parents-first "
            "with build_process_v2_chain()"
        ) from error


def _edge(repo_root: Path, relative_path: str, sealed: Mapping[str, bytes]) -> dict[str, Any]:
    """Build one declared edge: a physical pointer, plus a semantic one when defined."""

    raw = _read_target_bytes(repo_root, relative_path, sealed)
    provider = _PROVIDERS.get(relative_path, _CHAIN_PROVIDER)
    target_schema: str | None = None
    if relative_path.endswith(".json"):
        target_schema = _json_object(relative_path, raw).get("schema")
        if target_schema is not None:
            target_schema = str(target_schema)
    edge: dict[str, Any] = {
        "physical": typed_pointer(
            kind=PointerKind.REPOSITORY_CONFIG,
            provider=provider,
            target=relative_path,
            target_schema=target_schema,
            identity_role=IdentityRole.PHYSICAL,
            sha256=_sha256_bytes(raw),
        )
    }
    algorithm = semantic_pointer_algorithm(relative_path, raw)
    if algorithm is not None:
        edge["semantic"] = typed_pointer(
            kind=PointerKind.REPOSITORY_CONFIG,
            provider=provider,
            target=relative_path,
            target_schema=target_schema,
            identity_role=IdentityRole.SEMANTIC,
            hash_algorithm=algorithm,
            sha256=semantic_sha256_of(relative_path, raw),
        )
    return edge


def _parents_block(
    repo_root: Path, name: str, sealed: Mapping[str, bytes]
) -> dict[str, dict[str, Any]]:
    return {
        role: _edge(repo_root, relative_path, sealed)
        for role, relative_path in sorted(_PARENTS[name].items())
    }


def _process_identity_block() -> dict[str, Any]:
    """Bind the LIVE Process-V2 identity; never a stored or V1 value.

    Provider, identity schema and process semantics are declared alongside the
    value so a verifier resolves this to exactly one identity node rather than
    accepting whichever of the two live identities happens to match.
    """

    identity = editing_process_v2_identity()
    return {
        "identity_schema": str(identity["schema"]),
        "identity_schema_version": int(identity["schema_version"]),  # type: ignore[call-overload]
        "module": PROCESS_IDENTITY_MODULE,
        PROCESS_IDENTITY_PIN_FIELD: str(identity[PROCESS_IDENTITY_PIN_FIELD]),
        "process_semantics": str(identity["process_semantics"]),
        "provider": PROCESS_V2_IDENTITY_PROVIDER,
    }


def _superseded_lineage_block(name: str) -> list[dict[str, Any]]:
    """Preserve EVERY superseded generation as lineage, oldest first.

    A list rather than one block, because version 2 recorded only version 1: had
    version 3 kept that shape it would have dropped version 1 to record version 2,
    and a chain that can only name its immediately preceding shape cannot answer
    what a hash from two generations ago addressed.
    """

    return [
        {
            "contract_revision": revision,
            "physical": typed_pointer(
                kind=PointerKind.LINEAGE_REFERENCE,
                provider=_CHAIN_PROVIDER,
                target=name,
                target_schema=_ENVELOPE[name]["schema"],
                identity_role=IdentityRole.PHYSICAL,
                sha256=hashes[name]["file_sha256"],
            ),
            "schema_version": schema_version,
            "semantic": typed_pointer(
                kind=PointerKind.LINEAGE_REFERENCE,
                provider=_CHAIN_PROVIDER,
                target=name,
                target_schema=_ENVELOPE[name]["schema"],
                identity_role=IdentityRole.SEMANTIC,
                hash_algorithm=SELF_HASH_FIELD_ALGORITHM,
                sha256=hashes[name]["contract_sha256"],
            ),
        }
        for schema_version, revision, hashes in _SUPERSEDED_GENERATIONS
    ]


def _resolved_evidence_binding_declaration() -> dict[str, Any]:
    """Name the schema that will cite this contract with measured evidence.

    A constant, never a hash: a prospective contract must not address a later
    measured result, and the version-1 attempt to leave a fillable hash slot here
    was structurally impossible.
    """

    return {
        "cited_by_schema": RESOLVED_EVIDENCE_BINDING_SCHEMA,
        "cited_by_schema_version": RESOLVED_EVIDENCE_BINDING_SCHEMA_VERSION,
        "measured_evidence_in_this_contract": False,
    }


def _operator_capability_fingerprint(repo_root: Path, sealed: Mapping[str, bytes]) -> str:
    """Read the V2 operator capability expectation from the pinned Gate-0 contract.

    Reading it from the parent rather than restating it means the runtime's
    expectation cannot silently disagree with the contract this artifact pins.
    """

    payload = _json_object(
        GATE_ZERO_MODEL_PROCESS_V2,
        _read_target_bytes(repo_root, GATE_ZERO_MODEL_PROCESS_V2, sealed),
    )
    fingerprint = payload.get("model_identity", {}).get("operator_capability_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ProcessV2ChainError(
            f"{GATE_ZERO_MODEL_PROCESS_V2} declares no "
            "model_identity.operator_capability_fingerprint"
        )
    return fingerprint


# ---- Builders ----


def _project(block: str, repo_root: Path) -> Any:
    try:
        return project_shared_policy(block, repo_root=repo_root)
    except PolicyRegistryError as error:
        raise ProcessV2ChainError(
            f"the shared policy block {block!r} could not be projected: {error}"
        ) from error


def _project_body(body: str, repo_root: Path) -> dict[str, Any]:
    try:
        return project_shared_policy_body(body, repo_root=repo_root)
    except PolicyRegistryError as error:
        raise ProcessV2ChainError(
            f"the shared policy body {body!r} could not be projected: {error}"
        ) from error


def _registry_pin(repo_root: Path) -> dict[str, Any]:
    """Pin the registry by its own identity hash rather than copying its body.

    Copying the descriptor into all seven artifacts would reintroduce, at a
    smaller scale, exactly the duplication this registry exists to remove. The
    hash is enough: it moves if any frozen source or the allowlist moves, and the
    full descriptor is obtainable from the registry itself.
    """

    try:
        identity = policy_registry_identity(repo_root=repo_root)
    except PolicyRegistryError as error:
        raise ProcessV2ChainError(
            f"the shared policy registry refuses to identify itself: {error}"
        ) from error
    return {
        "registry_identity_sha256": str(identity["registry_identity_sha256"]),
        "schema": str(identity["schema"]),
        "schema_version": int(identity["schema_version"]),
    }


def _envelope(repo_root: Path, name: str, sealed: Mapping[str, bytes]) -> dict[str, Any]:
    meta = _ENVELOPE[name]
    payload: dict[str, Any] = {
        "active_families": list(ACTIVE8_FAMILIES),
        "contract_id": meta["contract_id"],
        "contract_revision": CHAIN_CONTRACT_REVISION,
        "parents": _parents_block(repo_root, name, sealed),
        "process_identity": _process_identity_block(),
        "resolved_evidence_binding": _resolved_evidence_binding_declaration(),
        "schema": meta["schema"],
        "schema_version": CHAIN_SCHEMA_VERSION,
        "shared_policy_registry": _registry_pin(repo_root),
        "status": meta["status"],
        "superseded_design_lineage": _superseded_lineage_block(name),
    }
    payload.update(authority_false_block())
    return payload


def _build_decision_runtime(repo_root: Path, sealed: Mapping[str, bytes]) -> dict[str, Any]:
    model = dict(_project("runtime_model", repo_root))
    model["operator_capability_fingerprint"] = _operator_capability_fingerprint(
        repo_root, sealed
    )
    payload = _envelope(repo_root, ACTIVE8_DECISION_RUNTIME, sealed)
    payload["model"] = model
    payload["software"] = _project("runtime_software", repo_root)
    return payload


def _build_capability_cells(repo_root: Path, sealed: Mapping[str, bytes]) -> dict[str, Any]:
    payload = _envelope(repo_root, CAPABILITY_CELLS, sealed)
    payload["bindings"] = {
        "action_codec_schema_version": _project("action_codec_schema_version", repo_root),
        "data_lanes": _project("data_lanes", repo_root),
        "partition_roles": _project("partition_roles", repo_root),
    }
    payload["cell_identity_policy"] = _project("cell_identity_policy", repo_root)
    payload["exact_evidence_strata"] = _project("exact_evidence_strata", repo_root)
    payload["fail_closed_policy"] = _project("fail_closed_policy", repo_root)
    payload["family_contexts"] = _project("family_contexts", repo_root)
    payload["scope"] = _project("cell_scope", repo_root)
    return payload


def _build_development_cell_roles(
    repo_root: Path, sealed: Mapping[str, bytes]
) -> dict[str, Any]:
    payload = _envelope(repo_root, DEVELOPMENT_CELL_ROLES, sealed)
    payload["conditional_cell_ids"] = _project("conditional_cell_ids", repo_root)
    payload["conditional_policy"] = _project("conditional_policy", repo_root)
    payload["partition_policy"] = _project("cell_role_partition_policy", repo_root)
    payload["required_cell_ids"] = _project("required_cell_ids", repo_root)
    payload["scope"] = _project("cell_role_scope", repo_root)
    payload["separate_lane_cell_ids"] = _project("separate_lane_cell_ids", repo_root)
    payload["separate_lane_policy"] = _project("separate_lane_policy", repo_root)
    return payload


def _process_v2_gate_zero_structural_checks(repo_root: Path) -> dict[str, Any]:
    """Project the frozen base checks and apply the explicit Process-V2 delta.

    The V1 gate measured full quotient size and alias multiplicity for every
    corpus transition.  Process V2 proves exact teacher support and productive
    nonself execution exhaustively, then measures full quotient geometry in the
    completion-bound release sentinel.  Requiring the old keys before removing
    them makes upstream policy drift fail instead of silently widening the delta.
    """

    checks = dict(_project("structural_checks", repo_root))
    retired = (
        "require_exactly_one_matching_mark",
        "require_positive_canonical_successor_count",
        "require_positive_raw_mark_count",
        "require_positive_successor_alias_count",
    )
    for field in retired:
        if checks.get(field) is not True:
            raise ProcessV2ChainError(
                f"the Process-V2 Gate-0 base policy no longer declares {field} true"
            )
        del checks[field]
    if "require_productive_nonself_successor" in checks:
        raise ProcessV2ChainError(
            "the Process-V2 Gate-0 base policy unexpectedly declares the V2 delta"
        )
    checks["require_productive_nonself_successor"] = True
    checks["require_teacher_coordinate_legal"] = True
    checks["require_teacher_executes_to_exact_successor"] = True
    return checks


def _build_gate_zero_structural(
    repo_root: Path, sealed: Mapping[str, bytes]
) -> dict[str, Any]:
    payload = _envelope(repo_root, GATE_ZERO_STRUCTURAL, sealed)
    payload["decision_policy"] = _project("gate_zero_decision_policy", repo_root)
    payload["required_architecture"] = _project("required_architecture", repo_root)
    payload["structural_checks"] = _process_v2_gate_zero_structural_checks(repo_root)
    return payload


def _build_t1_panel_policy(repo_root: Path, sealed: Mapping[str, bytes]) -> dict[str, Any]:
    payload = _envelope(repo_root, T1_PANEL_POLICY, sealed)
    payload.update(_project_body("t1_panel_body", repo_root))
    return payload


def _build_t1_capacity_policy(
    repo_root: Path, sealed: Mapping[str, bytes]
) -> dict[str, Any]:
    payload = _envelope(repo_root, T1_CAPACITY_POLICY, sealed)
    shared = _project_body("t1_capacity_body", repo_root)
    shared_optimization = shared.get("optimization")
    if not isinstance(shared_optimization, Mapping):
        raise ProcessV2ChainError(
            "the shared T1 capacity policy has no optimization mapping"
        )
    optimization = dict(shared_optimization)
    optimization.update(
        {
            "early_stop_rule": PROCESS_V2_T1_EARLY_STOP_RULE,
            "learning_rate_schedule": PROCESS_V2_T1_LEARNING_RATE_SCHEDULE,
            "maximum_optimizer_steps": PROCESS_V2_T1_MAXIMUM_OPTIMIZER_STEPS,
            "minimum_optimizer_steps_before_early_stop": (
                PROCESS_V2_T1_MINIMUM_STEPS_BEFORE_EARLY_STOP
            ),
            "operational_semantics_version": (
                PROCESS_V2_T1_OPERATIONAL_SEMANTICS_VERSION
            ),
            "report_points": PROCESS_V2_T1_REPORT_POINTS,
            "scheduler": PROCESS_V2_T1_SCHEDULER,
            "trajectory_evaluation": PROCESS_V2_T1_TRAJECTORY_EVALUATION,
        }
    )
    shared["sampling_law"] = PROCESS_V2_T1_SAMPLING_LAW
    shared["optimization"] = optimization
    payload.update(shared)
    return payload


def _build_p50_recipe_policy(repo_root: Path, sealed: Mapping[str, bytes]) -> dict[str, Any]:
    payload = _envelope(repo_root, P50_RECIPE_POLICY, sealed)
    payload.update(_project_body("p50_recipe_body", repo_root))
    return payload


_BUILDERS = {
    ACTIVE8_DECISION_RUNTIME: _build_decision_runtime,
    CAPABILITY_CELLS: _build_capability_cells,
    DEVELOPMENT_CELL_ROLES: _build_development_cell_roles,
    GATE_ZERO_STRUCTURAL: _build_gate_zero_structural,
    T1_PANEL_POLICY: _build_t1_panel_policy,
    T1_CAPACITY_POLICY: _build_t1_capacity_policy,
    P50_RECIPE_POLICY: _build_p50_recipe_policy,
}


def _require_known(name: str) -> str:
    if name not in _BUILDERS:
        raise ProcessV2ChainError(
            f"{name!r} is not a Process-V2 chain artifact; "
            f"expected one of {list(PROCESS_V2_CHAIN_ARTIFACTS)}"
        )
    return name


def build_process_v2_chain_artifact(
    name: str, *, repo_root: Path, sealed: Mapping[str, bytes] | None = None
) -> dict[str, Any]:
    """Build one chain artifact, sealed with its ``contract_sha256``.

    Args:
        name: one of :data:`PROCESS_V2_CHAIN_ARTIFACTS`.
        repo_root: the checkout holding the frozen external parents.
        sealed: bytes of chain members already sealed in this generation. A
            parent absent from both the overlay and disk is an error rather than
            a stale pin.

    Raises:
        ProcessV2ChainError: on an unknown name, an unreadable or ambiguous
            parent, or a frozen policy source that moved.
    """

    payload = _BUILDERS[_require_known(name)](Path(repo_root), dict(sealed or {}))
    return self_hashed(payload, field=SELF_HASH_FIELD)


def build_process_v2_chain(repo_root: Path) -> dict[str, bytes]:
    """Build all seven artifacts parents-first, in memory, and return their bytes.

    Pure: it writes nothing. Each child sees its parent's FINAL sealed bytes
    through the overlay, so the graph is consistent before anything is published.
    """

    repo_root = Path(repo_root)
    sealed: dict[str, bytes] = {}
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        payload = build_process_v2_chain_artifact(name, repo_root=repo_root, sealed=sealed)
        sealed[name] = serialize_process_v2_chain_artifact(payload)
    return sealed


def serialize_process_v2_chain_artifact(payload: Mapping[str, Any]) -> bytes:
    """Serialize to the chain's byte-stable form: ``indent=2, sort_keys=True`` + ``\\n``."""

    text = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode()


# ---- Validation ----


def _fail(name: str, detail: str) -> None:
    raise ProcessV2ChainError(f"{name}: {detail}")


def _first_difference(observed: object, expected: object, path: str = "") -> str | None:
    """Return a dotted path to the first structural difference, or ``None``."""

    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        for key in sorted(set(expected) | set(observed)):
            if key not in observed:
                return f"{path}.{key} (missing)"
            if key not in expected:
                return f"{path}.{key} (unexpected)"
            found = _first_difference(observed[key], expected[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if (
        isinstance(expected, Sequence)
        and isinstance(observed, Sequence)
        and not isinstance(expected, str)
        and not isinstance(observed, str)
    ):
        if len(expected) != len(observed):
            return f"{path} (length {len(observed)} != {len(expected)})"
        for index, (left, right) in enumerate(zip(observed, expected)):
            found = _first_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if type(observed) is not type(expected) or observed != expected:
        return f"{path} ({observed!r} != {expected!r})"
    return None


def _check_authority(name: str, payload: Mapping[str, Any]) -> None:
    """Both guards, over the whole body.

    A prospective contract is a non-granting artifact, so the vocabulary-free
    guard applies to it: the P50 recipe body carries ``p500_authorized`` as a
    policy field, which is outside ``AUTHORITY_FIELDS`` and therefore invisible to
    the vocabulary-aware guard, and a contract that shipped it granted would
    otherwise pass.
    """

    try:
        require_no_granted_authority(payload, label=name)
        require_authority_false(payload, label=name)
    except ProcessV2SchemaError as error:
        _fail(name, str(error))


def _check_active8(name: str, payload: Mapping[str, Any]) -> None:
    observed = payload.get("active_families")
    if not isinstance(observed, list) or tuple(observed) != tuple(ACTIVE8_FAMILIES):
        _fail(
            name,
            f"active_families is {observed!r}, which differs from the Active8 "
            f"operator order {list(ACTIVE8_FAMILIES)}",
        )


def _check_process_identity(name: str, payload: Mapping[str, Any]) -> None:
    """Resolve the declared identity edge to exactly one node, then compare.

    The declared provider, identity schema and process semantics select the node.
    A pin that carries the V1 VALUE under the V2 DECLARATION is therefore reported
    as binding V1, not silently accepted because it matched some live identity.
    """

    block = payload.get("process_identity")
    if not isinstance(block, Mapping) or set(block) != set(PROCESS_IDENTITY_EDGE_FIELDS):
        _fail(
            name,
            "process_identity must carry exactly "
            f"{sorted(PROCESS_IDENTITY_EDGE_FIELDS)}",
        )
    assert isinstance(block, Mapping)  # narrowed by the guard above
    if (
        block["provider"] != PROCESS_V2_IDENTITY_PROVIDER
        or block["identity_schema"] != PROCESS_V2_IDENTITY_SCHEMA
    ):
        _fail(
            name,
            f"process_identity declares provider {block['provider']!r} and schema "
            f"{block['identity_schema']!r}; the Process-V2 chain resolves only "
            f"{PROCESS_V2_IDENTITY_PROVIDER!r} / {PROCESS_V2_IDENTITY_SCHEMA!r}",
        )
    pinned = block[PROCESS_IDENTITY_PIN_FIELD]
    live_v2 = str(editing_process_v2_identity()[PROCESS_IDENTITY_PIN_FIELD])
    if pinned == live_v2:
        return
    live_v1 = str(editing_v2_process_identity()[PROCESS_IDENTITY_PIN_FIELD])
    if pinned == live_v1:
        _fail(
            name,
            "process_identity.process_identity_sha256 binds the V1 semantic process "
            f"identity {live_v1}; the Process-V2 chain must bind the live Process-V2 "
            f"identity {live_v2}",
        )
    _fail(
        name,
        f"process_identity.process_identity_sha256 is {pinned!r}, which is not the "
        f"live Process-V2 identity {live_v2}",
    )


def _check_parents(
    name: str, payload: Mapping[str, Any], repo_root: Path, sealed: Mapping[str, bytes]
) -> None:
    block = payload.get("parents")
    expected_roles = set(_PARENTS[name])
    if not isinstance(block, Mapping):
        _fail(name, "parents block is missing or is not an object")
    assert isinstance(block, Mapping)  # narrowed by the guard above
    if set(block) != expected_roles:
        missing = sorted(expected_roles - set(block))
        unexpected = sorted(set(block) - expected_roles)
        _fail(
            name,
            f"declared dependency edges differ from the chain graph; missing="
            f"{missing} unexpected={unexpected}",
        )
    for role in sorted(expected_roles):
        edge = block[role]
        live = _edge(repo_root, _PARENTS[name][role], sealed)
        if not isinstance(edge, Mapping) or set(edge) != set(live):
            _fail(
                name,
                f"edge {role!r} declares {sorted(edge) if isinstance(edge, Mapping) else edge!r}, "
                f"not the {sorted(live)} its target supports",
            )
        assert isinstance(edge, Mapping)  # narrowed by the guard above
        for slot in sorted(live):
            try:
                pointer = validate_typed_pointer(edge[slot], label=f"{name} edge {role}.{slot}")
            except ProcessV2SchemaError as error:
                _fail(name, str(error))
            target = pointer["target"]
            if _V1_STEM.search(Path(str(target)).stem):
                _fail(
                    name,
                    f"edge {role!r} points at the V1 artifact {target!r}; the "
                    "Process-V2 chain must never bind a _v1 config",
                )
            if target != _PARENTS[name][role]:
                _fail(
                    name,
                    f"edge {role!r} points at {target!r}, not {_PARENTS[name][role]!r}",
                )
            if pointer != live[slot]:
                _fail(
                    name,
                    f"edge {role!r} pins {target} {slot} at {pointer['sha256']!r}, "
                    f"but the live value is {live[slot]['sha256']!r}",
                )


def _live_chain_values(repo_root: Path, sealed: Mapping[str, bytes]) -> dict[str, str]:
    """``sha256 -> where that value is live now``, over every chain artifact.

    Read from the sealed overlay when this generation has already sealed the
    artifact, otherwise from disk.  An artifact that is absent contributes
    nothing: a lineage reference may outlive its target, which is what historical
    means.  Only bytes OUTSIDE the payload under test are read, which is what
    makes the comparison falsifiable -- see :func:`_check_superseded_lineage`.
    """

    live: dict[str, str] = {}
    for artifact in PROCESS_V2_CHAIN_ARTIFACTS:
        if artifact in sealed:
            raw = sealed[artifact]
        else:
            try:
                raw = (Path(repo_root) / artifact).read_bytes()
            except OSError:
                continue
        live.setdefault(_sha256_bytes(raw), f"physical hash of {artifact}")
        try:
            semantic = semantic_sha256_of(artifact, raw)
        except ProcessV2ChainError:
            continue
        live.setdefault(semantic, f"semantic hash of {artifact}")
    return live


def _check_superseded_lineage(
    name: str,
    payload: Mapping[str, Any],
    repo_root: Path,
    sealed: Mapping[str, bytes],
) -> None:
    """Every generation must be historical, distinct, and in order.

    "Historical" is the whole claim a lineage reference makes, so each of the
    ways it can be false is checked separately: a value equal to a live one is not
    superseded, a repeated value is not a second generation, and an out-of-order
    or duplicated version cannot be read as a sequence.
    """

    block = payload.get("superseded_design_lineage")
    if not isinstance(block, list):
        _fail(
            name,
            "superseded_design_lineage is missing or is not a list; it records every "
            "superseded generation, oldest first, rather than only the last one",
        )
    assert isinstance(block, list)  # narrowed by the guard above
    if len(block) != len(_SUPERSEDED_GENERATIONS):
        _fail(
            name,
            f"superseded_design_lineage records {len(block)} generations, not the "
            f"{len(_SUPERSEDED_GENERATIONS)} this chain has superseded; a generation is "
            "appended, never replaced",
        )
    live_semantic = payload.get(SELF_HASH_FIELD)
    # Two formulations of "this value is still live" exist, and only one of them
    # can be written down.
    #
    # The SELF-REFERENTIAL one -- expectation = sha256(serialize(the payload under
    # test)) -- is unfalsifiable: the lineage pointer is part of the body that IS
    # the file, so writing the live physical hash into it moves the live physical
    # hash, and a validator recomputing its expectation from the payload it is
    # checking compares the mutation against itself. That guard is not claimed
    # here, and `test_the_self_referential_lineage_physical_check_is_unfalsifiable`
    # pins the fixed point as unreachable.
    #
    # The ON-DISK one is falsifiable and IS checked, below: the expectation comes
    # from bytes outside the payload, which do not move when the payload is
    # mutated. It is the same check the sibling verifier already makes over
    # committed bytes (`verify_process_v2_chain._check_lineage_pointer` ->
    # `lineage_value_is_live`), and it additionally catches a lineage pin for one
    # artifact carrying ANOTHER chain artifact's live hash, which the `target`
    # check cannot see because the target is correct.
    #
    # The same caveat applies to `live_semantic` below, which compares against a
    # constant embedded in the body: it is reachable only by hand-mutating a
    # sealed payload, and it is kept as the diagnostic naming that specific lie.
    live_values = _live_chain_values(Path(repo_root), sealed)
    seen_hashes: dict[str, str] = {}
    seen_revisions: set[str] = set()
    previous_version = 0
    for index, entry in enumerate(block):
        where = f"superseded_design_lineage[{index}]"
        if not isinstance(entry, Mapping) or set(entry) != set(LINEAGE_GENERATION_FIELDS):
            _fail(name, f"{where} must carry exactly {sorted(LINEAGE_GENERATION_FIELDS)}")
        assert isinstance(entry, Mapping)  # narrowed by the guard above
        version = entry["schema_version"]
        if type(version) is not int:
            _fail(name, f"{where}.schema_version is not an exact int")
        if version <= previous_version:
            _fail(
                name,
                f"{where}.schema_version is {version}, which does not follow "
                f"{previous_version}; lineage is recorded oldest first and a repeated "
                "version is a duplicated generation rather than a second one",
            )
        if version >= CHAIN_SCHEMA_VERSION:
            _fail(
                name,
                f"{where}.schema_version is {version}, which is not older than the live "
                f"schema version {CHAIN_SCHEMA_VERSION}",
            )
        previous_version = version
        revision = entry["contract_revision"]
        if revision == CHAIN_CONTRACT_REVISION:
            _fail(
                name,
                f"{where}.contract_revision is the live revision {revision!r}, so it is "
                "not superseded",
            )
        if revision in seen_revisions:
            _fail(name, f"{where}.contract_revision {revision!r} is already recorded")
        seen_revisions.add(str(revision))
        for slot in ("physical", "semantic"):
            try:
                pointer = validate_typed_pointer(entry.get(slot), label=f"{name} {where}.{slot}")
            except ProcessV2SchemaError as error:
                _fail(name, str(error))
            if pointer["kind"] != PointerKind.LINEAGE_REFERENCE:
                _fail(
                    name,
                    f"{where}.{slot} declares kind {pointer['kind']!r}; a superseded "
                    f"value must be a {PointerKind.LINEAGE_REFERENCE!r} and must carry "
                    "no currency claim",
                )
            if pointer["target"] != name:
                _fail(
                    name,
                    f"{where}.{slot} targets {pointer['target']!r}; an artifact records "
                    "its OWN superseded hashes, not another artifact's",
                )
            digest = pointer["sha256"]
            if digest in seen_hashes:
                _fail(
                    name,
                    f"{where}.{slot} repeats {digest}, already recorded at "
                    f"{seen_hashes[digest]}; two generations that hash alike are one "
                    "generation written twice",
                )
            seen_hashes[digest] = f"{where}.{slot}"
        if entry["semantic"]["sha256"] == live_semantic:
            _fail(
                name,
                f"{where}.semantic equals the live {SELF_HASH_FIELD}, so it is not "
                "superseded",
            )
        for slot in ("physical", "semantic"):
            digest = entry[slot]["sha256"]
            where_live = live_values.get(digest)
            if where_live is not None:
                _fail(
                    name,
                    f"{where}.{slot} records {digest} as superseded, but that is the "
                    f"live {where_live}",
                )


def _check_registry(name: str, payload: Mapping[str, Any], repo_root: Path) -> None:
    observed = payload.get("shared_policy_registry")
    expected = _registry_pin(repo_root)
    if observed != expected:
        _fail(
            name,
            f"shared_policy_registry pins {observed!r}, but the live registry identity "
            f"is {expected!r}; a frozen policy source moved or the allowlist changed",
        )


def _check_t1_capacity_operational_delta(
    name: str,
    payload: Mapping[str, Any],
    repo_root: Path,
) -> None:
    """Prove the prospective Process-V2 T1 recovery delta exactly."""

    if name != T1_CAPACITY_POLICY:
        return
    shared = _project_body("t1_capacity_body", repo_root)
    shared_optimization = shared.get("optimization")
    observed_optimization = payload.get("optimization")
    if not isinstance(shared_optimization, Mapping) or not isinstance(
        observed_optimization, Mapping
    ):
        _fail(name, "T1 capacity optimization policy is not a mapping")

    for field, expected in shared.items():
        if field not in {"optimization", "sampling_law"} and payload.get(field) != expected:
            _fail(
                name,
                "mirrored policy values are frozen: "
                f"shared scientific T1 field {field!r} differs from frozen V1",
            )

    if payload.get("sampling_law") != PROCESS_V2_T1_SAMPLING_LAW:
        _fail(
            name,
            "Process-V2 T1 sampling law is not the prospective equal-entry recovery law",
        )

    changed = {
        "early_stop_rule",
        "maximum_optimizer_steps",
        "report_points",
        "scheduler",
        "trajectory_evaluation",
    }
    added = {
        "learning_rate_schedule",
        "minimum_optimizer_steps_before_early_stop",
        "operational_semantics_version",
    }
    expected_keys = set(shared_optimization) | added
    if set(observed_optimization) != expected_keys:
        _fail(
            name,
            "Process-V2 T1 optimization fields differ outside the frozen operational delta",
        )
    for field, expected in shared_optimization.items():
        if field not in changed and observed_optimization.get(field) != expected:
            _fail(
                name,
                "mirrored policy values are frozen: shared scientific T1 "
                f"optimization field {field!r} differs from frozen V1",
            )
    expected_operational = {
        "early_stop_rule": PROCESS_V2_T1_EARLY_STOP_RULE,
        "learning_rate_schedule": PROCESS_V2_T1_LEARNING_RATE_SCHEDULE,
        "maximum_optimizer_steps": PROCESS_V2_T1_MAXIMUM_OPTIMIZER_STEPS,
        "minimum_optimizer_steps_before_early_stop": (
            PROCESS_V2_T1_MINIMUM_STEPS_BEFORE_EARLY_STOP
        ),
        "operational_semantics_version": PROCESS_V2_T1_OPERATIONAL_SEMANTICS_VERSION,
        "report_points": PROCESS_V2_T1_REPORT_POINTS,
        "scheduler": PROCESS_V2_T1_SCHEDULER,
        "trajectory_evaluation": PROCESS_V2_T1_TRAJECTORY_EVALUATION,
    }
    for field, expected in expected_operational.items():
        if observed_optimization.get(field) != expected:
            _fail(
                name,
                f"Process-V2 T1 operational field {field!r} is "
                f"{observed_optimization.get(field)!r}, not {expected!r}",
            )


def validate_process_v2_chain_artifact(
    value: object,
    *,
    name: str,
    repo_root: Path,
    sealed: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Validate one chain artifact against the live chain and return it.

    Every mixed-V1/V2 defect is reported with a message naming that specific defect
    rather than collapsing into a generic hash mismatch, because a chain that fails
    only at the self-hash tells a reader nothing about WHICH binding drifted.

    Raises:
        ProcessV2ChainError: on any disagreement.
    """

    _require_known(name)
    repo_root = Path(repo_root)
    overlay = dict(sealed or {})
    if not isinstance(value, Mapping):
        _fail(name, f"artifact must be a JSON object, got {type(value).__name__}")
    assert isinstance(value, Mapping)  # narrowed by the guard above
    payload: dict[str, Any] = dict(value)

    expected = build_process_v2_chain_artifact(name, repo_root=repo_root, sealed=overlay)
    if set(payload) != set(expected):
        missing = sorted(set(expected) - set(payload))
        unexpected = sorted(set(payload) - set(expected))
        _fail(
            name,
            f"field set differs from the deterministic rebuild; missing={missing} "
            f"unexpected={unexpected}",
        )

    status = payload.get("status")
    if not isinstance(status, str) or not status.endswith(STATUS_SUFFIX):
        _fail(name, f"status {status!r} must end in {STATUS_SUFFIX!r}")
    if payload.get("schema_version") != CHAIN_SCHEMA_VERSION:
        _fail(
            name,
            f"schema_version is {payload.get('schema_version')!r}, not "
            f"{CHAIN_SCHEMA_VERSION}; adding dependency edges required a new version "
            "rather than resealing version 1 in place",
        )
    if payload.get("contract_revision") != CHAIN_CONTRACT_REVISION:
        _fail(
            name,
            f"contract_revision is {payload.get('contract_revision')!r}, not "
            f"{CHAIN_CONTRACT_REVISION!r}",
        )
    _check_authority(name, payload)
    _check_active8(name, payload)
    _check_process_identity(name, payload)
    _check_parents(name, payload, repo_root, overlay)
    _check_superseded_lineage(name, payload, repo_root, overlay)
    _check_registry(name, payload, repo_root)
    _check_t1_capacity_operational_delta(name, payload, repo_root)

    difference = _first_difference(
        {k: v for k, v in payload.items() if k != SELF_HASH_FIELD},
        {k: v for k, v in expected.items() if k != SELF_HASH_FIELD},
    )
    if difference is not None:
        _fail(
            name,
            f"body differs from the deterministic rebuild at {difference.lstrip('.')}; "
            "mirrored policy values are frozen",
        )

    try:
        verify_self_hash(payload, field=SELF_HASH_FIELD, label=name)
    except ProcessV2SchemaError as error:
        _fail(
            name,
            f"{SELF_HASH_FIELD} is {payload.get(SELF_HASH_FIELD)!r}, which disagrees "
            f"with the canonical hash of its own body "
            f"{process_v2_chain_self_hash(payload)!r} ({error})",
        )
    return payload


def load_process_v2_chain_artifact(
    name: str, *, repo_root: Path, sealed: Mapping[str, bytes] | None = None
) -> dict[str, Any]:
    """Read one committed chain artifact from disk and validate it.

    Raises:
        ProcessV2ChainError: if it is absent, unparseable, or inconsistent.
    """

    _require_known(name)
    repo_root = Path(repo_root)
    try:
        raw = (repo_root / name).read_bytes()
    except OSError as error:
        raise ProcessV2ChainError(f"{name}: chain artifact is not readable") from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProcessV2ChainError(f"{name}: chain artifact is not valid JSON") from error
    return validate_process_v2_chain_artifact(
        payload, name=name, repo_root=repo_root, sealed=sealed
    )


# ---- Dependency graph ----


def process_v2_superseded_generations() -> tuple[tuple[int, str], ...]:
    """Every superseded generation as ``(schema version, contract revision)``.

    Public so a consumer counting lineage pointers -- the chain verifier's test
    does -- can express "two per generation per artifact" rather than a literal
    that silently becomes wrong the next time a generation is appended.
    """

    return tuple(
        (schema_version, revision)
        for schema_version, revision, _hashes in _SUPERSEDED_GENERATIONS
    )


def process_v2_dependency_edges() -> dict[str, dict[str, str]]:
    """The declared graph, as ``artifact -> {edge role: target}``."""

    return {name: dict(edges) for name, edges in _PARENTS.items()}


def process_v2_transitive_dependencies(name: str) -> set[str]:
    """Every target reachable from ``name`` by declared edges.

    Raises:
        ProcessV2ChainError: on an unknown name or a cycle. A prospective contract
            chain that contained a cycle could not be built parents-first at all.
    """

    _require_known(name)
    reached: set[str] = set()
    stack = [(name, (name,))]
    while stack:
        current, path = stack.pop()
        for target in _PARENTS.get(current, {}).values():
            if target in path:
                raise ProcessV2ChainError(
                    f"the declared dependency graph contains a cycle: "
                    f"{' -> '.join((*path, target))}"
                )
            reached.add(target)
            if target in _PARENTS:
                stack.append((target, (*path, target)))
    return reached


# ---- Transactional publication ----

GENERATION_SELF_HASH_FIELD = "committed_sha256"
GENERATION_STATUS = "PROCESS_V2_CHAIN_GENERATION_COMMITTED" + STATUS_SUFFIX


def process_v2_generation_id(sealed: Mapping[str, bytes]) -> str:
    """Content-address one generation by the physical hashes of all seven artifacts."""

    missing = [name for name in PROCESS_V2_CHAIN_ARTIFACTS if name not in sealed]
    if missing:
        raise ProcessV2ChainError(
            f"a generation must contain all seven artifacts; missing {missing}"
        )
    return canonical_sha256(
        {name: _sha256_bytes(sealed[name]) for name in PROCESS_V2_CHAIN_ARTIFACTS}
    )


def stage_process_v2_chain_generation(repo_root: Path, *, generations_root: Path) -> Path:
    """Write and validate one complete generation WITHOUT its committed marker.

    This is deliberately a separate step from :func:`commit_process_v2_chain_generation`.
    Calling it alone is exactly the state an interrupted publisher leaves behind, so
    an interruption test needs no injected failure hook: it stages, does not commit,
    and asserts that every reader refuses the directory.

    Returns:
        The generation directory.

    Raises:
        ProcessV2ChainError: if any artifact fails validation inside the generation.
    """

    repo_root = Path(repo_root)
    sealed = build_process_v2_chain(repo_root)
    generation = Path(generations_root) / process_v2_generation_id(sealed)
    generation.mkdir(parents=True, exist_ok=True)
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        path = generation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sealed[name])
    # Validate the COMPLETE graph inside the generation, before anything is
    # visible to a reader. External parents still come from the repository; chain
    # members come from the generation's own bytes.
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        payload = json.loads((generation / name).read_bytes())
        validate_process_v2_chain_artifact(
            payload, name=name, repo_root=repo_root, sealed=sealed
        )
    return generation


def commit_process_v2_chain_generation(generation: Path) -> Path:
    """Publish the committed marker LAST, making the generation readable.

    A Modal volume commit is a visibility boundary, not a multi-file atomic
    transaction, so the marker is what makes a generation exist for a reader.

    Raises:
        ProcessV2ChainError: if an artifact is absent from the generation.
    """

    generation = Path(generation)
    artifacts: dict[str, dict[str, str]] = {}
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        path = generation / name
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ProcessV2ChainError(
                f"the generation at {generation} is incomplete: {name} is absent"
            ) from error
        artifacts[name] = {
            "contract_sha256": str(json.loads(raw)[SELF_HASH_FIELD]),
            "file_sha256": _sha256_bytes(raw),
        }
    sealed = {name: (generation / name).read_bytes() for name in PROCESS_V2_CHAIN_ARTIFACTS}
    body: dict[str, Any] = {
        "artifacts": artifacts,
        "generation_id": process_v2_generation_id(sealed),
        "schema": GENERATION_COMMITTED_SCHEMA,
        "schema_version": GENERATION_COMMITTED_SCHEMA_VERSION,
        "status": GENERATION_STATUS,
        **authority_false_block(),
    }
    require_authority_false(body, label=f"the generation marker at {generation}")
    marker = self_hashed(body, field=GENERATION_SELF_HASH_FIELD)
    text = json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path = generation / GENERATION_COMMITTED_MARKER
    handle, staged = tempfile.mkstemp(dir=str(generation), suffix=".partial")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(text.encode())
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return path


def publish_process_v2_chain(repo_root: Path, *, generations_root: Path) -> Path:
    """Stage a complete validated generation, then commit it. Returns its directory."""

    generation = stage_process_v2_chain_generation(
        repo_root, generations_root=generations_root
    )
    commit_process_v2_chain_generation(generation)
    return generation


def read_committed_generation(generation: Path) -> dict[str, bytes]:
    """Return the artifacts of a COMMITTED generation.

    Raises:
        ProcessV2ChainError: if the marker is absent, malformed, self-inconsistent,
            grants authority, or disagrees with the artifacts on disk. A generation
            without a valid marker does not exist for a reader; it is never read as
            a partial one.
    """

    generation = Path(generation)
    marker_path = generation / GENERATION_COMMITTED_MARKER
    if not marker_path.is_file():
        raise ProcessV2ChainError(
            f"the generation at {generation} carries no {GENERATION_COMMITTED_MARKER}, "
            "so it is not committed and must be ignored rather than read as partial"
        )
    try:
        marker = json.loads(marker_path.read_bytes())
    except json.JSONDecodeError as error:
        raise ProcessV2ChainError(
            f"the {GENERATION_COMMITTED_MARKER} at {generation} is not valid JSON"
        ) from error
    if not isinstance(marker, dict):
        raise ProcessV2ChainError(
            f"the {GENERATION_COMMITTED_MARKER} at {generation} is not a JSON object"
        )
    if (
        marker.get("schema") != GENERATION_COMMITTED_SCHEMA
        or marker.get("schema_version") != GENERATION_COMMITTED_SCHEMA_VERSION
    ):
        raise ProcessV2ChainError(
            f"the {GENERATION_COMMITTED_MARKER} at {generation} declares "
            f"{marker.get('schema')!r} v{marker.get('schema_version')!r}, not "
            f"{GENERATION_COMMITTED_SCHEMA!r} v{GENERATION_COMMITTED_SCHEMA_VERSION}"
        )
    try:
        verify_self_hash(
            marker, field=GENERATION_SELF_HASH_FIELD, label=f"the marker at {generation}"
        )
        require_authority_false(marker, label=f"the marker at {generation}")
    except ProcessV2SchemaError as error:
        raise ProcessV2ChainError(str(error)) from error
    declared = marker.get("artifacts")
    if not isinstance(declared, dict) or set(declared) != set(PROCESS_V2_CHAIN_ARTIFACTS):
        described = sorted(declared) if isinstance(declared, dict) else declared
        raise ProcessV2ChainError(
            f"the marker at {generation} declares {described!r}, not the seven chain "
            "artifacts"
        )
    sealed: dict[str, bytes] = {}
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        path = generation / name
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ProcessV2ChainError(
                f"the committed generation at {generation} is missing {name}"
            ) from error
        if _sha256_bytes(raw) != declared[name]["file_sha256"]:
            raise ProcessV2ChainError(
                f"{name} in the committed generation at {generation} does not match "
                "the physical hash its marker declares"
            )
        sealed[name] = raw
    if process_v2_generation_id(sealed) != marker.get("generation_id"):
        raise ProcessV2ChainError(
            f"the marker at {generation} declares generation_id "
            f"{marker.get('generation_id')!r}, which the artifacts do not reproduce"
        )
    return sealed


def committed_generations(generations_root: Path) -> list[Path]:
    """Every committed generation under ``generations_root``, ignoring the rest.

    A staged-but-uncommitted directory is not listed. That is the whole point of
    the marker: a reader never has to decide whether a partial directory is
    trustworthy, because it never sees one.
    """

    root = Path(generations_root)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            read_committed_generation(child)
        except ProcessV2ChainError:
            continue
        found.append(child)
    return found


def materialize_committed_generation(generation: Path, *, repo_root: Path) -> dict[str, str]:
    """Copy a COMMITTED generation into ``repo_root``; return ``name -> self-hash``.

    Refuses an uncommitted generation, so an interruption cannot reach the
    repository's canonical paths.
    """

    sealed = read_committed_generation(generation)
    repo_root = Path(repo_root)
    published: dict[str, str] = {}
    for name in PROCESS_V2_CHAIN_ARTIFACTS:
        path = repo_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sealed[name])
        published[name] = str(json.loads(sealed[name])[SELF_HASH_FIELD])
    return published


def write_process_v2_chain(repo_root: Path) -> dict[str, str]:
    """Regenerate the committed chain in ``repo_root``; return ``name -> self-hash``.

    Transactional: the seven artifacts are written and the complete graph is
    validated inside a temporary content-addressed generation, the committed marker
    is published last, and only then is the generation materialized at the
    repository's canonical paths. A failure anywhere before the marker leaves
    ``configs/`` untouched. Rebuilding over an already-written chain reproduces
    identical bytes.
    """

    repo_root = Path(repo_root)
    with tempfile.TemporaryDirectory() as staging:
        generation = publish_process_v2_chain(repo_root, generations_root=Path(staging))
        return materialize_committed_generation(generation, repo_root=repo_root)


def clear_generations(generations_root: Path) -> None:
    """Remove every generation under ``generations_root``. For fixtures only."""

    shutil.rmtree(Path(generations_root), ignore_errors=True)


__all__ = [
    "ACTIVE8_DECISION_RUNTIME",
    "AUTHORITY_FIELDS",
    "CAPABILITY_CELLS",
    "CHAIN_CONTRACT_REVISION",
    "CHAIN_SCHEMA_VERSION",
    "DEPENDENCY_EDGES_ADDED_IN_SCHEMA_VERSION_2",
    "DEVELOPMENT_CELL_ROLES",
    "EDITING_CORPUS_V2_CONTRACT",
    "GATE_ZERO_MODEL_PROCESS_V2",
    "GATE_ZERO_STRUCTURAL",
    "GENERATION_SELF_HASH_FIELD",
    "GENERATION_STATUS",
    "LINEAGE_GENERATION_FIELDS",
    "P50_RECIPE_POLICY",
    "PROCESS_IDENTITY_EDGE_FIELDS",
    "PROCESS_IDENTITY_PIN_FIELD",
    "PROCESS_V2_CHAIN_ARTIFACTS",
    "PROCESS_V2_T1_EARLY_STOP_RULE",
    "PROCESS_V2_T1_MINIMUM_STEPS_BEFORE_EARLY_STOP",
    "PROCESS_V2_T1_OPERATIONAL_SEMANTICS_VERSION",
    "PROCESS_V2_T1_TRAJECTORY_EVALUATION",
    "PROCESS_V2_IDENTITY_PROVIDER",
    "PROCESS_V2_IDENTITY_SCHEMA",
    "SELF_HASH_FIELD",
    "SELF_HASH_FIELD_ALGORITHM",
    "SEMANTIC_PROCESS_V2",
    "STATUS_SUFFIX",
    "SUPERSEDED_CHAIN_CONTRACT_REVISION",
    "SUPERSEDED_CHAIN_CONTRACT_REVISION_V2",
    "SUPERSEDED_CHAIN_CONTRACT_REVISION_V3",
    "T1_CAPACITY_POLICY",
    "T1_PANEL_POLICY",
    "WHOLE_CANONICAL_BODY_ALGORITHM",
    "ProcessV2ChainError",
    "build_process_v2_chain",
    "build_process_v2_chain_artifact",
    "clear_generations",
    "commit_process_v2_chain_generation",
    "committed_generations",
    "declared_self_hash",
    "load_process_v2_chain_artifact",
    "materialize_committed_generation",
    "process_v2_chain_self_hash",
    "process_v2_dependency_edges",
    "process_v2_generation_id",
    "process_v2_superseded_generations",
    "process_v2_transitive_dependencies",
    "publish_process_v2_chain",
    "read_committed_generation",
    "semantic_pointer_algorithm",
    "semantic_sha256_of",
    "serialize_process_v2_chain_artifact",
    "stage_process_v2_chain_generation",
    "validate_process_v2_chain_artifact",
    "write_process_v2_chain",
]
