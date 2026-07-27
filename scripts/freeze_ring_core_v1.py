#!/usr/bin/env python3
"""§1 of the RingCore preflight: freeze the supervised compositional ring core as RING_CORE_V1.

Emits an immutable, reproducible manifest recording the frozen capability: the source commit, operator-
registry + cycle-operator semantic versions, a capability hash, the subtype-supervision artifact, the exact
training recipe, and the DE-AMBIGUATED public<->internal operator alias mapping (the internal repurposed
slots cycle_insert/cycle_attach are never exposed as public/paper names -- the public contract uses
cycle_close/cycle_open).

RING_CORE_V1 config: enable_cycle_ops=True, enable_ring_macros=False, legacy ring_system_grow macro DISABLED.
Ring generation SUPPORT is the compositional cycle_close/cycle_open; a finite ring catalog is a LATER
(RING_HYBRID_V2 / P4) acceleration layer, never the definition of ring reachability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from compose_v4.chem.molecular_graph import ORGANIC_VOCABULARY
from compose_v4.data.organic_corpus import BROAD_ORGANIC_V1
from compose_v4.experiments.training_support_cache import RING_SUPPORT_SEMANTICS_VERSION
from compose_v4.model.factorized_tracelet_rate_model import MARK_RULE_NAMES

REPO = Path(__file__).resolve().parent.parent


def _hash_sources(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(paths):
        fp = REPO / rel
        digest.update(rel.encode())
        digest.update(fp.read_bytes() if fp.exists() else b"<MISSING>")
    return digest.hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


# The RingCore-V1 public operator contract. Internal slots cycle_insert/cycle_attach are REPURPOSED
# (enable_cycle_ops) to compositional bond insert/delete; they are ambiguous historical names and MUST NOT
# be exposed publicly -- the public + paper name is cycle_close / cycle_open.
_PUBLIC_OPERATORS = {
    "cycle_close": {
        "role": "bond-only cycle closure (ring SUPPORT: add one ring bond over a nonbonded same-component "
        "pair x order)",
        "internal_family_slot": "cycle_insert",  # repurposed under enable_cycle_ops -- NOT exposed publicly
        "executor_rule": "bond_insert",
        "status": "SUPPORTED_BY_CORE",
    },
    "cycle_open": {
        "role": "cycle opening (ring SUPPORT: remove one non-bridge cycle bond)",
        "internal_family_slot": "cycle_attach",  # repurposed under enable_cycle_ops -- NOT exposed publicly
        "executor_rule": "bond_delete",
        "status": "SUPPORTED_BY_CORE",
    },
    "ring_delete": {
        "role": "whole removable-ring deletion (decoration-preserving clean delete)",
        "internal_family_slot": "ring_system_delete",
        "executor_rule": "ring_system_delete",
        "status": "SUPPORTED_BY_CORE",
    },
    "ring_aromaticity_restate": {
        "role": "ring electronic (de-)aromatization restate",
        "internal_family_slot": "ring_system_restate",
        "executor_rule": "ring_system_restate",
        "status": "SUPPORTED_BY_CORE",
    },
    "ring_ear_insert": {
        "role": "path/ear insertion (2-anchor)",
        "internal_family_slot": None,
        "executor_rule": None,
        "status": "REQUIRES_STRUCTURED_P2_MODE",
    },
    "ring_spiro_attach": {
        "role": "one-anchor cyclic (spiro) attachment",
        "internal_family_slot": None,
        "executor_rule": None,
        "status": "REQUIRES_STRUCTURED_P2_MODE",
    },
    "ring_grow_macro": {
        "role": "legacy whole-ring growth macro (adds a full ring in one event)",
        "internal_family_slot": "ring_system_grow",
        "executor_rule": "ring_system_grow",
        "status": "DISABLED_IN_CORE",  # ring addition is compositional (cycle_close) in RingCore-V1
    },
}

# Non-ring editing families retained in the B-edit vocabulary (documented for completeness, not ring ops).
_EDITING_FAMILIES = {
    "atom_insert": "peripheral atom+bond insert",
    "atom_delete": "peripheral atom+bond delete",
    "atom_restate": "bioisostere / heteroatom (element,valence) restate",
    "bond_reorder": "non-aromatic bond-order change",
    "bond_reroute": "cyclic graft (pendant fragment relocation across a single bond)",
}

# Source files whose bytes define the compositional cycle-operator semantics.
_CYCLE_OP_SEMANTIC_SOURCES = [
    "src/compose_v4/experiments/cycle_op_prior.py",
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/kernel.py",
]
_OPERATOR_REGISTRY_SOURCES = [
    "src/compose_v4/rewrite/operators.py",
    "src/compose_v4/rewrite/kernel.py",
    "src/compose_v4/rewrite/tracelets.py",
    "src/compose_v4/rewrite/factorized_fiber.py",
]


def build_manifest() -> dict:
    catalog_fp = "639ff6078c32d43c"  # deterministic production catalog (5 seeds); recorded per checkpoint
    capability = {
        "version": "RING_CORE_V1",
        "flags": {
            "enable_cycle_ops": True,
            "enable_ring_grow_macro": False,  # legacy whole-ring macro disabled
            "enable_ring_macros": False,  # forward-looking P4 hybrid flag; always False in RingCore
            "corrupted_prior_mix": True,
            "organic_vocabulary": True,
            "enable_ring_restates": True,
            "enable_ring_opening": True,
            "enable_cyclic_graft": True,
            "enable_heteroatom_scan": True,
        },
        "ring_support_mechanism": "compositional cycle_close/cycle_open (NOT a finite macro catalog)",
        "supported_public_operators": [
            name for name, spec in _PUBLIC_OPERATORS.items() if spec["status"] == "SUPPORTED_BY_CORE"
        ],
        "disabled_public_operators": [
            name for name, spec in _PUBLIC_OPERATORS.items() if spec["status"] == "DISABLED_IN_CORE"
        ],
        "deferred_p2_operators": [
            name
            for name, spec in _PUBLIC_OPERATORS.items()
            if spec["status"] == "REQUIRES_STRUCTURED_P2_MODE"
        ],
        "atom_vocabulary": "ORGANIC_VOCABULARY",
        "atom_vocabulary_classes": len(ORGANIC_VOCABULARY.classes),
        "corpus_scope_hash": BROAD_ORGANIC_V1.scope_hash(),
        "ring_catalog_fingerprint": catalog_fp,
        "ring_support_semantics_version": int(RING_SUPPORT_SEMANTICS_VERSION),
        "operator_registry_hash": _hash_sources(_OPERATOR_REGISTRY_SOURCES),
        "cycle_op_semantic_hash": _hash_sources(_CYCLE_OP_SEMANTIC_SOURCES),
    }
    capability_hash = hashlib.sha256(
        json.dumps(capability, sort_keys=True).encode()
    ).hexdigest()[:16]

    subtype_artifact = REPO / "diagnostics/production_preflight/subtype_supervision.json"
    manifest = {
        "version": "RING_CORE_V1",
        "purpose": "frozen supervised compositional ring core; a separately-versioned production candidate. "
        "Ring reachability is defined by the compositional operators; the finite catalog is a later "
        "acceleration layer (RING_HYBRID_V2 / P4), never the support.",
        "source_commit": _git_commit(),
        "operator_registry": {
            "mark_rule_names": list(MARK_RULE_NAMES),
            "hash": capability["operator_registry_hash"],
        },
        "cycle_operator_semantic_version": {
            "ring_support_semantics_version": int(RING_SUPPORT_SEMANTICS_VERSION),
            "hash": capability["cycle_op_semantic_hash"],
            "sources": _CYCLE_OP_SEMANTIC_SOURCES,
        },
        "capability": capability,
        "capability_hash": capability_hash,
        "public_operator_contract": _PUBLIC_OPERATORS,
        "editing_families": _EDITING_FAMILIES,
        "alias_exposure_policy": "internal repurposed slots cycle_insert/cycle_attach are engineering aliases "
        "for cycle_close/cycle_open and MUST NOT appear in the paper or checkpoint metadata; the public "
        "names are cycle_close / cycle_open.",
        "subtype_supervision_artifact": {
            "path": "diagnostics/production_preflight/subtype_supervision.json",
            "sha256_16": _sha256(subtype_artifact),
        },
        "training_recipe": {
            "gate": "scripts/train_tracelet_cnof_gate.py --corrupted-prior-mix --cycle-op-mix "
            "--disable-ring-grow-macro --organic-vocabulary --initialize-compatible-from-source-checkpoint",
            "modal": "modal run modal_apps/train_tracelet_gm.py --train-only --corrupted-prior-mix "
            "--cycle-op-mix --disable-ring-grow-macro --organic-vocabulary "
            "--initialize-compatible-from-source-checkpoint",
            "base_checkpoint_sha256": "c9d927510360ec6eb84ff8dae1a222b0b693a9bef0ca23bb5d9cca063025876c",
        },
        "guardrails": "Do NOT implement P4 macros into this production path. Do NOT launch the full A100 run. "
        "Proceed only through the bounded RingCore preflight (<=750 steps).",
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "diagnostics/production_preflight/ring_core_v1_manifest.json",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "version": manifest["version"],
                "source_commit": manifest["source_commit"],
                "capability_hash": manifest["capability_hash"],
                "operator_registry_hash": manifest["operator_registry"]["hash"],
                "cycle_op_semantic_hash": manifest["cycle_operator_semantic_version"]["hash"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
