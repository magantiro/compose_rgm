"""Teacher-forced exact support compiler for the Editing-V3 protonation action.

This module is a support diagnostic, not an autonomous proposer.  It first
enumerates the complete narrow protonation-restatement fiber, then delegates the
remaining charge-preserving graph delta to the existing target-informed Active8
compiler.  Every returned action is re-encoded under V5 and replayed by the new
runtime from the original exact source state.
"""

from __future__ import annotations

from dataclasses import dataclass

from compose_v4.chem.molecular_graph import (
    MolecularGraph,
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.winner_paths import PathConfig, find_path_from_state
from compose_v4.rewrite import action_codec_v4 as v4
from compose_v4.rewrite.action_codec_v5 import decode_action, encode_action
from compose_v4.rewrite.kernel import (
    canonical_state_key,
    editing_v3_protonation_rewrite_system,
)
from compose_v4.rewrite.operators import enumerate_atom_protonation_restates
from compose_v4.rewrite.trace_shard import encode_state

PROGRAM_RECEIPT_SCHEMA = "atom_protonation_restate_program_receipt_v1"
COMPILER_RESULT_SCHEMA = "atom_protonation_restate_compiler_result_v1"


@dataclass(frozen=True)
class ProtonationProgramConfig:
    maximum_primitives: int = 32
    maximum_active_atoms: int = 40
    persistent_slots: int = 48
    mapping_timeout_seconds: int = 1
    matches_per_molecule: int = 4
    maximum_expansions: int = 512
    children_per_expansion: int = 4

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError(
                "protonation-program allocations must be positive integers"
            )
        if (
            self.maximum_primitives != 32
            or self.maximum_active_atoms != 40
            or self.persistent_slots != 48
        ):
            raise ValueError(
                "protonation program preserves the declared 32-step/40-active/48-slot support"
            )

    def active8_tail(self) -> PathConfig:
        return PathConfig(
            slots=self.persistent_slots,
            max_active=self.maximum_active_atoms,
            mapping_timeout_seconds=self.mapping_timeout_seconds,
            matches_per_molecule=self.matches_per_molecule,
            max_expansions=self.maximum_expansions,
            children_per_expansion=self.children_per_expansion,
            max_steps=self.maximum_primitives - 1,
        )


def execute_protonation_program(
    source: MolecularGraph,
    actions: tuple[dict, ...] | list[dict],
) -> tuple[MolecularGraph, dict]:
    """Execute one complete V5 program without exposing partial endpoints."""

    if source.n_atoms != 48 or not 1 <= source.n_real_atoms <= 40:
        raise ValueError("expected an exact supported 48-slot source")
    if not actions or len(actions) > 32:
        raise ValueError("complete protonation program requires 1..32 primitives")
    system = editing_v3_protonation_rewrite_system()
    current = source
    states = [encode_state(source)]
    for index, record in enumerate(actions):
        rule, action = decode_action(record)
        product = system.apply(current, rule, action)
        if not 1 <= product.n_real_atoms <= 40:
            raise ValueError(f"step {index}: active-atom support failed")
        if canonical_state_key(current) == canonical_state_key(product):
            raise ValueError(f"step {index}: program contains a canonical self-event")
        current = product
        states.append(encode_state(current))
    return current, {
        "schema_version": PROGRAM_RECEIPT_SCHEMA,
        "actions": list(actions),
        "states": states,
        "endpoint": canonical_state_key(current),
        "primitive_edits": len(actions),
        "committed_endpoint_count": 1,
        "partial_endpoint_evaluations": 0,
    }


def _lift_v4_action(record: dict) -> dict:
    rule, action = v4.decode_action(record)
    return encode_action(rule, action)


def compile_protonation_supported_target(
    source_smiles: str,
    target_smiles: str,
    *,
    config: ProtonationProgramConfig | None = None,
) -> dict:
    """Compile one answer-known target through the declared V5 action support."""

    config = ProtonationProgramConfig() if config is None else config
    source = pad_molecular_graph(
        smiles_to_molecular_graph(source_smiles),
        config.persistent_slots,
    )
    target = smiles_to_molecular_graph(target_smiles)
    if target.n_real_atoms > config.maximum_active_atoms:
        return {
            "schema_version": COMPILER_RESULT_SCHEMA,
            "status": "unsupported_target_size",
            "attempts": [],
        }
    target_charge = int(target.formal_charges.sum())
    runtime = editing_v3_protonation_rewrite_system()
    attempts = []
    for protonation_action in enumerate_atom_protonation_restates(source):
        rewritten = runtime.apply(
            source,
            "atom_protonation_restate",
            protonation_action,
        )
        if int(rewritten.formal_charges.sum()) != target_charge:
            continue
        bridge = find_path_from_state(
            encode_state(rewritten),
            target_smiles,
            config.active8_tail(),
            canonical_state_key(rewritten),
        )
        attempt = {
            "protonation_action": encode_action(
                "atom_protonation_restate",
                protonation_action,
            ),
            "tail_status": bridge["status"],
            "tail_witness_steps": bridge.get("witness_steps"),
        }
        attempts.append(attempt)
        if bridge["status"] != "witness_found":
            attempt["tail_attempts"] = bridge.get("attempts", [])
            continue
        actions = (
            attempt["protonation_action"],
            *(_lift_v4_action(record) for record in bridge["actions"]),
        )
        if len(actions) > config.maximum_primitives:
            attempt["tail_status"] = "combined_primitive_limit"
            continue
        endpoint, receipt = execute_protonation_program(source, actions)
        expected = canonical_state_key(
            pad_molecular_graph(target, config.persistent_slots)
        )
        if canonical_state_key(endpoint) != expected:
            raise RuntimeError("exact V5 replay disagrees with the compiler target")
        return {
            "schema_version": COMPILER_RESULT_SCHEMA,
            "status": "supported",
            "source": canonical_state_key(source),
            "target": expected,
            "attempts": attempts,
            "actions": list(actions),
            "states": receipt["states"],
            "primitive_edits": len(actions),
            "protonation_restate_count": 1,
            "active8_tail_primitives": len(actions) - 1,
            "exact_endpoint": True,
            "exact_execution_precision_numerator": 1,
            "exact_execution_precision_denominator": 1,
            "teacher_endpoint_injected_into_autonomous_proposals": False,
            "claim_boundary": "answer-known zero-oracle support certificate only",
        }
    return {
        "schema_version": COMPILER_RESULT_SCHEMA,
        "status": "unsupported",
        "source": canonical_state_key(source),
        "target": canonical_state_key(target),
        "attempts": attempts,
        "residual_blocker": (
            "no admitted protonation restatement reaches the target charge stratum"
            if not attempts
            else "the charge-preserving Active8 tail did not realize the target"
        ),
    }


__all__ = [
    "COMPILER_RESULT_SCHEMA",
    "PROGRAM_RECEIPT_SCHEMA",
    "ProtonationProgramConfig",
    "compile_protonation_supported_target",
    "execute_protonation_program",
]
