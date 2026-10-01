"""Replay-checked program inputs and frozen canonical-successor scoring.

No oracle values enter this module. Endpoint strings check identity only; they
are never parsed to reconstruct a slot-addressed source or intermediate state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from compose_v4.chem.aromaticity import resonance_invariant_bond_classes
from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.control.reference_guidance import (
    GuidanceConfig,
    GuidedPanel,
    ProgramScore,
    ScoredPanel,
    guide_panel,
)
from compose_v4.experiments.factorized_successor_training import (
    SuccessorTrainingError,
    compile_teacher_successor_fibers_support_only,
    forward_teacher_successor_batch,
    resolve_successor_process_runtime,
    rewrite_action_codec_sha256,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    _CYCLE_OP_EXECUTOR_TO_FAMILY,
    LEGACY_ATOM_RESTATE_ACTION_SEMANTICS,
    LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS,
    LEGACY_CYCLE_OPEN_ACTION_SEMANTICS,
    PROCESS_V2_EDITING_PROCESS_SEMANTICS,
    prepare_factorized_mark_batch,
)
from compose_v4.model.reference_checkpoint import LoadedReference
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import (
    InvalidRewrite,
    canonical_state_key,
    de_novo_rewrite_system,
    editing_v2_semantic_rewrite_system,
)
from compose_v4.rewrite.operators import (
    AtomRestate,
    BondDelete,
    BondInsert,
    CycleCloseEdge,
    CycleOpenEdge,
    SemanticAtomRestate,
)
from compose_v4.rewrite.trace_shard import decode_state, encode_state
from compose_v4.rewrite.tracelets import RingSystemRestate


@dataclass(frozen=True)
class ProgramInput:
    """An immutable snapshot of an executed program, or an explicit missing trace."""

    candidate_id: str
    endpoint: str
    trace_json: str | None
    missing_reason: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or not isinstance(self.endpoint, str)
            or not self.endpoint
        ):
            raise ValueError("program input requires candidate identity and endpoint")
        if self.trace_json is None and not self.missing_reason:
            raise ValueError("missing program traces require an explicit reason")
        if self.trace_json is not None and self.missing_reason:
            raise ValueError("a program input cannot have both a trace and a missing-trace reason")

    @classmethod
    def from_trace(cls, candidate_id: str, endpoint: str, trace: Mapping[str, Any]) -> ProgramInput:
        serialized = json.dumps(dict(trace), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return cls(candidate_id, endpoint, serialized)

    @property
    def trace_sha256(self) -> str | None:
        return (
            None
            if self.trace_json is None
            else hashlib.sha256(self.trace_json.encode()).hexdigest()
        )


def pmo_program_input(candidate: Mapping[str, Any]) -> ProgramInput:
    """Adapt a PMO pool record without importing the oracle or campaign driver."""
    key, endpoint = candidate["candidate_id"], candidate["endpoint"]
    trace = candidate.get("trace")
    if trace is None:
        return ProgramInput(key, endpoint, None, "PMO candidate has no executed trace")
    states = trace.get("states")
    if not states or candidate.get("source_state") != states[0]:
        raise ValueError(f"{key}: PMO source_state differs from the trace's exact source")
    return ProgramInput.from_trace(key, endpoint, trace)


def t4_program_input(
    candidate: Mapping[str, Any], *, candidate_id: str, source: MolecularGraph | None = None
) -> ProgramInput:
    """Adapt actual T4 realized actions; never score a pre-intervention recipe.

    The proposer must retain its exact padded source. Records with no realized
    primitive actions remain explicitly unscored, including older shallow lanes.
    """
    endpoint = candidate["smiles"]
    actions = candidate.get("realized_actions")
    if actions is None:
        return ProgramInput(candidate_id, endpoint, None, "T4 candidate has no realized_actions")
    if candidate.get("realized_endpoint_key") != endpoint:
        raise ValueError(f"{candidate_id}: T4 realized endpoint differs from the proposed endpoint")
    recorded_source = candidate.get("source_state")
    if source is None:
        if recorded_source is None:
            raise ValueError(
                f"{candidate_id}: T4 realized actions require their exact source_state"
            )
        source = decode_state(recorded_source)
    elif recorded_source is not None and encode_state(source) != recorded_source:
        raise ValueError(f"{candidate_id}: supplied T4 source differs from recorded source_state")
    current, states = source, [encode_state(source)]
    system = editing_v2_semantic_rewrite_system()
    for record in actions:
        rule, action = decode_action(record)
        current = system.apply(current, rule, action)
        states.append(encode_state(current))
    if canonical_state_key(current) != endpoint:
        raise ValueError(
            f"{candidate_id}: T4 realized actions do not produce the proposed endpoint"
        )
    return ProgramInput.from_trace(candidate_id, endpoint, {"actions": actions, "states": states})


class NativeMarkUnsupported(ValueError):
    """A legal executed transition has no score through this native mark adapter.

    This is not a claim that its canonical successor has zero reference mass.
    """


def _native_mark(model, predecessor, successor, rule, action):
    if (
        isinstance(action, CycleCloseEdge)
        and model.cycle_close_action_semantics == LEGACY_CYCLE_CLOSE_ACTION_SEMANTICS
    ):
        rule, action = "bond_insert", BondInsert(action.a, action.b, action.order)
    elif (
        isinstance(action, CycleOpenEdge)
        and model.cycle_open_action_semantics == LEGACY_CYCLE_OPEN_ACTION_SEMANTICS
    ):
        rule, action = "bond_delete", BondDelete(action.a, action.b)
    elif (
        isinstance(action, SemanticAtomRestate)
        and model.atom_restate_action_semantics == LEGACY_ATOM_RESTATE_ACTION_SEMANTICS
    ):
        slot = action.v
        rule, action = (
            "atom_restate",
            AtomRestate(
                slot,
                int(successor.atom_types[slot]),
                int(successor.formal_charges[slot]),
                int(successor.implicit_h_counts[slot]),
            ),
        )
    try:
        system = (
            editing_v2_semantic_rewrite_system()
            if model.editing_process_semantics == PROCESS_V2_EDITING_PROCESS_SEMANTICS
            else de_novo_rewrite_system()
        )
        native = system.apply(predecessor, rule, action)
    except InvalidRewrite as error:
        raise NativeMarkUnsupported(f"native replay rejected {rule}: {error}") from error
    if exact_graph_key(native) != exact_graph_key(successor):
        same_atoms = all(
            np.array_equal(getattr(native, name), getattr(successor, name))
            for name in ("atom_types", "formal_charges", "implicit_h_counts")
        )
        same_chemistry = (
            same_atoms
            and canonical_state_key(native) == canonical_state_key(successor)
            and np.array_equal(
                resonance_invariant_bond_classes(native),
                resonance_invariant_bond_classes(successor),
            )
        )
        if not same_chemistry:
            raise NativeMarkUnsupported("native mark changes the slot-mapped successor chemistry")
    return rule, action


class FrozenProgramReference:
    """Deterministic CPU float32 scoring, using the loaded model's actual support."""

    def __init__(self, reference: LoadedReference, *, progress: float = 0.5, batch_size: int = 16):
        if not math.isfinite(progress) or not 0 < progress < 1:
            raise ValueError("canonical-successor progress must be finite and in (0, 1)")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("reference batch_size must be a positive integer")
        self.reference, self.progress, self.batch_size = reference, progress, batch_size

    def identity(self) -> dict:
        """Serializable inference identity for run manifests and resume checks."""
        return {
            "checkpoint_sha256": self.reference.checkpoint_sha256,
            "catalog_fingerprint": self.reference.catalog_fingerprint,
            "catalog_sha256": self.reference.catalog_sha256,
            "max_active_atoms": self.reference.max_active_atoms,
            "progress": self.progress,
            "batch_size": self.batch_size,
            "device": "cpu",
            "dtype": "float32",
        }

    def score(self, programs: Sequence[ProgramInput]) -> ScoredPanel:
        model = self.reference.model
        if any(module.training for module in model.modules()) or any(
            p.requires_grad or p.device.type != "cpu" or p.dtype != torch.float32
            for p in model.parameters()
        ):
            raise ValueError("reference must remain frozen, in eval mode, on CPU float32")
        ids = tuple(program.candidate_id for program in programs)
        if len(set(ids)) != len(ids):
            raise ValueError("reference panel contains duplicate candidate IDs")
        with torch.inference_mode():
            scores = tuple(self._score_program(program) for program in programs)
        return ScoredPanel(self.reference.checkpoint_sha256, scores, self.progress)

    def _score_program(self, program: ProgramInput) -> ProgramScore:
        if program.trace_json is None:
            return ProgramScore(program.candidate_id, None, "missing_trace", program.missing_reason)
        trace = json.loads(program.trace_json)
        actions, states = trace["actions"], trace["states"]
        if not actions or len(states) != len(actions) + 1:
            raise ValueError(
                f"{program.candidate_id}: trace needs a nonempty program and L+1 states"
            )
        graphs = tuple(decode_state(state) for state in states)
        if canonical_state_key(graphs[-1]) != program.endpoint:
            raise ValueError(f"{program.candidate_id}: trace endpoint differs from the candidate")
        system = editing_v2_semantic_rewrite_system()
        decoded = tuple(decode_action(record) for record in actions)
        # Validate the entire trace before reporting native support. A malformed
        # later state must not be hidden by an unsupported early reference mark.
        for index, (rule, action) in enumerate(decoded):
            product = system.apply(graphs[index], rule, action)
            if exact_graph_key(product) != exact_graph_key(graphs[index + 1]):
                raise ValueError(f"{program.candidate_id}: exact replay mismatch at step {index}")
        outside = [
            (index, graph.n_real_atoms)
            for index, graph in enumerate(graphs)
            if not 1 <= graph.n_real_atoms <= self.reference.max_active_atoms
        ]
        if outside:
            return ProgramScore(
                program.candidate_id,
                None,
                "unsupported_state",
                f"states {outside} exceed checkpoint active-atom support "
                f"[1, {self.reference.max_active_atoms}]",
            )
        model = self.reference.model
        rows = []
        for index, (rule, action) in enumerate(decoded):
            try:
                native_rule, native_action = _native_mark(
                    model, graphs[index], graphs[index + 1], rule, action
                )
            except NativeMarkUnsupported as error:
                return ProgramScore(
                    program.candidate_id, None, "unsupported_native_mark", f"step {index}: {error}"
                )
            rows.append((graphs[index], graphs[index + 1], native_rule, native_action))
        options = {
            key: getattr(model, key)
            for key in (
                "editing_process_semantics",
                "atom_restate_action_semantics",
                "ring_restate_scorer_mode",
                "cycle_close_action_semantics",
                "cycle_open_action_semantics",
                "atom_delete_action_semantics",
            )
        }
        options.update(
            ring_catalog=model.ring_catalog,
            compute_ring_grow_support=model.enable_ring_grow_macro,
            compute_ring_restates=model.enable_ring_restates,
            compute_cyclic_graft=model.enable_cyclic_graft,
            compute_ring_opening=model.enable_ring_opening,
            compute_ring_system_delete=model.enable_ring_system_delete,
        )
        values = []
        process = resolve_successor_process_runtime(model)
        for start in range(0, len(rows), self.batch_size):
            part = rows[start : start + self.batch_size]
            batch = prepare_factorized_mark_batch(
                tuple(row[0] for row in part),
                (self.progress,) * len(part),
                tuple(row[3] for row in part),
                tuple(row[2] for row in part),
                (1.0,) * len(part),
                **options,
            )
            for index, action in enumerate(batch.teacher_actions):
                if (
                    isinstance(action, RingSystemRestate)
                    and action not in batch.ring_restate_actions[index]
                ):
                    return ProgramScore(
                        program.candidate_id,
                        None,
                        "unsupported_native_mark",
                        f"step {start + index}: ring restate outside native finite fiber",
                    )
            try:
                compiled = compile_teacher_successor_fibers_support_only(
                    model,
                    tuple(row[0] for row in part),
                    tuple(row[1] for row in part),
                    teacher_action_sha256s=tuple(
                        rewrite_action_codec_sha256(
                            row[2], row[3], schema_version=process.action_codec_schema_version
                        )
                        for row in part
                    ),
                    teacher_families=tuple(
                        _CYCLE_OP_EXECUTOR_TO_FAMILY.get(row[2], row[2]) for row in part
                    ),
                    times=(self.progress,) * len(part),
                )
            except SuccessorTrainingError as error:
                if "teacher action is not one exact coordinate" not in str(error):
                    raise
                return ProgramScore(
                    program.candidate_id,
                    None,
                    "unsupported_native_mark",
                    f"step {start}: {error}",
                )
            values.extend(
                forward_teacher_successor_batch(
                    model, batch, tuple(row.teacher_fiber for row in compiled)
                )
                .selected_productive_successor_log_probability.cpu()
                .tolist()
            )
        if any(math.isnan(value) or value == math.inf for value in values):
            raise FloatingPointError(f"{program.candidate_id}: invalid numerical reference output")
        if any(value == -math.inf for value in values):
            steps = [index for index, value in enumerate(values) if value == -math.inf]
            return ProgramScore(
                program.candidate_id,
                None,
                "unsupported_native_mark",
                f"native scorer has zero canonical-successor mass at steps {steps}",
            )
        return ProgramScore(program.candidate_id, math.fsum(values) / len(values), "scored")


@dataclass(frozen=True)
class ProgramPanelGuidance:
    """Join controller rows to exact programs before an exploration draw.

    PMO uses the campaign's candidate ID. Deduplicated T4 pools can use their
    canonical ``smiles`` key, with that same key assigned to each ProgramInput.
    The caller owns the frozen reference for the run and persists the receipts.
    """

    programs: tuple[ProgramInput, ...]
    reference: FrozenProgramReference | None
    config: GuidanceConfig
    identity_field: str = "candidate_id"

    def __post_init__(self) -> None:
        keys = [program.candidate_id for program in self.programs]
        if len(set(keys)) != len(keys):
            raise ValueError("program panel has duplicate candidate identities")
        if not isinstance(self.identity_field, str) or not self.identity_field:
            raise ValueError("reference candidate identity field must be explicit")
        if self.config.mode != "off" and self.reference is None:
            raise ValueError("shadow and active selection require a frozen reference")

    def __call__(self, candidates: Sequence[Mapping[str, Any]]) -> GuidedPanel:
        if not candidates:
            raise ValueError("reference exploration requires a nonempty candidate panel")
        ids = tuple(row[self.identity_field] for row in candidates)
        by_id = {program.candidate_id: program for program in self.programs}
        absent = [key for key in ids if key not in by_id]
        if absent:
            raise ValueError(f"no exact program record for candidate IDs: {absent}")
        programs = tuple(by_id[key] for key in ids)
        return guide_panel(
            ids,
            (1.0 / len(ids),) * len(ids),
            config=self.config,
            score=None if self.reference is None else lambda: self.reference.score(programs),
        )
