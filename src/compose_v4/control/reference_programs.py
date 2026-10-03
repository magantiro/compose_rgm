"""Replay-checked program inputs and frozen canonical-successor scoring.

No oracle values enter this module. Endpoint strings check identity only; they
are never parsed to reconstruct a slot-addressed source or intermediate state.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import torch

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.option_continuation import exact_graph_key
from compose_v4.control.reference_guidance import (
    GuidanceConfig,
    GuidedPanel,
    ProgramScore,
    ScoredPanel,
    guide_panel,
)
from compose_v4.experiments.factorized_mark_conditional import (
    operator_capability_batch_kwargs,
)
from compose_v4.experiments.factorized_successor_training import (
    SuccessorTrainingError,
    compile_teacher_successor_fibers_support_only,
    forward_teacher_successor_batch,
)
from compose_v4.model.factorized_tracelet_rate_model import (
    prepare_factorized_mark_batch,
)
from compose_v4.model.reference_checkpoint import LoadedReference
from compose_v4.rewrite.action_codec_v4 import decode_action
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_semantic_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state, encode_state


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
        rows = tuple(pairwise(graphs))
        options = operator_capability_batch_kwargs(model.operator_capabilities)
        options["ring_catalog"] = model.ring_catalog
        values = []
        for start in range(0, len(rows), self.batch_size):
            part = rows[start : start + self.batch_size]
            try:
                compiled = compile_teacher_successor_fibers_support_only(
                    model,
                    tuple(row[0] for row in part),
                    tuple(row[1] for row in part),
                    times=(self.progress,) * len(part),
                    require_exact_teacher_action=False,
                )
            except SuccessorTrainingError as error:
                if "canonical target is absent from production marked support" not in str(error):
                    raise
                return ProgramScore(
                    program.candidate_id,
                    None,
                    "unsupported_native_mark",
                    f"batch starting at step {start}: {error}",
                )
            # The exact executed edit has already been replay-checked.  The
            # reference law is over canonical molecular successors, so its
            # scoring batch uses a native alias family, not the program's
            # possibly different edit coordinate or family.
            batch = prepare_factorized_mark_batch(
                tuple(row[0] for row in part),
                (self.progress,) * len(part),
                (None,) * len(part),
                (None,) * len(part),
                (1.0,) * len(part),
                **options,
            )
            batch = replace(
                batch,
                teacher_rule_names=tuple(
                    row.teacher_fiber.aliases[0].family_name for row in compiled
                ),
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
