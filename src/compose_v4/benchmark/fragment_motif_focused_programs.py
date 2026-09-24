"""Whole coherent motif extension without unrelated post-completion refinements.

Content, context, size and ring cells are unchanged from the training-derived
joint completion law. The only task-topology choice is to offer eight complete
motif-rooted region programs rather than six T4 refinements of two such seeds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from compose_v4.benchmark.fragment_constrained import FragmentTask
from compose_v4.benchmark.fragment_joint_programs import propose_joint_region_completion
from compose_v4.benchmark.fragment_program_adapter import (
    CompleteProgram,
    learned_program_scores,
    select_learned_program,
)


@dataclass(frozen=True)
class MotifPanelResult:
    selected: CompleteProgram | None
    receipt: dict


def sample_motif_panel(context, sampler, model, rng) -> MotifPanelResult:
    if context.prompt.task is not FragmentTask.MOTIF_EXTENSION:
        raise ValueError("focused motif panel requires a motif-extension prompt")
    before = copy.deepcopy(rng.bit_generator.state)
    offered, candidates, scores, draw_indices, seen = [], [], [], [], set()
    for draw in range(8):
        record = {"draw": draw}
        try:
            candidate = propose_joint_region_completion(context, sampler, rng)
        except ValueError as error:
            record.update(status="compiler_or_constraint_abstention", reason=str(error))
        else:
            record.update(
                endpoint=candidate.smiles,
                actions=candidate.trace["actions"],
                provenance=candidate.provenance,
            )
            if candidate.smiles in seen:
                record["status"] = "duplicate_endpoint"
            else:
                seen.add(candidate.smiles)
                try:
                    [score] = learned_program_scores(model, (candidate,))
                except (ValueError, KeyError) as error:
                    record.update(status="model_support_abstention", reason=str(error))
                else:
                    if np.isfinite(score):
                        record.update(status="model_supported", mean_log_mark=float(score))
                        candidates.append(candidate)
                        scores.append(float(score))
                        draw_indices.append(draw)
                    else:
                        record.update(
                            status="model_support_abstention", reason="nonfinite_native_probability"
                        )
        offered.append(record)
    selected, selection = None, None
    if candidates:
        selected, selection = select_learned_program(tuple(candidates), tuple(scores), rng)
        selection["selected_draw"] = draw_indices[selection["selected_index"]]
    return MotifPanelResult(
        selected,
        {
            "schema": "fragment_motif_focused_panel_v1",
            "rng_state_before": before,
            "rng_state_after": copy.deepcopy(rng.bit_generator.state),
            "offered": offered,
            "offered_count": 8,
            "selected_smiles": selected.smiles if selected else None,
            "selection": selection,
            "output_count": int(selected is not None),
            "exact_compiled_count": sum("actions" in row for row in offered),
            "unique_compiled_endpoints": len(seen),
            "model_supported_count": len(candidates),
            "qed_sa_guidance": False,
            "benchmark_reference_used": False,
            "beam_search": False,
        },
    )
