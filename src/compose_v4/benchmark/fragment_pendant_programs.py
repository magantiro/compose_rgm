"""One complete COMPOSE decoration program from training-observed pendants."""

import copy
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from compose_v4.benchmark.fragment_program_adapter import (
    CompleteProgram,
    ProgramConstraint,
    admit_program_stages,
    compile_region,
    learned_program_scores,
    select_learned_program,
    verify_prompt_endpoint,
)
from compose_v4.benchmark.training_attachment_fragments import atom_context
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state


@dataclass(frozen=True)
class PendantPanelResult:
    selected: CompleteProgram | None
    receipt: dict


def propose_pendant_decoration(context, sampler, rng):
    """Bind all declared interfaces, compile once, and stop at completion.

    This uses the existing dependency/conflict-aware complete-program executor.
    It differs from broad motif growth by not appending a T4 refinement after
    the requested decorations are present.
    """
    constraint = ProgramConstraint.from_context(context)
    if context.prompt.task.value != "scaffold_decoration":
        raise ValueError("pendant-side policy applies only to scaffold decoration")
    source = context.start_state
    core = Chem.MolFromSmiles(context.start_smiles)
    if core is None:
        raise ValueError("declared decoration core does not parse")
    slots = [site for site, count in constraint.requirements for _ in range(count)]
    slots = [slots[int(index)] for index in rng.permutation(len(slots))]
    entries, receipt = sampler.sample(
        tuple(atom_context(core.GetAtomWithIdx(site)) for site in slots),
        40 - source.n_real_atoms,
        rng,
    )
    current, stages, regions = source, [], []
    for slot, entry in zip(slots, entries, strict=True):
        trace, provenance = compile_region(current, entry["rooted_smiles"], (slot,))
        stages.append(
            {
                "name": "observed_training_pendant",
                "actions": trace["actions"],
                "states": trace["states"],
                "endpoint": canonical_state_key(decode_state(trace["states"][-1])),
            }
        )
        current = decode_state(trace["states"][-1])
        regions.append(
            {
                **provenance,
                "source_rows": entry["source_rows"],
                "source_balanced_weight": entry["source_balanced_weight"],
                "training_occurrences": entry["occurrences"],
            }
        )
    candidate = admit_program_stages(
        source,
        stages,
        constraint,
        provenance={
            "lane": "training_pendant_complete_program",
            "regions": regions,
            "module_count": len(regions),
            "learned_head_flags_changed": False,
            "pendant_plan": receipt,
            "stop_reason": "all_declared_decoration_interfaces_completed",
            "t4_refinement": False,
        },
    )
    verify_prompt_endpoint(context, candidate)
    mol = Chem.MolFromSmiles(candidate.smiles)
    expected_atoms = source.n_real_atoms + sum(entry["heavy_atoms"] for entry in entries)
    if mol.GetNumHeavyAtoms() != expected_atoms:
        raise RuntimeError("exact decoration endpoint changed the sampled atom count")
    return candidate


def sample_pendant_panel(context, sampler, model, rng) -> PendantPanelResult:
    """Offer eight complete legal programs and sample by the frozen model score."""
    before = copy.deepcopy(rng.bit_generator.state)
    offered, candidates, scores, draw_indices, seen = [], [], [], [], set()
    for draw in range(8):
        record = {"draw": draw}
        try:
            candidate = propose_pendant_decoration(context, sampler, rng)
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
    return PendantPanelResult(
        selected,
        {
            "schema": "fragment_pendant_complete_panel_v1",
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
