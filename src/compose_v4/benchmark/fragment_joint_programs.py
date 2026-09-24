"""Whole-completion content allocation, shared COMPOSE compilation and T4 lanes."""

from rdkit import Chem

from compose_v4.benchmark.fragment_program_adapter import (
    CompleteProgram,
    ProgramConstraint,
    admit_program_stages,
    compile_region,
    verify_prompt_endpoint,
)
from compose_v4.benchmark.fragment_t4_programs import T4_LANES, refine_with_t4
from compose_v4.benchmark.training_attachment_fragments import atom_context
from compose_v4.rewrite.kernel import canonical_state_key
from compose_v4.rewrite.trace_shard import decode_state


def propose_joint_region_completion(context, sampler, rng):
    constraint = ProgramConstraint.from_context(context)
    source = context.start_state
    core = Chem.MolFromSmiles(context.start_smiles)
    slots = [site for site, count in constraint.requirements for _ in range(count)]
    slots = [slots[int(i)] for i in rng.permutation(len(slots))]
    selected, receipt = sampler.sample(
        [atom_context(core.GetAtomWithIdx(s)) for s in slots],
        source.n_real_atoms,
        core.GetRingInfo().NumRings(),
        rng,
    )
    current, stages, regions = source, [], []
    for slot, entry in zip(slots, selected, strict=True):
        trace, provenance = compile_region(current, entry["rooted_smiles"], (slot,))
        stages.append(
            {
                "name": "observed_training_region",
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
                "training_occurrences": entry["occurrences"],
            }
        )
    candidate = admit_program_stages(
        source,
        stages,
        constraint,
        provenance={
            "lane": "joint_training_regions",
            "regions": regions,
            "module_count": len(regions),
            "learned_head_flags_changed": False,
            "joint_plan": receipt,
        },
    )
    verify_prompt_endpoint(context, candidate)
    mol = Chem.MolFromSmiles(candidate.smiles)
    if (mol.GetNumHeavyAtoms(), mol.GetRingInfo().NumRings()) != (
        receipt["planned_heavy_atoms"],
        receipt["planned_rings"],
    ):
        raise RuntimeError("shared compiler endpoint differs from whole-completion plan")
    return candidate


def propose_joint_panel_member(context, sampler, rng, draw):
    if type(draw) is not int or draw < 0:
        raise ValueError("candidate draw must be a nonnegative integer")
    seed = propose_joint_region_completion(context, sampler, rng)
    choice = draw % 4
    result = seed if choice == 0 else refine_with_t4(context, seed, rng, T4_LANES[choice - 1])
    mol = Chem.MolFromSmiles(result.smiles)
    return CompleteProgram(
        result.endpoint,
        result.trace,
        {
            **result.provenance,
            "final_heavy_atoms": mol.GetNumHeavyAtoms(),
            "final_rings": mol.GetRingInfo().NumRings(),
            "structural_prior_role": "joint private seed proposal; T4 refinement may change final cell",
        },
    )
