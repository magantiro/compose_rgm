import numpy as np
from rdkit import Chem

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentPrompt, FragmentTask
from compose_v4.benchmark.fragment_joint_programs import propose_joint_region_completion
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior, JointCompletionSampler
from compose_v4.benchmark.training_attachment_fragments import atom_context


def test_joint_plan_uses_existing_exact_compiler_and_preserves_supplied_ring():
    prompt = FragmentPrompt(
        "synthetic", "Cc1ccccc1", FragmentTask.SCAFFOLD_DECORATION, ("[1*]c1ccccc1",)
    )
    context = build_prompt_context(prompt)
    core = Chem.MolFromSmiles(context.start_smiles)
    site = context.attachment.interfaces[0]
    key = atom_context(core.GetAtomWithIdx(site))
    entries = [
        {
            "contexts": [key],
            "rooted_smiles": text,
            "heavy_atoms": size,
            "ring_count": rings,
            "occurrences": 1,
            "source_rows": [1],
        }
        for text, size, rings in (("[1*]CC", 2, 0), ("[1*]c1ccccc1", 6, 1), ("[1*]C1CCNCC1", 6, 1))
    ]
    sampler = JointCompletionSampler(entries, JointCompletionPrior(((12, 2, 1000),)))
    result = propose_joint_region_completion(context, sampler, np.random.default_rng(0))
    mol = Chem.MolFromSmiles(result.smiles)
    assert mol.GetNumHeavyAtoms() == 12
    assert mol.GetRingInfo().NumRings() == 2
    assert mol.HasSubstructMatch(Chem.MolFromSmiles("c1ccccc1"))
    assert len(result.trace["actions"]) > 1
    assert result.provenance["joint_plan"]["all_interfaces_planned_together"]


def test_drug_name_does_not_change_joint_proposals():
    first = FragmentPrompt("arbitrary_a", "CCO", FragmentTask.SCAFFOLD_DECORATION, ("[1*]CC",))
    second = FragmentPrompt("arbitrary_b", "CCO", FragmentTask.SCAFFOLD_DECORATION, ("[1*]CC",))
    ctx = build_prompt_context(first)
    mol = Chem.MolFromSmiles(ctx.start_smiles)
    key = atom_context(mol.GetAtomWithIdx(ctx.attachment.interfaces[0]))
    entry = {
        "contexts": [key],
        "rooted_smiles": "[1*]CCO",
        "heavy_atoms": 3,
        "ring_count": 0,
        "occurrences": 1,
        "source_rows": [1],
    }
    sampler = JointCompletionSampler([entry], JointCompletionPrior(((5, 0, 1),)))
    a = propose_joint_region_completion(ctx, sampler, np.random.default_rng(10))
    b = propose_joint_region_completion(
        build_prompt_context(second), sampler, np.random.default_rng(10)
    )
    assert a.smiles == b.smiles and a.trace == b.trace
