"""Complete multi-interface pendant programs use the exact COMPOSE executor."""

from pathlib import Path

import numpy as np
import pytest

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import FragmentTask, load_genmol_prompts
from compose_v4.benchmark.fragment_pendant_programs import propose_pendant_decoration
from compose_v4.benchmark.fragment_program_adapter import ProgramConstraint
from compose_v4.chem.state import is_connected_or_null, is_valid_state

PROMPTS = Path("data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv")


class OneAtomTrainingSampler:
    def sample(self, contexts, capacity, rng):
        assert capacity >= len(contexts)
        return tuple(
            {
                "rooted_smiles": "[1*]C",
                "heavy_atoms": 1,
                "source_rows": [1],
                "source_balanced_weight": 1.0,
                "occurrences": 1,
            }
            for _ in contexts
        ), {"schema": "fixture", "contexts": contexts}


@pytest.mark.skipif(not PROMPTS.exists(), reason="documented fragment prompt asset absent")
def test_six_interface_decoration_is_one_complete_exact_program():
    prompt = next(
        row
        for row in load_genmol_prompts(PROMPTS)
        if row.task is FragmentTask.SCAFFOLD_DECORATION and row.drug_name == "MARIBAVIR"
    )
    context = build_prompt_context(prompt)
    required = len(ProgramConstraint.from_context(context).requirements)
    candidate = propose_pendant_decoration(
        context, OneAtomTrainingSampler(), np.random.default_rng(0)
    )
    assert required == 6
    assert len(candidate.trace["actions"]) == required
    assert candidate.endpoint.n_real_atoms == context.start_state.n_real_atoms + required
    assert is_valid_state(candidate.endpoint) and is_connected_or_null(candidate.endpoint)
    assert candidate.provenance["stop_reason"] == "all_declared_decoration_interfaces_completed"
    assert not candidate.provenance["t4_refinement"]
