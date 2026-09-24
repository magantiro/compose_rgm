from __future__ import annotations

import copy

import numpy as np
import pytest
from run_fragment_program_matched_pilot import (
    assigned_prompt,
    require_qualified_support,
    restore_attempt_rng,
    restore_candidates,
)

from compose_v4.benchmark.fragment_program_adapter import (
    ProgramConstraint,
    admit_complete_program,
    compile_region,
    select_learned_program,
)
from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph


def test_saved_candidates_replay_and_resume_selection_without_rescoring():
    source = pad_molecular_graph(smiles_to_molecular_graph("CC"), 48)
    records, original = [], []
    for region, score in (("[1*]CCO", -3.0), ("[1*]c1ccccc1", -4.0)):
        trace, provenance = compile_region(source, region, (0,))
        candidate = admit_complete_program(
            source, trace, ProgramConstraint((0, 1), (0,), ((0, 1),)), provenance=provenance
        )
        original.append(candidate)
        records.append(
            {
                "status": "model_supported",
                "actions": candidate.trace["actions"],
                "provenance": candidate.provenance,
                "endpoint": candidate.smiles,
                "mean_log_mark": score,
            }
        )
    records.insert(1, {"status": "model_support_abstention"})
    candidates, scores = restore_candidates(source, records)
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    first, receipt1 = select_learned_program(original, (-3.0, -4.0), rng1)
    second, receipt2 = select_learned_program(candidates, scores, rng2)
    assert (first.smiles, receipt1, rng1.bit_generator.state) == (
        second.smiles,
        receipt2,
        rng2.bit_generator.state,
    )
    records[0]["endpoint"] = "CC"
    with pytest.raises(ValueError, match="changed on exact replay"):
        restore_candidates(source, records)
    assert restore_candidates(source, [{"status": "compiler_or_constraint_abstention"}]) == (
        [],
        [],
    )


def test_rng_resume_preserves_refusal_draws_and_rejects_missing_records():
    uninterrupted = np.random.default_rng(12)
    before = copy.deepcopy(uninterrupted.bit_generator.state)
    uninterrupted.random(37)
    after = copy.deepcopy(uninterrupted.bit_generator.state)
    attempt = {
        "attempt_index": 0,
        "rng_state_before": before,
        "rng_state_after": after,
        "offered": [{"draw": i, "status": "compiler_or_constraint_abstention"} for i in range(3)],
        "complete": False,
    }
    resumed = np.random.default_rng(12)
    restore_attempt_rng(attempt, 0, resumed)
    assert resumed.random() == uninterrupted.random()
    with pytest.raises(ValueError, match="identity/RNG"):
        restore_attempt_rng(attempt, 1, np.random.default_rng(12))
    attempt["complete"] = True
    with pytest.raises(ValueError, match="eight candidate"):
        restore_attempt_rng(attempt, 0, np.random.default_rng(12))
    attempt["offered"][1]["draw"] = 0
    with pytest.raises(ValueError, match="reordered"):
        restore_attempt_rng(attempt, 0, np.random.default_rng(12))


def test_incomplete_or_negative_support_cannot_start_pilot():
    support = {
        "support_pass": True,
        "native_t4_support_pass": True,
        "every_prompt_has_output": True,
        "shard_hashes": {str(i): "hash" for i in range(20)},
        "tasks": {
            "motif_extension": {"attempts": 20},
            "scaffold_decoration": {"attempts": 20},
        },
    }
    manifest = {"mode": "support", "sample_count_per_prompt": 2, "panel_attempts": 8, "seed": 0}
    require_qualified_support(support, manifest)
    support["support_pass"] = False
    with pytest.raises(ValueError, match="not complete/passed"):
        require_qualified_support(support, manifest)
    support["support_pass"] = True
    support["tasks"]["motif_extension"]["attempts"] = 18
    with pytest.raises(ValueError, match="not complete/passed"):
        require_qualified_support(support, manifest)


def test_cpu_shards_partition_all_prompts_without_changing_drug_pairs():
    partitions = [{i for i in range(20) if assigned_prompt(i, shard, 2)} for shard in range(2)]
    assert not partitions[0] & partitions[1]
    assert partitions[0] | partitions[1] == set(range(20))
    assert all((i in part) == (i + 1 in part) for part in partitions for i in range(0, 20, 2))
    assert all(assigned_prompt(i, 0, 1) for i in range(20))
    with pytest.raises(ValueError, match="one or two"):
        assigned_prompt(0, 0, 3)
