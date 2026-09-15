from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from compose_v4.chem.molecular_graph import MolecularGraph
from compose_v4.control.autoregressive_program_decoder import (
    CHECKPOINT_SCHEMA,
    LEARNED,
    MARGINAL,
    AutoregressivePolicy,
    DecoderConfig,
    ProgramPolicyNetwork,
    TrainingConfig,
    decode_programs,
    dependency_region_count,
    fit_autoregressive_policy,
    fit_legal_where_how_ranker,
    legal_where_how_metrics,
    prefix_features,
    teacher_forced_metrics,
)
from compose_v4.control.docking_value import identity
from compose_v4.control.generic_legal_action_policy import (
    CHECKPOINT_SCHEMA as LEGAL_CHECKPOINT_SCHEMA,
)
from compose_v4.control.generic_legal_action_policy import (
    RULES,
    RankerConfig,
    enumerate_rule_successors,
)
from compose_v4.rewrite.trace_shard import encode_state

ROOT = Path(__file__).resolve().parents[1]


def _methane() -> MolecularGraph:
    atom_types = np.zeros(48, dtype=np.int32)
    charges = np.zeros(48, dtype=np.int32)
    hydrogens = np.zeros(48, dtype=np.int32)
    bonds = np.zeros((48, 48), dtype=np.int32)
    atom_types[0] = 2
    hydrogens[0] = 4
    return MolecularGraph(atom_types, charges, hydrogens, bonds)


def _zero_policy(source) -> AutoregressivePolicy:
    features = prefix_features(source, source, (), 1)
    network = ProgramPolicyNetwork(len(features), hidden=4)
    for parameter in network.parameters():
        parameter.data.zero_()
    return AutoregressivePolicy(
        network,
        tuple(np.zeros_like(features)),
        tuple(np.ones_like(features)),
        0.1,
        "synthetic-training-identity",
    )


def test_marginal_and_zero_learned_decoder_share_legal_support():
    source = _methane()
    config = DecoderConfig(
        maximum_primitives=1,
        maximum_regions=8,
        maximum_active_atoms=40,
        beam_width=1,
        successors_per_rule=1,
        maximum_outputs=1,
        snapshots=(1,),
    )
    rule = tuple([1 / len(RULES)] * len(RULES))
    control = (0.5, 0.5)

    def one_insert(graph, family):
        if family != "atom_insert":
            return ()
        return enumerate_rule_successors(graph, family)[:1]

    outputs = [
        decode_programs(
            source,
            decoder=decoder,
            config=config,
            marginal_rule_probabilities=rule,
            marginal_control_probabilities=control,
            policy=_zero_policy(source) if decoder == LEARNED else None,
            legal_checkpoint=(
                {"schema_version": LEGAL_CHECKPOINT_SCHEMA}
                if decoder == LEARNED
                else None
            ),
            successor_enumerator=one_insert,
            legal_scorer=lambda graph, candidate, checkpoint: 0.0,
        )
        for decoder in (MARGINAL, LEARNED)
    ]
    for output in outputs:
        rows = output["snapshots"]["1"]
        assert len(rows) == 1
        assert rows[0]["primitive_count"] == 1
        assert output["telemetry"]["exact_execution_precision"] == 1.0
        assert output["new_oracle_calls"] == 0
    assert outputs[0]["snapshots"]["1"][0]["endpoint_key"] == outputs[1][
        "snapshots"
    ]["1"][0]["endpoint_key"]


def test_created_handle_keeps_actions_in_one_dependency_region():
    actions = (
        {
            "executor_rule": "atom_insert",
            "payload": {"slot": 3, "neighbors": [[0, 1]]},
        },
        {"executor_rule": "atom_restate_semantic", "payload": {"v": 3}},
        {"executor_rule": "atom_delete", "payload": {"v": 8}},
    )
    assert dependency_region_count(actions[:2]) == 1
    assert dependency_region_count(actions) == 2


def test_runtime_checkpoint_round_trip_and_forbidden_key_rejection():
    source = _methane()
    policy = _zero_policy(source)
    checkpoint = policy.to_checkpoint()
    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA
    restored = AutoregressivePolicy.from_checkpoint(checkpoint)
    expected = policy.probabilities(prefix_features(source, source, (), 1))
    observed = restored.probabilities(prefix_features(source, source, (), 1))
    assert np.allclose(expected[0], observed[0])
    assert np.allclose(expected[1], observed[1])
    checkpoint["target"] = "forbidden"
    with pytest.raises(ValueError, match="forbidden"):
        AutoregressivePolicy.from_checkpoint(checkpoint)


def test_autoregressive_fit_uses_complete_trace_and_serializes_no_teacher():
    source = _methane()
    successor = enumerate_rule_successors(source, "atom_insert")[0]
    traces = [
        {
            "source_group": "source-a",
            "source": source,
            "trace": {
                "states": [encode_state(source), encode_state(successor.successor)],
                "actions": [successor.action_record],
            },
        }
    ]
    policy, history = fit_autoregressive_policy(
        traces,
        maximum=2,
        config=TrainingConfig(hidden=4, updates=2, batch_size=2, seed=7),
    )
    metrics = teacher_forced_metrics(policy, traces, maximum=2)
    assert metrics["teacher_rows"] == 2
    assert metrics["rule_rows"] == 1
    assert len(history) == 2
    checkpoint = policy.to_checkpoint()
    assert checkpoint["runtime_teacher_rows"] == 0
    assert "source-a" not in repr(checkpoint)


def test_legal_where_how_fit_preserves_teacher_coverage():
    source = _methane()
    successor = enumerate_rule_successors(source, "atom_insert")[0]
    traces = [
        {
            "source_group": "source-a",
            "source": source,
            "trace": {
                "states": [encode_state(source), encode_state(successor.successor)],
                "actions": [successor.action_record],
            },
        }
    ]
    checkpoint, fit = fit_legal_where_how_ranker(
        traces, maximum=2, config=RankerConfig(negatives_per_teacher=2)
    )
    evaluation = legal_where_how_metrics(traces, checkpoint)
    assert fit["teacher_successor_coverage"] == 1.0
    assert evaluation["teacher_successor_coverage"] == 1.0
    assert checkpoint["runtime_teacher_rows"] == 0


def test_contract_is_self_hashed_and_zero_oracle():
    envelope = json.loads(
        (ROOT / "configs/t4_complete_program_decoder_v1.json").read_text()
    )
    assert envelope["contract_sha256"] == identity(envelope["payload"])
    assert envelope["payload"]["oracle"]["calls_authorized"] == 0
    assert envelope["payload"]["support"]["maximum_primitives"] == 32
