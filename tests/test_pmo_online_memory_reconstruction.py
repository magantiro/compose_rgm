"""Arm B must not resume on a COLD memory when its snapshot predates persistence.

The 250-call A/B ran entirely inside one container, so its online memory lived in
process and was never serialised.  Resuming that snapshot at call 251 with an empty
memory would quietly turn a 1000-call memory experiment into a 250-call one followed
by a reset -- a change to what the experiment measures, invisible in the artifact.

Every quantity the memory holds is a sum or a bounded max over counted observations,
and those observations are durable in the archive, so the learned state is
reconstructible by replaying them through the production observation path.
"""

from __future__ import annotations

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.pmo_online_memory import OnlineProposalMemory
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.rewrite.trace_shard import encode_state

_PAIRS = [
    ("CCOc1ccc(cc1)C(=O)N", "CCOc1ccc(cc1)C(=O)NC", 0.31),
    ("c1ccc(cc1)S(=O)(=O)N", "c1ccc(cc1)S(=O)(=O)NC", 0.42),
    ("CC(=O)Nc1ccc(O)cc1", "CC(=O)Nc1ccc(OC)cc1", 0.27),
]


def _state(smiles):
    return encode_state(pad_molecular_graph(smiles_to_molecular_graph(smiles), 48))


def _archive():
    """Archive rows shaped as the production snapshot stores them."""
    entries, observations = {}, {}
    for index, (parent, child, score) in enumerate(_PAIRS):
        entries[f"entry_{index}"] = {
            "endpoint": child,
            "source_state": _state(parent),
            "trace": {
                "endpoint": child,
                "states": [_state(parent), _state(child)],
                "actions": [{"model_family": "atom_insert"}],
            },
            "provenance": {
                "parent_measured_score": score - 0.05,
                "actual_changes": {"changed_original_slots": [index]},
            },
        }
        # Observations are keyed by RECEIPT id, not entry id, so the replay must join
        # them on the endpoint. A join on the key would silently reconstruct nothing.
        observations[f"receipt_{index}"] = {"endpoint": child, "score": score}
    return entries, observations


class _Archive:
    """The two archive fields the replay reads, plus the REAL observation path."""

    def __init__(self, entries, observations, memory):
        self.entries, self.observations, self.online_memory = entries, observations, memory
        self.online_memory_attribution_failures = 0

    _observe_into_online_memory = PmoPopulationController._observe_into_online_memory
    _reconstruct_online_memory = PmoPopulationController._reconstruct_online_memory


def test_reconstruction_warms_the_memory_from_archived_observations():
    entries, observations = _archive()
    memory = OnlineProposalMemory()
    assert memory.ordinal == 0 and not memory.edits.rows
    report = _Archive(entries, observations, memory)._reconstruct_online_memory()
    assert report["replayed"] == len(_PAIRS)
    assert report["attribution_failures"] == 0
    assert report["skipped_unscored"] == 0
    assert report["byte_equality_with_live_memory_claimed"] is False
    # The memory is warm on every axis the proposal law reads, not merely non-empty.
    assert memory.ordinal == len(_PAIRS)
    assert len(memory.frontier.scores) == len(_PAIRS)
    assert memory.edits.rows and memory.edits.counts
    assert memory.size.counts


def test_reconstruction_survives_a_round_trip_through_the_persisted_payload():
    entries, observations = _archive()
    memory = OnlineProposalMemory()
    _Archive(entries, observations, memory)._reconstruct_online_memory()
    payload = memory.payload()
    restored = OnlineProposalMemory()
    restored.restore_payload(payload)
    assert restored.payload() == payload


def test_an_archive_that_reconstructs_nothing_is_refused_not_resumed_cold():
    # Scored entries that replay to nothing means the join or the entry shape moved.
    # Resuming there is exactly the silent arm change this path exists to prevent.
    entries, _ = _archive()
    unjoinable = {f"receipt_{i}": {"endpoint": "CCO", "score": 0.1} for i in range(3)}
    with pytest.raises(ValueError, match="refusing to resume arm B on an empty memory"):
        _Archive(entries, unjoinable, OnlineProposalMemory())._reconstruct_online_memory()


def test_unscored_entries_are_skipped_rather_than_charged_a_zero():
    # A missing score is absent evidence, not evidence of a zero: recording 0.0 would
    # teach the memory that every unscored edit was maximally bad.
    entries, observations = _archive()
    entries["entry_unscored"] = dict(entries["entry_0"], endpoint="CCCCCCO")
    report = _Archive(entries, observations, OnlineProposalMemory())._reconstruct_online_memory()
    assert (report["replayed"], report["skipped_unscored"]) == (len(_PAIRS), 1)
