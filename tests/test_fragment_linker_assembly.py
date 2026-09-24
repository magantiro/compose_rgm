"""Boundary/path semantics and exact shared-compiler assembly, without a model."""

from dataclasses import replace

import pytest

from compose_v4.benchmark.fragment_constrained import FragmentPrompt, FragmentTask
from compose_v4.benchmark.fragment_constrained_runner import ProposalLimits
from compose_v4.benchmark.fragment_linker_assembly import assemble_linker_program, linker_fidelity
from compose_v4.chem.state import is_connected_or_null, is_valid_state
from compose_v4.rewrite.trace_shard import decode_state


def prompt():
    return FragmentPrompt(
        "fixture", "CC", FragmentTask.LINKER_DESIGN, ("[1*]c1ccccc1", "[2*]N1CCCCC1")
    )


@pytest.mark.parametrize("length", [2, 3, 4])
def test_exact_path_length_without_seeded_bridge(length):
    candidate = assemble_linker_program(prompt(), "[1*]" + "C" * length + "[2*]")
    assert candidate.provenance["fidelity"]["linker_internal_atoms"] == length
    assert candidate.provenance["initial_bridge_atoms"] == 0
    assert candidate.provenance["initial_state_contains_only_one_core"]
    assert candidate.provenance["source_core_locked_all_states"]
    assert len(candidate.provenance["program_block_lengths"]) == 2
    assert sum(candidate.provenance["program_block_lengths"]) == len(candidate.trace["actions"])
    states = [decode_state(s) for s in candidate.trace["states"]]
    assert all(is_valid_state(s) and is_connected_or_null(s) for s in states)
    assert states[0].n_real_atoms == 6
    assert candidate.endpoint.n_real_atoms == 12 + length
    assert linker_fidelity(prompt(), candidate.smiles)["satisfied"]


@pytest.mark.parametrize(
    "connector",
    [
        "[1*]c1ccc([2*])cc1",
        "[1*]C1CC([2*])CC1",
        "[1*]CC(C)C[2*]",
    ],
)
def test_ring_and_branched_content_use_same_compiler(connector):
    candidate = assemble_linker_program(prompt(), connector)
    assert candidate.provenance["fidelity"]["satisfied"]
    assert len(candidate.trace["actions"]) > 1
    assert candidate.provenance["dependencies"]


def test_morphing_reference_and_drug_identity_do_not_change_program():
    original = prompt()
    candidate = assemble_linker_program(original, "[1*]CCC[2*]")
    changed = assemble_linker_program(
        replace(
            original,
            drug_name="different",
            original_smiles="O",
            task=FragmentTask.SCAFFOLD_MORPHING,
        ),
        "[1*]CCC[2*]",
    )
    assert candidate.trace == changed.trace
    assert candidate.smiles == changed.smiles


@pytest.mark.parametrize(
    "smiles,reason",
    [
        ("c1ccc(N2CCCCC2)cc1", "direct_core_shortcut"),
        ("c1ccccc1.N1CCCCC1", "disconnected"),
        ("c1ccccc1", "missing_core"),
        ("", "unparseable"),
    ],
)
def test_linker_fidelity_rejects_semantic_failures(smiles, reason):
    result = linker_fidelity(prompt(), smiles)
    assert not result["satisfied"]
    assert result["reason"] == reason


def test_fidelity_requires_designated_boundary_and_no_extra_core_contacts():
    p = replace(prompt(), fragments=("[1*]OC", "[2*]c1ccccc1"))
    assert linker_fidelity(p, "COCCc1ccccc1")["satisfied"]
    assert not linker_fidelity(p, "CCOc1ccccc1")["satisfied"]


@pytest.mark.parametrize("bad", ["CC", "[1*]CC[1*]", "[1*]CC.[2*]C", "[1*]=CC[2*]"])
def test_malformed_connector_rejected(bad):
    with pytest.raises(ValueError, match="connector"):
        assemble_linker_program(prompt(), bad)


def test_budget_is_not_silently_expanded():
    with pytest.raises(ValueError, match="program_budget"):
        assemble_linker_program(prompt(), "[1*]CC[2*]", limits=ProposalLimits(max_primitives=1))
    with pytest.raises(ValueError, match="may not increase"):
        assemble_linker_program(prompt(), "[1*]CC[2*]", limits=ProposalLimits(max_primitives=40))


def test_supplied_attachment_order_is_not_silently_changed():
    changed = replace(prompt(), fragments=("[1*]=CC", "[2*]N1CCCCC1"))
    with pytest.raises(ValueError, match="supplied single bonds"):
        assemble_linker_program(changed, "[1*]CC[2*]")


def test_invalid_supplied_core_fails_with_a_boundary_error():
    with pytest.raises(ValueError, match="invalid fragment SMILES"):
        replace(prompt(), fragments=("not-a-molecule", "[2*]N1CCCCC1"))


def test_overlapping_core_matches_are_not_linker_support():
    changed = replace(prompt(), fragments=("[1*]CC", "[2*]CC"))
    result = linker_fidelity(changed, "CC")
    assert not result["satisfied"]
    assert result["reason"] == "overlapping_cores"
