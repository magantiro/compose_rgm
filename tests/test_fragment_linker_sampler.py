"""Small exact-chemistry and sampling-law tests; no checkpoint or quality oracle."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

from compose_v4.benchmark import fragment_linker_sampler as sampler
from compose_v4.benchmark.fragment_constrained import FragmentPrompt, FragmentTask
from compose_v4.benchmark.fragment_constrained_runner import ProposalLimits
from compose_v4.benchmark.joint_completion_prior import JointCompletionPrior
from compose_v4.rewrite.trace_shard import decode_state


def prompt():
    return FragmentPrompt("fixture", "O", FragmentTask.LINKER_DESIGN, ("[1*]CC", "[2*]N"))


def entry(smiles, occurrences=1, contexts=("C:0:0", "N:0:0")):
    molecule = Chem.MolFromSmiles(smiles)
    return {
        "rooted_smiles": smiles,
        "contexts": list(contexts),
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "ring_count": molecule.GetRingInfo().NumRings(),
        "occurrences": occurrences,
        "source_rows": [1] * occurrences,
        "source_stereo_annotations": 0,
    }


def catalog(*entries):
    return {
        "schema": "split_first_training_region_catalog_v1",
        "split": {
            "algorithm": "murcko+carbonized-wl3",
            "version": 2,
            "salt": "ringcore-v1",
            "ratios": [0.9, 0.05, 0.05],
            "partition": "train",
        },
        "accepted_source_rows": [1],
        "training_molecules": 1,
        "entries": list(entries),
    }


def runtime(*entries):
    return sampler.prepare_linker_catalog(catalog(*entries), catalog_sha256="fixture")


class FixedRng:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.probabilities = []

    def choice(self, size, *, p):
        self.probabilities.append(p)
        index = next(self.choices)
        assert 0 <= index < size
        return index

    def integers(self, size):
        assert size > 0
        return 0


def test_cell_mass_is_independent_of_content_multiplicity_and_occurrence():
    library = runtime(entry("[1*]CC[2*]"), entry("[1*]CO[2*]", 9), entry("[1*]CCC[2*]", 100))
    prior = JointCompletionPrior(((5, 0, 2), (6, 0, 2)))
    rng = FixedRng([0, 1])
    candidate = sampler.propose_linker_completion(prompt(), library, prior, rng)
    assert np.allclose(rng.probabilities[0], (0.5, 0.5))
    assert np.allclose(rng.probabilities[1], (0.25, 0.75))
    assert candidate.provenance["training_connector"]["planned_final_cell"] == [5, 0]
    assert candidate.provenance["training_connector"]["source_rows"] == [1] * 9
    assert candidate.provenance["exact_mapped_core_identity_checked"]
    assert decode_state(candidate.trace["states"][0]).n_real_atoms == 2


def test_reversed_context_binding_retains_exact_role_provenance():
    library = runtime(entry("[1*]CO[2*]", contexts=("N:0:0", "C:0:0")))
    candidate = sampler.propose_linker_completion(
        prompt(), library, JointCompletionPrior(((5, 0, 1),)), np.random.default_rng(0)
    )
    assert candidate.provenance["training_connector"]["reversed_boundary_roles"]
    assert candidate.smiles == "CCOCN"


def test_symmetric_contexts_do_not_duplicate_content_mass():
    library = runtime(entry("[1*]CO[2*]", contexts=("C:0:0", "C:0:0")))
    query = replace(prompt(), fragments=("[1*]CC", "[2*]C"))
    groups, receipt = sampler.compatible_linker_cells(query, library)
    assert receipt["context_compatible_entries"] == 1
    assert len(groups[(5, 0)]) == 1
    candidate = sampler.propose_linker_completion(
        query, library, JointCompletionPrior(((5, 0, 1),)), np.random.default_rng(0)
    )
    assert candidate.provenance["training_connector"]["orientation_probability"] == 0.5


@pytest.mark.parametrize("connector", ["[1*]c1ccc([2*])cc1", "[1*]C1CN([2*])CC1"])
def test_ring_connector_counts_and_exact_execution(connector):
    library = runtime(entry(connector))
    candidate = sampler.propose_linker_completion(
        prompt(), library, JointCompletionPrior(((9, 1, 1),)), np.random.default_rng(0)
    )
    assert candidate.provenance["final_cell"][1] == 1
    assert candidate.provenance["fidelity"]["satisfied"]
    assert len(candidate.trace["actions"]) > 1


def test_seed_and_identical_linker_morphing_inputs_give_identical_programs():
    library = runtime(entry("[1*]CC[2*]"), entry("[1*]CO[2*]"))
    prior = JointCompletionPrior(((5, 0, 1),))
    first = sampler.propose_linker_completion(prompt(), library, prior, np.random.default_rng(9))
    second = sampler.propose_linker_completion(
        replace(
            prompt(), task=FragmentTask.SCAFFOLD_MORPHING, drug_name="other", original_smiles="F"
        ),
        library,
        prior,
        np.random.default_rng(9),
    )
    assert first.trace == second.trace
    assert first.provenance == second.provenance


def test_capacity_and_compile_budget_abstentions_keep_proposal_receipts():
    library = runtime(entry("[1*]CC[2*]"))
    prior = JointCompletionPrior(((5, 0, 1),))
    with pytest.raises(sampler.LinkerProposalAbstention, match="no context") as failure:
        sampler.propose_linker_completion(
            prompt(),
            library,
            prior,
            np.random.default_rng(0),
            limits=ProposalLimits(max_active_atoms=4),
        )
    assert failure.value.receipt["atom_budget_excluded_entries"] == 1
    with pytest.raises(sampler.LinkerProposalAbstention, match="program_budget") as failure:
        sampler.propose_linker_completion(
            prompt(),
            library,
            prior,
            np.random.default_rng(0),
            limits=ProposalLimits(max_primitives=1),
        )
    assert failure.value.receipt["planned_final_cell"] == [5, 0]


@pytest.mark.parametrize("mutation", ["partition", "source_rows", "occurrences"])
def test_catalog_rejects_nontraining_or_malformed_rows(mutation):
    raw = catalog(entry("[1*]CC[2*]"))
    if mutation == "partition":
        raw["split"]["partition"] = "test"
    elif mutation == "source_rows":
        raw["entries"][0]["source_rows"] = [2]
    else:
        raw["entries"][0]["occurrences"] = 0
    with pytest.raises(ValueError):
        sampler.prepare_linker_catalog(raw, catalog_sha256="fixture")


def test_selected_connector_metadata_drift_fails_closed():
    raw = entry("[1*]CC[2*]")
    raw["ring_count"] = 1
    with pytest.raises(ValueError, match="metadata changed"):
        sampler.propose_linker_completion(
            prompt(), runtime(raw), JointCompletionPrior(((5, 1, 1),)), np.random.default_rng(0)
        )


@pytest.mark.parametrize("score", [0.0, -np.inf])
def test_panel_uses_shared_scorer_and_records_all_draws_without_uniform_fallback(
    monkeypatch, score
):
    calls = []

    def scorer(model, candidates):
        calls.append(candidates)
        return (score,)

    monkeypatch.setattr(sampler, "learned_program_scores", scorer)
    rng = np.random.default_rng(0)
    before = deepcopy(rng.bit_generator.state)
    result = sampler.sample_linker_panel(
        prompt(), runtime(entry("[1*]CC[2*]")), JointCompletionPrior(((5, 0, 1),)), None, rng
    )
    receipt = result.receipt
    assert receipt["offered_count"] == receipt["exact_compiled_count"] == 8
    assert receipt["unique_compiled_endpoints"] == len(calls) == 1
    assert sum(row["status"] == "duplicate_endpoint" for row in receipt["offered"]) == 7
    assert receipt["rng_state_before"] == before
    assert receipt["rng_state_after"] == rng.bit_generator.state
    assert (result.selected is not None) == np.isfinite(score)
    assert receipt["output_count"] == receipt["model_supported_count"] == int(np.isfinite(score))
    if np.isfinite(score):
        assert receipt["selection"]["model_used"]
        assert receipt["selection"]["selected_draw"] == 0
    else:
        assert receipt["selected_smiles"] is receipt["selection"] is None
        assert receipt["offered"][0]["reason"] == "nonfinite_native_probability"


def test_no_compatible_content_records_eight_abstentions_and_no_model_call(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("scorer must not run without an exact candidate")

    monkeypatch.setattr(sampler, "learned_program_scores", forbidden)
    result = sampler.sample_linker_panel(
        prompt(), runtime(), JointCompletionPrior(((5, 0, 1),)), None, np.random.default_rng(0)
    )
    assert result.selected is None
    assert result.receipt["offered_count"] == 8
    assert result.receipt["exact_compiled_count"] == result.receipt["model_supported_count"] == 0
