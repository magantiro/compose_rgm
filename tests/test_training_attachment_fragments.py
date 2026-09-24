"""Focused checks for the train-only, opt-in attachment-fragment lane."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from rdkit import Chem, rdBase

from compose_v4.benchmark.fragment_conditioned_sampler import build_prompt_context
from compose_v4.benchmark.fragment_constrained import (
    FragmentTask,
    check_fragment_constraint,
    load_genmol_prompts,
)
from compose_v4.benchmark.training_attachment_fragments import (
    AttachmentCatalog,
    FragmentEntry,
    atom_context,
    build_attachment_catalog,
    catalog_bytes,
    sample_catalog_completion,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system


def _motif_context():
    prompts = load_genmol_prompts(
        "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    )
    prompt = next(
        prompt
        for prompt in prompts
        if prompt.drug_name == "BARICITINIB" and prompt.task == FragmentTask.MOTIF_EXTENSION
    )
    return build_prompt_context(prompt)


def _one_entry_catalog(context, rooted_smiles: str) -> AttachmentCatalog:
    ((site, _),) = context.attachment.requirements
    molecule = Chem.MolFromSmiles(context.start_smiles)
    key = atom_context(molecule.GetAtomWithIdx(site))
    return AttachmentCatalog(
        source_sha256="test-only",
        source_path="test-only",
        source_rows=1,
        unique_source_molecules=1,
        duplicate_source_rows=0,
        excluded_reference_rows=(),
        excluded={},
        entries=(FragmentEntry(key, rooted_smiles, 3, 1, (1,)),),
        rdkit_version=rdBase.rdkitVersion,
        max_fragment_atoms=6,
    )


def test_catalog_hash_split_and_round_trip(tmp_path):
    source = tmp_path / "train.smi"
    source.write_text("CCOC(=O)N\nCCOC(=O)N\ninvalid\nCCNCCO\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    catalog = build_attachment_catalog(source, expected_sha256=digest)
    assert catalog.source_rows == 4
    assert catalog.unique_source_molecules == 2
    assert catalog.duplicate_source_rows == 1
    assert catalog.excluded_reference_rows == ()
    assert catalog.excluded["unparseable_source_row"] == 1
    assert catalog.entries
    assert all(entry.source_rows and 2 not in entry.source_rows for entry in catalog.entries)
    assert catalog_bytes(catalog) == catalog_bytes(catalog)
    restored = AttachmentCatalog.from_dict(json.loads(catalog_bytes(catalog)))
    assert restored == catalog
    assert isinstance(restored.entries[0].source_rows, tuple)
    with pytest.raises(ValueError, match="SHA-256"):
        build_attachment_catalog(source, expected_sha256="0" * 64)


def test_reference_overlap_is_excluded_before_fragment_extraction(tmp_path):
    source = tmp_path / "train.smi"
    source.write_text("CCOC(=O)N\nCCNCCO\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    held_reference = Chem.MolToSmiles(Chem.MolFromSmiles("CCNCCO"), canonical=True)
    catalog = build_attachment_catalog(
        source,
        expected_sha256=digest,
        excluded_canonical=frozenset({held_reference}),
    )
    assert catalog.excluded_reference_rows == (2,)
    assert catalog.excluded["benchmark_reference_overlap"] == 1
    assert catalog.unique_source_molecules == 1
    assert all(2 not in entry.source_rows for entry in catalog.entries)


def test_observed_fragment_grafts_through_exact_lock():
    context = _motif_context()
    catalog = _one_entry_catalog(context, "*CCO")
    product, receipt = sample_catalog_completion(
        context, catalog, de_novo_rewrite_system(), np.random.default_rng(1)
    )
    assert product is not None
    assert check_fragment_constraint(context.prompt, product).satisfied
    assert len(receipt.accepted_actions) == 3
    assert receipt.fallback_count == 0
    assert receipt.failure_reason is None
    assert Chem.MolFromSmiles(product) is not None


def test_single_explicit_fallback_is_counted_not_hidden_retry():
    context = _motif_context()
    # An invalid library entry is used only to exercise the documented refusal
    # path; the production catalog builder never emits such an entry.
    catalog = _one_entry_catalog(context, "not-a-fragment")
    product, receipt = sample_catalog_completion(
        context, catalog, de_novo_rewrite_system(), np.random.default_rng(2)
    )
    assert product is not None
    assert check_fragment_constraint(context.prompt, product).satisfied
    assert receipt.fallback_count == 1
    assert len(receipt.attempted_fragments) == 2
    assert receipt.attempted_fragments[0]["refusal"]
    assert receipt.attempted_fragments[1]["minimal_fallback"] is True
    assert len(receipt.accepted_actions) == 1


def test_unsupported_task_is_an_explicit_abstention():
    prompts = load_genmol_prompts(
        "data/benchmarks/fragment_constrained/genmol_safe_drugs_fragments.csv"
    )
    prompt = next(
        prompt for prompt in prompts if prompt.task == FragmentTask.SUPERSTRUCTURE_GENERATION
    )
    context = build_prompt_context(prompt)
    with pytest.raises(ValueError, match="does not support"):
        sample_catalog_completion(
            context,
            _one_entry_catalog(_motif_context(), "*CCO"),
            de_novo_rewrite_system(),
            np.random.default_rng(3),
        )
