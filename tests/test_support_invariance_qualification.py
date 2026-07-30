"""Bounded qualification of score-free production marked support."""

from __future__ import annotations

from dataclasses import replace

import pytest

from compose_v4.experiments import support_invariance_qualification as qualification


@pytest.fixture(scope="module")
def qualified_fixture():
    models, panel = qualification.build_default_ringcore_support_fixture()
    report = qualification.qualify_stratified_support_invariance(
        models,
        panel,
    )
    return models, panel, report


def test_stratified_real_support_is_invariant_but_scores_change(
    qualified_fixture,
) -> None:
    models, panel, report = qualified_fixture

    assert len(models) == 2
    assert len({digest for _, digest in report.model_parameter_sha256s}) == 2
    assert report.scoring_times == (0.13, 0.79)
    assert report.support_invariant
    assert report.probability_variation_observed
    assert report.maximum_probability_l1 > 1e-8
    assert report.covered_families == tuple(
        sorted(qualification.ACTIVE_RINGCORE_FAMILIES)
    )
    assert {state.name for state in report.states} == {
        item.name for item in panel
    }
    assert all(
        len(state.support_signature_sha256) == 64
        and state.raw_mark_count > 0
        for state in report.states
    )

    payload = report.to_payload()
    assert payload["status"] == "pass"
    assert payload["corpus_wide_proof"] is False
    assert "mark_probability" in payload["support_signature_excludes"]
    assert any(
        "not a corpus-wide support proof" in limitation
        for limitation in payload["limitations"]
    )


def test_qualification_fails_closed_on_candidate_identity_drift(
    qualified_fixture,
    monkeypatch,
) -> None:
    models, panel, _report = qualified_fixture
    original_compile = qualification.compile_state_successor_map
    drift_model = models[1].model

    def compile_with_one_missing_group(model, state, *, time, system):
        compiled = original_compile(
            model,
            state,
            time=time,
            system=system,
        )
        if model is drift_model and time == 0.79:
            assert len(compiled.successor_groups) > 1
            return replace(
                compiled,
                successor_groups=compiled.successor_groups[1:],
            )
        return compiled

    monkeypatch.setattr(
        qualification,
        "compile_state_successor_map",
        compile_with_one_missing_group,
    )
    with pytest.raises(
        qualification.SupportInvarianceQualificationError,
        match="compiled candidate support disagree",
    ):
        qualification.qualify_stratified_support_invariance(
            models,
            (panel[0],),
            scoring_times=(0.13, 0.79),
            required_families=("atom_insert",),
        )


def test_qualification_rejects_nonindependent_models(
    qualified_fixture,
) -> None:
    models, panel, _report = qualified_fixture
    with pytest.raises(
        qualification.SupportInvarianceQualificationError,
        match="distinct model objects",
    ):
        qualification.qualify_stratified_support_invariance(
            (models[0], models[0]),
            panel,
        )


def test_qualification_rejects_duplicate_scored_candidate(
    qualified_fixture,
    monkeypatch,
) -> None:
    models, panel, _report = qualified_fixture
    original_enumerate = qualification.enumerate_factorized_marked_law

    def enumerate_with_duplicate(*args, **kwargs):
        law = original_enumerate(*args, **kwargs)
        assert law.marks
        return replace(law, marks=(law.marks[0], *law.marks))

    monkeypatch.setattr(
        qualification,
        "enumerate_factorized_marked_law",
        enumerate_with_duplicate,
    )
    with pytest.raises(
        qualification.SupportInvarianceQualificationError,
        match="repeats a candidate identity",
    ):
        qualification.qualify_stratified_support_invariance(
            models,
            (panel[0],),
            required_families=("atom_insert",),
        )
