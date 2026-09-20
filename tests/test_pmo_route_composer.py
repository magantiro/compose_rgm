"""Tests for the PMO route-distilled program composer preparation modules.

These cover the parts that carry a claim: the scale/mode vocabulary, the
leave-one-task-out fit (including that the held task cannot leak into the
vocabulary), the matched uniform control, the checkpoint round trip, and the
bounded-vocabulary restriction.  Generation itself is exercised by a small
end-to-end smoke on a real corpus state.

``scripts`` is on the path for these modules, matching the repo convention for
experiment drivers (``PYTHONPATH=src:scripts``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from pmo_route_composer import (
    PmoRouteComposer,
    fit_pmo_route_composer,
    fit_program_shape_law,
    restrict_vocabulary,
    teacher_template_coverage,
    uniform_composer_like,
)
from pmo_route_program_corpus import (
    MODES,
    SCALE_BANDS,
    ProgramRow,
    StructuralDeltaTemplate,
    program_mode,
    program_scale,
    template_shape,
)


def _template(*, outputs: int = 0, released: bool = False, restated: bool = False):
    """One minimal well-formed address-free patch."""
    before = (6, 0, 1, 3)
    after = None if released else ((7, 0, 1, 3) if restated else (6, 0, 0, 4))
    n_target = 1 + outputs
    bonds = [[0] * n_target for _ in range(n_target)]
    for index in range(outputs):
        bonds[0][1 + index] = 1
        bonds[1 + index][0] = 1
    return StructuralDeltaTemplate(
        input_atoms=(before,),
        input_bonds=((0,),),
        target_atoms=(after,),
        output_atoms=tuple((6, 0, 0, 3) for _ in range(outputs)),
        target_bonds=tuple(tuple(row) for row in bonds),
    )


def _row(task: str, templates, *, lineage: str = "L", index: int = 0) -> ProgramRow:
    shapes = [template_shape(row) for row in templates]
    created = sum(row["created"] for row in shapes)
    released = sum(row["released"] for row in shapes)
    from compose_v4.control.structural_subgoal_policy import structural_rewrite_event_count

    events = sum(structural_rewrite_event_count(row) for row in templates)
    return ProgramRow(
        task=task,
        route_id=f"{task}-route",
        source_group="S",
        lineage=lineage,
        task_family="family",
        window_index=index,
        window_primitives=4,
        templates=tuple(templates),
        region_count=len(templates),
        rewrite_events=events,
        scale=program_scale(events),
        mode=program_mode(created, released),
        created_atoms=created,
        released_atoms=released,
        restated_atoms=sum(row["restated"] for row in shapes),
        artifact="test",
        success_basis="test",
        endpoint_key=f"{task}-endpoint-{index}",
    )


def _corpus() -> list[ProgramRow]:
    grow = _template(outputs=2)
    prune = _template(released=True)
    replace = _template(restated=True)
    rows = []
    for task in ("alpha", "beta", "gamma"):
        rows.append(_row(task, [grow], index=0))
        rows.append(_row(task, [replace], index=1))
    # One template that exists only in the held task, so coverage cannot be 1.
    rows.append(_row("alpha", [prune], index=2))
    return rows


def test_scale_and_mode_vocabulary_is_total():
    """Every program lands in exactly one declared scale and one mode."""
    assert program_scale(1) == "small"
    assert program_scale(3) == "small"
    assert program_scale(4) == "medium"
    assert program_scale(11) == "medium"
    assert program_scale(12) == "large"
    assert set(SCALE_BANDS) == {"small", "medium", "large"}
    assert set(MODES) == {"grow", "prune", "replace", "remodel"}
    assert program_mode(2, 0) == "grow"
    assert program_mode(0, 2) == "prune"
    assert program_mode(0, 0) == "replace"
    assert program_mode(1, 1) == "remodel"


def test_shape_law_is_a_normalized_distribution_with_a_floor():
    law = fit_program_shape_law(_corpus(), exploration_floor=0.1)
    cells = law.cells()
    assert len(cells) == len(SCALE_BANDS) * len(MODES)
    assert abs(sum(cells.values()) - 1.0) < 1e-9
    # The floor guarantees an unobserved shape stays reachable.
    assert all(value > 0 for value in cells.values())
    assert law.probability("small", "grow") > 0


def test_held_task_cannot_enter_the_fitted_vocabulary():
    """A template seen only in the held task must be absent by construction."""
    rows = _corpus()
    held_only = {
        name
        for row in rows
        if row.task == "alpha"
        for name in row.template_ids
    } - {name for row in rows if row.task != "alpha" for name in row.template_ids}
    assert held_only, "fixture must contain an alpha-only template"
    composer, audit = fit_pmo_route_composer(rows, held_task="alpha")
    assert audit["held_task_absent_from_training"] is True
    assert audit["excluded_held_task_programs"] == 3
    assert not held_only & set(composer.marginal.template_ids)


def test_expressibility_reports_the_uncovered_fraction():
    coverage = teacher_template_coverage(_corpus(), "alpha")
    assert coverage["held_programs"] == 3
    assert 0.0 < coverage["distinct_template_coverage"] < 1.0
    assert coverage["fully_expressible_rate"] < 1.0


def test_uniform_control_shares_the_vocabulary_and_flattens_the_law():
    composer, _ = fit_pmo_route_composer(_corpus(), held_task="alpha")
    uniform = uniform_composer_like(composer)
    assert uniform.marginal.template_ids == composer.marginal.template_ids
    assert len(set(uniform.marginal.probabilities)) == 1
    assert uniform.marginal.goal_count_probabilities == (0.25, 0.25, 0.25, 0.25)
    assert len(set(uniform.shape.probabilities)) == 1


def test_joint_score_adds_the_shape_term():
    composer, _ = fit_pmo_route_composer(_corpus(), held_task="alpha", shape_weight=1.0)
    templates = (composer.templates[0],)
    scale, mode = composer.program_shape(templates)
    expected = composer.score(templates) + composer.shape.log_probability(scale, mode)
    assert composer.score_proposal(templates) == pytest.approx(expected)
    # A zero shape weight must recover the pure template law exactly.
    flat = PmoRouteComposer(
        composer.marginal, composer.shape, composer.templates, composer.held_task, 0.0,
        composer.training_identity,
    )
    assert flat.score_proposal(templates) == pytest.approx(flat.score(templates))


def test_checkpoint_round_trip_preserves_every_score():
    composer, _ = fit_pmo_route_composer(_corpus(), held_task="alpha")
    restored = PmoRouteComposer.from_checkpoint(composer.checkpoint())
    assert restored.marginal.template_ids == composer.marginal.template_ids
    assert restored.held_task == composer.held_task
    for template in composer.templates:
        assert restored.score_proposal((template,)) == pytest.approx(
            composer.score_proposal((template,))
        )


def test_vocabulary_restriction_renormalizes_and_reports_lost_mass():
    composer, _ = fit_pmo_route_composer(_corpus(), held_task="alpha")
    limit = max(1, len(composer.templates) - 1)
    bounded, audit = restrict_vocabulary(composer, limit)
    assert audit["restricted"] is True
    assert audit["vocabulary"] == limit
    assert 0 < audit["retained_mass"] < 1.0
    assert abs(sum(bounded.marginal.probabilities) - 1.0) < 1e-9
    assert len(bounded.templates) == limit
    # An unrestricted call must be a no-op that says so.
    same, audit_all = restrict_vocabulary(composer, len(composer.templates))
    assert audit_all["restricted"] is False
    assert same is composer


def test_fitting_without_a_held_task_uses_every_row():
    rows = _corpus()
    composer, audit = fit_pmo_route_composer(rows)
    assert audit["training_programs"] == len(rows)
    assert audit["held_task"] is None
    assert composer.held_task is None


# ---- End-to-end smoke on real corpus routes and a charged drug-like source ----

_REGION_CORPUS = (
    Path(__file__).resolve().parents[1]
    / "diagnostics/pmo_dependency_region_program_v2/attempt_1"
    / "training_dependency_region_corpus.json.gz"
)
# A real drug-like scaffold carrying two formal charges on a net-neutral nitro
# group.  Every source in the PMO route corpus is neutral, so a charged source
# has to be supplied here or the charge path is never exercised at all.
_CHARGED_DRUG_LIKE = "O=[N+]([O-])c1ccc(C(=O)NCc2ccccc2)cc1"


def _real_program_rows(route_limit: int = 12, window: int = 4) -> list[ProgramRow]:
    import gzip
    import json

    from pmo_route_program_corpus import PmoRoute, route_programs

    with gzip.open(_REGION_CORPUS, "rt") as handle:
        routes = json.loads(handle.read())["payload"]["routes"][:route_limit]
    rows: list[ProgramRow] = []
    for index, route in enumerate(routes):
        member = min(route["members"], key=lambda row: (row["task"], row["member_id"]))
        record = PmoRoute(
            task=member["task"],
            route_id=f"route_{index}",
            source_group="s",
            source_smiles=None,
            states=tuple(route["states"]),
            actions=tuple(route["actions"]),
            success_basis="test",
            artifact="test",
            lineage=route["lineage_identity"],
            task_family=route.get("task_family", ""),
        )
        rows.extend(route_programs(record, window=window)[0])
    return rows


@pytest.mark.skipif(not _REGION_CORPUS.exists(), reason="PMO region corpus artifact absent")
def test_real_routes_decompose_and_every_fitted_score_is_finite():
    """Smoke: real PMO routes -> programs -> fitted composer with finite scores."""
    import math

    rows = _real_program_rows()
    assert rows, "real PMO routes must yield at least one complete program"
    assert {row.scale for row in rows} <= set(SCALE_BANDS)
    assert {row.mode for row in rows} <= set(MODES)
    composer, audit = fit_pmo_route_composer(rows)
    assert audit["training_programs"] == len(rows)
    for row in rows:
        score = composer.score_proposal(row.templates)
        assert math.isfinite(score), f"nonfinite score for a real teacher program: {score}"


@pytest.mark.skipif(not _REGION_CORPUS.exists(), reason="PMO region corpus artifact absent")
def test_generation_smoke_on_a_charged_drug_like_source():
    """The composer must rebind and generate on a charged, drug-like molecule.

    ``production_state_from_smiles`` pads the graph, which is required: a tight
    graph silently removes the whole ``atom_insert`` family from legal support,
    so any coverage claim on an unpadded state is an artifact.
    """
    import math

    import numpy as np

    from compose_v4.chem.molecular_graph import is_element
    from compose_v4.experiments.editing_v2_evaluation_semantics import (
        production_state_from_smiles,
    )
    from compose_v4.rewrite.kernel import canonical_state_key

    source = production_state_from_smiles(_CHARGED_DRUG_LIKE)
    charges = np.asarray(source.formal_charges)[is_element(source.atom_types)]
    assert int(np.abs(charges).sum()) == 2, "the smoke source must actually carry charge"
    assert source.n_atoms > source.n_real_atoms, "source must be padded, not tight"

    rows = _real_program_rows()
    composer, _ = fit_pmo_route_composer(rows)
    bounded, restriction = restrict_vocabulary(composer, 64)
    proposals, telemetry = bounded.propose(
        source,
        pool_size=32,
        beam_width=6,
        expansion_width=6,
        max_bindings_per_template=2,
        scale_balanced=True,
    )
    assert telemetry["binding_visits"] > 0, "no template was even rebound to the charged source"
    for proposal in proposals:
        assert math.isfinite(bounded.score_proposal(proposal.templates))
        # Every generated endpoint must be a real canonicalizable molecule.
        assert canonical_state_key(proposal.endpoint)
    assert restriction["vocabulary"] <= 64


@pytest.mark.skipif(not _REGION_CORPUS.exists(), reason="PMO region corpus artifact absent")
def test_uniform_and_distilled_arms_share_the_pool_on_a_real_source():
    """The matched control must differ only in the frequency law."""
    rows = _real_program_rows()
    composer, _ = fit_pmo_route_composer(rows)
    bounded, _ = restrict_vocabulary(composer, 32)
    uniform = uniform_composer_like(bounded)
    assert set(uniform.marginal.template_ids) == set(bounded.marginal.template_ids)
    assert {row.template_id for row in uniform.templates} == {
        row.template_id for row in bounded.templates
    }
