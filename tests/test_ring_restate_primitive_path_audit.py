from __future__ import annotations

from compose_v4.chem.molecular_graph import (
    smiles_to_molecular_graph,
)
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.experiments.ring_restate_primitive_path_audit import (
    audit_ring_restate_teacher,
)
from compose_v4.rewrite.kernel import de_novo_rewrite_system
from compose_v4.rewrite.progress import TraceProgressCTMC
from compose_v4.rewrite.tracelet_compiler import (
    compile_null_to_target_tracelets,
)
from compose_v4.rewrite.tracelet_fiber import (
    enumerate_ring_system_restate_actions,
)


def _aromatizing_teacher(smiles: str):
    target = pad_molecular_graph(smiles_to_molecular_graph(smiles), 16)
    path = TraceProgressCTMC(compile_null_to_target_tracelets(target))
    for progress, step in enumerate(path.trace.steps):
        if step.rule_name == "ring_system_restate":
            return (
                path.state_at(progress),
                path.state_at(progress + 1),
                step.action,
            )
    raise AssertionError(f"{smiles!r} compiled without a ring restatement")


def test_aromatizing_carbocycle_lowering_is_exact_valid_and_budget_bounded() -> None:
    source, successor, action = _aromatizing_teacher("c1ccccc1")
    result = audit_ring_restate_teacher(source, successor, action)

    assert result["failures"] == ()
    assert result["lowering_length"] == 3
    assert result["primitive_family_sequence"] == ("bond_reorder",) * 3
    assert result["lowering_completed"] is True
    assert result["all_primitive_intermediates_valid"] is True
    assert result["all_primitive_intermediates_connected"] is True
    assert result["all_primitive_intermediates_charge_policy_preserving"] is True
    assert result["all_lowering_steps_in_declared_primitive7"] is False
    assert result["declared_primitive7_step_count"] == 0
    assert result["declared_primitive7_failure_reasons"] == {
        "current_primitive7_excludes_cyclic_bond_reorder": 3
    }
    assert result["strata"]["direction"] == "aromatizing"
    assert result["strata"]["heterocycle"] is False
    assert result["strata"]["source_system_saturated"] is True
    assert result["strata"]["target_system_saturated"] is False


def test_aromatizing_heterocycle_is_stratified() -> None:
    source, successor, action = _aromatizing_teacher("c1ccncc1")
    result = audit_ring_restate_teacher(source, successor, action)

    assert result["failures"] == ()
    assert result["strata"]["direction"] == "aromatizing"
    assert result["strata"]["heterocycle"] is True
    assert result["all_primitive_intermediates_charge_policy_preserving"] is True


def test_dearomatizing_teacher_has_exact_valid_inverse_lowering() -> None:
    aromatic = pad_molecular_graph(smiles_to_molecular_graph("c1ccccc1"), 16)
    runtime = de_novo_rewrite_system()
    actions = enumerate_ring_system_restate_actions(aromatic, system=runtime)
    action = next(
        candidate
        for candidate in actions
        if all(int(change.new_order) == 1 for change in candidate.changes)
    )
    saturated = runtime.apply(aromatic, "ring_system_restate", action)
    result = audit_ring_restate_teacher(aromatic, saturated, action)

    assert result["failures"] == ()
    assert result["strata"]["direction"] == "dearomatizing"
    assert result["strata"]["source_system_saturated"] is False
    assert result["strata"]["target_system_saturated"] is True
    assert result["all_primitive_intermediates_valid"] is True
    assert result["all_primitive_intermediates_connected"] is True
    assert result["all_primitive_intermediates_charge_policy_preserving"] is True
    assert result["all_lowering_steps_in_declared_primitive7"] is False
