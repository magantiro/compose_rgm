"""Compatibility shim: the realizer now lives in ``compose_v4.control.pmo_realization``.

It was promoted out of ``scripts/`` when it became the production PMO binder, because a
module that decides which programs a scored run proposes has to be importable, testable
and pinned in the contract's ``implementation_sha256``.  The measurement drivers
(``pmo_realization_repair_report_v1.py``, ``pmo_realization_witness_gate_v1.py``) and
``tests/test_pmo_realization_repair.py`` import from here, so this re-exports the module
surface RATHER THAN duplicating it: two copies of a search whose output seals a contract
is exactly the split-brain this repository has paid for before.
"""

from __future__ import annotations

from compose_v4.control.pmo_realization import (  # noqa: F401
    BINDING_ROW_FIELDS,
    DATAFLOW_OPERAND_FIELDS,
    OUTCOME_COMPLETED,
    OUTCOME_EXHAUSTED,
    OUTCOME_INCOMPATIBLE,
    OUTCOME_INVALID_PLAN,
    OUTCOMES,
    PRODUCTION_MAX_REALIZATIONS,
    PRODUCTION_NODE_BUDGET,
    PRODUCTION_SECONDS_CAP,
    PRODUCTION_SPECIFICATION,
    RELAXABLE_COMPONENTS,
    SCHEMA,
    SEMANTIC_OPERAND_FIELDS,
    SPEC_R1,
    SPEC_R2,
    SPEC_V1_EXACT,
    SPECIFICATIONS,
    RoleSpecification,
    _PlanDemand,
    _Prefix,
    bind_realized_plan,
    expand,
    finalize,
    operand_availability,
    plan_demand,
    plan_validity,
    propagate,
    realize,
    realize_plan_binding,
    retained_fraction,
    role_match,
    static_first_step_feasible,
    teacher_step_gaps,
)

__all__ = [
    "BINDING_ROW_FIELDS",
    "DATAFLOW_OPERAND_FIELDS",
    "OUTCOMES",
    "OUTCOME_COMPLETED",
    "OUTCOME_EXHAUSTED",
    "OUTCOME_INCOMPATIBLE",
    "OUTCOME_INVALID_PLAN",
    "PRODUCTION_MAX_REALIZATIONS",
    "PRODUCTION_NODE_BUDGET",
    "PRODUCTION_SECONDS_CAP",
    "PRODUCTION_SPECIFICATION",
    "RELAXABLE_COMPONENTS",
    "SCHEMA",
    "SEMANTIC_OPERAND_FIELDS",
    "SPECIFICATIONS",
    "SPEC_R1",
    "SPEC_R2",
    "SPEC_V1_EXACT",
    "RoleSpecification",
    "bind_realized_plan",
    "expand",
    "finalize",
    "operand_availability",
    "plan_demand",
    "plan_validity",
    "propagate",
    "realize",
    "realize_plan_binding",
    "retained_fraction",
    "role_match",
    "static_first_step_feasible",
    "teacher_step_gaps",
]
