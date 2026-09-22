"""Measurement-only census of the 95 PMO jump-lane joint role plans.

ZERO oracle calls. Reads the frozen ``shared_all_routes`` plan-latent checkpoint and
reports, per role / per operand, exactly what the address-free descriptor
(``compose_v4.control.pmo_action_roles.atom_role``) pins: which fields are present,
how many distinct values each field takes, and how specific the joint tuple of fields
is (a proxy for how transferable / rebindable a plan's operand slots are).

Answer-known offline diagnostic. No src/ edits. No Modal. No commit.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import rdkit

from compose_v4.chem.molecular_graph import ELEMENTS
from compose_v4.control.docking_value import identity as docking_value_identity
from compose_v4.rewrite.action_codec_v4 import ACTIVE8_EXECUTOR_RULES

CHECKPOINT_PATH = (
    REPO_ROOT
    / "diagnostics"
    / "pmo_joint_dependency_jump_gate_v1"
    / "attempt_2"
    / "checkpoints.json"
)
EXPECTED_IDENTITY = "730bc905c128b9feb802262e1f98bdcb6479431b88347a05defb26c2b84fab35"
FROZEN_ACTIVE8_LITERAL = {
    "atom_delete",
    "atom_insert",
    "atom_restate_semantic",
    "bond_reorder",
    "bond_reroute",
    "cycle_close",
    "cycle_open",
    "ring_system_restate",
}
OUTPUT_PATH = (
    REPO_ROOT
    / "diagnostics"
    / "pmo_jump_rebinding_v1"
    / "plan_representation_census_v1.json"
)

DESCRIPTOR_FIELDS = (
    "origin",
    "atom_type",
    "formal_charge",
    "implicit_hydrogens",
    "degree",
    "bond_class_histogram",
    "neighbor_element_histogram",
    "created_ordinal",
    "creation_lag",
)


def _hist_to_tuple(value):
    if value is None:
        return None
    return tuple(int(v) for v in value)


def _decode_element(atom_type: int) -> str:
    if 0 <= atom_type < len(ELEMENTS):
        return ELEMENTS[atom_type]
    return f"UNKNOWN({atom_type})"


def _histogram_summary(counter_values: list[int]) -> dict:
    if not counter_values:
        return {"n": 0, "min": None, "median": None, "max": None, "histogram": {}}
    hist = Counter(counter_values)
    return {
        "n": len(counter_values),
        "min": min(counter_values),
        "median": median(counter_values),
        "max": max(counter_values),
        "histogram": {str(k): v for k, v in sorted(hist.items())},
    }


def _top_values(counter: Counter, n: int = 5) -> list[dict]:
    return [
        {"value": _jsonable(value), "count": count}
        for value, count in counter.most_common(n)
    ]


def _jsonable(value):
    """Render a (possibly tuple-of-tuples) key back into a JSON-safe structure."""

    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    rdkit_version = rdkit.__version__
    if rdkit_version != "2023.09.6":
        print(
            f"STOP: expected rdkit 2023.09.6 (PMO production kernel), got {rdkit_version}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"rdkit.__version__ = {rdkit_version}")

    envelope = json.loads(CHECKPOINT_PATH.read_text())
    payload = envelope["payload"]["checkpoints"]["shared_all_routes"]

    computed_identity = docking_value_identity(payload)
    if computed_identity != EXPECTED_IDENTITY:
        print(
            "STOP: checkpoint payload identity mismatch.\n"
            f"  expected: {EXPECTED_IDENTITY}\n"
            f"  computed: {computed_identity}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"checkpoint identity verified: {computed_identity}")

    plans = payload["plan_latents"]
    n_plans = len(plans)
    print(f"n plans = {n_plans}")

    # ---- codec surface programmatic check ----
    codec_surface = set(ACTIVE8_EXECUTOR_RULES)
    codec_surface_matches_literal = codec_surface == FROZEN_ACTIVE8_LITERAL

    # ================= 1. PLAN-LEVEL SHAPE =================
    primitive_counts = [int(p["primitive_count"]) for p in plans]
    component_counts = [int(p["component_count"]) for p in plans]
    created_dependency_counts = [int(p["created_dependency_count"]) for p in plans]
    support_counts = [int(p["support_count"]) for p in plans]

    total_roles = sum(len(p["roles"]) for p in plans)
    total_operands = sum(len(r["operands"]) for p in plans for r in p["roles"])

    plan_level_shape = {
        "n_plans": n_plans,
        "primitive_count": _histogram_summary(primitive_counts),
        "component_count": _histogram_summary(component_counts),
        "created_dependency_count": _histogram_summary(created_dependency_counts),
        "support_count": _histogram_summary(support_counts),
        "total_roles": total_roles,
        "total_operands": total_operands,
    }

    # ================= 2. EXECUTOR RULE CENSUS =================
    executor_rule_counter: Counter[str] = Counter()
    model_family_counter: Counter[str] = Counter()
    for p in plans:
        for r in p["roles"]:
            executor_rule_counter[r["executor_rule"]] += 1
            model_family_counter[r["model_family"]] += 1

    rules_present = set(executor_rule_counter)
    rules_outside_surface = sorted(rules_present - codec_surface)
    every_rule_inside_surface = len(rules_outside_surface) == 0

    executor_rule_census = {
        "by_executor_rule": dict(sorted(executor_rule_counter.items())),
        "by_model_family": dict(sorted(model_family_counter.items())),
        "frozen_active8_literal_from_task_prompt": sorted(FROZEN_ACTIVE8_LITERAL),
        "codec_active8_executor_rules_programmatic": sorted(codec_surface),
        "literal_matches_programmatic_surface": codec_surface_matches_literal,
        "codec_source": "compose_v4.rewrite.action_codec_v4.ACTIVE8_EXECUTOR_RULES",
        "rules_present_in_corpus": sorted(rules_present),
        "every_rule_inside_frozen_surface": every_rule_inside_surface,
        "rules_outside_surface": rules_outside_surface,
        "note_bond_insert": (
            "bond_insert is OUTSIDE the frozen Active8 surface per the task prompt; "
            f"observed count of bond_insert in this corpus = {executor_rule_counter.get('bond_insert', 0)}"
        ),
    }

    # ================= 3/4/5/6: walk every operand once =================
    operand_rows: list[dict] = []  # persisted rows

    field_presence: Counter[str] = Counter()
    field_value_counters: dict[str, Counter] = {f: Counter() for f in DESCRIPTOR_FIELDS}

    neighbor_total_counts: list[int] = []
    neighbor_distinct_element_counts: list[int] = []
    generic_carbon_only = 0
    specific_nonzero_noncarbon = 0

    origin_counter: Counter[str] = Counter()
    creation_lag_values: list[int] = []
    created_ordinal_by_plan: dict[str, set] = {}

    # step-order pinning
    plans_with_route_created_operand = 0
    plans_with_nonzero_creation_lag = 0
    roles_with_created_output_ordinal = 0
    plans_with_created_handle_dependencies = 0

    # specificity views
    v_full_counter: Counter[tuple] = Counter()
    v_drop_neb_counter: Counter[tuple] = Counter()
    v_drop_lag_counter: Counter[tuple] = Counter()
    v_role_counter: Counter[tuple] = Counter()
    v_semantic_counter: Counter[tuple] = Counter()

    per_plan_distinct_full: dict[str, set] = {}
    per_plan_distinct_role: dict[str, set] = {}

    # role 0
    role0_executor_rule_counter: Counter[str] = Counter()
    role0_operand_origin_counter: Counter[str] = Counter()
    role0_neighbor_hist_counter: Counter[tuple] = Counter()
    role0_field_value_counters: dict[str, Counter] = {f: Counter() for f in DESCRIPTOR_FIELDS}
    role0_total_operands = 0

    carbon_index = ELEMENTS.index("C")

    try_element_index = {name: i for i, name in enumerate(ELEMENTS)}
    assert try_element_index["C"] == carbon_index

    for p in plans:
        plan_id = p["plan_id"]
        route_created_seen = False
        nonzero_lag_seen = False
        handle_dep_seen = False
        per_plan_distinct_full[plan_id] = set()
        per_plan_distinct_role[plan_id] = set()
        created_ordinal_by_plan.setdefault(plan_id, set())

        for role_idx, r in enumerate(p["roles"]):
            if r.get("created_output_ordinal") is not None:
                roles_with_created_output_ordinal += 1
            if r.get("created_handle_dependencies"):
                handle_dep_seen = True

            for op_idx, op in enumerate(r["operands"]):
                desc = op["descriptor"]
                origin = desc.get("origin")
                atom_type = desc.get("atom_type")
                formal_charge = desc.get("formal_charge")
                implicit_h = desc.get("implicit_hydrogens")
                degree = desc.get("degree")
                bond_hist = _hist_to_tuple(desc.get("bond_class_histogram"))
                neb_hist = _hist_to_tuple(desc.get("neighbor_element_histogram"))
                created_ordinal = desc.get("created_ordinal")
                creation_lag = desc.get("creation_lag")

                origin_counter[origin] += 1
                if origin == "route_created":
                    route_created_seen = True
                    if created_ordinal is not None:
                        created_ordinal_by_plan[plan_id].add(created_ordinal)
                    if creation_lag is not None:
                        creation_lag_values.append(creation_lag)
                        if creation_lag != 0:
                            nonzero_lag_seen = True

                # field presence + distinct values
                for field, value in (
                    ("origin", origin),
                    ("atom_type", atom_type),
                    ("formal_charge", formal_charge),
                    ("implicit_hydrogens", implicit_h),
                    ("degree", degree),
                    ("bond_class_histogram", bond_hist),
                    ("neighbor_element_histogram", neb_hist),
                    ("created_ordinal", created_ordinal),
                    ("creation_lag", creation_lag),
                ):
                    if value is not None:
                        field_presence[field] += 1
                        key = value
                        if field == "atom_type":
                            key = _decode_element(value)
                        field_value_counters[field][key] += 1

                # neighbor histogram derived stats
                total_neighbors = sum(neb_hist) if neb_hist is not None else 0
                neighbor_total_counts.append(total_neighbors)
                distinct_elems = sum(1 for c in neb_hist if c > 0) if neb_hist else 0
                neighbor_distinct_element_counts.append(distinct_elems)
                if neb_hist is not None:
                    nonzero_idx = [i for i, c in enumerate(neb_hist) if c > 0]
                    if len(nonzero_idx) == 0:
                        # no neighbours at all (isolated / degree 0) -- neither generic nor specific
                        pass
                    elif nonzero_idx == [carbon_index]:
                        generic_carbon_only += 1
                    elif any(i != carbon_index for i in nonzero_idx):
                        specific_nonzero_noncarbon += 1

                # specificity tuples
                v_full = (
                    origin,
                    atom_type,
                    formal_charge,
                    implicit_h,
                    degree,
                    bond_hist,
                    neb_hist,
                    created_ordinal,
                    creation_lag,
                )
                v_drop_neb = (
                    origin,
                    atom_type,
                    formal_charge,
                    implicit_h,
                    degree,
                    bond_hist,
                    created_ordinal,
                    creation_lag,
                )
                v_drop_lag = (
                    origin,
                    atom_type,
                    formal_charge,
                    implicit_h,
                    degree,
                    bond_hist,
                    neb_hist,
                    created_ordinal,
                )
                v_role_ = (origin, atom_type, formal_charge, created_ordinal)
                v_semantic = (atom_type, formal_charge)

                v_full_counter[v_full] += 1
                v_drop_neb_counter[v_drop_neb] += 1
                v_drop_lag_counter[v_drop_lag] += 1
                v_role_counter[v_role_] += 1
                v_semantic_counter[v_semantic] += 1

                per_plan_distinct_full[plan_id].add(v_full)
                per_plan_distinct_role[plan_id].add(v_role_)

                if role_idx == 0:
                    role0_total_operands += 1
                    role0_operand_origin_counter[origin] += 1
                    if neb_hist is not None:
                        role0_neighbor_hist_counter[neb_hist] += 1
                    for field, value in (
                        ("origin", origin),
                        ("atom_type", atom_type),
                        ("formal_charge", formal_charge),
                        ("implicit_hydrogens", implicit_h),
                        ("degree", degree),
                        ("bond_class_histogram", bond_hist),
                        ("neighbor_element_histogram", neb_hist),
                        ("created_ordinal", created_ordinal),
                        ("creation_lag", creation_lag),
                    ):
                        if value is not None:
                            key = value
                            if field == "atom_type":
                                key = _decode_element(value)
                            role0_field_value_counters[field][key] += 1

                operand_rows.append(
                    {
                        "plan_id": plan_id,
                        "step": role_idx,
                        "operand_index": op_idx,
                        "role": op.get("role"),
                        "executor_rule": r["executor_rule"],
                        "model_family": r["model_family"],
                        "origin": origin,
                        "atom_type_index": atom_type,
                        "atom_type_symbol": _decode_element(atom_type) if atom_type is not None else None,
                        "formal_charge": formal_charge,
                        "implicit_hydrogens": implicit_h,
                        "degree": degree,
                        "bond_class_histogram": list(bond_hist) if bond_hist is not None else None,
                        "neighbor_element_histogram": list(neb_hist) if neb_hist is not None else None,
                        "created_ordinal": created_ordinal,
                        "creation_lag": creation_lag,
                    }
                )

            if role_idx == 0:
                role0_executor_rule_counter[r["executor_rule"]] += 1

        if route_created_seen:
            plans_with_route_created_operand += 1
        if nonzero_lag_seen:
            plans_with_nonzero_creation_lag += 1
        if handle_dep_seen:
            plans_with_created_handle_dependencies += 1

    n_operands_total = total_operands

    def _field_report(field: str) -> dict:
        counter = field_value_counters[field]
        present = field_presence[field]
        distinct = len(counter)
        return {
            "n_carrying_field": present,
            "n_total_operands": n_operands_total,
            "distinct_values": distinct,
            "top5": _top_values(counter, 5),
        }

    per_field_report = {f: _field_report(f) for f in DESCRIPTOR_FIELDS}

    neighbor_histogram_report = {
        "n_distinct_histograms": len(field_value_counters["neighbor_element_histogram"]),
        "top5": _top_values(field_value_counters["neighbor_element_histogram"], 5),
        "total_neighbour_count_distribution": _histogram_summary(neighbor_total_counts),
        "distinct_nonzero_elements_per_histogram_distribution": _histogram_summary(
            neighbor_distinct_element_counts
        ),
        "generic_carbon_only_count": generic_carbon_only,
        "specific_ge1_noncarbon_neighbor_count": specific_nonzero_noncarbon,
        "neither_generic_nor_specific_zero_neighbors_count": n_operands_total
        - generic_carbon_only
        - specific_nonzero_noncarbon,
    }

    bond_class_histogram_report = {
        "distinct_values": len(field_value_counters["bond_class_histogram"]),
        "top5": _top_values(field_value_counters["bond_class_histogram"], 5),
    }

    origin_report = dict(sorted(origin_counter.items()))

    creation_lag_report = _histogram_summary(creation_lag_values)

    created_ordinal_per_plan = [len(v) for v in created_ordinal_by_plan.values()]
    created_ordinal_report = {
        "distinct_ordinals_per_plan_distribution": _histogram_summary(created_ordinal_per_plan),
        "median_distinct_ordinals_per_plan": median(created_ordinal_per_plan)
        if created_ordinal_per_plan
        else None,
    }

    section3 = {
        "field_presence_and_distinct_values": per_field_report,
        "neighbor_element_histogram_detail": neighbor_histogram_report,
        "bond_class_histogram_detail": bond_class_histogram_report,
        "origin_counts": origin_report,
        "creation_lag_distribution": creation_lag_report,
        "created_ordinal_detail": created_ordinal_report,
    }

    # ================= 4. specificity / transferability proxy =================
    def _view_report(counter: Counter, full_n: int) -> dict:
        n = len(counter)
        return {
            "n_distinct_tuples": n,
            "n_total_operands": n_operands_total,
            "ratio_to_v_full": (n / full_n) if full_n else None,
        }

    n_full = len(v_full_counter)
    specificity = {
        "V_full": {
            "definition": "(origin, atom_type, formal_charge, implicit_hydrogens, degree, "
            "bond_class_histogram, neighbor_element_histogram, created_ordinal, creation_lag)",
            **_view_report(v_full_counter, n_full),
        },
        "V_drop_neb": {
            "definition": "V_full minus neighbor_element_histogram",
            **_view_report(v_drop_neb_counter, n_full),
        },
        "V_drop_lag": {
            "definition": "V_full minus creation_lag",
            **_view_report(v_drop_lag_counter, n_full),
        },
        "V_role": {
            "definition": "(origin, atom_type, formal_charge, created_ordinal)",
            **_view_report(v_role_counter, n_full),
        },
        "V_semantic": {
            "definition": "(atom_type, formal_charge)",
            **_view_report(v_semantic_counter, n_full),
        },
    }

    per_plan_full_counts = [len(v) for v in per_plan_distinct_full.values()]
    per_plan_role_counts = [len(v) for v in per_plan_distinct_role.values()]
    specificity["per_plan_distinct_tuple_counts"] = {
        "V_full": {
            "median": median(per_plan_full_counts) if per_plan_full_counts else None,
            "min": min(per_plan_full_counts) if per_plan_full_counts else None,
            "max": max(per_plan_full_counts) if per_plan_full_counts else None,
        },
        "V_role": {
            "median": median(per_plan_role_counts) if per_plan_role_counts else None,
            "min": min(per_plan_role_counts) if per_plan_role_counts else None,
            "max": max(per_plan_role_counts) if per_plan_role_counts else None,
        },
    }

    # ================= 5. step-order pinning =================
    step_order_pinning = {
        "n_plans": n_plans,
        "plans_with_ge1_route_created_operand": plans_with_route_created_operand,
        "plans_with_ge1_nonzero_creation_lag": plans_with_nonzero_creation_lag,
        "roles_with_created_output_ordinal_not_none": roles_with_created_output_ordinal,
        "total_roles": total_roles,
        "plans_with_ge1_role_with_created_handle_dependencies": plans_with_created_handle_dependencies,
    }

    # ================= 6. role 0 =================
    role0_report = {
        "n_plans": n_plans,
        "executor_rule_distribution": dict(sorted(role0_executor_rule_counter.items())),
        "operand_origin_distribution": dict(sorted(role0_operand_origin_counter.items())),
        "n_role0_operands": role0_total_operands,
        "all_role0_operands_preexisting": (
            role0_operand_origin_counter.get("preexisting", 0) == role0_total_operands
        ),
        "distinct_neighbor_element_histograms": len(role0_neighbor_hist_counter),
        "neighbor_element_histogram_top": [
            {
                "histogram_decoded": {
                    ELEMENTS[i]: c
                    for i, c in enumerate(hist)
                    if c > 0
                },
                "histogram_raw": list(hist),
                "count": count,
            }
            for hist, count in role0_neighbor_hist_counter.most_common(20)
        ],
        "field_distinct_value_counts": {
            f: {
                "n_carrying_field": sum(role0_field_value_counters[f].values()),
                "distinct_values": len(role0_field_value_counters[f]),
                "top5": _top_values(role0_field_value_counters[f], 5),
            }
            for f in DESCRIPTOR_FIELDS
        },
    }

    artifact = {
        "schema_version": "pmo_jump_plan_representation_census_v1",
        "answer_known_offline_diagnostic": True,
        "measurement_only_zero_oracle_calls": True,
        "rdkit_version": rdkit_version,
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)),
        "checkpoint_identity_sha256": computed_identity,
        "checkpoint_identity_expected_sha256": EXPECTED_IDENTITY,
        "checkpoint_identity_verified": computed_identity == EXPECTED_IDENTITY,
        "elements_vocabulary": ELEMENTS,
        "sections": {
            "1_plan_level_shape": plan_level_shape,
            "2_executor_rule_census": executor_rule_census,
            "3_operand_descriptor_field_census": section3,
            "4_specificity_transferability_proxy": specificity,
            "5_step_order_pinning": step_order_pinning,
            "6_role0_census": role0_report,
        },
        "operands": operand_rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=False))

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"wrote {OUTPUT_PATH} ({size_mb:.2f} MB)")
    print(f"n_operands_total = {n_operands_total}")
    print(json.dumps(plan_level_shape, indent=2))
    print(json.dumps(executor_rule_census, indent=2))
    print("V_full n_distinct_tuples =", specificity["V_full"]["n_distinct_tuples"])
    print("V_role n_distinct_tuples =", specificity["V_role"]["n_distinct_tuples"])
    print("V_role/V_full ratio =", specificity["V_role"]["ratio_to_v_full"])
    print(json.dumps(step_order_pinning, indent=2))


if __name__ == "__main__":
    main()
