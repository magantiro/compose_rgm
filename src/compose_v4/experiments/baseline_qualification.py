"""Validation for the external-baseline qualification registry (Workstream D).

The registry answers one question per external method: *which COMPOSE claim is
this method a valid comparator for, and under what matched-budget definition?*
It is deliberately not a leaderboard and carries no results.

The invariants enforced here exist because the two ways this registry can go
wrong are both silent:

1. **An unsupported capability claim.** A cell asserting that a method does or
   does not do something, with no primary source behind it, reads exactly like a
   verified cell once it is rendered into a table. Every capability cell must
   therefore carry a primary-source ``evidence`` string, or declare itself
   ``UNVERIFIED``. ``UNVERIFIED`` cells are legal; unsourced ``YES``/``NO``
   cells are not.

2. **A COMPOSE novelty claim resting on an unverified absence.** If a baseline
   natively does something COMPOSE claims as distinctive, that is the most
   important fact the registry can carry, and it must not be discoverable only
   by reading prose. Any capability whose verdict is ``YES`` on a COMPOSE
   sub-capability axis must appear in ``compose_claims_a_baseline_does_natively``
   -- the registry refuses to validate otherwise.

The rendered markdown in ``docs/workstreams/baseline-qualification/`` is
generated from this JSON by ``scripts/render_comparator_registry.py``; the JSON
is the single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


EXPECTED_SCHEMA = "compose.baseline_qualification"
EXPECTED_SCHEMA_VERSION = 3
EXPECTED_REGISTRY_ID = "compose-external-baseline-qualification-v3"

#: Result-status vocabulary shared by every COMPOSE workstream artifact.
ARTIFACT_STATUSES = frozenset(
    {
        "DESIGN_ONLY",
        "SMOKE_HELD_IN",
        "DEVELOPMENT",
        "CONFIRMATORY_HELD_OUT",
        "SUPERSEDED",
        "INVALID_INSTRUMENT",
    }
)

#: The COMPOSE Claim-4 sub-capabilities an external method can be qualified
#: against. C1-C3 are internal-arm territory and are not listed: an external
#: method does not share the executable support or the training target, so it
#: cannot be an arm there at all.
SUBCAPABILITIES = ("C4a_static_optimization", "C4b_exact_target", "C4c_retargeting", "C4d_pathwise")

CLASSIFICATIONS = frozenset({"MUST_RUN", "CONDITIONAL", "CONTEXT_ONLY"})
VERDICTS = frozenset({"YES", "NO", "PARTIAL", "UNVERIFIED"})

#: Capability axes every method row must answer. A missing axis is a validation
#: error rather than a blank cell.
#:
#: ``restart_at_supplied_state_under_new_objective`` is separated from
#: ``dynamic_goal_switching_native`` deliberately. They are routinely conflated
#: and they are not the same capability: an iterative editor whose state is a
#: complete molecule can always be *relaunched* from a supplied molecule under a
#: different objective, and several baselines can do exactly that with stock
#: code. What COMPOSE claims is the stronger thing -- the goal changes
#: mid-trajectory, the realized history is preserved, and only the control law
#: is recomputed. Collapsing the two axes would let a real reviewer objection
#: disappear into a single cell.
REQUIRED_CAPABILITY_AXES = frozenset(
    {
        "source_conditioning",
        "objective_specific_retraining",
        "intermediate_states_exposed",
        "pathwise_constraints_native",
        "dynamic_goal_switching_native",
        "restart_at_supplied_state_under_new_objective",
    }
)

#: Axes that map onto a COMPOSE novelty claim. ``YES`` or ``PARTIAL`` on any of
#: these must be surfaced at the top of the rendered document -- as a native
#: capability if ``YES``, or as a near miss if ``PARTIAL``. Burying either is
#: the specific failure this registry exists to prevent.
NOVELTY_BEARING_AXES = {
    "pathwise_constraints_native": "C4d_pathwise",
    "dynamic_goal_switching_native": "C4c_retargeting",
    "restart_at_supplied_state_under_new_objective": "C4c_retargeting",
}

REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "paper_url",
        "code_repository",
        "code_reference",
        "license",
        "checkpoints",
        "cpu_feasible",
    }
)

REQUIRED_ORACLE_FIELDS = frozenset(
    {
        "counts_rejected_proposals",
        "training_phase_oracle_calls",
        "default_total_budget",
        "definition",
    }
)

REQUIRED_METHODS = frozenset(
    {"MARS", "GraphXForm", "DDSBM", "HN_GFN", "GraphGA", "REINVENT"}
)


class BaselineQualificationError(ValueError):
    """The qualification registry cannot support a fair external comparison."""


def load_qualification_registry(path: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text())
    validate_qualification_registry(registry)
    return registry


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineQualificationError(message)


def _validate_capability_cell(method_id: str, axis: str, cell: Any) -> None:
    _require(
        isinstance(cell, dict),
        f"{method_id}.{axis} must be an object with verdict/evidence",
    )
    verdict = cell.get("verdict")
    _require(
        verdict in VERDICTS,
        f"{method_id}.{axis} verdict {verdict!r} is not one of {sorted(VERDICTS)}",
    )
    evidence = str(cell.get("evidence") or "").strip()
    if verdict == "UNVERIFIED":
        # An honest gap still has to say what was looked at, otherwise
        # "UNVERIFIED" becomes a way to skip the work silently.
        _require(
            bool(evidence),
            f"{method_id}.{axis} is UNVERIFIED and must record what was searched",
        )
        return
    _require(
        bool(evidence),
        f"{method_id}.{axis} asserts {verdict} with no primary source; "
        "use UNVERIFIED instead of an unsourced assertion",
    )


def _validate_provenance(method_id: str, provenance: Any) -> None:
    _require(isinstance(provenance, dict), f"{method_id} lacks a provenance block")
    missing = REQUIRED_PROVENANCE_FIELDS - set(provenance)
    _require(not missing, f"{method_id} provenance omits {sorted(missing)}")
    for field in ("paper_url", "code_repository"):
        value = str(provenance.get(field) or "")
        if value in {"", "NONE"}:
            continue
        parsed = urlparse(value)
        _require(
            parsed.scheme == "https" and bool(parsed.netloc),
            f"{method_id} provenance.{field} is not an https URL: {value!r}",
        )
    license_value = str(provenance.get("license") or "").strip()
    _require(
        bool(license_value),
        f"{method_id} provenance.license must be stated (use UNVERIFIED if the "
        "repository declares none)",
    )


def validate_qualification_registry(registry: dict[str, Any]) -> None:
    _require(registry.get("schema") == EXPECTED_SCHEMA, "unexpected qualification schema")
    _require(
        registry.get("schema_version") == EXPECTED_SCHEMA_VERSION,
        "unexpected qualification schema version",
    )
    _require(
        registry.get("registry_id") == EXPECTED_REGISTRY_ID,
        "unexpected qualification registry identity",
    )
    _require(
        registry.get("artifact_status") in ARTIFACT_STATUSES,
        "registry must declare one of the shared artifact statuses",
    )
    _require(
        registry.get("held_out_data_opened") is False,
        "this lane must declare explicitly that no held-out data were opened",
    )

    # "Future-aware control helps" is established on hard exact-target recovery
    # and measured ABSENT on an easy target-free property goal. A registry that
    # does not carry both halves invites a comparison that reads as testing a
    # claim COMPOSE does not make -- in either direction.
    # The retargeting claim's exact wording is project-wide and binding. It lives
    # here so the registry and the paper cannot drift; "without retraining" is
    # barred because REINVENT 4 carries its optimizer state across a stage
    # boundary, which makes the phrase arguable.
    wording = registry.get("retargeting_claim_wording") or {}
    for field in (
        "operational_claim",
        "barred_phrase",
        "mol2mol_belongs_beside_it",
        "mars_naming_rule",
    ):
        _require(
            bool(wording.get(field)),
            f"retargeting_claim_wording omits {field}",
        )
    claim = str(wording.get("operational_claim", "")).lower()
    # Both phrases are barred for the same reason: each asserts something the
    # experiments have not established. "without retraining" is arguable because
    # REINVENT's optimizer state survives a stage boundary; "at an arbitrary
    # step" claims switch-time invariance, and the scalable experiment fixes
    # tau=3, H=6.
    for barred in ("without retraining", "at an arbitrary step"):
        _require(
            barred not in claim,
            f"the operational claim must not reuse the barred phrase {barred!r}",
        )
    _require(
        bool(wording.get("sanctioned_phrasing")),
        "retargeting_claim_wording must record the sanctioned phrasing",
    )

    # "Outperforms state of the art" is barred project-wide: there will be no
    # broad apples-to-apples sweep, and the claim is not needed. Checked over
    # the WHOLE registry, not one field, because the phrase is most likely to
    # appear in a method note written in passing.
    presentation = registry.get("manuscript_presentation") or {}
    # The field that DECLARES the ban has to quote the phrase, so exempt it --
    # otherwise the guard fires on its own prohibition. Everything else is
    # checked, because the phrase is most likely to slip into a method note
    # written in passing rather than into a field anyone reviews.
    searchable = dict(registry)
    searchable["manuscript_presentation"] = {
        k: v for k, v in presentation.items() if k != "barred_claim"
    }
    serialized = json.dumps(searchable).lower()
    for barred in ("outperforms state of the art", "outperforms the state of the art"):
        _require(
            barred not in serialized,
            f"the barred claim {barred!r} appears in the registry; the sanctioned "
            "framing is competitive optimization plus stateful inference-time control",
        )

    for field in ("rule", "reviewer_facing_hierarchy", "success_criterion"):
        _require(
            bool(presentation.get(field)),
            f"manuscript_presentation omits {field}; qualify broadly, present narrowly",
        )

    scoping = registry.get("compose_claim_scoping") or {}
    for field in (
        "headline",
        "where_it_is_established",
        "where_it_is_measured_absent",
        "consequences_for_every_comparison_in_this_registry",
    ):
        _require(
            bool(scoping.get(field)),
            f"compose_claim_scoping omits {field}; both the positive and the "
            "negative result must be carried, not just the flattering one",
        )
    for half in ("where_it_is_established", "where_it_is_measured_absent"):
        _require(
            bool((scoping[half] or {}).get("artifact")),
            f"compose_claim_scoping.{half} must name the diagnostics artifact "
            "that established it",
        )

    methods = registry.get("methods") or []
    ids = [str(row.get("id")) for row in methods]
    _require(len(ids) == len(set(ids)), "duplicate method identifiers")
    missing_methods = REQUIRED_METHODS - set(ids)
    _require(not missing_methods, f"registry omits required methods {sorted(missing_methods)}")

    escalated = {
        (str(row.get("method")), str(row.get("subcapability")))
        for row in registry.get("compose_claims_a_baseline_does_natively") or []
    }
    near_missed = {
        (str(row.get("method")), str(row.get("axis")))
        for row in registry.get("near_misses") or []
    }

    for method in methods:
        method_id = str(method.get("id"))
        _require(bool(method.get("native_task")), f"{method_id} lacks a native-task statement")
        _require(bool(method.get("citation_key")), f"{method_id} lacks a citation key")
        _validate_provenance(method_id, method.get("provenance"))

        capabilities = method.get("capabilities") or {}
        missing_axes = REQUIRED_CAPABILITY_AXES - set(capabilities)
        _require(not missing_axes, f"{method_id} omits capability axes {sorted(missing_axes)}")
        for axis, cell in capabilities.items():
            _validate_capability_cell(method_id, axis, cell)

        oracle = method.get("oracle_accounting") or {}
        missing_oracle = REQUIRED_ORACLE_FIELDS - set(oracle)
        _require(
            not missing_oracle,
            f"{method_id} oracle_accounting omits {sorted(missing_oracle)}",
        )

        _require(
            bool(method.get("edit_budget")),
            f"{method_id} must state an edit/generation budget (use N/A with a reason)",
        )
        adapter = method.get("adapter") or {}
        _require(bool(adapter.get("status")), f"{method_id} lacks an adapter status")

        classification = method.get("classification") or {}
        missing_caps = set(SUBCAPABILITIES) - set(classification)
        _require(
            not missing_caps,
            f"{method_id} classification omits sub-capabilities {sorted(missing_caps)}",
        )
        for subcap, value in classification.items():
            _require(
                value in CLASSIFICATIONS,
                f"{method_id}.{subcap} classification {value!r} is invalid",
            )
        _require(
            method.get("method_verdict") in CLASSIFICATIONS,
            f"{method_id} lacks a valid aggregate verdict",
        )
        # The aggregate verdict is the strongest per-sub-capability verdict; it
        # must not be quietly softer than the rows it summarises.
        order = {"CONTEXT_ONLY": 0, "CONDITIONAL": 1, "MUST_RUN": 2}
        strongest = max(order[value] for value in classification.values())
        _require(
            order[str(method["method_verdict"])] == strongest,
            f"{method_id} aggregate verdict disagrees with its sub-capability rows",
        )

        _require(
            bool(method.get("fairness_contract")),
            f"{method_id} lacks a fairness contract",
        )
        if method["method_verdict"] == "MUST_RUN":
            contract = method["fairness_contract"]
            for field in ("matched_quantity", "not_fair_because"):
                _require(
                    bool(contract.get(field)),
                    f"MUST_RUN method {method_id} fairness_contract omits {field}",
                )

        for axis, subcap in NOVELTY_BEARING_AXES.items():
            verdict = capabilities[axis]["verdict"]
            if verdict == "YES":
                _require(
                    (method_id, subcap) in escalated,
                    f"{method_id} natively supports {axis} but is not escalated in "
                    "compose_claims_a_baseline_does_natively",
                )
            elif verdict == "PARTIAL":
                _require(
                    (method_id, axis) in near_missed or (method_id, subcap) in escalated,
                    f"{method_id} partially supports {axis} and must appear in "
                    "near_misses with the reason it is not a native match",
                )

    for row in registry.get("compose_claims_a_baseline_does_natively") or []:
        _require(
            str(row.get("method")) in set(ids),
            f"escalation names unknown method {row.get('method')!r}",
        )
        _require(
            str(row.get("subcapability")) in set(SUBCAPABILITIES),
            f"escalation names unknown sub-capability {row.get('subcapability')!r}",
        )
        _require(
            bool(row.get("evidence")),
            "every escalation must carry the primary source that established it",
        )
        # An escalation without the counterweight is half a finding, and the
        # missing half is the one that decides what the paper may claim.
        _require(
            bool(row.get("what_compose_still_has")),
            f"escalation for {row.get('method')} must state what COMPOSE still "
            "has once the baseline's native capability is granted",
        )

    for row in registry.get("near_misses") or []:
        _require(
            str(row.get("method")) in set(ids),
            f"near miss names unknown method {row.get('method')!r}",
        )
        for field in ("axis", "why_not", "evidence"):
            _require(bool(row.get(field)), f"near miss for {row.get('method')} omits {field}")

    smoke = registry.get("smoke_plan") or {}
    _require(bool(smoke.get("sources")), "registry must state the held-in smoke source count")
    _require(
        smoke.get("executed") is False,
        "no smoke has been executed in this lane; the plan must say so",
    )
