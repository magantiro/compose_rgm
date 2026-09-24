"""Does the historical T4 panel qualify as replicate 1 of the unified policy?

This consolidates evidence that already exists rather than producing any of its
own, and it READS the committed artifacts so it cannot drift from them.  Every
number here is traceable to a file, and the script fails rather than reporting a
verdict if an input is missing.

The criterion is branch-level, not implementation-level: does the unified
controller select and execute the same molecular process that produced each
historical row?  Wrapper module, contract hash, controller fingerprint and code
revision are NOT criteria -- they differ because the mechanisms were discovered
incrementally, which is a fact about the code's history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EQUIVALENCE = ROOT / "diagnostics/t4_behavioral_equivalence/audit_v1.json"
GATE = ROOT / "diagnostics/t4_unified_production_gate/t4_unified_production_gate_v1.json"
MANIFEST = ROOT / "diagnostics/t4_replication_manifest/manifest_v1.json"
TARGETS = ("5ht1b", "braf", "fa7", "jak2", "parp1")


def _payload(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"required evidence is missing: {path}")
    data = json.loads(path.read_text())
    return data.get("payload", data)


def _seed_correspondence() -> dict:
    """Do the sealed unified arms carry replicate 1's seeds, cell by cell?"""

    manifest = _payload(MANIFEST)
    historical = {
        (row["cell"], row["delta"]): row["controller_seed"] for row in manifest["rows"]
    }
    checked = matched = 0
    mismatches = []
    for target in TARGETS:
        for tag, delta in (("d04", 0.4), ("d06", 0.6)):
            path = ROOT / f"configs/t4_unified_controller_{target}_{tag}_v1.json"
            if not path.exists():
                continue
            for cell in _payload(path)["cells"]:
                key = (cell["cell"], delta)
                if key not in historical:
                    continue
                checked += 1
                if historical[key] == cell["controller_seed"]:
                    matched += 1
                else:
                    mismatches.append(
                        {
                            "cell": cell["cell"],
                            "delta": delta,
                            "historical": historical[key],
                            "unified": cell["controller_seed"],
                        }
                    )
    return {
        "cells_checked": checked,
        "cells_matching": matched,
        "mismatches": mismatches,
        "meaning": (
            "replicate 1 is not re-seeded: the sealed unified arms already carry "
            "the seed each historical row ran under"
        ),
    }


def main() -> int:
    equivalence = _payload(EQUIVALENCE)
    gate = _payload(GATE)
    seeds = _seed_correspondence()

    coverage = gate["coverage"]
    blocker = (
        "the production gate verdict is INCOMPLETE: trigger rows "
        f"{coverage['trigger_rows_present']}/{coverage['trigger_rows_expected']}, "
        f"control rows {coverage['control_rows_present']}/"
        f"{coverage['control_rows_expected']}, controls_run_declared="
        f"{gate['headline']['controls_run_declared']}"
    )
    branch_ok = equivalence["verdict"] == "ALL_FIRED_ROWS_BEHAVIORALLY_EQUIVALENT"
    seeds_ok = seeds["cells_checked"] > 0 and not seeds["mismatches"]
    gate_complete = gate["verdict"] != "INCOMPLETE"

    qualifies = branch_ok and seeds_ok
    report = {
        "schema_version": "t4_replicate_one_qualification_v1",
        "new_oracle_calls": 0,
        "docking_calls": 0,
        "question": (
            "does the historical T4 panel qualify as replicate 1 of the unified "
            "state-adaptive policy?"
        ),
        "criterion": (
            "branch-level equivalence: the unified router selects and executes the "
            "same molecular process from the same state. Implementation identity "
            "(wrapper, contract hash, fingerprint, revision) is NOT a criterion."
        ),
        "branch_equivalence": {
            "source": str(EQUIVALENCE.relative_to(ROOT)),
            "rows_audited": equivalence["rows_audited"],
            "fired_rows": equivalence["fired_rows"],
            "agreement_on_fired_rows": equivalence["agreement_on_fired_rows"],
            "primary_path_rows": equivalence["primary_path_rows"],
            "verdict": equivalence["verdict"],
            "scope_limit": equivalence["scope_limit"]["warning"],
        },
        "seed_correspondence": seeds,
        "production_consumption": {
            "source": str(GATE.relative_to(ROOT)),
            "entrypoint": gate.get("entrypoint"),
            "oracle_calls": gate.get("oracle_calls"),
            "docking_calls": gate.get("docking_calls"),
            "routing_agrees_with_every_historical_arm": gate["headline"][
                "routing_agrees_with_every_historical_arm"
            ],
            "state_routing_proven_consumed_on_every_trigger_event": gate[
                "consumption"
            ]["state_routing_proven_consumed_on_every_trigger_event"],
            "terminal_guard_passed_on_every_trigger_event": gate["consumption"][
                "terminal_guard_passed_on_every_trigger_event"
            ],
            "trigger_cells_passing": gate["headline"]["trigger_cells_passing"],
            "ladder_unreachable_on_every_control": gate["headline"][
                "ladder_unreachable_on_every_control"
            ],
            "alternate_kernel_never_consulted_on_a_control": gate["headline"][
                "alternate_kernel_never_consulted_on_a_control"
            ],
            "controls_law_off_reproduce_committed_audit": gate["headline"][
                "controls_law_off_reproduce_committed_audit"
            ],
            "coverage": gate["coverage"],
            "gate_verdict": gate["verdict"],
        },
        "what_is_claimed": (
            "the unified state-adaptive controller SUBSUMES the execution rules "
            "that produced the historical panel"
        ),
        "what_is_NOT_claimed": (
            "that every historical cell was literally executed through "
            "modal_apps/t4_unified_controller_app.py -- it was not, and the "
            "replication manifest records the fifteen arms that did run it"
        ),
        "qualifies_as_replicate_1": qualifies,
        "blocking_before_scoring": [] if gate_complete else [blocker],
    }

    destination = ROOT / "diagnostics/t4_replicate_one_qualification/qualification_v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"qualifies_as_replicate_1: {qualifies}")
    print(f"  branch equivalence : {equivalence['agreement_on_fired_rows']} fired rows")
    print(f"  seed correspondence: {seeds['cells_matching']}/{seeds['cells_checked']}")
    print(f"  gate verdict       : {gate['verdict']}  coverage={gate['coverage']}")
    for item in report["blocking_before_scoring"]:
        print(f"  BLOCKING: {item}")
    print(f"written: {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
