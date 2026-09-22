"""The target-blind gating table: which kernel each T4 state selects, names hidden.

PROTOCOL (from `diagnostics/T4_FROZEN_RESULT_v1.json` -> `target_blind_gating_test`):
run the gating logic on all 30 starting states with target names hidden, docking and
reward inaccessible, the same constraint definitions, and molecular/search state
visible; report the kernel each state selects.

HOW "HIDDEN" IS ENFORCED HERE, because a promise is not a mechanism
--------------------------------------------------------------------
The routing functions accept two frozen dataclasses (`MolecularApplicability`,
`SearchProgress`) whose field sets are pinned by
`tests/test_t4_unified_routing.py`; neither can carry a name, an identifier or a
score.  This driver therefore reads the contracts to find the starting molecules, and
then passes the routing NOTHING BUT those two records.  The target label is re-attached
only when the table is printed, after every decision has been made.  A mutation
battery (`scripts/t4_routing_isolation_mutations.py`) proves the guards bind, including
one mutation that discriminates a single cell through a legitimate state feature and
names no protein at all.

WHAT IS A REGRESSION AND WHAT IS OPEN
---------------------------------------
The routing rule is already derived and committed
(`diagnostics/t4_state_routing_v1/blind_routing_v1.json`): on an empty pool,
`state_aware` if the molecule carries formal charge, else `region`.  This driver is a
REGRESSION check that the implementation reproduces that table, not a re-derivation.
Any disagreement is a defect in the implementation and is reported as one.

THE OUT-OF-PANEL CONTROL, and why the table alone is not enough
----------------------------------------------------------------
On these fifteen leads, "carries a formal charge" is perfectly confounded with "is a
5HT1B lead": all three 5HT1B sources are cations and no other source is charged.  The
panel therefore cannot by itself distinguish routing on charge from routing on
identity.  `--control-smiles` scores task-independent molecules through the same
functions so the weights can be seen tracking charge on molecules that are not T4
leads at all.

Zero oracle calls.  Zero docking.  Zero Modal.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.t4_unified_routing import (
    PROPOSAL_SLOTS,
    SCHEMA_VERSION,
    MolecularApplicability,
    SearchProgress,
    applicable_kernel_if_exhausted,
    kernel_allocation_agreement,
    kernel_weights,
    molecular_applicability,
)

ROOT = Path(__file__).resolve().parents[1]
BLIND = ROOT / "diagnostics/t4_state_routing_v1/blind_routing_v1.json"
AUDIT = ROOT / "diagnostics/t4_support_stage_audit_v1.json"

#: (protein, delta) -> contract. delta is READ FROM THE CONTRACT and verified; the
#: mapping here only says which FILE to open. `..._jak2_d06_250.json` carries
#: delta 0.4 despite its name, which is exactly why the file name is never trusted.
CONTRACTS = {
    ("parp1", 0.4): "configs/t4_held_target_distilled_parp1_d04_250.json",
    ("parp1", 0.6): "configs/t4_held_target_distilled_parp1_d06_250.json",
    ("braf", 0.4): "configs/t4_held_target_distilled_braf_d04_250.json",
    ("braf", 0.6): "configs/t4_held_target_distilled_braf_d06_250.json",
    ("fa7", 0.4): "configs/t4_held_target_distilled_fa7_d04_250.json",
    ("fa7", 0.6): "configs/t4_held_target_distilled_fa7_d06_250.json",
    ("5ht1b", 0.4): "configs/t4_held_target_distilled_5ht1b_d04_250.json",
    ("5ht1b", 0.6): "configs/t4_held_target_distilled_5ht1b_d06_250.json",
    ("jak2", 0.4): "configs/t4_held_target_distilled_jak2_d06_250.json",
    ("jak2", 0.6): "configs/t4_held_target_distilled_jak2_true_d06_250.json",
}

#: Task-independent control molecules: charged and neutral, none a T4 lead. They exist
#: to show the weights track CHARGE rather than provenance.
CONTROL_SMILES = {
    "neutral_diphenyl_ether": "c1ccc(Oc2ccccc2)cc1",
    "neutral_tertiary_amine": "CCN(CC)CCc1ccccc1",
    "cation_protonated_amine": "CC[NH+](CC)CCc1ccccc1",
    "cation_quaternary_ammonium": "C[N+](C)(C)CCc1ccccc1",
    "anion_carboxylate": "[O-]C(=O)Cc1ccc(Cl)cc1",
    "zwitterion_amino_acid": "[NH3+]C(Cc1ccccc1)C(=O)[O-]",
}

EMPTY = SearchProgress(eligible_pool_size=0, consecutive_empty_rounds=1, rounds_completed=1)
HEALTHY = SearchProgress(eligible_pool_size=8, consecutive_empty_rounds=0, rounds_completed=1)


def _payload(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())["payload"]


def _blind_reference() -> dict[str, dict]:
    payload = json.loads(BLIND.read_text())
    payload = payload.get("payload", payload)
    return {row["cell"]: row for row in payload["cells"]}


#: The audit that measures round-one pool emptiness was run against the DELTA 0.6
#: contracts. Its numbers are therefore facts about the delta 0.6 arms only, and
#: applying them to the delta 0.4 rows would manufacture a measurement that was never
#: taken. The 0.4 rows carry `null` and are excluded from the history comparison.
AUDITED_DELTA = 0.6

#: Cells whose DELTA 0.6 arm historically exhausted at round one and was rescued.
#: Taken from the committed blind-routing evidence, which is a delta 0.6 artifact.
#: Exhaustion is a per-(cell, delta) fact: at delta 0.4 only two cells were rescued
#: (fa7_2 region repair, 5ht1b_2 protonation) and no round-one audit exists for them,
#: so this driver reports the 0.4 rows as UNMEASURED rather than asserting either way.


def _round_one_pool() -> dict[str, int]:
    """Round-one eligible pool size per cell at DELTA 0.6, from the committed audit.

    This is the SEARCH-state input. It is a measurement, not an assumption, and it is
    the reason a molecular feature alone is never asked to predict which cells are
    support-limited -- heavy-atom count does not separate them (`braf_2` at 37 heavy
    atoms searches while `fa7_2` at 35 does not).
    """

    if not AUDIT.exists():
        return {}
    payload = json.loads(AUDIT.read_text())
    payload = payload.get("payload", payload)
    cells = payload.get("cells", {})
    out: dict[str, int] = {}
    if isinstance(cells, dict):
        for cell, row in cells.items():
            for key in ("selected", "selected_count", "round_one_selected"):
                if isinstance(row, dict) and key in row:
                    out[cell] = int(row[key])
                    break
            else:
                decision = (row or {}).get("round_one_decision") if isinstance(row, dict) else None
                if isinstance(decision, dict) and "selected" in decision:
                    value = decision["selected"]
                    out[cell] = len(value) if isinstance(value, list) else int(value)
    return out


def _describe(applicability: MolecularApplicability) -> dict:
    agreement = kernel_allocation_agreement(applicability)
    return {
        **applicability.as_record(),
        "kernel_if_exhausted": applicable_kernel_if_exhausted(applicability),
        "weights_empty_pool": kernel_weights(applicability, EMPTY),
        "weights_healthy_pool": kernel_weights(applicability, HEALTHY),
        "graded_kernel": agreement["graded"],
        "graded_weights": agreement["graded_weights"],
        "binary_and_graded_agree": agreement["agree"],
        "protonation_precondition_matches_binary": agreement[
            "protonation_precondition_matches_binary"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "diagnostics/t4_state_routing_v1/blind_routing_gate_v1.json")
    args = parser.parse_args()

    reference = _blind_reference()
    pools = _round_one_pool()

    rows = []
    cache: dict[str, MolecularApplicability] = {}
    for (protein, delta), relative in sorted(CONTRACTS.items()):
        payload = _payload(relative)
        declared = float(payload["delta"])
        for cell in payload["cells"]:
            smiles = cell["smiles"]
            if smiles not in cache:
                cache[smiles] = molecular_applicability(
                    pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
                )
            described = _describe(cache[smiles])
            expected = reference.get(cell["cell"], {})
            audited = abs(declared - AUDITED_DELTA) < 1e-12
            exhausted = expected.get("exhausted_historically") if audited else None
            historical = expected.get("historical_kernel") if audited else None
            rows.append(
                {
                    "cell": cell["cell"],
                    # The label is attached AFTER the decision, never before it.
                    "target_label_attached_after_decision": protein,
                    "contract_path": relative,
                    "delta_declared_by_contract": declared,
                    "delta_the_file_name_suggests": delta,
                    "delta_name_matches_contract": abs(declared - delta) < 1e-12,
                    "round_one_audit_available": audited,
                    "round_one_eligible_pool": pools.get(cell["cell"]) if audited else None,
                    "exhausted_historically": exhausted,
                    "historical_kernel": historical,
                    "agrees_with_history": (
                        None
                        if not exhausted
                        else described["kernel_if_exhausted"] == historical
                    ),
                    "reproduces_committed_table": (
                        described["kernel_if_exhausted"]
                        == expected.get("applicable_kernel_if_exhausted")
                    ),
                    **described,
                }
            )

    controls = []
    for name, smiles in sorted(CONTROL_SMILES.items()):
        applicability = molecular_applicability(
            pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
        )
        controls.append({"name": name, "smiles": smiles, **_describe(applicability)})

    fired = [row for row in rows if row["exhausted_historically"]]
    regressions = [row["cell"] for row in rows if not row["reproduces_committed_table"]]
    disagreements = [row["cell"] for row in fired if row["agrees_with_history"] is False]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "evidence_role": "target_blind_routing_regression_and_out_of_panel_control",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "new_oracle_calls": 0,
        "kernel": "rdkit 2024.03.5 (T4 production)",
        "rule": (
            "Q_t empty -> state_aware if formal charge(G) != 0 else region. "
            "No target identity, no reward."
        ),
        "isolation": (
            "the routing accepts only MolecularApplicability and SearchProgress, whose "
            "field sets are pinned by tests/test_t4_unified_routing.py and contain no "
            "name and no value; the target label is attached after every decision"
        ),
        "rows": rows,
        "out_of_panel_controls": controls,
        "regression_against_committed_table": {
            "cells": len(rows),
            "distinct_molecules": len(cache),
            "disagreeing_cells": regressions,
            "verdict": "REPRODUCED" if not regressions else "IMPLEMENTATION_DEFECT",
        },
        "agreement_on_historically_fired_cells": {
            "scope": (
                f"delta {AUDITED_DELTA} only: the committed round-one audit and the "
                "blind-routing evidence are both delta 0.6 artifacts. The delta 0.4 "
                "rows carry a routing DECISION (the molecular feature does not depend "
                "on delta) but no exhaustion measurement, and are excluded here rather "
                "than assumed. At delta 0.4 two cells were historically rescued "
                "(fa7_2 region repair, 5ht1b_2 protonation) and no round-one audit "
                "exists for them."
            ),
            "fired_cells": sorted({row["cell"] for row in fired}),
            "disagreements": sorted(set(disagreements)),
            "verdict": f"{len(fired) - len(disagreements)}/{len(fired)}",
        },
        "scope_limit": (
            "n = 5 fired cells, separated by ONE binary feature. That is a weak test "
            "which could not have failed in many ways. Do NOT claim formal charge "
            "universally determines the correct molecular-search operator. On this "
            "panel the feature is perfectly confounded with 'is a 5HT1B lead', which "
            "the out-of-panel controls address and the panel alone cannot."
        ),
        "graded_alternative": (
            "the region kernel measurably loses support where the executor's charge "
            "policy refuses its excisions; that share is recorded per row as a "
            "continuous signal and is deliberately not consulted by the shipped rule"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"{'cell':10} {'delta':>5} {'netQ':>5} {'pool':>5} {'kernel':>11} "
          f"{'graded':>11} {'hist':>11} {'ok':>4}")
    for row in rows:
        print(f"{row['cell']:10} {row['delta_declared_by_contract']:5.1f} "
              f"{row['net_formal_charge']:5d} {row['round_one_eligible_pool']!s:>5} "
              f"{row['kernel_if_exhausted']:>11} {row['graded_kernel']!s:>11} "
              f"{row['historical_kernel']!s:>11} "
              f"{'-' if row['agrees_with_history'] is None else ('yes' if row['agrees_with_history'] else 'NO'):>4}")
    print()
    for control in controls:
        print(f"CONTROL {control['name']:28} netQ={control['net_formal_charge']:+d} "
              f"kernel={control['kernel_if_exhausted']:11} "
              f"graded={control['graded_kernel']!s:11} "
              f"charge_refused={control['charge_refused_regions']}/{control['total_regions']}")
    print()
    print(json.dumps(payload["regression_against_committed_table"], indent=2))
    print(json.dumps(payload["agreement_on_historically_fired_cells"], indent=2))
    return 0 if not regressions and not disagreements else 1


if __name__ == "__main__":
    raise SystemExit(main())
