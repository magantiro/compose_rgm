"""Guards for the unified T4 routing policy.

The load-bearing guard is ISOLATION: no target identity and no reward value may reach
the routing decision.  It is enforced three ways -- structurally (the routing accepts
two frozen dataclasses whose field sets are pinned here), textually (an AST sweep of
the module's own source), and behaviourally (states that agree on the declared
features must agree on the decision, which a rule keyed on a cell id would fail).

The blind-routing table is checked as a REGRESSION against the committed evidence
(`diagnostics/t4_state_routing_v1/blind_routing_v1.json`), recomputed from the shipped
contracts through the production chemistry -- not read back from the artifact it is
compared with.
"""

from __future__ import annotations

import ast
import json
from dataclasses import fields
from pathlib import Path

import pytest

from compose_v4.chem.molecular_graph import smiles_to_molecular_graph
from compose_v4.chem.state import pad_molecular_graph
from compose_v4.control.t4_unified_routing import (
    KERNELS,
    PROPOSAL_SLOTS,
    MolecularApplicability,
    SearchProgress,
    activated_kernel,
    applicability_mask,
    applicable_kernel_if_exhausted,
    charge_refused_region_share,
    kernel_weights,
    molecular_applicability,
    routing_decision,
    support_ramp,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/compose_v4/control/t4_unified_routing.py"
BLIND = ROOT / "diagnostics/t4_state_routing_v1/blind_routing_v1.json"

PANEL = {
    "parp1": "configs/t4_held_target_distilled_parp1_d06_250.json",
    "braf": "configs/t4_held_target_distilled_braf_d06_250.json",
    "fa7": "configs/t4_held_target_distilled_fa7_d06_250.json",
    "5ht1b": "configs/t4_held_target_distilled_5ht1b_d06_250.json",
    "jak2": "configs/t4_held_target_distilled_jak2_true_d06_250.json",
}

#: Names that would carry a target identity or an objective value into the routing.
FORBIDDEN_TOKENS = {
    "parp1", "braf", "fa7", "5ht1b", "jak2", "receptor", "protein", "target_name",
    "cell_id", "seed_index", "docking", "dock_t4", "score", "reward", "incumbent",
    "parent_score", "archive", "ivg", "best_so_far", "final_best", "utility",
}


def _sources() -> dict[str, str]:
    out = {}
    for relative in PANEL.values():
        payload = json.loads((ROOT / relative).read_text())["payload"]
        for row in payload["cells"]:
            out[row["cell"]] = row["smiles"]
    return out


def _applicability(smiles: str) -> MolecularApplicability:
    return molecular_applicability(
        pad_molecular_graph(smiles_to_molecular_graph(smiles), PROPOSAL_SLOTS)
    )


EMPTY = SearchProgress(eligible_pool_size=0, consecutive_empty_rounds=1, rounds_completed=1)
HEALTHY = SearchProgress(eligible_pool_size=8, consecutive_empty_rounds=0, rounds_completed=1)


# ---- ISOLATION: identity and reward cannot reach the decision ----------------------


def test_the_routing_inputs_carry_no_identity_and_no_value():
    """STRUCTURAL guard. The field sets are pinned, so adding a name or a score to
    either record is the only way to smuggle one in -- and it fails here."""

    assert {field.name for field in fields(MolecularApplicability)} == {
        "heavy_atoms", "growth_headroom", "net_formal_charge", "charged_centres",
        "total_regions", "executable_regions", "charge_refused_regions", "large_regions",
        "protonation_sites",
    }
    assert {field.name for field in fields(SearchProgress)} == {
        "eligible_pool_size", "consecutive_empty_rounds", "rounds_completed",
    }


def test_the_routing_functions_accept_only_those_two_records():
    import inspect

    for function in (kernel_weights, activated_kernel, routing_decision):
        parameters = list(inspect.signature(function).parameters)
        assert parameters == ["applicability", "search"], (
            f"{function.__name__} takes {parameters}; a third argument is a route by "
            "which an identity or a score could enter"
        )
    assert list(inspect.signature(applicability_mask).parameters) == ["applicability"]
    assert list(inspect.signature(molecular_applicability).parameters) == ["source"]


def test_the_module_source_names_no_target_and_no_objective_value():
    """TEXTUAL guard, over identifiers and string constants in real code.

    Docstrings are excluded deliberately: the module documents the measured
    per-protein evidence, and a guard that banned naming a protein in prose would
    force the evidence out of the file where it belongs.
    """

    tree = ast.parse(MODULE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body[0].value = ast.Constant(value="")
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            found.add(node.attr.lower())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value.lower())
    leaked = sorted(found & FORBIDDEN_TOKENS)
    assert not leaked, f"routing code references {leaked}; it must read state only"


def test_states_agreeing_on_the_declared_features_agree_on_the_decision():
    """BEHAVIOURAL guard, on the real panel.

    All three 5HT1B sources carry net charge +1 and must therefore receive the SAME
    applicable kernel. A rule that had secretly keyed on `5ht1b_2` -- the only one of
    the three that historically exhausted -- would separate them here and fail.
    """

    applicabilities = {cell: _applicability(smiles) for cell, smiles in _sources().items()}
    by_charge: dict[bool, set[str]] = {}
    for applicability in applicabilities.values():
        charged = applicability.net_formal_charge != 0
        by_charge.setdefault(charged, set()).add(
            applicable_kernel_if_exhausted(applicability)
        )
    assert by_charge[True] == {"state_aware"}
    assert by_charge[False] == {"region"}


# ---- The non-trigger guarantee ----------------------------------------------------


def test_a_healthy_pool_consults_no_alternate_kernel():
    """Byte-identical to the primary lane while the pool is non-empty."""

    for cell, smiles in _sources().items():
        applicability = _applicability(smiles)
        assert support_ramp(HEALTHY) == 0.0
        assert kernel_weights(applicability, HEALTHY) == {
            "local": 1.0, "region": 0.0, "state_aware": 0.0
        }, cell
        assert activated_kernel(applicability, HEALTHY) is None, cell


@pytest.mark.parametrize("pool", [1, 2, 8, 64])
def test_the_ramp_is_zero_for_any_non_empty_pool(pool):
    assert support_ramp(SearchProgress(pool, 0, 3)) == 0.0


def test_the_ramp_fires_on_an_empty_pool():
    """NEGATIVE CONTROL for the test above: a ramp that never fired would pass it."""

    assert support_ramp(EMPTY) == 1.0


def test_exactly_one_alternate_kernel_is_ever_applicable():
    for smiles in _sources().values():
        mask = applicability_mask(_applicability(smiles))
        assert mask["local"] == 1
        assert mask["region"] + mask["state_aware"] == 1


# ---- REGRESSION against the committed blind-routing evidence ----------------------


def _blind_rows() -> dict[str, dict]:
    payload = json.loads(BLIND.read_text())
    payload = payload.get("payload", payload)
    return {row["cell"]: row for row in payload["cells"]}


def test_the_blind_routing_table_is_reproduced_cell_by_cell():
    """Recomputed from the shipped contracts through the production chemistry, then
    compared against the committed artifact. Nothing is read back from the artifact
    to build the expectation."""

    expected = _blind_rows()
    sources = _sources()
    assert set(expected) == set(sources), "the panel and the committed table disagree"
    disagreements = []
    for cell, smiles in sorted(sources.items()):
        applicability = _applicability(smiles)
        row = expected[cell]
        if int(applicability.net_formal_charge) != int(row["net_formal_charge"]):
            disagreements.append((cell, "net_formal_charge",
                                  applicability.net_formal_charge, row["net_formal_charge"]))
        if int(applicability.heavy_atoms) != int(row["heavy_atoms"]):
            disagreements.append((cell, "heavy_atoms",
                                  applicability.heavy_atoms, row["heavy_atoms"]))
        computed = applicable_kernel_if_exhausted(applicability)
        if computed != row["applicable_kernel_if_exhausted"]:
            disagreements.append((cell, "kernel", computed,
                                  row["applicable_kernel_if_exhausted"]))
    assert not disagreements, f"blind-routing regression: {disagreements}"


def test_every_historically_fired_cell_agrees_with_its_historical_kernel():
    expected = _blind_rows()
    sources = _sources()
    fired = {cell: row for cell, row in expected.items() if row["exhausted_historically"]}
    assert len(fired) == 5, f"expected five fired cells, found {sorted(fired)}"
    for cell, row in sorted(fired.items()):
        computed = applicable_kernel_if_exhausted(_applicability(sources[cell]))
        assert computed == row["historical_kernel"], (
            f"{cell}: routing chooses {computed}, history used {row['historical_kernel']}"
        )


# ---- The graded fallback is measured, recorded, and NOT consulted -----------------


def test_the_graded_signal_separates_the_charged_cells_from_every_neutral_one():
    """MEASURED: the executor's charge policy refuses pendant excisions on exactly the
    charged sources. Recorded as the declared fallback if the binary proves brittle."""

    shares = {
        cell: charge_refused_region_share(_applicability(smiles))
        for cell, smiles in _sources().items()
    }
    charged = {cell: value for cell, value in shares.items() if cell.startswith("5ht1b")}
    neutral = {cell: value for cell, value in shares.items() if not cell.startswith("5ht1b")}
    assert min(charged.values()) > 0.0, charged
    assert max(neutral.values()) == 0.0, neutral


def test_the_graded_signal_does_not_change_any_decision():
    """It is a diagnostic. Swapping the rule after seeing a result is the failure this
    separation exists to prevent, so the mask must not read it."""

    high = MolecularApplicability(
        heavy_atoms=30, growth_headroom=10, net_formal_charge=0, charged_centres=0,
        total_regions=16, executable_regions=7, charge_refused_regions=9, large_regions=5,
        protonation_sites=2,
    )
    assert charge_refused_region_share(high) > 0.5
    assert applicability_mask(high)["region"] == 1
    assert applicability_mask(high)["state_aware"] == 0


# ---- Slot semantics ---------------------------------------------------------------


def test_a_tight_graph_is_refused_rather_than_measured():
    """48 is the SLOT capacity of a T4 proposal source; a tight graph deletes the whole
    insertion family from the legal support and would change every count here."""

    tight = smiles_to_molecular_graph("CCOc1ccccc1")
    with pytest.raises(ValueError, match="48 slots"):
        molecular_applicability(tight)


def test_the_kernel_names_are_the_committed_ones():
    assert KERNELS == ("local", "region", "state_aware")


# ---- The graded portfolio, and the confound it exists to address ------------------


def test_the_graded_portfolio_agrees_on_every_cell_that_historically_fired():
    """The five cells where a kernel actually activated are where the choice matters."""

    from compose_v4.control.t4_unified_routing import kernel_allocation_agreement

    fired = {cell for cell, row in _blind_rows().items() if row["exhausted_historically"]}
    assert len(fired) == 5
    rows = {
        cell: kernel_allocation_agreement(_applicability(smiles))
        for cell, smiles in _sources().items()
        if cell in fired
    }
    bad = {cell: row for cell, row in rows.items() if not row["agree"]}
    assert not bad, f"binary and graded allocations disagree on fired cells {sorted(bad)}"


def test_the_only_disagreement_anywhere_is_an_exact_tie_on_a_cell_that_never_fires():
    """MEASURED. `5ht1b_1` splits its ten pendant regions exactly 5 executable / 5
    charge-refused, so the graded portfolio is tied at 0.5/0.5 and resolves toward the
    region kernel while the binary says state_aware.

    It is recorded rather than smoothed away for two reasons. The cell never exhausted,
    so no kernel ever activates there and the disagreement is inert. And a tie is a
    different thing from a reversal: if a future panel produced a genuine reversal this
    test goes red, which a looser "they mostly agree" assertion would not.
    """

    from compose_v4.control.t4_unified_routing import kernel_allocation_agreement

    rows = {
        cell: kernel_allocation_agreement(_applicability(smiles))
        for cell, smiles in _sources().items()
    }
    disagreeing = {cell: row for cell, row in rows.items() if not row["agree"]}
    assert set(disagreeing) == {"5ht1b_1"}, sorted(disagreeing)
    weights = disagreeing["5ht1b_1"]["graded_weights"]
    assert weights["region"] == weights["state_aware"] == 0.5, weights
    fired = {cell for cell, row in _blind_rows().items() if row["exhausted_historically"]}
    assert "5ht1b_1" not in fired


def test_the_mechanisms_own_precondition_does_not_match_the_binary_and_this_is_recorded():
    """MEASURED, and it is the sharpest statement of the scope limit.

    The protonation kernel's OWN precondition -- an admissible tertiary-amine site,
    per `configs/t4_5ht1b2_protonation_rescue_d06_v1.json` -- holds on TEN of the
    fifteen sources, including eight neutral ones. The shipped binary keys on net
    formal charge instead, which holds on three. So the binary is not a restatement
    of the mechanism's applicability: it is a coarser feature that happens to agree
    with the historical choices on this panel.

    This test PINS that disagreement rather than hiding it. It is not a defect in the
    rule -- the rule reproduces all five historical choices -- but anyone reading the
    mask as "zero where chemically inapplicable" would be wrong, and this is where
    they find out.
    """

    from compose_v4.control.t4_unified_routing import kernel_allocation_agreement

    rows = {
        cell: kernel_allocation_agreement(_applicability(smiles))
        for cell, smiles in _sources().items()
    }
    with_sites = {cell for cell, row in rows.items() if row["protonation_sites"] > 0}
    state_aware = {cell for cell, row in rows.items() if row["binary"] == "state_aware"}
    assert state_aware < with_sites, (
        "the binary is expected to be STRICTLY narrower than the mechanism's own "
        f"precondition; state_aware={sorted(state_aware)} sites={sorted(with_sites)}"
    )
    assert len(with_sites) == 10, sorted(with_sites)
    assert len(state_aware) == 3, sorted(state_aware)
