"""A synthetic exact-slot closure tests the read-only sampled-path auditor."""

from copy import deepcopy

import pytest

from compose_v4.control import region_rewrite as RR
from compose_v4.rewrite.kernel import canonical_state_key, editing_v2_rewrite_system
from compose_v4.rewrite.trace_shard import decode_state
from tests.test_t4_append_contract_audit import fixture
from tools.t4_program_replay_audit import verify_transition


def inputs():
    attempt = {**fixture()["rows"][0], "status": "executed"}
    source = decode_state(attempt["source"])
    transition = {k: attempt[k] for k in ("source", "product")}
    transition["canonical_product"] = canonical_state_key(decode_state(attempt["product"]))
    context = RR.RewriteContext(
        frozenset(range(6)), frozenset(range(6, 12)), ((5, 6, 1.0),), "pendant", 1
    )
    return transition, attempt, source, context


def test_valid_recorded_closure_replays_without_model_or_oracle():
    transition, attempt, source, context = inputs()
    assert (
        verify_transition(
            transition, attempt, "append_system", source, context, editing_v2_rewrite_system()
        )
        == context
    )


@pytest.mark.parametrize("defect", ["ledger", "product", "macro", "canonical"])
def test_corrupt_path_fails_closed(defect):
    transition, attempt, source, context = deepcopy(inputs())
    macro = "append_system"
    if defect == "ledger":
        attempt["status"] = "invalid_rewrite"
    elif defect == "product":
        transition["product"] = attempt["product"] = attempt["source"]
    elif defect == "macro":
        macro = "grow"
    else:
        transition["canonical_product"] = "CC"
    with pytest.raises(ValueError):
        verify_transition(transition, attempt, macro, source, context, editing_v2_rewrite_system())
